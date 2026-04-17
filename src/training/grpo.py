from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .checkpointing import CheckpointManager
from .collector import RolloutCollector
from .ppo import compute_clipped_policy_objective
from .rewards import (
    RewardConfig,
    build_optimization_samples,
    compute_episode_reward_breakdown,
)
from .trajectory import EpisodeTrajectory, OptimizationSample


@dataclass(slots=True)
class GRPOConfig:
    """Configuration for single-task grouped relative policy optimization."""

    iterations: int = 1
    ppo_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    clip_range: float = 0.2
    beta: float = 0.01
    max_grad_norm: float = 1.0


def compute_group_advantages(scores: list[float]) -> list[float]:
    """Normalize episode scores within a rollout group."""

    if not scores:
        return []
    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / max(1, len(scores))
    std = math.sqrt(variance + 1e-8)
    if std <= 1e-6:
        return [0.0 for _ in scores]
    return [(score - mean_score) / std for score in scores]

class GRPOTrainer:
    """Small grouped-relative optimizer on top of Task 2 rollouts and Task 3 policy."""

    def __init__(
        self,
        *,
        policy: Any,
        collector: RolloutCollector,
        reward_config: RewardConfig,
        config: GRPOConfig,
        checkpoint_manager: CheckpointManager,
        eval_callback: Any | None = None,
    ) -> None:
        self.policy = policy
        self.collector = collector
        self.reward_config = reward_config
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.eval_callback = eval_callback

    def train(self) -> list[dict[str, Any]]:
        """Run all configured iterations and return iteration-level metrics."""

        iteration_results: list[dict[str, Any]] = []
        optimizer_step = 0

        for iteration_idx in range(self.config.iterations):
            grouped_episodes = self.collector.collect_iteration(self.policy, iteration_idx)
            samples, rollout_summary = self._build_samples(grouped_episodes)
            if not samples:
                raise RuntimeError("Collected rollouts did not produce any trainable prompt/response samples.")

            # Rollout collection can leave CUDA memory fragmented; clean up before PPO scoring.
            self._cleanup_policy_memory()
            old_logprobs = self._score_samples(samples, use_reference=False, requires_grad=False).detach().cpu()
            reference_logprobs = self._score_samples(samples, use_reference=True, requires_grad=False).detach().cpu()
            self._cleanup_policy_memory()

            latest_loss = 0.0
            latest_reference_penalty = 0.0
            latest_ratio_mean = 1.0
            latest_approx_kl = 0.0
            accumulation_steps = max(1, self.config.gradient_accumulation_steps)

            for _ in range(self.config.ppo_epochs):
                self.policy.zero_grad()
                pending_steps = 0
                for batch_start in range(0, len(samples), self.config.batch_size):
                    batch_end = batch_start + self.config.batch_size
                    batch_samples = samples[batch_start:batch_end]
                    current_logprobs = self._score_samples(batch_samples, use_reference=False)
                    device = current_logprobs.device
                    batch_old = old_logprobs[batch_start:batch_end].to(device)
                    batch_reference = reference_logprobs[batch_start:batch_end].to(device)
                    batch_advantages = torch.tensor(
                        [sample.advantage for sample in batch_samples],
                        dtype=torch.float32,
                        device=device,
                    )
                    batch_weights = torch.tensor(
                        [sample.sample_weight for sample in batch_samples],
                        dtype=torch.float32,
                        device=device,
                    )
                    ppo_stats = compute_clipped_policy_objective(
                        current_logprobs=current_logprobs,
                        old_logprobs=batch_old,
                        advantages=batch_advantages,
                        clip_range=self.config.clip_range,
                        sample_weights=batch_weights,
                    )
                    normalized_weights = batch_weights / batch_weights.sum().clamp(min=1e-8)
                    reference_penalty = ((current_logprobs - batch_reference) ** 2 * normalized_weights).sum()
                    loss = ppo_stats.loss + (self.config.beta * reference_penalty)
                    scaled_loss = loss / accumulation_steps

                    scaled_loss.backward()
                    pending_steps += 1
                    is_last_batch = batch_end >= len(samples)
                    if pending_steps >= accumulation_steps or is_last_batch:
                        self.policy.clip_grad_norm_(self.config.max_grad_norm)
                        self.policy.step_optimizer()
                        self.policy.zero_grad()
                        optimizer_step += 1
                        pending_steps = 0
                        self._cleanup_policy_memory()

                    latest_loss = float(loss.detach().item())
                    latest_reference_penalty = float(reference_penalty.detach().item())
                    latest_ratio_mean = ppo_stats.ratio_mean
                    latest_approx_kl = ppo_stats.approx_kl

            metrics = {
                "iteration": iteration_idx,
                "policy_version": getattr(self.policy, "name", "qwen-grpo"),
                "episode_reward": rollout_summary["average_episode_score"],
                "success_rate": rollout_summary["success_rate"],
                "step_count": rollout_summary["step_count"],
                "invalid_actions": rollout_summary["invalid_actions"],
                "parse_failures": rollout_summary["parse_failures"],
                "repeat_actions": rollout_summary["repeat_actions"],
                "same_page_repeat_actions": rollout_summary["same_page_repeat_actions"],
                "unique_urls": rollout_summary["unique_urls"],
                "reference_penalty": latest_reference_penalty,
                "approx_kl": latest_approx_kl,
                "ratio_mean": latest_ratio_mean,
                "loss": latest_loss,
            }
            checkpoint_record = self.checkpoint_manager.save(
                self.policy,
                iteration_idx=iteration_idx,
                optimizer_step=optimizer_step,
                metrics=metrics,
            )
            metrics["checkpoint_path"] = checkpoint_record.checkpoint_path

            if self.eval_callback is not None:
                metrics["eval_metrics"] = self.eval_callback(iteration_idx, checkpoint_record.checkpoint_path)

            iteration_results.append(metrics)

        return iteration_results

    def _build_samples(self, grouped_episodes: list[list[EpisodeTrajectory]]) -> tuple[list[OptimizationSample], dict[str, Any]]:
        samples: list[OptimizationSample] = []
        all_scores: list[float] = []
        success_count = 0
        total_episodes = 0
        step_count = 0
        invalid_actions = 0
        parse_failures = 0
        repeat_actions = 0
        same_page_repeat_actions = 0
        unique_urls = 0

        for group_idx, group in enumerate(grouped_episodes):
            group_breakdowns = [compute_episode_reward_breakdown(episode, self.reward_config) for episode in group]
            group_scores = [float(breakdown["total_score"]) for breakdown in group_breakdowns]
            group_advantages = compute_group_advantages(group_scores)
            model_system_prompt = getattr(getattr(self.policy, "config", None), "system_prompt", None)
            for episode, episode_score, advantage, reward_breakdown in zip(
                group,
                group_scores,
                group_advantages,
                group_breakdowns,
            ):
                samples.extend(
                    build_optimization_samples(
                        episode,
                        group_id=group_idx,
                        episode_score=episode_score,
                        advantage=advantage,
                        model_system_prompt=model_system_prompt,
                    )
                )
                all_scores.append(episode_score)
                success_count += int(episode.success)
                total_episodes += 1
                step_count += episode.steps_taken
                invalid_actions += episode.invalid_action_count
                parse_failures += episode.parse_failure_count
                repeat_actions += int(reward_breakdown["repeat_action_count"])
                same_page_repeat_actions += int(reward_breakdown["same_page_repeat_action_count"])
                unique_urls += int(reward_breakdown["unique_url_count"])

        summary = {
            "average_episode_score": sum(all_scores) / max(1, len(all_scores)),
            "success_rate": success_count / max(1, total_episodes),
            "step_count": step_count,
            "invalid_actions": invalid_actions,
            "parse_failures": parse_failures,
            "repeat_actions": repeat_actions,
            "same_page_repeat_actions": same_page_repeat_actions,
            "unique_urls": unique_urls,
        }
        return samples, summary

    def _score_samples(
        self,
        samples: list[OptimizationSample],
        *,
        use_reference: bool,
        requires_grad: bool = True,
    ) -> torch.Tensor:
        prompts = [sample.prompt for sample in samples]
        responses = [sample.response_text for sample in samples]
        return self.policy.score_responses(
            prompts,
            responses,
            use_reference=use_reference,
            requires_grad=requires_grad,
        )

    def _cleanup_policy_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
