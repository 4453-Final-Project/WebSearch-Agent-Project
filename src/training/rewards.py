from __future__ import annotations

from dataclasses import dataclass

from src.agent.compat import normalize_observation
from src.agent.prompting import build_full_prompt

from .warmup import build_compact_warmup_prompt
from .trajectory import EpisodeTrajectory, OptimizationSample


@dataclass(slots=True)
class RewardConfig:
    """Reward shaping knobs used for single-task policy optimization."""

    reward_scale: float = 1.0
    success_bonus: float = 1.0
    invalid_action_penalty: float = 0.15
    parse_failure_penalty: float = 0.10
    truncation_penalty: float = 0.05
    repeat_action_penalty: float = 0.02


def compute_episode_score(trajectory: EpisodeTrajectory, config: RewardConfig) -> float:
    """Compute a shaped scalar score for one episode."""

    score = trajectory.reward * config.reward_scale
    if trajectory.success:
        score += config.success_bonus
    score -= trajectory.invalid_action_count * config.invalid_action_penalty
    score -= trajectory.parse_failure_count * config.parse_failure_penalty
    if trajectory.truncated and not trajectory.success:
        score -= config.truncation_penalty
    score -= _count_repeated_actions(trajectory) * config.repeat_action_penalty
    return float(score)


def build_optimization_samples(
    trajectory: EpisodeTrajectory,
    *,
    group_id: int,
    episode_score: float,
    advantage: float,
    model_system_prompt: str | None = None,
) -> list[OptimizationSample]:
    """Convert a trajectory into prompt/response training samples."""

    samples: list[OptimizationSample] = []
    episode_id = f"seed-{trajectory.seed}"
    step_count = max(1, sum(1 for step in trajectory.steps if step.response_text))
    sample_weight = 1.0 / step_count
    for step in trajectory.steps:
        if not step.response_text:
            continue
        prompt = step.prompt
        if step.observation:
            compact_prompt = build_compact_warmup_prompt(
                step,
                model_system_prompt=model_system_prompt,
            )
            prompt = compact_prompt or build_full_prompt(
                normalize_observation(step.observation),
                step.step_idx,
                model_system_prompt=model_system_prompt,
            )
        samples.append(
            OptimizationSample(
                prompt=prompt,
                response_text=step.response_text,
                action_text=step.action_text,
                step_idx=step.step_idx,
                group_id=group_id,
                episode_id=episode_id,
                episode_score=episode_score,
                advantage=advantage,
                sample_weight=sample_weight,
                env_reward=trajectory.reward,
                step_reward=step.reward,
                success=trajectory.success,
                invalid_action=step.last_action_error is not None,
                parse_failed=step.parse_error is not None,
            )
        )
    return samples


def _count_repeated_actions(trajectory: EpisodeTrajectory) -> int:
    repeated = 0
    previous_action: str | None = None
    for step in trajectory.steps:
        action = (step.action_text or "").strip()
        if not action:
            continue
        if action == previous_action:
            repeated += 1
        previous_action = action
    return repeated
