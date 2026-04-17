from __future__ import annotations

from dataclasses import dataclass

from src.agent.compat import normalize_observation
from src.agent.prompting import build_full_prompt

from .warmup import build_compact_warmup_prompt
from .trajectory import EpisodeTrajectory, OptimizationSample, TrajectoryStep, compute_step_sample_weights


@dataclass(slots=True)
class RewardConfig:
    """Reward shaping knobs used for single-task policy optimization."""

    reward_scale: float = 1.0
    success_bonus: float = 1.0
    success_step_bonus: float = 0.05
    invalid_action_penalty: float = 0.15
    parse_failure_penalty: float = 0.10
    truncation_penalty: float = 0.05
    repeat_action_penalty: float = 0.02
    same_page_repeat_action_penalty: float = 0.03
    per_step_penalty: float = 0.01
    success_unique_url_bonus: float = 0.02
    max_unique_url_bonus_urls: int = 4


def compute_episode_score(trajectory: EpisodeTrajectory, config: RewardConfig) -> float:
    """Compute a shaped scalar score for one episode."""

    return float(compute_episode_reward_breakdown(trajectory, config)["total_score"])


def compute_episode_reward_breakdown(
    trajectory: EpisodeTrajectory,
    config: RewardConfig,
) -> dict[str, float | int]:
    """Return the shaped reward plus the component counts used to build it."""

    repeated_action_count = _count_repeated_actions(trajectory)
    same_page_repeat_action_count = _count_same_page_repeated_actions(trajectory)
    unique_url_count = _count_unique_urls(trajectory)
    success_step_bonus = (
        _compute_success_step_bonus(trajectory.steps_taken, config.success_step_bonus)
        if trajectory.success
        else 0.0
    )
    success_unique_url_bonus = (
        _compute_success_unique_url_bonus(
            unique_url_count,
            config.success_unique_url_bonus,
            config.max_unique_url_bonus_urls,
        )
        if trajectory.success
        else 0.0
    )
    truncation_penalty = config.truncation_penalty if trajectory.truncated and not trajectory.success else 0.0
    per_step_penalty = max(0, trajectory.steps_taken - 1) * config.per_step_penalty
    invalid_action_penalty = trajectory.invalid_action_count * config.invalid_action_penalty
    parse_failure_penalty = trajectory.parse_failure_count * config.parse_failure_penalty
    repeat_action_penalty = repeated_action_count * config.repeat_action_penalty
    same_page_repeat_action_penalty = same_page_repeat_action_count * config.same_page_repeat_action_penalty

    score = trajectory.reward * config.reward_scale
    if trajectory.success:
        score += config.success_bonus
        score += success_step_bonus
        score += success_unique_url_bonus
    score -= invalid_action_penalty
    score -= parse_failure_penalty
    score -= truncation_penalty
    score -= repeat_action_penalty
    score -= same_page_repeat_action_penalty
    score -= per_step_penalty

    return {
        "base_reward": float(trajectory.reward * config.reward_scale),
        "success_bonus": float(config.success_bonus if trajectory.success else 0.0),
        "success_step_bonus": float(success_step_bonus),
        "unique_url_count": unique_url_count,
        "success_unique_url_bonus": float(success_unique_url_bonus),
        "invalid_action_count": trajectory.invalid_action_count,
        "invalid_action_penalty": float(invalid_action_penalty),
        "parse_failure_count": trajectory.parse_failure_count,
        "parse_failure_penalty": float(parse_failure_penalty),
        "truncation_penalty": float(truncation_penalty),
        "repeat_action_count": repeated_action_count,
        "repeat_action_penalty": float(repeat_action_penalty),
        "same_page_repeat_action_count": same_page_repeat_action_count,
        "same_page_repeat_action_penalty": float(same_page_repeat_action_penalty),
        "per_step_penalty": float(per_step_penalty),
        "total_score": float(score),
    }


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
    weighted_steps = compute_step_sample_weights(trajectory)
    for step, sample_weight in weighted_steps:
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


def _count_same_page_repeated_actions(trajectory: EpisodeTrajectory) -> int:
    repeated = 0
    previous_step: TrajectoryStep | None = None
    for step in trajectory.steps:
        action = (step.action_text or "").strip()
        current_url = _step_current_url(step)
        if (
            previous_step is not None
            and action
            and action == (previous_step.action_text or "").strip()
            and current_url
            and current_url == _step_current_url(previous_step)
            and not action.startswith("send_msg_to_user(")
            and not previous_step.done
        ):
            repeated += 1
        previous_step = step
    return repeated


def _count_unique_urls(trajectory: EpisodeTrajectory) -> int:
    urls = {_step_current_url(step) for step in trajectory.steps if _step_current_url(step)}
    return len(urls)


def _step_current_url(step: TrajectoryStep) -> str:
    observation = step.observation or {}
    current_url = observation.get("current_url") or observation.get("url") or ""
    return str(current_url).strip()


def _compute_success_step_bonus(steps_taken: int, success_step_bonus: float) -> float:
    if success_step_bonus <= 0.0:
        return 0.0
    normalized_steps = max(1, steps_taken)
    return success_step_bonus / normalized_steps


def _compute_success_unique_url_bonus(
    unique_url_count: int,
    success_unique_url_bonus: float,
    max_unique_url_bonus_urls: int,
) -> float:
    if success_unique_url_bonus <= 0.0 or max_unique_url_bonus_urls <= 1:
        return 0.0
    rewarded_unique_urls = max(0, min(unique_url_count, max_unique_url_bonus_urls) - 1)
    return rewarded_unique_urls * success_unique_url_bonus

