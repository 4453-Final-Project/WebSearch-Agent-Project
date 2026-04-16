from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.agent.compat import normalize_observation
from src.agent.prompting import build_full_prompt


@dataclass(slots=True)
class TrajectoryStep:
    """One runner-emitted environment step converted into training-friendly data."""

    step_idx: int
    prompt: str
    response_text: str
    action_text: str
    raw_text: str
    parse_error: str | None
    last_action_error: str | None
    reward: float
    done: bool
    terminated: bool
    truncated: bool
    observation: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpisodeTrajectory:
    """Stable training representation for one BrowserGym episode."""

    task_id: int
    seed: int
    raw_goal: str
    success: bool
    reward: float
    steps_taken: int
    invalid_action_count: int
    parse_failure_count: int
    terminated: bool
    truncated: bool
    failure_reasons: list[str]
    output_dir: str
    steps_path: str
    episode_path: str
    policy_name: str
    steps: list[TrajectoryStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OptimizationSample:
    """One prompt/response pair used for policy optimization."""

    prompt: str
    response_text: str
    action_text: str
    step_idx: int
    group_id: int
    episode_id: str
    episode_score: float
    advantage: float
    sample_weight: float
    env_reward: float
    step_reward: float
    success: bool
    invalid_action: bool
    parse_failed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_episode_trajectory(out_dir: str | Path) -> EpisodeTrajectory:
    """Load runner artifacts and reconstruct prompts for training."""

    output_dir = Path(out_dir)
    episode_data = json.loads((output_dir / "episode.json").read_text(encoding="utf-8"))
    steps: list[TrajectoryStep] = []

    steps_path = output_dir / "steps.jsonl"
    with steps_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            step_record = json.loads(line)
            observation = step_record.get("observation", {})
            normalized = normalize_observation(observation)
            prompt = build_full_prompt(normalized, int(step_record["step_idx"]))
            action_text = str(step_record.get("parsed_action", "") or "")
            response_text = f"ACTION: {action_text}" if action_text else ""
            steps.append(
                TrajectoryStep(
                    step_idx=int(step_record["step_idx"]),
                    prompt=prompt,
                    response_text=response_text,
                    action_text=action_text,
                    raw_text=str(step_record.get("raw_agent_output", "") or ""),
                    parse_error=_as_optional_text(step_record.get("parse_error")),
                    last_action_error=_as_optional_text(step_record.get("last_action_error")),
                    reward=float(step_record.get("reward", 0.0)),
                    done=bool(step_record.get("done", False)),
                    terminated=bool(step_record.get("terminated", False)),
                    truncated=bool(step_record.get("truncated", False)),
                    observation=dict(observation),
                    info=dict(step_record.get("info", {})),
                )
            )

    return EpisodeTrajectory(
        task_id=int(episode_data["task_id"]),
        seed=int(episode_data["seed"]),
        raw_goal=str(episode_data.get("raw_goal", "")),
        success=bool(episode_data.get("success", False)),
        reward=float(episode_data.get("reward", 0.0)),
        steps_taken=int(episode_data.get("steps", len(steps))),
        invalid_action_count=int(episode_data.get("invalid_action_count", 0)),
        parse_failure_count=int(episode_data.get("parse_failure_count", 0)),
        terminated=bool(episode_data.get("terminated", False)),
        truncated=bool(episode_data.get("truncated", False)),
        failure_reasons=[str(reason) for reason in episode_data.get("failure_reasons", [])],
        output_dir=str(output_dir),
        steps_path=str(steps_path),
        episode_path=str(output_dir / "episode.json"),
        policy_name=str(episode_data.get("policy_name", "")),
        steps=steps,
    )


def _as_optional_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)
