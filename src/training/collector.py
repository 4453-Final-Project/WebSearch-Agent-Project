from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.utils.config import DEFAULT_MAX_STEPS, DEFAULT_SEED, DEFAULT_TASK_ID

from .trajectory import EpisodeTrajectory, load_episode_trajectory


EpisodeRunner = Callable[[Any, int, int, int, str], dict[str, Any]]


@dataclass(slots=True)
class RolloutConfig:
    """Collection-time configuration for grouped single-task rollouts."""

    task_id: int = DEFAULT_TASK_ID
    seed: int = DEFAULT_SEED
    max_steps: int = DEFAULT_MAX_STEPS
    groups_per_iteration: int = 2
    group_size: int = 2
    headless: bool = True
    output_dir: str = ""


class RolloutCollector:
    """Collect grouped episodes through the shared Task 2 runner."""

    def __init__(self, config: RolloutConfig, runner: EpisodeRunner | None = None) -> None:
        self.config = config
        self.runner = runner if runner is not None else self._load_runner()

    def collect_iteration(self, policy: Any, iteration_idx: int) -> list[list[EpisodeTrajectory]]:
        """Collect all groups for one optimization iteration."""

        root_dir = Path(self.config.output_dir) / "rollouts" / f"iteration_{iteration_idx:04d}"
        root_dir.mkdir(parents=True, exist_ok=True)

        groups: list[list[EpisodeTrajectory]] = []
        for group_idx in range(self.config.groups_per_iteration):
            group: list[EpisodeTrajectory] = []
            for member_idx in range(self.config.group_size):
                episode_index = group_idx * self.config.group_size + member_idx
                seed = self.config.seed + (iteration_idx * self.config.groups_per_iteration * self.config.group_size) + episode_index
                episode_out_dir = root_dir / f"group_{group_idx:04d}" / f"episode_{member_idx:04d}"
                runner_kwargs: dict[str, Any] = {}
                try:
                    runner_signature = inspect.signature(self.runner)
                except (TypeError, ValueError):
                    runner_signature = None

                if runner_signature is not None:
                    parameters = runner_signature.parameters
                    if "headless" in parameters or any(
                        parameter.kind == inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters.values()
                    ):
                        runner_kwargs["headless"] = self.config.headless

                self.runner(
                    policy,
                    self.config.task_id,
                    seed,
                    self.config.max_steps,
                    str(episode_out_dir),
                    **runner_kwargs,
                )
                group.append(load_episode_trajectory(episode_out_dir))
            groups.append(group)
        return groups

    def _load_runner(self) -> EpisodeRunner:
        from src.env.webarena_runner import run_episode

        return run_episode
