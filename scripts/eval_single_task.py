"""Evaluation scaffold for single-task Task 3 policy runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any, Callable


EpisodeRunner = Callable[[Any, int, int, int, str], dict[str, Any]]


class FakeEpisodeRunner:
    """Simple queued runner for tests and local harness validation."""

    def __init__(self, episode_results: list[dict[str, Any]] | None = None) -> None:
        """Initialize the fake runner with optional precomputed episode results."""

        self._episode_results = list(episode_results or [])
        self.calls: list[dict[str, Any]] = []

    def run_episode(
        self,
        policy: Any,
        task_id: int,
        seed: int,
        max_steps: int,
        out_dir: str,
    ) -> dict[str, Any]:
        """Return the next precomputed result or a deterministic default result."""

        call = {
            "policy": policy,
            "task_id": task_id,
            "seed": seed,
            "max_steps": max_steps,
            "out_dir": out_dir,
        }
        self.calls.append(call)
        if self._episode_results:
            return self._episode_results.pop(0)
        return {
            "success": False,
            "reward": 0.0,
            "steps": 0,
            "invalid_action_count": 0,
            "parse_failure_count": 0,
            "failure_reasons": ["fake_runner_default"],
        }


def try_load_task2_runner() -> EpisodeRunner | None:
    """Try to load Task 2's episode runner without requiring it to exist."""

    try:
        module = import_module("src.env.webarena_runner")
    except ImportError:
        return None

    runner = getattr(module, "run_episode", None)
    if callable(runner):
        return runner
    return None


def resolve_runner(
    runner: EpisodeRunner | Any | None = None,
    *,
    use_fake_runner: bool = False,
) -> EpisodeRunner:
    """Resolve the runner function from injection, fake fallback, or Task 2."""

    if runner is not None:
        if callable(runner):
            return runner
        if hasattr(runner, "run_episode") and callable(runner.run_episode):
            return runner.run_episode
        raise TypeError("Runner must be callable or expose a callable run_episode method.")

    if use_fake_runner:
        return FakeEpisodeRunner().run_episode

    task2_runner = try_load_task2_runner()
    if task2_runner is not None:
        return task2_runner

    raise RuntimeError(
        "No episode runner is available. Task 2 runner was not found and no fake runner was requested."
    )


def normalize_episode_result(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a runner-produced episode result into a stable metric schema."""

    failure_reasons = raw_result.get("failure_reasons")
    if failure_reasons is None:
        single_reason = raw_result.get("failure_reason")
        failure_reasons = [single_reason] if single_reason else []

    return {
        "success": bool(raw_result.get("success", False)),
        "reward": float(raw_result.get("reward", 0.0)),
        "steps": int(raw_result.get("steps", 0)),
        "invalid_action_count": int(raw_result.get("invalid_action_count", 0)),
        "parse_failure_count": int(raw_result.get("parse_failure_count", 0)),
        "failure_reasons": [str(reason) for reason in failure_reasons if reason],
    }


def aggregate_metrics(episode_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode-level results into stable evaluation metrics."""

    normalized_results = [normalize_episode_result(result) for result in episode_results]
    episode_count = len(normalized_results)
    if episode_count == 0:
        return {
            "episode_count": 0,
            "success_rate": 0.0,
            "average_reward": 0.0,
            "average_steps": 0.0,
            "invalid_action_count": 0,
            "parse_failure_count": 0,
            "most_common_failure_reasons": [],
        }

    failure_counter = Counter(
        reason
        for result in normalized_results
        for reason in result["failure_reasons"]
    )

    return {
        "episode_count": episode_count,
        "success_rate": sum(1 for result in normalized_results if result["success"]) / episode_count,
        "average_reward": sum(result["reward"] for result in normalized_results) / episode_count,
        "average_steps": sum(result["steps"] for result in normalized_results) / episode_count,
        "invalid_action_count": sum(result["invalid_action_count"] for result in normalized_results),
        "parse_failure_count": sum(result["parse_failure_count"] for result in normalized_results),
        "most_common_failure_reasons": [
            {"reason": reason, "count": count}
            for reason, count in failure_counter.most_common()
        ],
    }


def write_metrics_json(metrics: dict[str, Any], out_dir: str | Path) -> Path:
    """Write aggregated metrics as JSON to the requested output directory."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics_path


def evaluate_single_task(
    policy: Any,
    task_id: int,
    seed: int,
    max_steps: int,
    episodes: int,
    out_dir: str | Path,
    *,
    runner: EpisodeRunner | Any | None = None,
    use_fake_runner: bool = False,
) -> dict[str, Any]:
    """Run repeated single-task evaluations through a runner interface."""

    resolved_runner = resolve_runner(runner, use_fake_runner=use_fake_runner)
    output_dir = Path(out_dir)
    episode_results: list[dict[str, Any]] = []

    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        episode_out_dir = output_dir / f"episode_{episode_idx}"
        result = resolved_runner(
            policy,
            task_id,
            episode_seed,
            max_steps,
            str(episode_out_dir),
        )
        episode_results.append(normalize_episode_result(result))

    metrics = aggregate_metrics(episode_results)
    metrics.update(
        {
            "task_id": task_id,
            "seed": seed,
            "max_steps": max_steps,
            "episodes": episodes,
            "out_dir": str(output_dir),
        }
    )
    write_metrics_json(metrics, output_dir)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for single-task evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a single task with a runner-backed policy harness.")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--use-fake-runner", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point and print machine-readable aggregated metrics."""

    args = build_arg_parser().parse_args(argv)
    metrics = evaluate_single_task(
        policy=None,
        task_id=args.task_id,
        seed=args.seed,
        max_steps=args.max_steps,
        episodes=args.episodes,
        out_dir=args.out_dir,
        use_fake_runner=args.use_fake_runner,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
