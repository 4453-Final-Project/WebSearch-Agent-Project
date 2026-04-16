"""Collect deterministic warmup demonstrations in runner format."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.webarena_runner import run_episode  # noqa: E402
from src.training.demo_policies import (  # noqa: E402
    BOOTSTRAP41_TASK_IDS,
    BOOTSTRAP_TASK_IDS,
    available_scripted_task_ids,
    get_scripted_warmup_policy,
)
from src.utils.config import DEFAULT_SEED, get_output_dir  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect deterministic warmup demonstrations for scripted WebArena tasks.")
    parser.add_argument(
        "--task-id",
        type=int,
        action="append",
        default=[],
        help="Task id to collect. Repeat to collect multiple tasks.",
    )
    parser.add_argument(
        "--preset",
        choices=("bootstrap9", "bootstrap41", "coverage41"),
        default=None,
        help="Collect a predefined multi-task warmup set.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--episodes", type=int, default=4, help="Episodes to collect per task.")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--headed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--out-dir",
        default=str(get_output_dir("warmup_demos", create=True)),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_ids = _resolve_task_ids(args.task_id, args.preset)
    all_task_summaries: list[dict[str, object]] = []
    all_episodes: list[dict[str, object]] = []

    for task_offset, task_id in enumerate(task_ids):
        policy = get_scripted_warmup_policy(task_id)
        task_dir = out_dir / f"task_{task_id:04d}"
        task_dir.mkdir(parents=True, exist_ok=True)

        episodes: list[dict[str, object]] = []
        for episode_idx in range(args.episodes):
            seed = args.seed + (task_offset * 1000) + episode_idx
            episode_out_dir = task_dir / f"episode_{episode_idx:04d}"
            episode = run_episode(
                policy=policy,
                task_id=task_id,
                seed=seed,
                max_steps=args.max_steps,
                out_dir=episode_out_dir,
                headless=not args.headed,
            )
            episodes.append(episode)
            all_episodes.append(episode)

        success_count = sum(int(episode.get("success", False)) for episode in episodes)
        task_summary = {
            "task_id": task_id,
            "policy_name": policy.name,
            "episodes": len(episodes),
            "success_count": success_count,
            "success_rate": success_count / max(1, len(episodes)),
            "average_reward": sum(float(episode.get("reward", 0.0)) for episode in episodes) / max(1, len(episodes)),
            "out_dir": str(task_dir),
        }
        (task_dir / "summary.json").write_text(json.dumps(task_summary, indent=2, sort_keys=True), encoding="utf-8")
        (task_dir / "summary.txt").write_text(_format_task_summary_text(task_summary), encoding="utf-8")
        all_task_summaries.append(task_summary)

    summary = {
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "episodes_per_task": args.episodes,
        "episodes": len(all_episodes),
        "success_count": sum(int(episode.get("success", False)) for episode in all_episodes),
        "success_rate": sum(int(episode.get("success", False)) for episode in all_episodes) / max(1, len(all_episodes)),
        "average_reward": sum(float(episode.get("reward", 0.0)) for episode in all_episodes) / max(1, len(all_episodes)),
        "out_dir": str(out_dir),
        "tasks": all_task_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "summary.txt").write_text(_format_collection_summary_text(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _resolve_task_ids(task_ids: list[int], preset: str | None) -> list[int]:
    selected = list(task_ids)
    if preset == "bootstrap9":
        selected.extend(BOOTSTRAP_TASK_IDS)
    if preset in {"bootstrap41", "coverage41"}:
        selected.extend(BOOTSTRAP41_TASK_IDS)
    if not selected:
        selected = [310]
    deduped = sorted(set(selected))
    unsupported = [task_id for task_id in deduped if task_id not in available_scripted_task_ids()]
    if unsupported:
        raise SystemExit(
            "Unsupported scripted warmup task ids: "
            + ", ".join(str(task_id) for task_id in unsupported)
            + ". Available tasks: "
            + ", ".join(str(task_id) for task_id in available_scripted_task_ids())
        )
    return deduped


def _format_task_summary_text(summary: dict[str, object]) -> str:
    return (
        "Warmup Task Summary\n"
        f"Task ID: {summary['task_id']}\n"
        f"Policy: {summary['policy_name']}\n"
        f"Episodes: {summary['episodes']}\n"
        f"Success Count: {summary['success_count']}\n"
        f"Success Rate: {float(summary['success_rate']):.3f}\n"
        f"Average Reward: {float(summary['average_reward']):.3f}\n"
        f"Output Directory: {summary['out_dir']}\n"
    )


def _format_collection_summary_text(summary: dict[str, object]) -> str:
    lines = [
        "Warmup Demo Collection Summary",
        f"Task Count: {summary['task_count']}",
        f"Task IDs: {', '.join(str(task_id) for task_id in summary['task_ids'])}",
        f"Episodes Per Task: {summary['episodes_per_task']}",
        f"Total Episodes: {summary['episodes']}",
        f"Success Count: {summary['success_count']}",
        f"Success Rate: {float(summary['success_rate']):.3f}",
        f"Average Reward: {float(summary['average_reward']):.3f}",
        "",
        "Per-Task Results:",
    ]
    for task_summary in summary["tasks"]:
        lines.append(
            f"- task {task_summary['task_id']}: success_rate={float(task_summary['success_rate']):.3f}, "
            f"episodes={task_summary['episodes']}, out_dir={task_summary['out_dir']}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
