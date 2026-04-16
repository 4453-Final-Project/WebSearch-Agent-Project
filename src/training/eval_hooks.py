from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.eval_single_task import evaluate_single_task
from src.utils.config import DEFAULT_MAX_STEPS, DEFAULT_SEED, DEFAULT_TASK_ID


@dataclass(slots=True)
class EvalConfig:
    """Evaluation configuration shared by before/after hooks."""

    task_id: int = DEFAULT_TASK_ID
    seed: int = DEFAULT_SEED
    max_steps: int = DEFAULT_MAX_STEPS
    episodes: int = 1
    out_dir: str = ""
    headless: bool = True


def run_eval(policy: Any, config: EvalConfig, suffix: str) -> dict[str, Any]:
    """Run the shared single-task evaluation harness into a namespaced directory."""

    output_dir = Path(config.out_dir) / suffix
    return evaluate_single_task(
        policy=policy,
        task_id=config.task_id,
        seed=config.seed,
        max_steps=config.max_steps,
        episodes=config.episodes,
        out_dir=output_dir,
        headless=config.headless,
    )


def write_metrics_table(
    out_dir: str | Path,
    *,
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
    iteration_metrics: list[dict[str, Any]],
) -> Path:
    """Write a compact before/after table plus iteration-level details."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table = {
        "before": before_metrics,
        "after": after_metrics,
        "delta": {
            "success_rate": after_metrics.get("success_rate", 0.0) - before_metrics.get("success_rate", 0.0),
            "average_reward": after_metrics.get("average_reward", 0.0) - before_metrics.get("average_reward", 0.0),
            "average_steps": after_metrics.get("average_steps", 0.0) - before_metrics.get("average_steps", 0.0),
            "invalid_action_count": after_metrics.get("invalid_action_count", 0) - before_metrics.get("invalid_action_count", 0),
            "parse_failure_count": after_metrics.get("parse_failure_count", 0) - before_metrics.get("parse_failure_count", 0),
        },
        "iterations": iteration_metrics,
    }
    path = output_dir / "metrics_table.json"
    path.write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")
    _write_metrics_table_text(output_dir, table)
    return path


def _write_metrics_table_text(output_dir: Path, table: dict[str, Any]) -> Path:
    before = table["before"]
    after = table["after"]
    delta = table["delta"]
    lines = [
        "Training Metrics Summary",
        "",
        "Before:",
        f"  success_rate={before.get('success_rate', 0.0):.3f}",
        f"  average_reward={before.get('average_reward', 0.0):.3f}",
        f"  average_steps={before.get('average_steps', 0.0):.3f}",
        f"  invalid_action_count={before.get('invalid_action_count', 0)}",
        f"  parse_failure_count={before.get('parse_failure_count', 0)}",
        "",
        "After:",
        f"  success_rate={after.get('success_rate', 0.0):.3f}",
        f"  average_reward={after.get('average_reward', 0.0):.3f}",
        f"  average_steps={after.get('average_steps', 0.0):.3f}",
        f"  invalid_action_count={after.get('invalid_action_count', 0)}",
        f"  parse_failure_count={after.get('parse_failure_count', 0)}",
        "",
        "Delta:",
        f"  success_rate={delta.get('success_rate', 0.0):+.3f}",
        f"  average_reward={delta.get('average_reward', 0.0):+.3f}",
        f"  average_steps={delta.get('average_steps', 0.0):+.3f}",
        f"  invalid_action_count={delta.get('invalid_action_count', 0):+d}",
        f"  parse_failure_count={delta.get('parse_failure_count', 0):+d}",
        "",
        "Iteration Metrics:",
    ]
    iterations = table.get("iterations", [])
    if iterations:
        for item in iterations:
            lines.extend(
                [
                    f"- iteration={item.get('iteration', '(unknown)')}",
                    f"  episode_reward={item.get('episode_reward', 0.0):.3f}",
                    f"  success_rate={item.get('success_rate', 0.0):.3f}",
                    f"  step_count={item.get('step_count', 0)}",
                    f"  invalid_actions={item.get('invalid_actions', 0)}",
                    f"  parse_failures={item.get('parse_failures', 0)}",
                    f"  loss={item.get('loss', 0.0):.6f}",
                    f"  checkpoint_path={item.get('checkpoint_path', '(none)')}",
                ]
            )
    else:
        lines.append("- none")

    path = output_dir / "metrics_table.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
