"""Validate the committed final-result expectations against saved output artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT.parent / "outputs"


EXPECTED_STAGE_RATES = {
    "baseline": 0.60,
    "warmup_only": 0.80,
    "warmup_plus_grpo": 1.00,
}

EXPECTED_PER_TASK = {
    "baseline": {"324": 0.0, "325": 1.0, "326": 1.0, "327": 0.0, "328": 1.0},
    "warmup_only": {"324": 0.0, "325": 1.0, "326": 1.0, "327": 1.0, "328": 1.0},
    "warmup_plus_grpo": {"324": 1.0, "325": 1.0, "326": 1.0, "327": 1.0, "328": 1.0},
}

EXPECTED_RUN_CARD = {
    "run_name": "liquid_shopping_disjoint_v1",
    "model": "LFM2.5-350M",
    "task_family": "shopping_searchsort_324_328",
    "task_split": {
        "warmup_task_ids": [325, 326],
        "grpo_task_ids": [327, 328],
        "eval_task_ids": [324, 325, 326, 327, 328],
    },
    "warmup": {
        "selected_demo_count": 6,
        "success_demo_count": 6,
        "epochs": 1,
        "family_success_rate": 0.80,
    },
    "rollout_collection": {
        "episode_count": 20,
        "success_count": 19,
        "success_rate": 0.95,
        "task_success_counts": {"327": 9, "328": 10},
    },
    "grpo": {
        "iteration_count": 1,
        "family_success_rate": 1.00,
    },
}


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_equal(label: str, actual: object, expected: object, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, found {actual!r}")


def _assert_close(label: str, actual: float, expected: float, errors: list[str], tolerance: float = 1e-9) -> None:
    if abs(actual - expected) > tolerance:
        errors.append(f"{label}: expected {expected:.6f}, found {actual:.6f}")


def validate_final_results(summary: dict[str, object], run_card: dict[str, object]) -> list[str]:
    errors: list[str] = []

    for stage, expected_rate in EXPECTED_STAGE_RATES.items():
        stage_summary = summary.get(stage)
        if not isinstance(stage_summary, dict):
            errors.append(f"summary.{stage}: missing stage data")
            continue
        _assert_close(
            f"summary.{stage}.family_success_rate",
            float(stage_summary["family_success_rate"]),
            expected_rate,
            errors,
        )
        per_task = stage_summary.get("per_task_success_rate")
        if not isinstance(per_task, dict):
            errors.append(f"summary.{stage}.per_task_success_rate: missing per-task data")
            continue
        for task_id, expected_value in EXPECTED_PER_TASK[stage].items():
            _assert_close(
                f"summary.{stage}.per_task_success_rate.{task_id}",
                float(per_task[task_id]),
                expected_value,
                errors,
            )

    _assert_equal("run_card.run_name", run_card.get("run_name"), EXPECTED_RUN_CARD["run_name"], errors)
    _assert_equal("run_card.model", run_card.get("model"), EXPECTED_RUN_CARD["model"], errors)
    _assert_equal("run_card.task_family", run_card.get("task_family"), EXPECTED_RUN_CARD["task_family"], errors)

    task_split = run_card.get("task_split")
    if not isinstance(task_split, dict):
        errors.append("run_card.task_split: missing task split data")
    else:
        for key, expected_value in EXPECTED_RUN_CARD["task_split"].items():
            _assert_equal(f"run_card.task_split.{key}", task_split.get(key), expected_value, errors)

    warmup = run_card.get("warmup")
    if not isinstance(warmup, dict):
        errors.append("run_card.warmup: missing warmup section")
    else:
        for key, expected_value in EXPECTED_RUN_CARD["warmup"].items():
            actual = warmup.get(key)
            if isinstance(expected_value, float):
                _assert_close(f"run_card.warmup.{key}", float(actual), expected_value, errors)
            else:
                _assert_equal(f"run_card.warmup.{key}", actual, expected_value, errors)

    rollout = run_card.get("rollout_collection")
    if not isinstance(rollout, dict):
        errors.append("run_card.rollout_collection: missing rollout section")
    else:
        for key, expected_value in EXPECTED_RUN_CARD["rollout_collection"].items():
            actual = rollout.get(key)
            if isinstance(expected_value, float):
                _assert_close(f"run_card.rollout_collection.{key}", float(actual), expected_value, errors)
            else:
                _assert_equal(f"run_card.rollout_collection.{key}", actual, expected_value, errors)

    grpo = run_card.get("grpo")
    if not isinstance(grpo, dict):
        errors.append("run_card.grpo: missing GRPO section")
    else:
        for key, expected_value in EXPECTED_RUN_CARD["grpo"].items():
            actual = grpo.get(key)
            if isinstance(expected_value, float):
                _assert_close(f"run_card.grpo.{key}", float(actual), expected_value, errors)
            else:
                _assert_equal(f"run_card.grpo.{key}", actual, expected_value, errors)

    return errors


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the saved final Shopping result artifacts.")
    parser.add_argument(
        "--summary-path",
        default=str(OUTPUTS_ROOT / "shopping_report_figures" / "shopping_searchsort_summary.json"),
    )
    parser.add_argument(
        "--run-card-path",
        default=str(OUTPUTS_ROOT / "shopping_report_figures" / "shopping_searchsort_run_card.json"),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    summary_path = Path(args.summary_path)
    run_card_path = Path(args.run_card_path)
    summary = _load_json(summary_path)
    run_card = _load_json(run_card_path)
    errors = validate_final_results(summary, run_card)
    if errors:
        print(json.dumps({"ok": False, "errors": errors}, indent=2))
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "summary_path": str(summary_path),
                "run_card_path": str(run_card_path),
                "baseline_family_success_rate": EXPECTED_STAGE_RATES["baseline"],
                "warmup_family_success_rate": EXPECTED_STAGE_RATES["warmup_only"],
                "grpo_family_success_rate": EXPECTED_STAGE_RATES["warmup_plus_grpo"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
