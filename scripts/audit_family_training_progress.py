from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.json_io import write_json_atomic  # noqa: E402
from scripts.validate_family_summary import validate_family_summary  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit stage-to-stage training progress inside one family summary.")
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--stage-a", default="baseline")
    parser.add_argument("--stage-b", default="warmup_only")
    parser.add_argument("--benchmark-blocker", action="append", default=[])
    parser.add_argument("--out", type=Path, default=None)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sorted_task_ids(task_ids: list[str]) -> list[int]:
    return sorted((int(task_id) for task_id in task_ids), key=int)


def _build_task_delta_report(
    per_task_a: dict[str, float],
    per_task_b: dict[str, float],
) -> dict[str, object]:
    common_task_ids = sorted(set(per_task_a) & set(per_task_b), key=int)
    task_deltas = {
        task_id: {
            "stage_a_success_rate": float(per_task_a[task_id]),
            "stage_b_success_rate": float(per_task_b[task_id]),
            "delta": float(per_task_b[task_id]) - float(per_task_a[task_id]),
        }
        for task_id in common_task_ids
    }
    improved_task_ids = [task_id for task_id in common_task_ids if task_deltas[task_id]["delta"] > 0.0]
    regressed_task_ids = [task_id for task_id in common_task_ids if task_deltas[task_id]["delta"] < 0.0]
    unchanged_task_ids = [task_id for task_id in common_task_ids if task_deltas[task_id]["delta"] == 0.0]
    return {
        "task_count": len(common_task_ids),
        "improved_task_count": len(improved_task_ids),
        "regressed_task_count": len(regressed_task_ids),
        "unchanged_task_count": len(unchanged_task_ids),
        "improved_task_ids": _sorted_task_ids(improved_task_ids),
        "regressed_task_ids": _sorted_task_ids(regressed_task_ids),
        "unchanged_task_ids": _sorted_task_ids(unchanged_task_ids),
        "task_deltas": task_deltas,
    }


def audit_family_training_progress(
    summary: dict[str, object],
    *,
    stage_a: str,
    stage_b: str,
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], dict[str, object]]:
    errors, derived = validate_family_summary(summary, benchmark_blockers=benchmark_blockers)
    if errors:
        return errors, {}

    stage_metrics = derived["stage_metrics"]
    if stage_a not in stage_metrics:
        return [f"stage_a '{stage_a}' not found in summary"], {}
    if stage_b not in stage_metrics:
        return [f"stage_b '{stage_b}' not found in summary"], {}

    summary_stage_a = summary[stage_a]
    summary_stage_b = summary[stage_b]
    per_task_a = {str(task_id): float(value) for task_id, value in summary_stage_a["per_task_success_rate"].items()}
    per_task_b = {str(task_id): float(value) for task_id, value in summary_stage_b["per_task_success_rate"].items()}
    task_delta_report = _build_task_delta_report(per_task_a, per_task_b)

    task_groups_a = summary_stage_a.get("task_groups", {})
    task_groups_b = summary_stage_b.get("task_groups", {})
    common_task_groups = sorted(set(task_groups_a) & set(task_groups_b))
    task_group_deltas = {}
    for group_name in common_task_groups:
        group_a = task_groups_a[group_name]
        group_b = task_groups_b[group_name]
        group_report = _build_task_delta_report(
            {str(task_id): float(value) for task_id, value in group_a["per_task_success_rate"].items()},
            {str(task_id): float(value) for task_id, value in group_b["per_task_success_rate"].items()},
        )
        group_report.update(
            {
                "stage_a_success_rate": float(group_a["success_rate"]),
                "stage_b_success_rate": float(group_b["success_rate"]),
                "success_rate_delta": float(group_b["success_rate"]) - float(group_a["success_rate"]),
            }
        )
        task_group_deltas[group_name] = group_report

    return [], {
        "family_name": derived["family_name"],
        "stage_a": stage_a,
        "stage_b": stage_b,
        "benchmark_blockers": sorted({str(task_id) for task_id in benchmark_blockers}, key=int),
        "stage_a_family_success_rate": float(stage_metrics[stage_a]["family_success_rate"]),
        "stage_b_family_success_rate": float(stage_metrics[stage_b]["family_success_rate"]),
        "family_success_rate_delta": (
            float(stage_metrics[stage_b]["family_success_rate"]) - float(stage_metrics[stage_a]["family_success_rate"])
        ),
        "task_progress": task_delta_report,
        "common_task_groups": common_task_groups,
        "task_groups_only_in_stage_a": sorted(set(task_groups_a) - set(task_groups_b)),
        "task_groups_only_in_stage_b": sorted(set(task_groups_b) - set(task_groups_a)),
        "task_group_progress": task_group_deltas,
    }


def main() -> int:
    args = build_arg_parser().parse_args()
    summary = _load_json(args.summary_path)
    errors, report = audit_family_training_progress(
        summary,
        stage_a=args.stage_a,
        stage_b=args.stage_b,
        benchmark_blockers=args.benchmark_blocker,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    if args.out is not None:
        write_json_atomic(args.out, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
