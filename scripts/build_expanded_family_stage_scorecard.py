from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_family_override_summary import _load_run_provenance  # noqa: E402
from scripts.json_io import write_json_atomic  # noqa: E402
from scripts.merge_family_reevals import load_json, parse_override_spec  # noqa: E402
from scripts.run_family_curriculum import (  # noqa: E402
    _build_family_metadata,
    _build_stage_summary,
    _load_split_from_manifest,
)
from src.training.task_families import get_family_task_groups  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a stage-only scorecard for an expanded family from a finished base stage plus added-task overrides."
    )
    parser.add_argument("--expanded-split-manifest", type=Path, required=True)
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument("--base-stage", default="current_stack")
    parser.add_argument("--label", default="current_stack")
    parser.add_argument("--override", action="append", default=[], metavar="TASK_ID=METRICS_JSON")
    parser.add_argument("--benchmark-blocker", action="append", default=[], metavar="TASK_ID")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def _as_float_per_task_map(stage_summary: object) -> dict[str, float]:
    if not isinstance(stage_summary, dict):
        return {}
    per_task = stage_summary.get("per_task_success_rate", {})
    if not isinstance(per_task, dict):
        return {}
    return {str(task_id): float(value) for task_id, value in per_task.items()}


def build_expanded_family_stage_scorecard(
    *,
    expanded_split_manifest_path: Path,
    base_summary_path: Path,
    base_stage: str,
    label: str,
    overrides: dict[str, Path],
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> dict[str, object]:
    split = _load_split_from_manifest(expanded_split_manifest_path)
    task_groups = get_family_task_groups(split.family_name)
    base_summary = load_json(base_summary_path)
    inherited_per_task = _as_float_per_task_map(base_summary.get(base_stage))
    expanded_task_ids = {str(task_id) for task_id in split.task_ids}

    inherited_task_ids = sorted(
        (task_id for task_id in inherited_per_task if task_id in expanded_task_ids),
        key=int,
    )
    missing_task_ids = sorted(
        expanded_task_ids.difference(inherited_task_ids).difference(overrides.keys()),
        key=int,
    )
    if missing_task_ids:
        raise ValueError(
            "Expanded family task ids are missing both inherited stage values and overrides: "
            + ", ".join(missing_task_ids)
        )

    merged_per_task = {task_id: inherited_per_task[task_id] for task_id in inherited_task_ids}
    applied_overrides: dict[str, object] = {}
    for task_id, metrics_path in sorted(overrides.items(), key=lambda item: int(item[0])):
        metrics = load_json(metrics_path)
        merged_per_task[task_id] = float(metrics["success_rate"])
        applied_overrides[task_id] = {
            "metrics_path": str(metrics_path),
            "success_rate": float(metrics["success_rate"]),
            "episodes": int(metrics["episodes"]),
            "average_steps": float(metrics["average_steps"]),
        }

    merged_task_ids = sorted(merged_per_task, key=int)
    stage_summary = _build_stage_summary(merged_per_task, task_groups)
    stage_summary["applied_overrides"] = applied_overrides

    added_task_ids = sorted((int(task_id) for task_id in applied_overrides), key=int)
    result = {
        "scorecard_type": "expanded_family_stage_scorecard",
        "family_name": split.family_name,
        "task_count": len(split.task_ids),
        "training_task_count": len(split.training_task_ids),
        "holdout_task_count": len(split.holdout_task_ids),
        "family_metadata": _build_family_metadata(split, task_groups),
        "split": split.to_dict(),
        "run_provenance": _load_run_provenance(expanded_split_manifest_path),
        "expanded_split_manifest": str(expanded_split_manifest_path),
        "base_summary_path": str(base_summary_path),
        label: stage_summary,
        f"{label}_meta": {
            "base_summary": str(base_summary_path),
            "base_stage": base_stage,
            "benchmark_blockers": sorted({str(task_id) for task_id in benchmark_blockers}, key=int),
            "inherited_task_count": len(inherited_task_ids),
            "inherited_task_ids": [int(task_id) for task_id in inherited_task_ids],
            "added_task_ids": added_task_ids,
            "override_count": len(applied_overrides),
            "override_metrics_paths": {
                task_id: str(metrics_path)
                for task_id, metrics_path in sorted(overrides.items(), key=lambda item: int(item[0]))
            },
            "applied_override_metrics": applied_overrides,
            "base_run_provenance": base_summary.get("run_provenance"),
            "merged_task_count": len(merged_task_ids),
        },
    }
    return result


def main() -> int:
    args = build_arg_parser().parse_args()
    overrides = dict(parse_override_spec(spec) for spec in args.override)
    scorecard = build_expanded_family_stage_scorecard(
        expanded_split_manifest_path=args.expanded_split_manifest,
        base_summary_path=args.base_summary,
        base_stage=args.base_stage,
        label=args.label,
        overrides=overrides,
        benchmark_blockers=args.benchmark_blocker,
    )
    write_json_atomic(args.out, scorecard)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "label": args.label,
                "family_name": scorecard["family_name"],
                "family_success_rate": scorecard[args.label]["family_success_rate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
