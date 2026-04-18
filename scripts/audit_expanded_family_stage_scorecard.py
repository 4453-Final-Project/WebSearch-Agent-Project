from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_family_training_progress import _build_task_delta_report  # noqa: E402
from scripts.json_io import write_json_atomic  # noqa: E402
from scripts.validate_expanded_family_stage_scorecard import (  # noqa: E402
    validate_expanded_family_stage_scorecard,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit an expanded family stage scorecard against its inherited base stage."
    )
    parser.add_argument("--scorecard-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--benchmark-blocker", action="append", default=[])
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def audit_expanded_family_stage_scorecard(
    scorecard: dict[str, object],
    *,
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], dict[str, object]]:
    errors, derived = validate_expanded_family_stage_scorecard(
        scorecard,
        benchmark_blockers=benchmark_blockers,
    )
    if errors:
        return errors, {}

    label = derived["label"]
    assert isinstance(label, str)
    meta_key = f"{label}_meta"
    stage_summary = scorecard[label]
    stage_meta = scorecard[meta_key]

    base_summary_path = Path(stage_meta["base_summary"])
    base_stage = str(stage_meta["base_stage"])
    base_summary = _load_json(base_summary_path)
    if base_stage not in base_summary or not isinstance(base_summary[base_stage], dict):
        return [f"Base stage {base_stage!r} not found in {base_summary_path}"], {}

    base_per_task = {
        str(task_id): float(value)
        for task_id, value in base_summary[base_stage]["per_task_success_rate"].items()
    }
    expanded_per_task = {
        str(task_id): float(value)
        for task_id, value in stage_summary["per_task_success_rate"].items()
    }

    inherited_task_ids = [str(task_id) for task_id in stage_meta.get("inherited_task_ids", [])]
    added_task_ids = [str(task_id) for task_id in stage_meta.get("added_task_ids", [])]

    inherited_base_map = {task_id: base_per_task[task_id] for task_id in inherited_task_ids}
    inherited_expanded_map = {task_id: expanded_per_task[task_id] for task_id in inherited_task_ids}
    added_map = {task_id: expanded_per_task[task_id] for task_id in added_task_ids}

    inherited_delta = _build_task_delta_report(inherited_base_map, inherited_expanded_map)
    added_success_task_ids = sorted((int(task_id) for task_id, value in added_map.items() if value >= 1.0), key=int)
    added_unresolved_task_ids = sorted((int(task_id) for task_id, value in added_map.items() if value < 1.0), key=int)

    stage_task_groups = stage_summary.get("task_groups", {})
    added_task_group_success: dict[str, object] = {}
    if isinstance(stage_task_groups, dict):
        for group_name, group_value in stage_task_groups.items():
            if not isinstance(group_value, dict):
                continue
            group_per_task = group_value.get("per_task_success_rate", {})
            if not isinstance(group_per_task, dict):
                continue
            group_added_task_ids = sorted(
                (
                    int(task_id)
                    for task_id in added_task_ids
                    if task_id in group_per_task
                ),
                key=int,
            )
            if not group_added_task_ids:
                continue
            group_added_values = [float(group_per_task[str(task_id)]) for task_id in group_added_task_ids]
            added_task_group_success[group_name] = {
                "added_task_count": len(group_added_task_ids),
                "added_task_ids": group_added_task_ids,
                "added_success_task_count": sum(1 for value in group_added_values if value >= 1.0),
                "added_success_rate": _mean(group_added_values),
            }

    report = {
        "family_name": scorecard.get("family_name"),
        "label": label,
        "base_summary_path": str(base_summary_path),
        "base_stage": base_stage,
        "expanded_family_success_rate": float(stage_summary["family_success_rate"]),
        "expanded_task_count": len(expanded_per_task),
        "inherited_slice_success_rate": _mean(list(inherited_expanded_map.values())),
        "inherited_task_count": len(inherited_task_ids),
        "added_task_success_rate": _mean(list(added_map.values())),
        "added_task_count": len(added_task_ids),
        "added_success_task_count": sum(1 for value in added_map.values() if value >= 1.0),
        "added_success_task_ids": added_success_task_ids,
        "added_unresolved_task_ids": added_unresolved_task_ids,
        "added_task_group_success": added_task_group_success,
        "inherited_progress": inherited_delta,
        "expansion_summary": {
            "base_family_name": base_summary.get("family_name"),
            "base_stage_success_rate_on_inherited_tasks": _mean(list(inherited_base_map.values())),
            "expanded_stage_success_rate_on_inherited_tasks": _mean(list(inherited_expanded_map.values())),
            "changed_inherited_task_ids": inherited_delta["improved_task_ids"] + inherited_delta["regressed_task_ids"],
            "added_task_ids": sorted((int(task_id) for task_id in added_task_ids), key=int),
        },
    }
    return [], report


def main() -> int:
    args = build_arg_parser().parse_args()
    scorecard = _load_json(args.scorecard_path)
    errors, report = audit_expanded_family_stage_scorecard(
        scorecard,
        benchmark_blockers=args.benchmark_blocker,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    report["scorecard_path"] = str(args.scorecard_path)

    if args.out is not None:
        write_json_atomic(args.out, report)
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
