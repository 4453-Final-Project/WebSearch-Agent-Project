from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.json_io import write_json_atomic  # noqa: E402
from scripts.validate_combined_family_stage_scorecard import (  # noqa: E402
    validate_combined_family_stage_scorecard,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a combined family stage scorecard against its source scorecards."
    )
    parser.add_argument("--scorecard-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--benchmark-blocker", action="append", default=[])
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_combined_family_stage_scorecard(
    scorecard: dict[str, object],
    *,
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], dict[str, object]]:
    errors, derived = validate_combined_family_stage_scorecard(
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

    merged_per_task = {
        str(task_id): float(value)
        for task_id, value in stage_summary["per_task_success_rate"].items()
    }
    sources = stage_meta["sources"]
    overlap_task_ids = stage_meta.get("overlap_task_ids", {})

    source_reports: dict[str, object] = {}
    overlap_consistency: dict[str, object] = {}
    overlap_task_id_set = {str(task_id) for task_id in overlap_task_ids}
    for source_name, source_value in sorted(sources.items()):
        summary_path = Path(source_value["summary_path"])
        stage_name = str(source_value["stage_name"])
        source_summary = _load_json(summary_path)
        source_stage = source_summary.get(stage_name)
        if not isinstance(source_stage, dict):
            return [f"Source stage {stage_name!r} not found in {summary_path}"], {}
        source_per_task = {
            str(task_id): float(value)
            for task_id, value in source_stage.get("per_task_success_rate", {}).items()
        }
        matched_task_ids = [str(task_id) for task_id in source_value.get("matched_task_ids", [])]
        unique_task_ids = sorted(
            (int(task_id) for task_id in matched_task_ids if task_id not in overlap_task_id_set),
            key=int,
        )
        source_reports[source_name] = {
            "summary_path": str(summary_path),
            "stage_name": stage_name,
            "matched_task_count": len(matched_task_ids),
            "matched_task_ids": [int(task_id) for task_id in matched_task_ids],
            "unique_task_count": len(unique_task_ids),
            "unique_task_ids": unique_task_ids,
            "overlap_task_ids": sorted(
                (int(task_id) for task_id in matched_task_ids if task_id in overlap_task_id_set),
                key=int,
            ),
            "matched_success_rate": _mean([merged_per_task[task_id] for task_id in matched_task_ids]),
        }
        for task_id in matched_task_ids:
            if task_id not in overlap_task_id_set:
                continue
            overlap_entry = overlap_consistency.setdefault(
                task_id,
                {
                    "source_scores": {},
                    "merged_score": merged_per_task[task_id],
                },
            )
            overlap_entry["source_scores"][source_name] = source_per_task.get(task_id)

    for task_id, overlap_entry in overlap_consistency.items():
        score_values = [
            float(score)
            for score in overlap_entry["source_scores"].values()
            if score is not None
        ]
        overlap_entry["consistent_with_merged_score"] = (
            len({round(score, 12) for score in score_values}.union({round(float(overlap_entry["merged_score"]), 12)}))
            == 1
        )

    report = {
        "audit_type": "combined_family_stage_scorecard_audit",
        "family_name": scorecard.get("family_name"),
        "label": label,
        "scorecard_success_rate": float(stage_summary["family_success_rate"]),
        "task_count": len(merged_per_task),
        "success_task_count": sum(1 for value in merged_per_task.values() if value >= 1.0),
        "unresolved_task_ids": [int(task_id) for task_id, value in merged_per_task.items() if value < 1.0],
        "source_contributions": source_reports,
        "overlap_consistency": {
            str(task_id): overlap_consistency[str(task_id)]
            for task_id in sorted(overlap_consistency, key=int)
        },
        "consistent_overlap_task_ids": [
            int(task_id)
            for task_id, overlap_entry in sorted(overlap_consistency.items(), key=lambda item: int(item[0]))
            if overlap_entry["consistent_with_merged_score"]
        ],
    }
    return [], report


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> int:
    args = build_arg_parser().parse_args()
    scorecard = _load_json(args.scorecard_path)
    errors, report = audit_combined_family_stage_scorecard(
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
