from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_subset_family_stage_scorecard import _load_json, validate_subset_family_stage_scorecard  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a subset family stage scorecard artifact.")
    parser.add_argument("--scorecard-path", required=True)
    parser.add_argument("--benchmark-blocker", action="append", default=[])
    parser.add_argument("--out", type=Path, default=None)
    return parser


def audit_subset_family_stage_scorecard(
    scorecard: dict[str, object],
    *,
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], dict[str, object]]:
    errors, derived = validate_subset_family_stage_scorecard(
        scorecard,
        benchmark_blockers=benchmark_blockers,
    )
    if errors:
        return errors, {}

    label = derived["label"]
    assert isinstance(label, str)
    stage_summary = scorecard[label]
    meta = scorecard[f"{label}_meta"]
    audit = {
        "audit_type": "subset_family_stage_scorecard_audit",
        "family_name": scorecard.get("family_name"),
        "label": label,
        "family_success_rate": stage_summary["family_success_rate"],
        "task_count": scorecard.get("task_count"),
        "matched_task_count": meta.get("matched_task_count"),
        "matched_task_ids": meta.get("matched_task_ids"),
        "omitted_source_task_count": meta.get("omitted_source_task_count"),
        "omitted_source_task_ids": meta.get("omitted_source_task_ids"),
        "source_summary": meta.get("source_summary"),
        "source_stage": meta.get("source_stage"),
        "source_family_name": meta.get("source_family_name"),
        "benchmark_blockers": sorted({str(task_id) for task_id in benchmark_blockers}, key=int),
        "unresolved_task_ids": [
            int(task_id)
            for task_id, success_rate in sorted(stage_summary["per_task_success_rate"].items(), key=lambda item: int(item[0]))
            if float(success_rate) < 1.0
        ],
    }
    return [], audit


def main() -> int:
    args = build_arg_parser().parse_args()
    scorecard_path = Path(args.scorecard_path)
    scorecard = _load_json(scorecard_path)
    errors, audit = audit_subset_family_stage_scorecard(
        scorecard,
        benchmark_blockers=args.benchmark_blocker,
    )
    payload = {
        "ok": not errors,
        "scorecard_path": str(scorecard_path),
        "audit": audit,
    }
    if errors:
        payload["errors"] = errors
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    if args.out is not None:
        args.out.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
        payload["out"] = str(args.out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
