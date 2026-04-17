"""Write a checked-in family split manifest from a recommended split or saved manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.json_io import write_json_atomic  # noqa: E402
from scripts.run_family_curriculum import (  # noqa: E402
    _compare_split_to_recommended,
    _get_family_requirement_status,
    _load_split_from_manifest,
    _build_split_provenance,
    build_split_manifest_payload,
)
from src.training.task_families import (  # noqa: E402
    build_task_split,
    list_task_family_names,
    recommend_task_split,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh a family split manifest from code defaults or a saved manifest.")
    parser.add_argument("--family", choices=list_task_family_names(), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=None)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--warmup-task-count", type=int, default=None)
    parser.add_argument("--holdout-task-count", type=int, default=None)
    return parser


def resolve_split(args: argparse.Namespace):
    if args.source_manifest is not None:
        return _load_split_from_manifest(args.source_manifest, expected_family_name=args.family)
    if args.warmup_task_count is None and args.holdout_task_count is None:
        return recommend_task_split(args.family, split_seed=args.split_seed)

    recommended = recommend_task_split(args.family, split_seed=args.split_seed)
    warmup_count = args.warmup_task_count or len(recommended.warmup_task_ids)
    holdout_count = args.holdout_task_count or len(recommended.holdout_task_ids)
    return build_task_split(
        recommended.task_ids,
        family_name=recommended.family_name,
        warmup_count=warmup_count,
        holdout_count=holdout_count,
        split_seed=args.split_seed,
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    split = resolve_split(args)
    if args.source_manifest is not None:
        split_provenance = _build_split_provenance(
            split,
            split_source="manifest",
            source_manifest=str(args.source_manifest.resolve()),
        )
    elif args.warmup_task_count is None and args.holdout_task_count is None:
        split_provenance = _build_split_provenance(split, split_source="recommended")
    else:
        split_provenance = _build_split_provenance(
            split,
            split_source="custom_counts",
            warmup_task_count=args.warmup_task_count,
            holdout_task_count=args.holdout_task_count,
        )
    payload = build_split_manifest_payload(
        split,
        split_provenance=split_provenance,
        judge_requirements=_get_family_requirement_status(split.family_name),
        recommended_split_alignment=_compare_split_to_recommended(split),
    )
    write_json_atomic(args.out, payload)
    print(
        json.dumps(
            {
                "family_name": split.family_name,
                "out": str(args.out),
                "task_count": len(split.task_ids),
                "training_task_count": len(split.training_task_ids),
                "holdout_task_count": len(split.holdout_task_ids),
                "source_manifest": str(args.source_manifest) if args.source_manifest else None,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
