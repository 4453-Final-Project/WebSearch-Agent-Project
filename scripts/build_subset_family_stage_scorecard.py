from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_subset_family_stage_scorecard import audit_subset_family_stage_scorecard  # noqa: E402
from scripts.build_family_override_summary import _load_run_provenance  # noqa: E402
from scripts.json_io import write_json_atomic  # noqa: E402
from scripts.merge_family_reevals import load_json  # noqa: E402
from scripts.run_family_curriculum import _build_family_metadata, _build_stage_summary, _load_split_from_manifest  # noqa: E402
from scripts.validate_subset_family_stage_scorecard import validate_subset_family_stage_scorecard  # noqa: E402
from src.training.task_families import get_family_task_groups  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a stage scorecard for a family that is a strict subset of a validated source artifact."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--subset-split-manifest", type=Path, default=None)
    parser.add_argument("--source-summary", type=Path, default=None)
    parser.add_argument("--source-stage", default="current_stack")
    parser.add_argument("--label", default="current_stack")
    parser.add_argument("--benchmark-blocker", action="append", default=[], metavar="TASK_ID")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--audit-out", type=Path, default=None)
    return parser


def _load_manifest(path: Path) -> dict[str, object]:
    manifest = load_json(path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return manifest


def _resolve_manifest_path(manifest_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _merge_manifest_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.manifest is None:
        return args

    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)

    if args.subset_split_manifest is None and "subset_split_manifest" in manifest:
        args.subset_split_manifest = _resolve_manifest_path(manifest_path, str(manifest["subset_split_manifest"]))
    if args.source_summary is None and "source_summary" in manifest:
        args.source_summary = _resolve_manifest_path(manifest_path, str(manifest["source_summary"]))
    if args.out is None and "out" in manifest:
        args.out = _resolve_manifest_path(manifest_path, str(manifest["out"]))
    if args.audit_out is None and "audit_out" in manifest:
        args.audit_out = _resolve_manifest_path(manifest_path, str(manifest["audit_out"]))
    if args.label == "current_stack" and "label" in manifest:
        args.label = str(manifest["label"])
    if args.source_stage == "current_stack" and "source_stage" in manifest:
        args.source_stage = str(manifest["source_stage"])
    if not args.benchmark_blocker and "benchmark_blockers" in manifest:
        args.benchmark_blocker = [str(task_id) for task_id in manifest["benchmark_blockers"]]
    return args


def _as_float_per_task_map(stage_summary: object) -> dict[str, float]:
    if not isinstance(stage_summary, dict):
        return {}
    per_task = stage_summary.get("per_task_success_rate", {})
    if not isinstance(per_task, dict):
        return {}
    return {str(task_id): float(value) for task_id, value in per_task.items()}


def build_subset_family_stage_scorecard(
    *,
    subset_split_manifest_path: Path,
    source_summary_path: Path,
    source_stage: str,
    label: str,
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> dict[str, object]:
    split = _load_split_from_manifest(subset_split_manifest_path)
    task_groups = get_family_task_groups(split.family_name)
    source_summary = load_json(source_summary_path)
    source_stage_map = _as_float_per_task_map(source_summary.get(source_stage))
    if not source_stage_map:
        raise ValueError(f"Source summary is missing stage {source_stage!r}: {source_summary_path}")

    subset_task_ids = sorted((str(task_id) for task_id in split.task_ids), key=int)
    missing_task_ids = sorted((task_id for task_id in subset_task_ids if task_id not in source_stage_map), key=int)
    if missing_task_ids:
        raise ValueError(
            "Subset family task ids are missing from the source stage: "
            + ", ".join(missing_task_ids)
        )

    subset_per_task = {task_id: source_stage_map[task_id] for task_id in subset_task_ids}
    omitted_source_task_ids = sorted((task_id for task_id in source_stage_map if task_id not in subset_per_task), key=int)
    stage_summary = _build_stage_summary(subset_per_task, task_groups)

    return {
        "scorecard_type": "subset_family_stage_scorecard",
        "family_name": split.family_name,
        "task_count": len(split.task_ids),
        "training_task_count": len(split.training_task_ids),
        "holdout_task_count": len(split.holdout_task_ids),
        "family_metadata": _build_family_metadata(split, task_groups),
        "split": split.to_dict(),
        "run_provenance": _load_run_provenance(subset_split_manifest_path),
        "subset_split_manifest": str(subset_split_manifest_path),
        "source_summary_path": str(source_summary_path),
        label: stage_summary,
        f"{label}_meta": {
            "source_summary": str(source_summary_path),
            "source_stage": source_stage,
            "source_family_name": source_summary.get("family_name"),
            "benchmark_blockers": sorted({str(task_id) for task_id in benchmark_blockers}, key=int),
            "source_stage_task_count": len(source_stage_map),
            "matched_task_count": len(subset_per_task),
            "matched_task_ids": [int(task_id) for task_id in subset_task_ids],
            "omitted_source_task_count": len(omitted_source_task_ids),
            "omitted_source_task_ids": [int(task_id) for task_id in omitted_source_task_ids],
            "source_run_provenance": source_summary.get("run_provenance"),
        },
    }


def validate_and_audit_subset_family_stage_scorecard(
    scorecard: dict[str, object],
    *,
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], dict[str, object], dict[str, object] | None]:
    errors, derived = validate_subset_family_stage_scorecard(
        scorecard,
        benchmark_blockers=benchmark_blockers,
    )
    if errors:
        return errors, derived, None
    audit_errors, audit_report = audit_subset_family_stage_scorecard(
        scorecard,
        benchmark_blockers=benchmark_blockers,
    )
    if audit_errors:
        return audit_errors, derived, None
    return [], derived, audit_report


def main() -> int:
    args = _merge_manifest_args(build_arg_parser().parse_args())
    if args.subset_split_manifest is None:
        raise ValueError("A subset split manifest is required via --subset-split-manifest or --manifest")
    if args.source_summary is None:
        raise ValueError("A source summary is required via --source-summary or --manifest")
    if args.out is None:
        raise ValueError("An output path is required via --out or --manifest")

    scorecard = build_subset_family_stage_scorecard(
        subset_split_manifest_path=args.subset_split_manifest,
        source_summary_path=args.source_summary,
        source_stage=args.source_stage,
        label=args.label,
        benchmark_blockers=args.benchmark_blocker,
    )
    errors, derived, audit_report = validate_and_audit_subset_family_stage_scorecard(
        scorecard,
        benchmark_blockers=args.benchmark_blocker,
    )
    payload = {
        "ok": not errors,
        "out": str(args.out),
        "label": args.label,
        "family_name": scorecard["family_name"],
        "family_success_rate": scorecard[args.label]["family_success_rate"],
        "derived": derived,
    }
    if args.audit_out is not None:
        payload["audit_out"] = str(args.audit_out)
    if errors:
        payload["errors"] = errors
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    write_json_atomic(args.out, scorecard)
    if args.audit_out is not None and audit_report is not None:
        audit_report["scorecard_path"] = str(args.out)
        write_json_atomic(args.audit_out, audit_report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
