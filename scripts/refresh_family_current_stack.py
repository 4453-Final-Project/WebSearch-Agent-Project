"""Refresh a merged current-stack family summary and optional compare report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_family_summaries import compare_family_summaries  # noqa: E402
from scripts.json_io import write_json_atomic  # noqa: E402
from scripts.merge_family_reevals import (  # noqa: E402
    build_label_meta,
    build_merged_stage,
    build_stage_holdout_summary,
    load_json,
    parse_override_spec,
)
from scripts.validate_family_summary import validate_family_summary  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refresh a merged current-stack summary, validate it, and optionally emit a compare report.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--base-summary", type=Path, default=None)
    parser.add_argument("--base-stage", default="warmup_plus_grpo")
    parser.add_argument("--label", default="current_stack")
    parser.add_argument("--override", action="append", default=[], metavar="TASK_ID=METRICS_JSON")
    parser.add_argument("--benchmark-blocker", action="append", default=[], metavar="TASK_ID")
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--compare-report-out", type=Path, default=None)
    parser.add_argument(
        "--compare-base-stage",
        default="warmup_plus_grpo",
        help="Stage from the base summary to compare against the refreshed label stage.",
    )
    parser.add_argument("--compare-label-a", default="base")
    parser.add_argument("--compare-label-b", default="current_stack")
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

    def manifest_value(key: str, default):
        return manifest.get(key, default)

    if getattr(args, "_base_summary_explicit", False) is False and "base_summary" in manifest:
        args.base_summary = _resolve_manifest_path(manifest_path, str(manifest["base_summary"]))
    if getattr(args, "_summary_out_explicit", False) is False and "summary_out" in manifest:
        args.summary_out = _resolve_manifest_path(manifest_path, str(manifest["summary_out"]))
    if getattr(args, "_compare_report_out_explicit", False) is False and "compare_report_out" in manifest:
        args.compare_report_out = _resolve_manifest_path(manifest_path, str(manifest["compare_report_out"]))
    if getattr(args, "_base_stage_explicit", False) is False:
        args.base_stage = str(manifest_value("base_stage", args.base_stage))
    if getattr(args, "_label_explicit", False) is False:
        args.label = str(manifest_value("label", args.label))
    if getattr(args, "_compare_base_stage_explicit", False) is False:
        args.compare_base_stage = str(manifest_value("compare_base_stage", args.compare_base_stage))
    if getattr(args, "_compare_label_a_explicit", False) is False:
        args.compare_label_a = str(manifest_value("compare_label_a", args.compare_label_a))
    if getattr(args, "_compare_label_b_explicit", False) is False:
        args.compare_label_b = str(manifest_value("compare_label_b", args.compare_label_b))
    if not getattr(args, "_benchmark_blocker_explicit", False):
        args.benchmark_blocker = [str(task_id) for task_id in manifest_value("benchmark_blockers", args.benchmark_blocker)]
    manifest_overrides = manifest.get("overrides", {})
    if not isinstance(manifest_overrides, dict):
        raise ValueError(f"manifest.overrides must be an object: {manifest_path}")
    merged_overrides = {
        str(task_id): str(_resolve_manifest_path(manifest_path, str(metrics_path)))
        for task_id, metrics_path in manifest_overrides.items()
    }
    cli_overrides = dict(parse_override_spec(spec) for spec in args.override)
    for task_id, metrics_path in cli_overrides.items():
        merged_overrides[task_id] = str(metrics_path)
    args.override = [f"{task_id}={metrics_path}" for task_id, metrics_path in sorted(merged_overrides.items(), key=lambda item: int(item[0]))]
    return args


def build_refreshed_summary(
    base_summary: dict[str, object],
    *,
    base_summary_path: Path,
    base_stage: str,
    label: str,
    overrides: dict[str, Path],
    benchmark_blockers: list[str] | tuple[str, ...],
) -> dict[str, object]:
    merged_stage = build_merged_stage(base_summary, base_stage=base_stage, overrides=overrides)
    refreshed = dict(base_summary)
    refreshed[label] = merged_stage
    merged_holdout = build_stage_holdout_summary(
        base_summary,
        stage_per_task_success_rate=merged_stage["per_task_success_rate"],
    )
    if merged_holdout is not None:
        refreshed[f"{label}_holdout"] = merged_holdout
    refreshed[f"{label}_meta"] = build_label_meta(
        base_summary,
        base_summary_path=base_summary_path,
        base_stage=base_stage,
        benchmark_blockers=benchmark_blockers,
        overrides=overrides,
        merged_stage=merged_stage,
    )
    return refreshed


def validate_refresh_inputs(
    base_summary: dict[str, object],
    *,
    base_stage: str,
    overrides: dict[str, Path],
) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    stage_summary = base_summary.get(base_stage)
    if not isinstance(stage_summary, dict):
        errors.append(f"Base summary is missing stage {base_stage!r}")
        return errors, {"base_stage": base_stage, "override_count": len(overrides), "override_task_ids": []}
    per_task_raw = stage_summary.get("per_task_success_rate")
    if not isinstance(per_task_raw, dict):
        errors.append(f"Base stage {base_stage!r} is missing per_task_success_rate")
        return errors, {"base_stage": base_stage, "override_count": len(overrides), "override_task_ids": []}

    stage_task_ids = set(str(task_id) for task_id in per_task_raw)
    for task_id, metrics_path in sorted(overrides.items(), key=lambda item: int(item[0])):
        if task_id not in stage_task_ids:
            errors.append(f"Override task {task_id} is not present in base stage {base_stage!r}")
        if not metrics_path.exists():
            errors.append(f"Override metrics file is missing for task {task_id}: {metrics_path}")

    preview = {
        "base_stage": base_stage,
        "base_stage_task_count": len(stage_task_ids),
        "override_count": len(overrides),
        "override_task_ids": sorted((int(task_id) for task_id in overrides), key=int),
        "override_metrics_paths": {
            task_id: str(metrics_path) for task_id, metrics_path in sorted(overrides.items(), key=lambda item: int(item[0]))
        },
    }
    return errors, preview


def refresh_family_current_stack(
    *,
    base_summary: dict[str, object],
    base_summary_path: Path,
    base_stage: str,
    label: str,
    overrides: dict[str, Path],
    benchmark_blockers: list[str] | tuple[str, ...],
    compare_base_stage: str,
    compare_label_a: str,
    compare_label_b: str,
) -> tuple[dict[str, object], list[str], dict[str, object], list[str], dict[str, object]]:
    refreshed = build_refreshed_summary(
        base_summary,
        base_summary_path=base_summary_path,
        base_stage=base_stage,
        label=label,
        overrides=overrides,
        benchmark_blockers=benchmark_blockers,
    )
    validation_errors, derived = validate_family_summary(refreshed, benchmark_blockers=benchmark_blockers)
    compare_errors, compare_payload = compare_family_summaries(
        base_summary,
        refreshed,
        label_a=compare_label_a,
        label_b=compare_label_b,
        benchmark_blockers=benchmark_blockers,
        compare_stage_specs=[f"{compare_base_stage}={label}"],
        summary_a_path=base_summary_path,
    )
    return refreshed, validation_errors, derived, compare_errors, compare_payload


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    explicit_flags = {
        "base_summary": "--base-summary" in sys.argv,
        "summary_out": "--summary-out" in sys.argv,
        "compare_report_out": "--compare-report-out" in sys.argv,
        "base_stage": "--base-stage" in sys.argv,
        "label": "--label" in sys.argv,
        "compare_base_stage": "--compare-base-stage" in sys.argv,
        "compare_label_a": "--compare-label-a" in sys.argv,
        "compare_label_b": "--compare-label-b" in sys.argv,
        "benchmark_blocker": "--benchmark-blocker" in sys.argv,
    }
    for key, value in explicit_flags.items():
        setattr(args, f"_{key}_explicit", value)
    args = _merge_manifest_args(args)
    if args.base_summary is None:
        raise ValueError("A base summary is required via --base-summary or --manifest")
    if args.summary_out is None:
        raise ValueError("A summary output path is required via --summary-out or --manifest")
    base_summary = load_json(args.base_summary)
    overrides = dict(parse_override_spec(spec) for spec in args.override)
    preflight_errors, preflight = validate_refresh_inputs(
        base_summary,
        base_stage=args.base_stage,
        overrides=overrides,
    )
    refreshed, validation_errors, derived, compare_errors, compare_payload = refresh_family_current_stack(
        base_summary=base_summary,
        base_summary_path=args.base_summary,
        base_stage=args.base_stage,
        label=args.label,
        overrides=overrides,
        benchmark_blockers=args.benchmark_blocker,
        compare_base_stage=args.compare_base_stage,
        compare_label_a=args.compare_label_a,
        compare_label_b=args.compare_label_b,
    )
    payload = {
        "ok": not preflight_errors and not validation_errors and not compare_errors,
        "summary_out": str(args.summary_out),
        "preflight": {
            "ok": not preflight_errors,
            "errors": preflight_errors,
            "preview": preflight,
        },
        "validation": {
            "ok": not validation_errors,
            "derived": derived,
            "errors": validation_errors,
        },
        "comparison": {
            "ok": not compare_errors,
            "errors": compare_errors,
            "payload": compare_payload,
        },
    }
    if preflight_errors or validation_errors or compare_errors:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    write_json_atomic(args.summary_out, refreshed)
    if args.compare_report_out:
        compare_report = {
            "ok": True,
            "summary_a": str(args.base_summary),
            "summary_b": str(args.summary_out),
            "comparison": compare_payload,
        }
        write_json_atomic(args.compare_report_out, compare_report)
        payload["compare_report_out"] = str(args.compare_report_out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
