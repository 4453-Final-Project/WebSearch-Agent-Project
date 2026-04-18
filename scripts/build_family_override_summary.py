from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.json_io import write_json_atomic  # noqa: E402
from scripts.merge_family_reevals import (  # noqa: E402
    build_label_meta,
    build_merged_stage,
    load_json,
    parse_override_spec,
)
from scripts.run_family_curriculum import (  # noqa: E402
    _build_family_metadata,
    _build_stage_summary,
    _compare_split_to_recommended,
    _load_split_from_manifest,
    _success_rates_from_eval,
)
from scripts.validate_family_summary import validate_family_summary  # noqa: E402
from src.training.task_families import family_requires_openai_judge, get_family_task_groups  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a family-style summary from a completed baseline eval plus targeted task overrides."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split-manifest", type=Path, default=None)
    parser.add_argument("--baseline-eval-summary", type=Path, default=None)
    parser.add_argument("--label", default="current_stack")
    parser.add_argument("--override", action="append", default=[], metavar="TASK_ID=METRICS_JSON")
    parser.add_argument("--benchmark-blocker", action="append", default=[], metavar="TASK_ID")
    parser.add_argument("--out", type=Path, default=None)
    return parser


def _load_run_provenance(split_manifest_path: Path) -> dict[str, object]:
    manifest_payload = load_json(split_manifest_path)
    split_provenance = manifest_payload.get("split_provenance")
    judge_requirements = manifest_payload.get("judge_requirements")
    recommended_split_alignment = manifest_payload.get("recommended_split_alignment")
    split = _load_split_from_manifest(split_manifest_path)
    return {
        "split_manifest_path": str(split_manifest_path),
        "split_provenance": split_provenance
        if isinstance(split_provenance, dict)
        else {"split_source": "manifest", "split_seed": split.split_seed, "source_manifest": str(split_manifest_path)},
        "judge_requirements": judge_requirements
        if isinstance(judge_requirements, dict)
        else {
            "requires_openai_judge": family_requires_openai_judge(split.family_name),
            "openai_api_key_present": None,
            "run_ready": None,
        },
        "recommended_split_alignment": recommended_split_alignment
        if isinstance(recommended_split_alignment, dict)
        else _compare_split_to_recommended(split),
    }


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

    if getattr(args, "_split_manifest_explicit", False) is False and "split_manifest" in manifest:
        args.split_manifest = _resolve_manifest_path(manifest_path, str(manifest["split_manifest"]))
    if getattr(args, "_baseline_eval_summary_explicit", False) is False and "baseline_eval_summary" in manifest:
        args.baseline_eval_summary = _resolve_manifest_path(manifest_path, str(manifest["baseline_eval_summary"]))
    if getattr(args, "_out_explicit", False) is False and "out" in manifest:
        args.out = _resolve_manifest_path(manifest_path, str(manifest["out"]))
    if getattr(args, "_label_explicit", False) is False:
        args.label = str(manifest_value("label", args.label))
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


def build_family_override_summary(
    *,
    split_manifest_path: Path,
    baseline_eval_summary_path: Path,
    label: str,
    overrides: dict[str, Path],
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> dict[str, object]:
    split = _load_split_from_manifest(split_manifest_path)
    baseline_eval = load_json(baseline_eval_summary_path)
    task_groups = get_family_task_groups(split.family_name)
    baseline_per_task = _success_rates_from_eval(baseline_eval)
    baseline_stage = _build_stage_summary(baseline_per_task, task_groups)

    summary = {
        "family_name": split.family_name,
        "task_count": len(split.task_ids),
        "training_task_count": len(split.training_task_ids),
        "holdout_task_count": len(split.holdout_task_ids),
        "family_metadata": _build_family_metadata(split, task_groups),
        "split": split.to_dict(),
        "baseline": baseline_stage,
        "run_provenance": _load_run_provenance(split_manifest_path),
        "baseline_eval_summary_path": str(baseline_eval_summary_path),
        "split_manifest": str(split_manifest_path),
    }

    merged_stage = build_merged_stage(
        summary,
        base_stage="baseline",
        overrides=overrides,
    )
    summary[label] = merged_stage
    summary[f"{label}_meta"] = build_label_meta(
        summary,
        base_summary_path=baseline_eval_summary_path,
        base_stage="baseline",
        benchmark_blockers=benchmark_blockers,
        overrides=overrides,
        merged_stage=merged_stage,
    )
    return summary


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    explicit_flags = {
        "split_manifest": "--split-manifest" in sys.argv,
        "baseline_eval_summary": "--baseline-eval-summary" in sys.argv,
        "out": "--out" in sys.argv,
        "label": "--label" in sys.argv,
        "benchmark_blocker": "--benchmark-blocker" in sys.argv,
    }
    for key, value in explicit_flags.items():
        setattr(args, f"_{key}_explicit", value)
    args = _merge_manifest_args(args)
    if args.split_manifest is None:
        raise ValueError("A split manifest is required via --split-manifest or --manifest")
    if args.baseline_eval_summary is None:
        raise ValueError("A baseline eval summary is required via --baseline-eval-summary or --manifest")
    if args.out is None:
        raise ValueError("An output path is required via --out or --manifest")
    overrides = dict(parse_override_spec(spec) for spec in args.override)
    summary = build_family_override_summary(
        split_manifest_path=args.split_manifest,
        baseline_eval_summary_path=args.baseline_eval_summary,
        label=args.label,
        overrides=overrides,
        benchmark_blockers=args.benchmark_blocker,
    )
    errors, derived = validate_family_summary(summary, benchmark_blockers=args.benchmark_blocker)
    payload = {
        "ok": not errors,
        "out": str(args.out),
        "label": args.label,
        "derived": derived,
    }
    if errors:
        payload["errors"] = errors
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1

    write_json_atomic(args.out, summary)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
