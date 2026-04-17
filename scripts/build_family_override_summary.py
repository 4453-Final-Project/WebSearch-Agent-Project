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
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--baseline-eval-summary", type=Path, required=True)
    parser.add_argument("--label", default="current_stack")
    parser.add_argument("--override", action="append", default=[], metavar="TASK_ID=METRICS_JSON")
    parser.add_argument("--benchmark-blocker", action="append", default=[], metavar="TASK_ID")
    parser.add_argument("--out", type=Path, required=True)
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
    args = build_arg_parser().parse_args()
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
