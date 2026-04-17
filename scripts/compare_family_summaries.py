"""Compare two normalized family summary artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_family_summary import validate_family_summary  # noqa: E402
from scripts.json_io import write_json_atomic  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two family summary artifacts.")
    parser.add_argument("--summary-a", type=Path, required=True)
    parser.add_argument("--summary-b", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--benchmark-blocker", action="append", default=[])
    parser.add_argument(
        "--compare-stage",
        action="append",
        default=[],
        metavar="STAGE_A=STAGE_B",
        help="Compare a stage from summary A against a differently named stage from summary B.",
    )
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_family_summaries(
    summary_a: dict[str, object],
    summary_b: dict[str, object],
    *,
    label_a: str,
    label_b: str,
    benchmark_blockers: list[str] | tuple[str, ...] = (),
    compare_stage_specs: list[str] | tuple[str, ...] = (),
    summary_a_path: Path | str | None = None,
    summary_b_path: Path | str | None = None,
) -> tuple[list[str], dict[str, object]]:
    errors_a, derived_a = validate_family_summary(summary_a, benchmark_blockers=benchmark_blockers)
    errors_b, derived_b = validate_family_summary(summary_b, benchmark_blockers=benchmark_blockers)
    errors = [f"{label_a}: {error}" for error in errors_a] + [f"{label_b}: {error}" for error in errors_b]
    if errors:
        return errors, {}

    stage_names_a = set(derived_a["stage_metrics"])
    stage_names_b = set(derived_b["stage_metrics"])
    common_stages = sorted(stage_names_a & stage_names_b)
    stage_comparison = {
        stage_name: _compare_stage(
            stage_name,
            summary_a[stage_name],
            summary_b[stage_name],
            derived_a["stage_metrics"][stage_name],
            derived_b["stage_metrics"][stage_name],
        )
        for stage_name in common_stages
    }

    holdout_keys_a = set(derived_a["holdout_metrics"])
    holdout_keys_b = set(derived_b["holdout_metrics"])
    common_holdouts = sorted(holdout_keys_a & holdout_keys_b)
    holdout_comparison = {
        holdout_key: _compare_holdout(summary_a[holdout_key], summary_b[holdout_key])
        for holdout_key in common_holdouts
    }

    stage_meta_a = derived_a.get("stage_meta", {})
    stage_meta_b = derived_b.get("stage_meta", {})
    stage_meta_names_a = set(stage_meta_a)
    stage_meta_names_b = set(stage_meta_b)
    common_stage_meta_names = sorted(stage_meta_names_a & stage_meta_names_b)

    custom_stage_comparison, custom_holdout_comparison, custom_stage_errors = _build_custom_stage_comparisons(
        summary_a,
        summary_b,
        derived_a,
        derived_b,
        compare_stage_specs=compare_stage_specs,
    )
    errors.extend(custom_stage_errors)
    if errors:
        return errors, {}

    return errors, {
        "family_name_a": derived_a["family_name"],
        "family_name_b": derived_b["family_name"],
        "label_a": label_a,
        "label_b": label_b,
        "artifact_provenance": {
            label_a: _summarize_run_provenance(summary_a),
            label_b: _summarize_run_provenance(summary_b),
        },
        "artifact_stage_meta": {
            label_a: stage_meta_a,
            label_b: stage_meta_b,
        },
        "provenance_comparison": _compare_run_provenance(summary_a, summary_b),
        "benchmark_blockers": sorted({str(task_id) for task_id in benchmark_blockers}, key=int),
        "common_stage_names": common_stages,
        "stage_names_only_in_a": sorted(stage_names_a - stage_names_b),
        "stage_names_only_in_b": sorted(stage_names_b - stage_names_a),
        "common_stage_meta_names": common_stage_meta_names,
        "stage_meta_names_only_in_a": sorted(stage_meta_names_a - stage_meta_names_b),
        "stage_meta_names_only_in_b": sorted(stage_meta_names_b - stage_meta_names_a),
        "stage_meta_lineage_only_in_a": {
            meta_name: _summarize_stage_meta_lineage(
                stage_meta_a[meta_name],
                counterpart_summary_path=summary_b_path,
            )
            for meta_name in sorted(stage_meta_names_a - stage_meta_names_b)
        },
        "stage_meta_lineage_only_in_b": {
            meta_name: _summarize_stage_meta_lineage(
                stage_meta_b[meta_name],
                counterpart_summary_path=summary_a_path,
            )
            for meta_name in sorted(stage_meta_names_b - stage_meta_names_a)
        },
        "common_holdout_keys": common_holdouts,
        "holdout_keys_only_in_a": sorted(holdout_keys_a - holdout_keys_b),
        "holdout_keys_only_in_b": sorted(holdout_keys_b - holdout_keys_a),
        "stage_comparison": stage_comparison,
        "stage_meta_comparison": {
            meta_name: _compare_stage_meta(stage_meta_a[meta_name], stage_meta_b[meta_name])
            for meta_name in common_stage_meta_names
        },
        "holdout_comparison": holdout_comparison,
        "custom_stage_comparison": custom_stage_comparison,
        "custom_holdout_comparison": custom_holdout_comparison,
}


def _summarize_stage_meta_lineage(
    meta: dict[str, object],
    *,
    counterpart_summary_path: Path | str | None = None,
) -> dict[str, object]:
    base_summary = meta.get("base_summary")
    counterpart_summary = str(counterpart_summary_path) if counterpart_summary_path is not None else None
    normalized_base_summary = _normalize_path_for_compare(base_summary)
    normalized_counterpart_summary = _normalize_path_for_compare(counterpart_summary)
    return {
        "stage_name": meta.get("stage_name"),
        "base_summary": base_summary,
        "base_stage": meta.get("base_stage"),
        "base_run_provenance_split_source": meta.get("base_run_provenance_split_source"),
        "counterpart_summary_path": counterpart_summary,
        "normalized_base_summary": normalized_base_summary,
        "normalized_counterpart_summary_path": normalized_counterpart_summary,
        "base_summary_matches_counterpart_summary_path": (
            normalized_base_summary is not None
            and normalized_counterpart_summary is not None
            and normalized_base_summary == normalized_counterpart_summary
        ),
        "override_count": int(meta.get("override_count", 0)),
        "override_task_ids": [int(task_id) for task_id in meta.get("override_task_ids", [])],
        "benchmark_blockers": [str(task_id) for task_id in meta.get("benchmark_blockers", [])],
    }


def _normalize_path_for_compare(path_value: object) -> str | None:
    if not isinstance(path_value, str):
        return None
    path_text = path_value.strip()
    if not path_text:
        return None
    lowered = path_text.replace("\\", "/")
    if len(lowered) >= 3 and lowered[1:3] == ":/":
        return f"{lowered[0].lower()}:{lowered[2:]}".replace("//", "/")
    if lowered.startswith("/mnt/") and len(lowered) > 6 and lowered[5].isalpha() and lowered[6] == "/":
        return f"{lowered[5].lower()}:/{lowered[7:]}".replace("//", "/")
    return lowered


def _summarize_run_provenance(summary: dict[str, object]) -> dict[str, object]:
    run_provenance = summary.get("run_provenance")
    if not isinstance(run_provenance, dict):
        return {
            "present": False,
            "split_source": None,
            "split_seed": None,
            "split_manifest_path": None,
            "matches_recommended_split": None,
            "requires_openai_judge": None,
            "run_ready": None,
            "inferred_from_legacy_artifact": None,
        }

    split_provenance = run_provenance.get("split_provenance")
    recommended_alignment = run_provenance.get("recommended_split_alignment")
    judge_requirements = run_provenance.get("judge_requirements")
    inferred_flag = None
    if isinstance(split_provenance, dict) and "inferred_from_legacy_artifact" in split_provenance:
        inferred_flag = bool(split_provenance["inferred_from_legacy_artifact"])
    elif isinstance(judge_requirements, dict) and "inferred_from_legacy_artifact" in judge_requirements:
        inferred_flag = bool(judge_requirements["inferred_from_legacy_artifact"])

    return {
        "present": True,
        "split_source": split_provenance.get("split_source") if isinstance(split_provenance, dict) else None,
        "split_seed": split_provenance.get("split_seed") if isinstance(split_provenance, dict) else None,
        "split_manifest_path": run_provenance.get("split_manifest_path"),
        "matches_recommended_split": (
            recommended_alignment.get("matches_recommended_split")
            if isinstance(recommended_alignment, dict)
            else None
        ),
        "requires_openai_judge": (
            judge_requirements.get("requires_openai_judge")
            if isinstance(judge_requirements, dict)
            else None
        ),
        "run_ready": judge_requirements.get("run_ready") if isinstance(judge_requirements, dict) else None,
        "inferred_from_legacy_artifact": inferred_flag,
    }


def _compare_run_provenance(
    summary_a: dict[str, object],
    summary_b: dict[str, object],
) -> dict[str, object]:
    provenance_a = _summarize_run_provenance(summary_a)
    provenance_b = _summarize_run_provenance(summary_b)
    return {
        "provenance_present_in_a": provenance_a["present"],
        "provenance_present_in_b": provenance_b["present"],
        "same_split_source": provenance_a["split_source"] == provenance_b["split_source"],
        "same_split_seed": provenance_a["split_seed"] == provenance_b["split_seed"],
        "same_split_manifest_path": provenance_a["split_manifest_path"] == provenance_b["split_manifest_path"],
        "same_recommended_split_alignment": (
            provenance_a["matches_recommended_split"] == provenance_b["matches_recommended_split"]
        ),
        "same_requires_openai_judge": (
            provenance_a["requires_openai_judge"] == provenance_b["requires_openai_judge"]
        ),
        "same_run_ready": provenance_a["run_ready"] == provenance_b["run_ready"],
        "same_legacy_inference_state": (
            provenance_a["inferred_from_legacy_artifact"] == provenance_b["inferred_from_legacy_artifact"]
        ),
    }


def _compare_stage(
    stage_name: str,
    stage_summary_a: dict[str, object],
    stage_summary_b: dict[str, object],
    stage_metrics_a: dict[str, object],
    stage_metrics_b: dict[str, object],
) -> dict[str, object]:
    per_task_a = {str(task_id): float(value) for task_id, value in stage_summary_a["per_task_success_rate"].items()}
    per_task_b = {str(task_id): float(value) for task_id, value in stage_summary_b["per_task_success_rate"].items()}
    common_task_ids = sorted(set(per_task_a) & set(per_task_b), key=int)
    improved_task_ids = [int(task_id) for task_id in common_task_ids if per_task_b[task_id] > per_task_a[task_id]]
    regressed_task_ids = [int(task_id) for task_id in common_task_ids if per_task_b[task_id] < per_task_a[task_id]]
    unchanged_task_ids = [int(task_id) for task_id in common_task_ids if per_task_b[task_id] == per_task_a[task_id]]
    return {
        "stage_name": stage_name,
        "family_success_rate_a": float(stage_metrics_a["family_success_rate"]),
        "family_success_rate_b": float(stage_metrics_b["family_success_rate"]),
        "family_success_rate_delta_b_minus_a": float(stage_metrics_b["family_success_rate"]) - float(stage_metrics_a["family_success_rate"]),
        "effective_family_success_rate_excluding_blockers_a": float(stage_metrics_a["effective_family_success_rate_excluding_blockers"]),
        "effective_family_success_rate_excluding_blockers_b": float(stage_metrics_b["effective_family_success_rate_excluding_blockers"]),
        "effective_family_success_rate_excluding_blockers_delta_b_minus_a": float(stage_metrics_b["effective_family_success_rate_excluding_blockers"]) - float(stage_metrics_a["effective_family_success_rate_excluding_blockers"]),
        "success_task_count_a": int(stage_metrics_a["success_task_count"]),
        "success_task_count_b": int(stage_metrics_b["success_task_count"]),
        "success_task_count_delta_b_minus_a": int(stage_metrics_b["success_task_count"]) - int(stage_metrics_a["success_task_count"]),
        "improved_task_ids_b_over_a": improved_task_ids,
        "regressed_task_ids_b_over_a": regressed_task_ids,
        "unchanged_task_count": len(unchanged_task_ids),
        "task_group_comparison": _compare_task_groups(stage_summary_a, stage_summary_b),
    }


def _compare_holdout(holdout_a: dict[str, object], holdout_b: dict[str, object]) -> dict[str, object]:
    return {
        "grpo_success_rate_a": float(holdout_a["grpo_success_rate"]),
        "grpo_success_rate_b": float(holdout_b["grpo_success_rate"]),
        "grpo_success_rate_delta_b_minus_a": float(holdout_b["grpo_success_rate"]) - float(holdout_a["grpo_success_rate"]),
        "gain_vs_baseline_a": float(holdout_a["gain_vs_baseline"]),
        "gain_vs_baseline_b": float(holdout_b["gain_vs_baseline"]),
        "gain_vs_baseline_delta_b_minus_a": float(holdout_b["gain_vs_baseline"]) - float(holdout_a["gain_vs_baseline"]),
        "gain_vs_warmup_a": float(holdout_a["gain_vs_warmup"]),
        "gain_vs_warmup_b": float(holdout_b["gain_vs_warmup"]),
        "gain_vs_warmup_delta_b_minus_a": float(holdout_b["gain_vs_warmup"]) - float(holdout_a["gain_vs_warmup"]),
    }


def _compare_stage_meta(
    meta_a: dict[str, object],
    meta_b: dict[str, object],
) -> dict[str, object]:
    override_task_ids_a = [int(task_id) for task_id in meta_a.get("override_task_ids", [])]
    override_task_ids_b = [int(task_id) for task_id in meta_b.get("override_task_ids", [])]
    benchmark_blockers_a = [str(task_id) for task_id in meta_a.get("benchmark_blockers", [])]
    benchmark_blockers_b = [str(task_id) for task_id in meta_b.get("benchmark_blockers", [])]
    return {
        "stage_name_a": meta_a.get("stage_name"),
        "stage_name_b": meta_b.get("stage_name"),
        "base_summary_a": meta_a.get("base_summary"),
        "base_summary_b": meta_b.get("base_summary"),
        "base_stage_a": meta_a.get("base_stage"),
        "base_stage_b": meta_b.get("base_stage"),
        "base_run_provenance_split_source_a": meta_a.get("base_run_provenance_split_source"),
        "base_run_provenance_split_source_b": meta_b.get("base_run_provenance_split_source"),
        "override_count_a": int(meta_a.get("override_count", 0)),
        "override_count_b": int(meta_b.get("override_count", 0)),
        "override_count_delta_b_minus_a": int(meta_b.get("override_count", 0)) - int(meta_a.get("override_count", 0)),
        "override_task_ids_a": override_task_ids_a,
        "override_task_ids_b": override_task_ids_b,
        "benchmark_blockers_a": benchmark_blockers_a,
        "benchmark_blockers_b": benchmark_blockers_b,
        "same_base_summary": meta_a.get("base_summary") == meta_b.get("base_summary"),
        "same_base_stage": meta_a.get("base_stage") == meta_b.get("base_stage"),
        "same_base_run_provenance_split_source": (
            meta_a.get("base_run_provenance_split_source") == meta_b.get("base_run_provenance_split_source")
        ),
        "same_override_task_ids": override_task_ids_a == override_task_ids_b,
        "same_benchmark_blockers": benchmark_blockers_a == benchmark_blockers_b,
        "has_base_run_provenance_a": bool(meta_a.get("has_base_run_provenance")),
        "has_base_run_provenance_b": bool(meta_b.get("has_base_run_provenance")),
        "has_applied_override_metrics_a": bool(meta_a.get("has_applied_override_metrics")),
        "has_applied_override_metrics_b": bool(meta_b.get("has_applied_override_metrics")),
    }


def _compare_task_groups(
    stage_summary_a: dict[str, object],
    stage_summary_b: dict[str, object],
) -> dict[str, object]:
    task_groups_a = _read_task_groups(stage_summary_a)
    task_groups_b = _read_task_groups(stage_summary_b)
    group_names_a = set(task_groups_a)
    group_names_b = set(task_groups_b)
    common_group_names = sorted(group_names_a & group_names_b)
    return {
        "common_group_names": common_group_names,
        "group_names_only_in_a": sorted(group_names_a - group_names_b),
        "group_names_only_in_b": sorted(group_names_b - group_names_a),
        "groups": {
            group_name: _compare_task_group(task_groups_a[group_name], task_groups_b[group_name])
            for group_name in common_group_names
        },
    }


def _read_task_groups(stage_summary: dict[str, object]) -> dict[str, dict[str, object]]:
    task_groups = stage_summary.get("task_groups")
    if not isinstance(task_groups, dict):
        return {}
    return {str(group_name): group_value for group_name, group_value in task_groups.items() if isinstance(group_value, dict)}


def _compare_task_group(
    group_summary_a: dict[str, object],
    group_summary_b: dict[str, object],
) -> dict[str, object]:
    per_task_a = {
        str(task_id): float(value)
        for task_id, value in dict(group_summary_a.get("per_task_success_rate", {})).items()
    }
    per_task_b = {
        str(task_id): float(value)
        for task_id, value in dict(group_summary_b.get("per_task_success_rate", {})).items()
    }
    common_task_ids = sorted(set(per_task_a) & set(per_task_b), key=int)
    improved_task_ids = [int(task_id) for task_id in common_task_ids if per_task_b[task_id] > per_task_a[task_id]]
    regressed_task_ids = [int(task_id) for task_id in common_task_ids if per_task_b[task_id] < per_task_a[task_id]]
    unchanged_task_ids = [int(task_id) for task_id in common_task_ids if per_task_b[task_id] == per_task_a[task_id]]
    return {
        "success_rate_a": float(group_summary_a.get("success_rate", 0.0)),
        "success_rate_b": float(group_summary_b.get("success_rate", 0.0)),
        "success_rate_delta_b_minus_a": float(group_summary_b.get("success_rate", 0.0)) - float(group_summary_a.get("success_rate", 0.0)),
        "task_count_a": int(group_summary_a.get("task_count", len(per_task_a))),
        "task_count_b": int(group_summary_b.get("task_count", len(per_task_b))),
        "task_count_delta_b_minus_a": int(group_summary_b.get("task_count", len(per_task_b))) - int(group_summary_a.get("task_count", len(per_task_a))),
        "improved_task_ids_b_over_a": improved_task_ids,
        "regressed_task_ids_b_over_a": regressed_task_ids,
        "unchanged_task_count": len(unchanged_task_ids),
    }


def _build_custom_stage_comparisons(
    summary_a: dict[str, object],
    summary_b: dict[str, object],
    derived_a: dict[str, object],
    derived_b: dict[str, object],
    *,
    compare_stage_specs: list[str] | tuple[str, ...],
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    errors: list[str] = []
    stage_results: dict[str, object] = {}
    holdout_results: dict[str, object] = {}
    for spec in compare_stage_specs:
        stage_a, stage_b = _parse_stage_spec(spec)
        if stage_a not in derived_a["stage_metrics"]:
            errors.append(f"summary A is missing stage {stage_a!r}")
            continue
        if stage_b not in derived_b["stage_metrics"]:
            errors.append(f"summary B is missing stage {stage_b!r}")
            continue
        comparison_key = f"{stage_a}__vs__{stage_b}"
        stage_results[comparison_key] = _compare_stage(
            comparison_key,
            summary_a[stage_a],
            summary_b[stage_b],
            derived_a["stage_metrics"][stage_a],
            derived_b["stage_metrics"][stage_b],
        )

        holdout_key_a = _resolve_holdout_key(summary_a, stage_a)
        holdout_key_b = _resolve_holdout_key(summary_b, stage_b)
        if holdout_key_a and holdout_key_b:
            holdout_results[f"{holdout_key_a}__vs__{holdout_key_b}"] = _compare_holdout(
                summary_a[holdout_key_a],
                summary_b[holdout_key_b],
            )
    return stage_results, holdout_results, errors


def _parse_stage_spec(spec: str) -> tuple[str, str]:
    stage_a, sep, stage_b = spec.partition("=")
    if not sep or not stage_a.strip() or not stage_b.strip():
        raise ValueError(f"Invalid stage comparison spec: {spec!r}")
    return stage_a.strip(), stage_b.strip()


def _resolve_holdout_key(summary: dict[str, object], stage_name: str) -> str | None:
    if stage_name == "warmup_plus_grpo" and "holdout" in summary:
        return "holdout"
    candidate = f"{stage_name}_holdout"
    if candidate in summary:
        return candidate
    return None


def main() -> int:
    args = build_arg_parser().parse_args()
    summary_a = _load_json(args.summary_a)
    summary_b = _load_json(args.summary_b)
    errors, comparison = compare_family_summaries(
        summary_a,
        summary_b,
        label_a=args.label_a,
        label_b=args.label_b,
        benchmark_blockers=args.benchmark_blocker,
        compare_stage_specs=args.compare_stage,
        summary_a_path=args.summary_a,
        summary_b_path=args.summary_b,
    )
    payload = {
        "ok": not errors,
        "summary_a": str(args.summary_a),
        "summary_b": str(args.summary_b),
    }
    if errors:
        payload["errors"] = errors
    else:
        payload["comparison"] = comparison
    if args.out:
        write_json_atomic(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
