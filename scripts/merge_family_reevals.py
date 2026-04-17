from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.json_io import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge targeted task re-evaluations into a family summary stage.",
    )
    parser.add_argument("--base-summary", type=Path, required=True)
    parser.add_argument(
        "--base-stage",
        default="warmup_plus_grpo",
        help="Stage inside the family summary to override.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="TASK_ID=METRICS_JSON",
        help="Per-task metrics override to merge into the selected stage.",
    )
    parser.add_argument(
        "--label",
        default="current_stack",
        help="Top-level key name for the merged stage in the output JSON.",
    )
    parser.add_argument(
        "--benchmark-blocker",
        action="append",
        default=[],
        metavar="TASK_ID",
        help="Task ids that should be called out as known benchmark blockers.",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_override_spec(spec: str) -> tuple[str, Path]:
    task_id, sep, metrics_path = spec.partition("=")
    if not sep or not task_id.strip() or not metrics_path.strip():
        raise ValueError(f"Invalid override spec: {spec!r}")
    return task_id.strip(), Path(metrics_path.strip())


def build_merged_stage(
    base_summary: dict[str, object],
    *,
    base_stage: str,
    overrides: dict[str, Path],
) -> dict[str, object]:
    stage = dict(base_summary[base_stage])
    per_task = dict(stage["per_task_success_rate"])
    applied_overrides: dict[str, object] = {}

    for task_id, metrics_path in overrides.items():
        metrics = load_json(metrics_path)
        per_task[task_id] = float(metrics["success_rate"])
        applied_overrides[task_id] = {
            "metrics_path": str(metrics_path),
            "success_rate": float(metrics["success_rate"]),
            "episodes": int(metrics["episodes"]),
            "average_steps": float(metrics["average_steps"]),
        }

    task_ids = sorted(per_task, key=int)
    family_success_rate = sum(float(per_task[task_id]) for task_id in task_ids) / len(task_ids)
    stage["family_success_rate"] = family_success_rate
    stage["per_task_success_rate"] = {task_id: float(per_task[task_id]) for task_id in task_ids}
    stage["applied_overrides"] = applied_overrides
    task_groups = _build_task_groups(base_summary, stage["per_task_success_rate"], base_stage=base_stage)
    if task_groups:
        stage["task_groups"] = task_groups
    return stage


def build_stage_holdout_summary(
    base_summary: dict[str, object],
    *,
    stage_per_task_success_rate: dict[str, float],
) -> dict[str, object] | None:
    split = base_summary.get("split")
    if not isinstance(split, dict):
        return None
    holdout_task_ids = [int(task_id) for task_id in split.get("holdout_task_ids", [])]
    if not holdout_task_ids:
        return None

    baseline = _as_float_per_task_map(base_summary.get("baseline", {}))
    warmup = _as_float_per_task_map(base_summary.get("warmup_only", {}))
    grpo = {str(task_id): float(value) for task_id, value in stage_per_task_success_rate.items()}
    holdout_keys = [str(task_id) for task_id in holdout_task_ids]

    baseline_rate = _mean(baseline[task_id] for task_id in holdout_keys)
    warmup_rate = _mean(warmup[task_id] for task_id in holdout_keys)
    grpo_rate = _mean(grpo[task_id] for task_id in holdout_keys)
    return {
        "task_ids": holdout_task_ids,
        "baseline_success_rate": baseline_rate,
        "warmup_success_rate": warmup_rate,
        "grpo_success_rate": grpo_rate,
        "gain_vs_baseline": grpo_rate - baseline_rate,
        "gain_vs_warmup": grpo_rate - warmup_rate,
        "per_task": {
            task_id: {
                "baseline_success_rate": baseline[task_id],
                "warmup_success_rate": warmup[task_id],
                "grpo_success_rate": grpo[task_id],
                "gain_vs_baseline": grpo[task_id] - baseline[task_id],
                "gain_vs_warmup": grpo[task_id] - warmup[task_id],
            }
            for task_id in holdout_keys
        },
    }


def build_label_meta(
    base_summary: dict[str, object],
    *,
    base_summary_path: Path,
    base_stage: str,
    benchmark_blockers: list[str] | tuple[str, ...],
    overrides: dict[str, Path],
    merged_stage: dict[str, object] | None = None,
) -> dict[str, object]:
    override_task_ids = sorted((int(task_id) for task_id in overrides), key=int)
    meta = {
        "base_summary": str(base_summary_path),
        "base_stage": base_stage,
        "benchmark_blockers": sorted({str(task_id) for task_id in benchmark_blockers}, key=int),
        "override_count": len(overrides),
        "override_task_ids": override_task_ids,
        "override_metrics_paths": {
            task_id: str(metrics_path)
            for task_id, metrics_path in sorted(overrides.items(), key=lambda item: int(item[0]))
        },
    }
    base_run_provenance = base_summary.get("run_provenance")
    if isinstance(base_run_provenance, dict):
        meta["base_run_provenance"] = base_run_provenance
    if isinstance(merged_stage, dict):
        applied_overrides = merged_stage.get("applied_overrides")
        if isinstance(applied_overrides, dict):
            meta["applied_override_metrics"] = applied_overrides
    return meta


def _build_task_groups(
    base_summary: dict[str, object],
    per_task_success_rate: dict[str, float],
    *,
    base_stage: str,
) -> dict[str, object]:
    base_stage_summary = base_summary.get(base_stage, {})
    task_group_sources: dict[str, list[int]] = {}
    if isinstance(base_stage_summary, dict) and isinstance(base_stage_summary.get("task_groups"), dict):
        for group_name, group_value in base_stage_summary["task_groups"].items():
            if isinstance(group_value, dict):
                task_group_sources[group_name] = [int(task_id) for task_id in group_value.get("task_ids", [])]
    elif isinstance(base_summary.get("family_metadata"), dict):
        family_metadata = base_summary["family_metadata"]
        judge_free_task_ids = [int(task_id) for task_id in family_metadata.get("judge_free_task_ids", [])]
        judge_gated_task_ids = [int(task_id) for task_id in family_metadata.get("judge_gated_task_ids", [])]
        if judge_free_task_ids:
            task_group_sources["judge_free"] = judge_free_task_ids
        if judge_gated_task_ids:
            task_group_sources["judge_gated"] = judge_gated_task_ids

    task_groups: dict[str, object] = {}
    for group_name, task_ids in task_group_sources.items():
        group_keys = [str(task_id) for task_id in task_ids if str(task_id) in per_task_success_rate]
        if not group_keys:
            continue
        task_groups[group_name] = {
            "task_ids": [int(task_id) for task_id in group_keys],
            "task_count": len(group_keys),
            "success_rate": _mean(per_task_success_rate[task_id] for task_id in group_keys),
            "per_task_success_rate": {task_id: float(per_task_success_rate[task_id]) for task_id in group_keys},
        }
    return task_groups


def _as_float_per_task_map(stage_summary: object) -> dict[str, float]:
    if not isinstance(stage_summary, dict):
        return {}
    per_task = stage_summary.get("per_task_success_rate", {})
    if not isinstance(per_task, dict):
        return {}
    return {str(task_id): float(value) for task_id, value in per_task.items()}


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    base_summary = load_json(args.base_summary)
    overrides = dict(parse_override_spec(spec) for spec in args.override)
    merged_stage = build_merged_stage(
        base_summary,
        base_stage=args.base_stage,
        overrides=overrides,
    )

    result = dict(base_summary)
    result[args.label] = merged_stage
    merged_holdout = build_stage_holdout_summary(
        base_summary,
        stage_per_task_success_rate=merged_stage["per_task_success_rate"],
    )
    if merged_holdout is not None:
        result[f"{args.label}_holdout"] = merged_holdout
    result[f"{args.label}_meta"] = build_label_meta(
        base_summary,
        base_summary_path=args.base_summary,
        base_stage=args.base_stage,
        benchmark_blockers=args.benchmark_blocker,
        overrides=overrides,
        merged_stage=merged_stage,
    )

    write_json_atomic(args.out, result)
    print(json.dumps({"out": str(args.out), "label": args.label, "family_success_rate": merged_stage["family_success_rate"]}, indent=2))


if __name__ == "__main__":
    main()
