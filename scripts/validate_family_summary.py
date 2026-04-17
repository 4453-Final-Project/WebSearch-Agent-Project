"""Validate family summary artifacts and surface blocker-aware derived metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_ROOT = REPO_ROOT.parent / "outputs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_family_curriculum import _compare_split_to_recommended  # noqa: E402
from src.training.task_families import TaskSplit  # noqa: E402


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a family_summary.json artifact.")
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--benchmark-blocker", action="append", default=[])
    return parser


def validate_family_summary(
    summary: dict[str, object],
    *,
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    split = summary.get("split")
    split_task_ids: list[int] = []
    holdout_task_ids: list[int] = []
    split_dict: dict[str, object] | None = None
    if isinstance(split, dict):
        split_dict = split
        split_task_ids = [int(task_id) for task_id in split.get("eval_task_ids", [])]
        holdout_task_ids = [int(task_id) for task_id in split.get("holdout_task_ids", [])]
        _validate_split_metadata(summary, split, errors)
    else:
        errors.append("summary.split: missing split metadata")

    benchmark_blocker_set = {str(task_id) for task_id in benchmark_blockers}
    stage_metrics: dict[str, object] = {}
    stage_maps: dict[str, dict[str, float]] = {}
    detected_stage_names = _detect_stage_names(summary)
    for stage_name in detected_stage_names:
        stage_summary = summary[stage_name]
        if not isinstance(stage_summary, dict):
            continue
        stage_errors, metrics, per_task_success = _validate_stage_summary(
            stage_name,
            stage_summary,
            split_task_ids=split_task_ids,
            benchmark_blockers=benchmark_blocker_set,
        )
        errors.extend(stage_errors)
        stage_metrics[stage_name] = metrics
        stage_maps[stage_name] = per_task_success

    holdout_metrics: dict[str, object] = {}
    holdout_sections = _detect_holdout_sections(summary)
    for holdout_key, stage_name in holdout_sections.items():
        holdout = summary.get(holdout_key)
        if isinstance(holdout, dict):
            errors.extend(
                _validate_holdout_summary(
                    holdout_key,
                    holdout,
                    split_holdout_task_ids=holdout_task_ids,
                    baseline_map=stage_maps.get("baseline"),
                    warmup_map=stage_maps.get("warmup_only"),
                    stage_map=stage_maps.get(stage_name),
                )
            )
            holdout_metrics[holdout_key] = {
                "stage_name": stage_name,
                "grpo_success_rate": float(holdout.get("grpo_success_rate", 0.0)),
                "task_count": len(holdout.get("task_ids", [])),
            }
        elif holdout is not None:
            errors.append(f"summary.{holdout_key}: expected an object when present")

    stage_meta: dict[str, object] = {}
    meta_sections = _detect_meta_sections(summary)
    for meta_key, stage_name in meta_sections.items():
        meta_value = summary.get(meta_key)
        if isinstance(meta_value, dict):
            errors.extend(
                _validate_stage_meta(
                    meta_key,
                    meta_value,
                    stage_name=stage_name,
                    summary=summary,
                )
            )
            stage_meta[meta_key] = {
                "stage_name": stage_name,
                "base_summary": meta_value.get("base_summary"),
                "base_stage": meta_value.get("base_stage"),
                "override_count": meta_value.get("override_count"),
                "override_task_ids": [int(task_id) for task_id in meta_value.get("override_task_ids", [])],
                "benchmark_blockers": [str(task_id) for task_id in meta_value.get("benchmark_blockers", [])],
                "has_base_run_provenance": isinstance(meta_value.get("base_run_provenance"), dict),
                "has_applied_override_metrics": isinstance(meta_value.get("applied_override_metrics"), dict),
                "base_run_provenance_split_source": (
                    meta_value.get("base_run_provenance", {})
                    .get("split_provenance", {})
                    .get("split_source")
                    if isinstance(meta_value.get("base_run_provenance"), dict)
                    and isinstance(meta_value.get("base_run_provenance", {}).get("split_provenance"), dict)
                    else None
                ),
            }
        elif meta_value is not None:
            errors.append(f"summary.{meta_key}: expected an object when present")

    derived = {
        "family_name": summary.get("family_name"),
        "stage_metrics": stage_metrics,
        "holdout_metrics": holdout_metrics,
        "stage_meta": stage_meta,
        "benchmark_blockers": sorted(benchmark_blocker_set, key=int),
        "run_provenance": _validate_run_provenance(summary, split_dict, errors),
    }
    return errors, derived


def _detect_stage_names(summary: dict[str, object]) -> list[str]:
    stage_names: list[str] = []
    for key, value in summary.items():
        if isinstance(value, dict) and "family_success_rate" in value and "per_task_success_rate" in value:
            stage_names.append(key)
    return stage_names


def _detect_holdout_sections(summary: dict[str, object]) -> dict[str, str]:
    holdout_sections: dict[str, str] = {}
    if "holdout" in summary:
        holdout_sections["holdout"] = "warmup_plus_grpo"
    for key in summary:
        if key.endswith("_holdout"):
            stage_name = key[: -len("_holdout")]
            holdout_sections[key] = stage_name
    return holdout_sections


def _detect_meta_sections(summary: dict[str, object]) -> dict[str, str]:
    meta_sections: dict[str, str] = {}
    for key in summary:
        if key.endswith("_meta"):
            stage_name = key[: -len("_meta")]
            meta_sections[key] = stage_name
    return meta_sections


def _validate_split_metadata(summary: dict[str, object], split: dict[str, object], errors: list[str]) -> None:
    eval_task_ids = [int(task_id) for task_id in split.get("eval_task_ids", [])]
    warmup_task_ids = [int(task_id) for task_id in split.get("warmup_task_ids", [])]
    grpo_task_ids = [int(task_id) for task_id in split.get("grpo_task_ids", [])]
    holdout_task_ids = [int(task_id) for task_id in split.get("holdout_task_ids", [])]

    if summary.get("task_count") != len(eval_task_ids):
        errors.append(f"summary.task_count: expected {len(eval_task_ids)}, found {summary.get('task_count')!r}")
    if summary.get("training_task_count") != len(warmup_task_ids) + len(grpo_task_ids):
        errors.append(
            "summary.training_task_count: expected "
            f"{len(warmup_task_ids) + len(grpo_task_ids)}, found {summary.get('training_task_count')!r}"
        )
    if summary.get("holdout_task_count") != len(holdout_task_ids):
        errors.append(
            f"summary.holdout_task_count: expected {len(holdout_task_ids)}, found {summary.get('holdout_task_count')!r}"
        )


def _validate_run_provenance(
    summary: dict[str, object],
    split: dict[str, object] | None,
    errors: list[str],
) -> dict[str, object]:
    run_provenance = summary.get("run_provenance")
    if run_provenance is None:
        return {"present": False}
    if not isinstance(run_provenance, dict):
        errors.append("summary.run_provenance: expected object when present")
        return {"present": True}

    split_manifest_path = run_provenance.get("split_manifest_path")
    if split_manifest_path is not None and not isinstance(split_manifest_path, str):
        errors.append("summary.run_provenance.split_manifest_path: expected string when present")

    split_provenance = run_provenance.get("split_provenance")
    if split_provenance is not None and not isinstance(split_provenance, dict):
        errors.append("summary.run_provenance.split_provenance: expected object when present")
        split_provenance = None
    if isinstance(split_provenance, dict):
        split_source = split_provenance.get("split_source")
        if split_source is not None and not isinstance(split_source, str):
            errors.append("summary.run_provenance.split_provenance.split_source: expected string when present")
        split_seed = split_provenance.get("split_seed")
        if split_seed is not None and not isinstance(split_seed, int):
            errors.append("summary.run_provenance.split_provenance.split_seed: expected integer when present")
        if (
            isinstance(split_seed, int)
            and isinstance(split, dict)
            and isinstance(split.get("split_seed"), int)
            and split_seed != split.get("split_seed")
        ):
            errors.append(
                f"summary.run_provenance.split_provenance.split_seed: expected {split.get('split_seed')}, found {split_seed}"
            )
        inferred_from_legacy = split_provenance.get("inferred_from_legacy_artifact")
        if inferred_from_legacy is not None and not isinstance(inferred_from_legacy, bool):
            errors.append(
                "summary.run_provenance.split_provenance.inferred_from_legacy_artifact: expected boolean when present"
            )

    judge_requirements = run_provenance.get("judge_requirements")
    if judge_requirements is not None and not isinstance(judge_requirements, dict):
        errors.append("summary.run_provenance.judge_requirements: expected object when present")
        judge_requirements = None
    if isinstance(judge_requirements, dict):
        _assert_optional_bool(
            "summary.run_provenance.judge_requirements.requires_openai_judge",
            judge_requirements.get("requires_openai_judge"),
            errors,
        )
        _assert_optional_bool(
            "summary.run_provenance.judge_requirements.openai_api_key_present",
            judge_requirements.get("openai_api_key_present"),
            errors,
        )
        _assert_optional_bool(
            "summary.run_provenance.judge_requirements.run_ready",
            judge_requirements.get("run_ready"),
            errors,
        )
        _assert_optional_bool(
            "summary.run_provenance.judge_requirements.inferred_from_legacy_artifact",
            judge_requirements.get("inferred_from_legacy_artifact"),
            errors,
        )

    recommended_alignment = run_provenance.get("recommended_split_alignment")
    if recommended_alignment is not None and not isinstance(recommended_alignment, dict):
        errors.append("summary.run_provenance.recommended_split_alignment: expected object when present")
        recommended_alignment = None
    if isinstance(recommended_alignment, dict):
        matches_recommended_split = recommended_alignment.get("matches_recommended_split")
        if matches_recommended_split is not None and not isinstance(matches_recommended_split, bool):
            errors.append(
                "summary.run_provenance.recommended_split_alignment.matches_recommended_split: expected boolean when present"
            )
        recommended_split_seed = recommended_alignment.get("recommended_split_seed")
        if recommended_split_seed is not None and not isinstance(recommended_split_seed, int):
            errors.append(
                "summary.run_provenance.recommended_split_alignment.recommended_split_seed: expected integer when present"
            )
        differences = recommended_alignment.get("differences")
        if differences is not None and not isinstance(differences, dict):
            errors.append("summary.run_provenance.recommended_split_alignment.differences: expected object when present")
        split_obj = _build_task_split_if_possible(split)
        if split_obj is not None:
            expected_alignment = _compare_split_to_recommended(split_obj)
            if (
                isinstance(matches_recommended_split, bool)
                and matches_recommended_split != expected_alignment["matches_recommended_split"]
            ):
                errors.append(
                    "summary.run_provenance.recommended_split_alignment.matches_recommended_split: "
                    f"expected {expected_alignment['matches_recommended_split']!r}, found {matches_recommended_split!r}"
                )
            if (
                isinstance(recommended_split_seed, int)
                and recommended_split_seed != expected_alignment["recommended_split_seed"]
            ):
                errors.append(
                    "summary.run_provenance.recommended_split_alignment.recommended_split_seed: "
                    f"expected {expected_alignment['recommended_split_seed']}, found {recommended_split_seed}"
                )
            if isinstance(differences, dict) and differences != expected_alignment["differences"]:
                errors.append(
                    "summary.run_provenance.recommended_split_alignment.differences: "
                    f"expected {expected_alignment['differences']!r}, found {differences!r}"
                )

    return {
        "present": True,
        "split_manifest_path": split_manifest_path,
        "split_source": split_provenance.get("split_source") if isinstance(split_provenance, dict) else None,
        "split_seed": split_provenance.get("split_seed") if isinstance(split_provenance, dict) else None,
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
    }


def _build_task_split_if_possible(split: dict[str, object] | None) -> TaskSplit | None:
    if not isinstance(split, dict):
        return None
    required_fields = {
        "family_name",
        "task_ids",
        "warmup_task_ids",
        "grpo_task_ids",
        "holdout_task_ids",
        "eval_task_ids",
        "split_seed",
    }
    if not required_fields.issubset(split):
        return None
    return TaskSplit(
        family_name=str(split["family_name"]),
        task_ids=tuple(int(task_id) for task_id in split["task_ids"]),
        warmup_task_ids=tuple(int(task_id) for task_id in split["warmup_task_ids"]),
        grpo_task_ids=tuple(int(task_id) for task_id in split["grpo_task_ids"]),
        holdout_task_ids=tuple(int(task_id) for task_id in split["holdout_task_ids"]),
        eval_task_ids=tuple(int(task_id) for task_id in split["eval_task_ids"]),
        split_seed=int(split["split_seed"]),
    )


def _validate_stage_summary(
    stage_name: str,
    stage_summary: dict[str, object],
    *,
    split_task_ids: list[int],
    benchmark_blockers: set[str],
) -> tuple[list[str], dict[str, object], dict[str, float]]:
    errors: list[str] = []
    per_task_raw = stage_summary.get("per_task_success_rate")
    if not isinstance(per_task_raw, dict):
        errors.append(f"summary.{stage_name}.per_task_success_rate: missing per-task map")
        return errors, {}, {}

    per_task_success = {str(task_id): float(value) for task_id, value in per_task_raw.items()}
    actual_rate = float(stage_summary.get("family_success_rate", 0.0))
    expected_rate = _mean(per_task_success.values())
    if abs(actual_rate - expected_rate) > 1e-9:
        errors.append(
            f"summary.{stage_name}.family_success_rate: expected {expected_rate:.6f}, found {actual_rate:.6f}"
        )

    if split_task_ids:
        expected_task_keys = [str(task_id) for task_id in split_task_ids]
        if sorted(per_task_success, key=int) != expected_task_keys:
            errors.append(
                f"summary.{stage_name}.per_task_success_rate: expected task ids {expected_task_keys}, "
                f"found {sorted(per_task_success, key=int)}"
            )

    task_groups = stage_summary.get("task_groups")
    if isinstance(task_groups, dict):
        errors.extend(_validate_task_groups(stage_name, task_groups, per_task_success))

    unresolved_task_ids = [task_id for task_id, value in per_task_success.items() if value < 1.0]
    effective_map = {
        task_id: value for task_id, value in per_task_success.items() if task_id not in benchmark_blockers
    }
    metrics = {
        "family_success_rate": actual_rate,
        "task_count": len(per_task_success),
        "success_task_count": sum(1 for value in per_task_success.values() if value >= 1.0),
        "unresolved_task_ids": [int(task_id) for task_id in unresolved_task_ids],
        "effective_family_success_rate_excluding_blockers": _mean(effective_map.values()),
        "effective_task_count_excluding_blockers": len(effective_map),
        "unresolved_task_ids_excluding_blockers": [
            int(task_id) for task_id, value in effective_map.items() if value < 1.0
        ],
    }
    return errors, metrics, per_task_success


def _validate_task_groups(
    stage_name: str,
    task_groups: dict[str, object],
    per_task_success: dict[str, float],
) -> list[str]:
    errors: list[str] = []
    grouped_task_ids: set[str] = set()
    for group_name, group_value in task_groups.items():
        if not isinstance(group_value, dict):
            errors.append(f"summary.{stage_name}.task_groups.{group_name}: expected object")
            continue
        task_ids = [str(task_id) for task_id in group_value.get("task_ids", [])]
        task_count = int(group_value.get("task_count", len(task_ids)))
        success_rate = float(group_value.get("success_rate", 0.0))
        group_per_task_raw = group_value.get("per_task_success_rate")
        if not isinstance(group_per_task_raw, dict):
            errors.append(f"summary.{stage_name}.task_groups.{group_name}.per_task_success_rate: missing per-task map")
            continue
        group_per_task = {str(task_id): float(value) for task_id, value in group_per_task_raw.items()}
        if task_count != len(task_ids):
            errors.append(
                f"summary.{stage_name}.task_groups.{group_name}.task_count: expected {len(task_ids)}, found {task_count}"
            )
        if sorted(group_per_task, key=int) != sorted(task_ids, key=int):
            errors.append(
                f"summary.{stage_name}.task_groups.{group_name}.per_task_success_rate: expected task ids {sorted(task_ids, key=int)}, "
                f"found {sorted(group_per_task, key=int)}"
            )
        expected_rate = _mean(group_per_task.values())
        if abs(success_rate - expected_rate) > 1e-9:
            errors.append(
                f"summary.{stage_name}.task_groups.{group_name}.success_rate: expected {expected_rate:.6f}, found {success_rate:.6f}"
            )
        for task_id in task_ids:
            grouped_task_ids.add(task_id)
            if task_id not in per_task_success:
                errors.append(f"summary.{stage_name}.task_groups.{group_name}: unknown task id {task_id}")
                continue
            if abs(group_per_task[task_id] - per_task_success[task_id]) > 1e-9:
                errors.append(
                    f"summary.{stage_name}.task_groups.{group_name}.per_task_success_rate.{task_id}: "
                    f"expected {per_task_success[task_id]:.6f}, found {group_per_task[task_id]:.6f}"
                )
    if grouped_task_ids and grouped_task_ids != set(per_task_success):
        errors.append(
            f"summary.{stage_name}.task_groups: grouped task ids {sorted(grouped_task_ids, key=int)} do not match "
            f"stage task ids {sorted(per_task_success, key=int)}"
        )
    return errors


def _validate_holdout_summary(
    holdout_key: str,
    holdout: dict[str, object],
    *,
    split_holdout_task_ids: list[int],
    baseline_map: dict[str, float] | None,
    warmup_map: dict[str, float] | None,
    stage_map: dict[str, float] | None,
) -> list[str]:
    errors: list[str] = []
    holdout_task_ids = [int(task_id) for task_id in holdout.get("task_ids", [])]
    if split_holdout_task_ids and holdout_task_ids != split_holdout_task_ids:
        errors.append(
            f"summary.{holdout_key}.task_ids: expected {split_holdout_task_ids}, found {holdout_task_ids}"
        )

    if not holdout_task_ids:
        return errors

    if not baseline_map or not warmup_map or not stage_map:
        return errors

    holdout_keys = [str(task_id) for task_id in holdout_task_ids]
    expected_baseline = _mean(baseline_map[task_id] for task_id in holdout_keys)
    expected_warmup = _mean(warmup_map[task_id] for task_id in holdout_keys)
    expected_grpo = _mean(stage_map[task_id] for task_id in holdout_keys)

    _assert_close(f"summary.{holdout_key}.baseline_success_rate", float(holdout.get("baseline_success_rate", 0.0)), expected_baseline, errors)
    _assert_close(f"summary.{holdout_key}.warmup_success_rate", float(holdout.get("warmup_success_rate", 0.0)), expected_warmup, errors)
    _assert_close(f"summary.{holdout_key}.grpo_success_rate", float(holdout.get("grpo_success_rate", 0.0)), expected_grpo, errors)
    _assert_close(f"summary.{holdout_key}.gain_vs_baseline", float(holdout.get("gain_vs_baseline", 0.0)), expected_grpo - expected_baseline, errors)
    _assert_close(f"summary.{holdout_key}.gain_vs_warmup", float(holdout.get("gain_vs_warmup", 0.0)), expected_grpo - expected_warmup, errors)

    per_task = holdout.get("per_task")
    if isinstance(per_task, dict):
        expected_keys = sorted(holdout_keys, key=int)
        if sorted(per_task, key=int) != expected_keys:
            errors.append(
                f"summary.{holdout_key}.per_task: expected task ids {expected_keys}, found {sorted(per_task, key=int)}"
            )
        for task_id in expected_keys:
            metrics = per_task.get(task_id)
            if not isinstance(metrics, dict):
                errors.append(f"summary.{holdout_key}.per_task.{task_id}: expected object")
                continue
            _assert_close(
                f"summary.{holdout_key}.per_task.{task_id}.baseline_success_rate",
                float(metrics.get("baseline_success_rate", 0.0)),
                baseline_map[task_id],
                errors,
            )
            _assert_close(
                f"summary.{holdout_key}.per_task.{task_id}.warmup_success_rate",
                float(metrics.get("warmup_success_rate", 0.0)),
                warmup_map[task_id],
                errors,
            )
            _assert_close(
                f"summary.{holdout_key}.per_task.{task_id}.grpo_success_rate",
                float(metrics.get("grpo_success_rate", 0.0)),
                stage_map[task_id],
                errors,
            )
            _assert_close(
                f"summary.{holdout_key}.per_task.{task_id}.gain_vs_baseline",
                float(metrics.get("gain_vs_baseline", 0.0)),
                stage_map[task_id] - baseline_map[task_id],
                errors,
            )
            _assert_close(
                f"summary.{holdout_key}.per_task.{task_id}.gain_vs_warmup",
                float(metrics.get("gain_vs_warmup", 0.0)),
                stage_map[task_id] - warmup_map[task_id],
                errors,
            )
    return errors


def _validate_stage_meta(
    meta_key: str,
    meta_value: dict[str, object],
    *,
    stage_name: str,
    summary: dict[str, object],
) -> list[str]:
    errors: list[str] = []
    if stage_name not in summary or not isinstance(summary.get(stage_name), dict):
        errors.append(f"summary.{meta_key}: expected matching stage summary {stage_name!r}")

    base_summary = meta_value.get("base_summary")
    if base_summary is not None and not isinstance(base_summary, str):
        errors.append(f"summary.{meta_key}.base_summary: expected string when present")

    base_stage = meta_value.get("base_stage")
    if base_stage is not None and not isinstance(base_stage, str):
        errors.append(f"summary.{meta_key}.base_stage: expected string when present")
    if isinstance(base_stage, str) and base_stage not in summary:
        errors.append(f"summary.{meta_key}.base_stage: unknown stage {base_stage!r}")

    benchmark_blockers = meta_value.get("benchmark_blockers")
    normalized_blockers: list[str] = []
    if benchmark_blockers is not None:
        if not isinstance(benchmark_blockers, list):
            errors.append(f"summary.{meta_key}.benchmark_blockers: expected list when present")
        else:
            try:
                normalized_blockers = [str(task_id) for task_id in benchmark_blockers]
            except Exception:
                errors.append(f"summary.{meta_key}.benchmark_blockers: expected scalar task ids")

    override_count = meta_value.get("override_count")
    if override_count is not None and not isinstance(override_count, int):
        errors.append(f"summary.{meta_key}.override_count: expected integer when present")

    override_task_ids_raw = meta_value.get("override_task_ids")
    normalized_override_task_ids: list[str] = []
    if override_task_ids_raw is not None:
        if not isinstance(override_task_ids_raw, list):
            errors.append(f"summary.{meta_key}.override_task_ids: expected list when present")
        else:
            try:
                normalized_override_task_ids = [str(task_id) for task_id in override_task_ids_raw]
            except Exception:
                errors.append(f"summary.{meta_key}.override_task_ids: expected scalar task ids")

    override_metrics_paths = meta_value.get("override_metrics_paths")
    normalized_override_path_keys: list[str] = []
    if override_metrics_paths is not None:
        if not isinstance(override_metrics_paths, dict):
            errors.append(f"summary.{meta_key}.override_metrics_paths: expected object when present")
        else:
            normalized_override_path_keys = sorted((str(task_id) for task_id in override_metrics_paths), key=int)
            for task_id, metrics_path in override_metrics_paths.items():
                if not isinstance(metrics_path, str):
                    errors.append(
                        f"summary.{meta_key}.override_metrics_paths.{task_id}: expected string path"
                    )

    if isinstance(override_count, int) and normalized_override_task_ids and override_count != len(normalized_override_task_ids):
        errors.append(
            f"summary.{meta_key}.override_count: expected {len(normalized_override_task_ids)}, found {override_count}"
        )

    if normalized_override_task_ids and normalized_override_path_keys:
        if sorted(normalized_override_task_ids, key=int) != normalized_override_path_keys:
            errors.append(
                f"summary.{meta_key}.override_metrics_paths: expected task ids {sorted(normalized_override_task_ids, key=int)}, "
                f"found {normalized_override_path_keys}"
            )

    base_run_provenance = meta_value.get("base_run_provenance")
    if base_run_provenance is not None and not isinstance(base_run_provenance, dict):
        errors.append(f"summary.{meta_key}.base_run_provenance: expected object when present")

    applied_override_metrics = meta_value.get("applied_override_metrics")
    stage_summary = summary.get(stage_name)
    stage_applied_overrides = (
        stage_summary.get("applied_overrides")
        if isinstance(stage_summary, dict) and isinstance(stage_summary.get("applied_overrides"), dict)
        else None
    )
    if applied_override_metrics is not None and not isinstance(applied_override_metrics, dict):
        errors.append(f"summary.{meta_key}.applied_override_metrics: expected object when present")
    if isinstance(applied_override_metrics, dict):
        applied_override_keys = sorted((str(task_id) for task_id in applied_override_metrics), key=int)
        if normalized_override_task_ids and applied_override_keys != sorted(normalized_override_task_ids, key=int):
            errors.append(
                f"summary.{meta_key}.applied_override_metrics: expected task ids {sorted(normalized_override_task_ids, key=int)}, "
                f"found {applied_override_keys}"
            )
        if isinstance(stage_applied_overrides, dict) and applied_override_metrics != stage_applied_overrides:
            errors.append(
                f"summary.{meta_key}.applied_override_metrics: expected stage {stage_name!r} applied_overrides to match"
            )

    if isinstance(stage_applied_overrides, dict) and normalized_override_task_ids:
        stage_override_keys = sorted((str(task_id) for task_id in stage_applied_overrides), key=int)
        if stage_override_keys != sorted(normalized_override_task_ids, key=int):
            errors.append(
                f"summary.{meta_key}.override_task_ids: expected stage {stage_name!r} override ids {stage_override_keys}, "
                f"found {sorted(normalized_override_task_ids, key=int)}"
            )

    return errors


def _assert_close(label: str, actual: float, expected: float, errors: list[str], tolerance: float = 1e-9) -> None:
    if abs(actual - expected) > tolerance:
        errors.append(f"{label}: expected {expected:.6f}, found {actual:.6f}")


def _assert_optional_bool(label: str, value: object, errors: list[str]) -> None:
    if value is not None and not isinstance(value, bool):
        errors.append(f"{label}: expected boolean when present")


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def main() -> int:
    args = build_arg_parser().parse_args()
    summary_path = Path(args.summary_path)
    summary = _load_json(summary_path)
    errors, derived = validate_family_summary(summary, benchmark_blockers=args.benchmark_blocker)
    payload = {
        "ok": not errors,
        "summary_path": str(summary_path),
        "derived": derived,
    }
    if errors:
        payload["errors"] = errors
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 1
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
