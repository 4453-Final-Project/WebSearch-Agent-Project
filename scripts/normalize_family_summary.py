"""Backfill legacy family summary artifacts to the current schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.merge_family_reevals import build_stage_holdout_summary  # noqa: E402
from scripts.json_io import write_json_atomic  # noqa: E402
from scripts.run_family_curriculum import (  # noqa: E402
    _build_family_metadata,
    _build_split_provenance,
    _build_stage_summary,
    _compare_split_to_recommended,
)
from src.training.task_families import TaskSplit, family_requires_openai_judge, get_family_task_groups  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize a family summary to the current schema.")
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_family_summary(summary: dict[str, object]) -> dict[str, object]:
    split_obj = _build_split(summary)
    task_groups = get_family_task_groups(split_obj.family_name)
    split_manifest_payload = _load_split_manifest_payload(summary)

    normalized = dict(summary)
    normalized["family_metadata"] = _build_family_metadata(split_obj, task_groups)
    normalized["run_provenance"] = _build_run_provenance(summary, split_obj, split_manifest_payload)

    stage_names = _detect_stage_names(summary)
    for stage_name in stage_names:
        stage_summary = summary.get(stage_name)
        if not isinstance(stage_summary, dict):
            continue
        extras = {
            key: value
            for key, value in stage_summary.items()
            if key not in {"family_success_rate", "per_task_success_rate", "task_groups"}
        }
        rebuilt = _build_stage_summary(_coerce_per_task_success_rate(stage_summary), task_groups)
        rebuilt.update(extras)
        normalized[stage_name] = rebuilt

    if {"baseline", "warmup_only", "warmup_plus_grpo"}.issubset(stage_names):
        normalized["holdout"] = build_stage_holdout_summary(
            normalized,
            stage_per_task_success_rate=normalized["warmup_plus_grpo"]["per_task_success_rate"],
        )

    for stage_name in stage_names:
        if stage_name in {"baseline", "warmup_only", "warmup_plus_grpo"}:
            continue
        holdout = build_stage_holdout_summary(
            normalized,
            stage_per_task_success_rate=normalized[stage_name]["per_task_success_rate"],
        )
        if holdout is not None:
            normalized[f"{stage_name}_holdout"] = holdout

    return normalized


def _build_run_provenance(
    summary: dict[str, object],
    split_obj: TaskSplit,
    split_manifest_payload: dict[str, object] | None,
) -> dict[str, object]:
    existing = summary.get("run_provenance")
    if isinstance(existing, dict):
        provenance = dict(existing)
    else:
        provenance = {}

    if "split_manifest_path" not in provenance:
        split_manifest_path = summary.get("split_manifest")
        if isinstance(split_manifest_path, str):
            provenance["split_manifest_path"] = split_manifest_path

    if "split_provenance" not in provenance:
        manifest_split_provenance = None
        if isinstance(split_manifest_payload, dict):
            manifest_split_provenance = split_manifest_payload.get("split_provenance")
        if isinstance(manifest_split_provenance, dict):
            provenance["split_provenance"] = manifest_split_provenance
        else:
            recommended_alignment = _compare_split_to_recommended(split_obj)
            split_manifest_path = provenance.get("split_manifest_path")
            if recommended_alignment["matches_recommended_split"]:
                inferred = _build_split_provenance(split_obj, split_source="recommended")
            elif isinstance(split_manifest_path, str):
                inferred = _build_split_provenance(
                    split_obj,
                    split_source="legacy_manifest",
                    source_manifest=split_manifest_path,
                )
            else:
                inferred = _build_split_provenance(split_obj, split_source="legacy_summary")
            inferred["inferred_from_legacy_artifact"] = True
            provenance["split_provenance"] = inferred

    if "judge_requirements" not in provenance:
        manifest_judge_requirements = None
        if isinstance(split_manifest_payload, dict):
            manifest_judge_requirements = split_manifest_payload.get("judge_requirements")
        if isinstance(manifest_judge_requirements, dict):
            provenance["judge_requirements"] = manifest_judge_requirements
        else:
            provenance["judge_requirements"] = {
                "requires_openai_judge": family_requires_openai_judge(split_obj.family_name),
                "openai_api_key_present": None,
                "run_ready": None,
                "inferred_from_legacy_artifact": True,
            }

    if "recommended_split_alignment" not in provenance:
        manifest_alignment = None
        if isinstance(split_manifest_payload, dict):
            manifest_alignment = split_manifest_payload.get("recommended_split_alignment")
        provenance["recommended_split_alignment"] = (
            manifest_alignment if isinstance(manifest_alignment, dict) else _compare_split_to_recommended(split_obj)
        )

    return provenance


def _load_split_manifest_payload(summary: dict[str, object]) -> dict[str, object] | None:
    split_manifest_path = summary.get("split_manifest")
    if not isinstance(split_manifest_path, str):
        return None
    resolved_path = _resolve_local_path(split_manifest_path)
    if resolved_path is None or not resolved_path.exists():
        return None
    return _load_json(resolved_path)


def _resolve_local_path(path_str: str) -> Path | None:
    path = Path(path_str)
    if path.exists():
        return path
    if path_str.startswith("/mnt/"):
        parts = path_str.split("/")
        if len(parts) >= 4 and len(parts[2]) == 1:
            drive = parts[2].upper() + ":"
            remainder = parts[3:]
            windows_path = Path(f"{drive}\\", *remainder)
            if windows_path.exists():
                return windows_path
    return None


def _build_split(summary: dict[str, object]) -> TaskSplit:
    split = summary.get("split")
    if not isinstance(split, dict):
        raise ValueError("summary.split is required")
    return TaskSplit(
        family_name=str(split["family_name"]),
        task_ids=tuple(int(task_id) for task_id in split["task_ids"]),
        warmup_task_ids=tuple(int(task_id) for task_id in split["warmup_task_ids"]),
        grpo_task_ids=tuple(int(task_id) for task_id in split["grpo_task_ids"]),
        holdout_task_ids=tuple(int(task_id) for task_id in split["holdout_task_ids"]),
        eval_task_ids=tuple(int(task_id) for task_id in split["eval_task_ids"]),
        split_seed=int(split["split_seed"]),
    )


def _detect_stage_names(summary: dict[str, object]) -> set[str]:
    return {
        key
        for key, value in summary.items()
        if isinstance(value, dict) and "family_success_rate" in value and "per_task_success_rate" in value
    }


def _coerce_per_task_success_rate(stage_summary: dict[str, object]) -> dict[str, float]:
    per_task = stage_summary.get("per_task_success_rate", {})
    if not isinstance(per_task, dict):
        raise ValueError("stage summary is missing per_task_success_rate")
    return {str(task_id): float(value) for task_id, value in per_task.items()}


def main() -> int:
    args = build_arg_parser().parse_args()
    summary_path = args.summary_path
    out_path = args.out or summary_path
    summary = _load_json(summary_path)
    normalized = normalize_family_summary(summary)
    write_json_atomic(out_path, normalized)
    print(json.dumps({"summary_path": str(summary_path), "out": str(out_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
