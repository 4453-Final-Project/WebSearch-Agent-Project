from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_family_summary import (  # noqa: E402
    _detect_meta_sections,
    _detect_stage_names,
    _load_json,
    _validate_run_provenance,
    _validate_split_metadata,
    _validate_stage_meta,
    _validate_stage_summary,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a combined family stage scorecard artifact.")
    parser.add_argument("--scorecard-path", required=True)
    parser.add_argument("--benchmark-blocker", action="append", default=[])
    return parser


def validate_combined_family_stage_scorecard(
    scorecard: dict[str, object],
    *,
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []

    if scorecard.get("scorecard_type") != "combined_family_stage_scorecard":
        errors.append(
            "scorecard.scorecard_type: expected 'combined_family_stage_scorecard', "
            f"found {scorecard.get('scorecard_type')!r}"
        )

    split = scorecard.get("split")
    split_dict: dict[str, object] | None = None
    split_task_ids: list[int] = []
    if isinstance(split, dict):
        split_dict = split
        split_task_ids = [int(task_id) for task_id in split.get("eval_task_ids", [])]
        _validate_split_metadata(scorecard, split, errors)
    else:
        errors.append("scorecard.split: missing split metadata")

    detected_stage_names = _detect_stage_names(scorecard)
    if len(detected_stage_names) != 1:
        errors.append(
            "scorecard: expected exactly one stage summary, "
            f"found {detected_stage_names!r}"
        )
    label = detected_stage_names[0] if detected_stage_names else None

    benchmark_blocker_set = {str(task_id) for task_id in benchmark_blockers}
    stage_metrics: dict[str, object] = {}
    stage_maps: dict[str, dict[str, float]] = {}
    for stage_name in detected_stage_names:
        stage_summary = scorecard.get(stage_name)
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

    meta_sections = _detect_meta_sections(scorecard)
    if label is not None:
        expected_meta_key = f"{label}_meta"
        if meta_sections != {expected_meta_key: label}:
            errors.append(
                f"scorecard: expected only meta section {expected_meta_key!r}, found {sorted(meta_sections)!r}"
            )

    stage_meta: dict[str, object] = {}
    for meta_key, stage_name in meta_sections.items():
        meta_value = scorecard.get(meta_key)
        if not isinstance(meta_value, dict):
            errors.append(f"scorecard.{meta_key}: expected object")
            continue
        errors.extend(
            _validate_stage_meta(
                meta_key,
                meta_value,
                stage_name=stage_name,
                summary=scorecard,
            )
        )
        _validate_combined_scorecard_meta_invariants(
            meta_key,
            meta_value,
            summary=scorecard,
            split_task_ids=split_task_ids,
            stage_map=stage_maps.get(stage_name, {}),
            errors=errors,
        )
        stage_meta[meta_key] = {
            "stage_name": stage_name,
            "source_count": meta_value.get("source_count"),
            "source_names": sorted(str(source_name) for source_name in meta_value.get("sources", {}).keys())
            if isinstance(meta_value.get("sources"), dict)
            else [],
            "overlap_task_ids": sorted(
                (int(task_id) for task_id in meta_value.get("overlap_task_ids", {}).keys()),
                key=int,
            )
            if isinstance(meta_value.get("overlap_task_ids"), dict)
            else [],
        }

    derived = {
        "family_name": scorecard.get("family_name"),
        "label": label,
        "stage_metrics": stage_metrics,
        "stage_meta": stage_meta,
        "benchmark_blockers": sorted(benchmark_blocker_set, key=int),
        "run_provenance": _validate_run_provenance(scorecard, split_dict, errors),
    }
    return errors, derived


def _validate_combined_scorecard_meta_invariants(
    meta_key: str,
    meta_value: dict[str, object],
    *,
    summary: dict[str, object],
    split_task_ids: list[int],
    stage_map: dict[str, float],
    errors: list[str],
) -> None:
    source_count = meta_value.get("source_count")
    if source_count is not None and not isinstance(source_count, int):
        errors.append(f"scorecard.{meta_key}.source_count: expected integer when present")

    sources = meta_value.get("sources")
    if not isinstance(sources, dict):
        errors.append(f"scorecard.{meta_key}.sources: expected object")
        return
    if isinstance(source_count, int) and source_count != len(sources):
        errors.append(
            f"scorecard.{meta_key}.source_count: expected {len(sources)}, found {source_count}"
        )

    overlap_task_ids = meta_value.get("overlap_task_ids")
    if not isinstance(overlap_task_ids, dict):
        errors.append(f"scorecard.{meta_key}.overlap_task_ids: expected object")
        overlap_task_ids = {}

    stage_name = meta_key[: -len("_meta")]
    stage_summary = summary.get(stage_name)
    source_coverage = {}
    if isinstance(stage_summary, dict):
        source_coverage = stage_summary.get("source_coverage", {})
    if not isinstance(source_coverage, dict):
        errors.append(f"scorecard.{stage_name}.source_coverage: expected object")
        source_coverage = {}

    split_set = {str(task_id) for task_id in split_task_ids}
    covered_task_ids: set[str] = set()
    source_names = set(str(source_name) for source_name in sources)

    for source_name, source_value in sorted(sources.items()):
        source_label = f"scorecard.{meta_key}.sources.{source_name}"
        if not isinstance(source_value, dict):
            errors.append(f"{source_label}: expected object")
            continue

        summary_path = source_value.get("summary_path")
        if not isinstance(summary_path, str):
            errors.append(f"{source_label}.summary_path: expected string")
        stage_name_value = source_value.get("stage_name")
        if not isinstance(stage_name_value, str):
            errors.append(f"{source_label}.stage_name: expected string")

        matched_task_ids = _normalize_task_id_list(
            source_value.get("matched_task_ids"),
            f"{source_label}.matched_task_ids",
            errors,
        )
        covered_task_ids.update(matched_task_ids)
        matched_task_count = source_value.get("matched_task_count")
        if matched_task_count is not None and not isinstance(matched_task_count, int):
            errors.append(f"{source_label}.matched_task_count: expected integer when present")
        if isinstance(matched_task_count, int) and matched_task_count != len(matched_task_ids):
            errors.append(
                f"{source_label}.matched_task_count: expected {len(matched_task_ids)}, found {matched_task_count}"
            )

        if split_set and not set(matched_task_ids).issubset(split_set):
            errors.append(
                f"{source_label}.matched_task_ids: expected subset of split task ids, found {sorted(set(matched_task_ids) - split_set, key=int)}"
            )

        stage_source_value = source_coverage.get(source_name)
        if not isinstance(stage_source_value, dict):
            errors.append(f"scorecard.{stage_name}.source_coverage.{source_name}: expected object")
            continue

        stage_matched_task_ids = _normalize_task_id_list(
            stage_source_value.get("matched_task_ids"),
            f"scorecard.{stage_name}.source_coverage.{source_name}.matched_task_ids",
            errors,
        )
        stage_matched_task_count = stage_source_value.get("matched_task_count")
        if stage_matched_task_count is not None and not isinstance(stage_matched_task_count, int):
            errors.append(
                f"scorecard.{stage_name}.source_coverage.{source_name}.matched_task_count: expected integer when present"
            )
        if isinstance(stage_matched_task_count, int) and stage_matched_task_count != len(stage_matched_task_ids):
            errors.append(
                "scorecard."
                f"{stage_name}.source_coverage.{source_name}.matched_task_count: expected {len(stage_matched_task_ids)}, "
                f"found {stage_matched_task_count}"
            )
        if matched_task_ids != stage_matched_task_ids:
            errors.append(
                f"scorecard.{meta_key}.sources.{source_name}.matched_task_ids does not match "
                f"scorecard.{stage_name}.source_coverage.{source_name}.matched_task_ids"
            )

    if set(source_coverage) != source_names:
        errors.append(
            f"scorecard.{stage_name}.source_coverage: expected sources {sorted(source_names)!r}, found {sorted(source_coverage)!r}"
        )

    if split_set and covered_task_ids != split_set:
        errors.append(
            f"scorecard.{meta_key}.sources: matched task ids {sorted(covered_task_ids, key=int)} do not match split task ids {sorted(split_set, key=int)}"
        )
    if stage_map and sorted(stage_map, key=int) != sorted(split_set, key=int):
        errors.append(
            f"scorecard.{meta_key}: stage task ids {sorted(stage_map, key=int)} do not match split task ids {sorted(split_set, key=int)}"
        )

    for task_id, source_list in sorted(overlap_task_ids.items(), key=lambda item: int(item[0])):
        overlap_label = f"scorecard.{meta_key}.overlap_task_ids.{task_id}"
        if task_id not in split_set:
            errors.append(f"{overlap_label}: expected task id in split")
        if not isinstance(source_list, list):
            errors.append(f"{overlap_label}: expected list of source names")
            continue
        normalized_source_list = [str(source_name) for source_name in source_list]
        if len(normalized_source_list) < 2:
            errors.append(f"{overlap_label}: expected at least two covering sources")
        unknown_sources = sorted(set(normalized_source_list) - source_names)
        if unknown_sources:
            errors.append(f"{overlap_label}: unknown source names {unknown_sources!r}")
        for source_name in normalized_source_list:
            source_value = sources.get(source_name)
            if not isinstance(source_value, dict):
                continue
            matched_task_ids = [str(task) for task in source_value.get("matched_task_ids", [])]
            if task_id not in matched_task_ids:
                errors.append(f"{overlap_label}: source {source_name!r} does not cover task {task_id}")


def _normalize_task_id_list(value: object, label: str, errors: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        errors.append(f"{label}: expected list when present")
        return []
    try:
        return [str(task_id) for task_id in value]
    except Exception:
        errors.append(f"{label}: expected scalar task ids")
        return []


def main() -> int:
    args = build_arg_parser().parse_args()
    scorecard_path = Path(args.scorecard_path)
    scorecard = _load_json(scorecard_path)
    errors, derived = validate_combined_family_stage_scorecard(
        scorecard,
        benchmark_blockers=args.benchmark_blocker,
    )
    payload = {
        "ok": not errors,
        "scorecard_path": str(scorecard_path),
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
