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
    parser = argparse.ArgumentParser(description="Validate a subset family stage scorecard artifact.")
    parser.add_argument("--scorecard-path", required=True)
    parser.add_argument("--benchmark-blocker", action="append", default=[])
    return parser


def validate_subset_family_stage_scorecard(
    scorecard: dict[str, object],
    *,
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []

    if scorecard.get("scorecard_type") != "subset_family_stage_scorecard":
        errors.append(
            "scorecard.scorecard_type: expected 'subset_family_stage_scorecard', "
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
        errors.append(f"scorecard: expected exactly one stage summary, found {detected_stage_names!r}")
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
            errors.append(f"scorecard: expected only meta section {expected_meta_key!r}, found {sorted(meta_sections)!r}")

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
        _validate_scorecard_meta_invariants(
            meta_key,
            meta_value,
            split_task_ids=split_task_ids,
            stage_map=stage_maps.get(stage_name, {}),
            errors=errors,
        )
        stage_meta[meta_key] = {
            "stage_name": stage_name,
            "source_summary": meta_value.get("source_summary"),
            "source_stage": meta_value.get("source_stage"),
            "matched_task_count": meta_value.get("matched_task_count"),
            "omitted_source_task_count": meta_value.get("omitted_source_task_count"),
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


def _validate_scorecard_meta_invariants(
    meta_key: str,
    meta_value: dict[str, object],
    *,
    split_task_ids: list[int],
    stage_map: dict[str, float],
    errors: list[str],
) -> None:
    matched_task_ids = _normalize_task_id_list(meta_value.get("matched_task_ids"), f"scorecard.{meta_key}.matched_task_ids", errors)
    omitted_source_task_ids = _normalize_task_id_list(
        meta_value.get("omitted_source_task_ids"),
        f"scorecard.{meta_key}.omitted_source_task_ids",
        errors,
    )

    matched_task_count = meta_value.get("matched_task_count")
    if matched_task_count is not None and not isinstance(matched_task_count, int):
        errors.append(f"scorecard.{meta_key}.matched_task_count: expected integer when present")
    if isinstance(matched_task_count, int) and matched_task_ids and matched_task_count != len(matched_task_ids):
        errors.append(
            f"scorecard.{meta_key}.matched_task_count: expected {len(matched_task_ids)}, found {matched_task_count}"
        )

    omitted_task_count = meta_value.get("omitted_source_task_count")
    if omitted_task_count is not None and not isinstance(omitted_task_count, int):
        errors.append(f"scorecard.{meta_key}.omitted_source_task_count: expected integer when present")
    if isinstance(omitted_task_count, int) and omitted_source_task_ids and omitted_task_count != len(omitted_source_task_ids):
        errors.append(
            f"scorecard.{meta_key}.omitted_source_task_count: expected {len(omitted_source_task_ids)}, found {omitted_task_count}"
        )

    split_set = {str(task_id) for task_id in split_task_ids}
    matched_set = set(matched_task_ids)
    omitted_set = set(omitted_source_task_ids)
    overlap = sorted(matched_set.intersection(omitted_set), key=int)
    if overlap:
        errors.append(f"scorecard.{meta_key}: matched and omitted source task ids overlap: {overlap}")

    if split_task_ids and matched_set != split_set:
        errors.append(
            f"scorecard.{meta_key}: matched task ids {sorted(matched_set, key=int)} do not match split task ids {sorted(split_set, key=int)}"
        )

    if stage_map and sorted(stage_map, key=int) != sorted(split_set, key=int):
        errors.append(
            f"scorecard.{meta_key}: stage task ids {sorted(stage_map, key=int)} do not match split task ids {sorted(split_set, key=int)}"
        )


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
    errors, derived = validate_subset_family_stage_scorecard(
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
