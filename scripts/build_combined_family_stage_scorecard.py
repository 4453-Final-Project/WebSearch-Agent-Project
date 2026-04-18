from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_family_override_summary import _load_run_provenance  # noqa: E402
from scripts.json_io import write_json_atomic  # noqa: E402
from scripts.merge_family_reevals import load_json  # noqa: E402
from scripts.run_family_curriculum import _build_family_metadata, _build_stage_summary, _load_split_from_manifest  # noqa: E402
from scripts.validate_combined_family_stage_scorecard import (  # noqa: E402
    validate_combined_family_stage_scorecard as _validate_combined_family_stage_scorecard,
)
from src.training.task_families import get_family_task_groups  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a stage scorecard for a family that is the union of multiple validated source artifacts."
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--combined-split-manifest", type=Path, default=None)
    parser.add_argument("--label", default="current_stack")
    parser.add_argument("--source", action="append", default=[], metavar="NAME=SUMMARY_JSON")
    parser.add_argument("--source-stage", action="append", default=[], metavar="NAME=STAGE")
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


def _parse_source_spec(spec: str) -> tuple[str, Path]:
    name, sep, summary_path = spec.partition("=")
    if not sep or not name.strip() or not summary_path.strip():
        raise ValueError(f"Invalid source spec: {spec!r}")
    return name.strip(), Path(summary_path.strip())


def _parse_source_stage_spec(spec: str) -> tuple[str, str]:
    name, sep, stage_name = spec.partition("=")
    if not sep or not name.strip() or not stage_name.strip():
        raise ValueError(f"Invalid source-stage spec: {spec!r}")
    return name.strip(), stage_name.strip()


def _merge_manifest_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.manifest is None:
        return args

    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)

    if args.combined_split_manifest is None and "combined_split_manifest" in manifest:
        args.combined_split_manifest = _resolve_manifest_path(
            manifest_path,
            str(manifest["combined_split_manifest"]),
        )
    if args.out is None and "out" in manifest:
        args.out = _resolve_manifest_path(manifest_path, str(manifest["out"]))
    if args.audit_out is None and "audit_out" in manifest:
        args.audit_out = _resolve_manifest_path(manifest_path, str(manifest["audit_out"]))
    if args.label == "current_stack" and "label" in manifest:
        args.label = str(manifest["label"])
    if not args.benchmark_blocker and "benchmark_blockers" in manifest:
        args.benchmark_blocker = [str(task_id) for task_id in manifest["benchmark_blockers"]]

    manifest_sources = manifest.get("sources", {})
    if not isinstance(manifest_sources, dict):
        raise ValueError(f"manifest.sources must be an object: {manifest_path}")

    merged_sources: dict[str, str] = {}
    merged_source_stages: dict[str, str] = {}
    for source_name, source_value in manifest_sources.items():
        if not isinstance(source_value, dict):
            raise ValueError(f"manifest.sources.{source_name} must be an object: {manifest_path}")
        summary_path = source_value.get("summary")
        if not isinstance(summary_path, str):
            raise ValueError(f"manifest.sources.{source_name}.summary must be a string: {manifest_path}")
        merged_sources[str(source_name)] = str(_resolve_manifest_path(manifest_path, summary_path))
        merged_source_stages[str(source_name)] = str(source_value.get("stage", args.label))

    for source_name, summary_path in (_parse_source_spec(spec) for spec in args.source):
        merged_sources[source_name] = str(summary_path)
    for source_name, stage_name in (_parse_source_stage_spec(spec) for spec in args.source_stage):
        merged_source_stages[source_name] = stage_name

    args.source = [f"{source_name}={summary_path}" for source_name, summary_path in sorted(merged_sources.items())]
    args.source_stage = [
        f"{source_name}={merged_source_stages.get(source_name, args.label)}"
        for source_name in sorted(merged_sources)
    ]
    return args


def _as_float_per_task_map(stage_summary: object) -> dict[str, float]:
    if not isinstance(stage_summary, dict):
        return {}
    per_task = stage_summary.get("per_task_success_rate", {})
    if not isinstance(per_task, dict):
        return {}
    return {str(task_id): float(value) for task_id, value in per_task.items()}


def build_combined_family_stage_scorecard(
    *,
    combined_split_manifest_path: Path,
    label: str,
    source_specs: dict[str, tuple[Path, str]],
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> tuple[dict[str, object], dict[str, object]]:
    split = _load_split_from_manifest(combined_split_manifest_path)
    task_groups = get_family_task_groups(split.family_name)
    split_task_ids = [str(task_id) for task_id in split.task_ids]

    source_stage_maps: dict[str, dict[str, float]] = {}
    for source_name, (summary_path, stage_name) in sorted(source_specs.items()):
        summary = load_json(summary_path)
        stage_map = _as_float_per_task_map(summary.get(stage_name))
        if not stage_map:
            raise ValueError(f"Source {source_name!r} is missing stage {stage_name!r}: {summary_path}")
        source_stage_maps[source_name] = stage_map

    merged_per_task: dict[str, float] = {}
    chosen_source_for_task: dict[str, str] = {}
    overlap_task_ids: dict[str, list[str]] = {}
    for task_id in split_task_ids:
        covering_sources = [
            source_name for source_name, stage_map in sorted(source_stage_maps.items()) if task_id in stage_map
        ]
        if not covering_sources:
            raise ValueError(f"Combined family task {task_id} is missing from every source stage")
        candidate_scores = {source_name: source_stage_maps[source_name][task_id] for source_name in covering_sources}
        if len({round(score, 12) for score in candidate_scores.values()}) != 1:
            raise ValueError(f"Combined family task {task_id} has inconsistent source scores: {candidate_scores}")
        merged_per_task[task_id] = next(iter(candidate_scores.values()))
        chosen_source_for_task[task_id] = covering_sources[0]
        if len(covering_sources) > 1:
            overlap_task_ids[task_id] = covering_sources

    stage_summary = _build_stage_summary(merged_per_task, task_groups)
    stage_summary["source_coverage"] = {
        source_name: {
            "matched_task_count": sum(1 for task_id in split_task_ids if task_id in stage_map),
            "matched_task_ids": [int(task_id) for task_id in split_task_ids if task_id in stage_map],
        }
        for source_name, stage_map in sorted(source_stage_maps.items())
    }

    scorecard = {
        "scorecard_type": "combined_family_stage_scorecard",
        "family_name": split.family_name,
        "task_count": len(split.task_ids),
        "training_task_count": len(split.training_task_ids),
        "holdout_task_count": len(split.holdout_task_ids),
        "family_metadata": _build_family_metadata(split, task_groups),
        "split": split.to_dict(),
        "run_provenance": _load_run_provenance(combined_split_manifest_path),
        "combined_split_manifest": str(combined_split_manifest_path),
        label: stage_summary,
        f"{label}_meta": {
            "benchmark_blockers": sorted({str(task_id) for task_id in benchmark_blockers}, key=int),
            "source_count": len(source_specs),
            "sources": {
                source_name: {
                    "summary_path": str(summary_path),
                    "stage_name": stage_name,
                    "matched_task_count": sum(1 for task_id in split_task_ids if task_id in source_stage_maps[source_name]),
                    "matched_task_ids": [int(task_id) for task_id in split_task_ids if task_id in source_stage_maps[source_name]],
                }
                for source_name, (summary_path, stage_name) in sorted(source_specs.items())
            },
            "overlap_task_ids": {
                task_id: covering_sources
                for task_id, covering_sources in sorted(overlap_task_ids.items(), key=lambda item: int(item[0]))
            },
        },
    }

    audit = {
        "audit_type": "combined_family_stage_scorecard_audit",
        "family_name": split.family_name,
        "label": label,
        "scorecard_success_rate": stage_summary["family_success_rate"],
        "task_count": len(split.task_ids),
        "success_task_count": sum(1 for value in merged_per_task.values() if float(value) >= 1.0),
        "unresolved_task_ids": [int(task_id) for task_id, value in merged_per_task.items() if float(value) < 1.0],
        "source_contributions": {
            source_name: {
                "summary_path": str(summary_path),
                "stage_name": stage_name,
                "matched_task_count": sum(1 for task_id in split_task_ids if task_id in source_stage_maps[source_name]),
                "matched_task_ids": [int(task_id) for task_id in split_task_ids if task_id in source_stage_maps[source_name]],
                "unique_task_count": sum(1 for task_id, chosen in chosen_source_for_task.items() if chosen == source_name),
                "unique_task_ids": [int(task_id) for task_id, chosen in chosen_source_for_task.items() if chosen == source_name],
                "overlap_task_ids": [int(task_id) for task_id, covering_sources in overlap_task_ids.items() if source_name in covering_sources],
            }
            for source_name, (summary_path, stage_name) in sorted(source_specs.items())
        },
        "consistent_overlap_task_ids": [int(task_id) for task_id in sorted(overlap_task_ids, key=int)],
    }
    return scorecard, audit


def validate_combined_family_stage_scorecard(
    scorecard: dict[str, object],
    *,
    benchmark_blockers: list[str] | tuple[str, ...] = (),
) -> tuple[list[str], dict[str, object]]:
    return _validate_combined_family_stage_scorecard(scorecard, benchmark_blockers=benchmark_blockers)


def main() -> int:
    args = _merge_manifest_args(build_arg_parser().parse_args())
    if args.combined_split_manifest is None:
        raise ValueError("A combined split manifest is required via --combined-split-manifest or --manifest")
    if args.out is None:
        raise ValueError("An output path is required via --out or --manifest")

    source_paths = dict(_parse_source_spec(spec) for spec in args.source)
    source_stages = dict(_parse_source_stage_spec(spec) for spec in args.source_stage)
    source_specs = {
        source_name: (summary_path, source_stages.get(source_name, args.label))
        for source_name, summary_path in sorted(source_paths.items())
    }
    if not source_specs:
        raise ValueError("At least one source is required via --source or --manifest")

    scorecard, audit = build_combined_family_stage_scorecard(
        combined_split_manifest_path=args.combined_split_manifest,
        label=args.label,
        source_specs=source_specs,
        benchmark_blockers=args.benchmark_blocker,
    )
    errors, derived = validate_combined_family_stage_scorecard(
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
    if args.audit_out is not None:
        audit["scorecard_path"] = str(args.out)
        write_json_atomic(args.audit_out, audit)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
