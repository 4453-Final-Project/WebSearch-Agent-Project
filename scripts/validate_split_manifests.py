"""Validate checked-in family split manifests and launcher wiring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_family_curriculum import _load_split_from_manifest  # noqa: E402
from src.training.task_families import recommend_task_split  # noqa: E402


DEFAULT_MANIFESTS = {
    "bootstrap41": REPO_ROOT / "scripts" / "local" / "bootstrap41_curriculum_manifest.json",
    "shopping_exact": REPO_ROOT / "scripts" / "local" / "shopping_exact_curriculum_manifest.json",
    "shopping_full": REPO_ROOT / "scripts" / "local" / "shopping_full_curriculum_manifest.json",
    "web_mix88": REPO_ROOT / "scripts" / "local" / "web_mix88_curriculum_manifest.json",
}

DEFAULT_LAUNCHERS = {
    "bootstrap41": (
        REPO_ROOT / "scripts" / "local" / "run_qwen_bootstrap41_curriculum.sh",
        REPO_ROOT / "scripts" / "local" / "run_liquid_bootstrap41_curriculum.sh",
    ),
    "shopping_exact": (
        REPO_ROOT / "scripts" / "local" / "run_qwen_shopping_exact_curriculum.sh",
        REPO_ROOT / "scripts" / "local" / "run_liquid_shopping_exact_curriculum.sh",
    ),
    "shopping_full": (
        REPO_ROOT / "scripts" / "local" / "run_qwen_shopping_full_curriculum.sh",
        REPO_ROOT / "scripts" / "local" / "run_liquid_shopping_full_curriculum.sh",
    ),
    "web_mix88": (
        REPO_ROOT / "scripts" / "local" / "run_qwen_web_mix88_curriculum.sh",
        REPO_ROOT / "scripts" / "local" / "run_liquid_web_mix88_curriculum.sh",
    ),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate checked-in split manifests and launcher wiring.")
    parser.add_argument("--family", choices=sorted(DEFAULT_MANIFESTS), action="append", default=[])
    return parser


def _validate_manifest(family_name: str, manifest_path: Path) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    split = _load_split_from_manifest(manifest_path, expected_family_name=family_name)
    recommended = recommend_task_split(family_name, split_seed=split.split_seed)

    if split.task_ids != recommended.task_ids:
        errors.append("task_ids do not match recommended split")
    if split.warmup_task_ids != recommended.warmup_task_ids:
        errors.append("warmup_task_ids do not match recommended split")
    if split.grpo_task_ids != recommended.grpo_task_ids:
        errors.append("grpo_task_ids do not match recommended split")
    if split.holdout_task_ids != recommended.holdout_task_ids:
        errors.append("holdout_task_ids do not match recommended split")
    if split.eval_task_ids != recommended.eval_task_ids:
        errors.append("eval_task_ids do not match recommended split")

    return errors, {
        "manifest_path": str(manifest_path),
        "split_seed": split.split_seed,
        "task_count": len(split.task_ids),
        "training_task_count": len(split.training_task_ids),
        "holdout_task_count": len(split.holdout_task_ids),
    }


def _validate_launchers(family_name: str, manifest_path: Path) -> tuple[list[str], dict[str, object]]:
    errors: list[str] = []
    manifest_name = manifest_path.name
    launcher_results: dict[str, object] = {}

    for launcher_path in DEFAULT_LAUNCHERS[family_name]:
        text = launcher_path.read_text(encoding="utf-8")
        launcher_errors: list[str] = []
        if manifest_name not in text:
            launcher_errors.append(f"missing manifest reference {manifest_name}")
        if "--split-manifest" not in text:
            launcher_errors.append("missing --split-manifest flag")
        if launcher_errors:
            errors.extend(f"{launcher_path.name}: {error}" for error in launcher_errors)
        launcher_results[launcher_path.name] = {
            "path": str(launcher_path),
            "ok": not launcher_errors,
            "errors": launcher_errors,
        }

    return errors, launcher_results


def validate_checked_in_split_manifests(families: list[str] | tuple[str, ...]) -> tuple[list[str], dict[str, object]]:
    selected_families = list(families) or sorted(DEFAULT_MANIFESTS)
    errors: list[str] = []
    payload: dict[str, object] = {}

    for family_name in selected_families:
        manifest_path = DEFAULT_MANIFESTS[family_name]
        manifest_errors, manifest_payload = _validate_manifest(family_name, manifest_path)
        launcher_errors, launcher_payload = _validate_launchers(family_name, manifest_path)
        errors.extend(f"{family_name}: {error}" for error in manifest_errors)
        errors.extend(f"{family_name}: {error}" for error in launcher_errors)
        payload[family_name] = {
            "manifest": {
                "ok": not manifest_errors,
                "errors": manifest_errors,
                **manifest_payload,
            },
            "launchers": launcher_payload,
        }

    return errors, payload


def main() -> int:
    args = build_arg_parser().parse_args()
    errors, payload = validate_checked_in_split_manifests(args.family)
    result = {
        "ok": not errors,
        "errors": errors,
        "families": payload,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
