"""Run a scalable staged warmup+GRPO curriculum over a scripted task family."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from argparse import Namespace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_warmup_demos import collect_scripted_warmup_demos  # noqa: E402
from scripts.eval_single_task import evaluate_single_task, load_policy  # noqa: E402
from scripts.json_io import write_json_atomic  # noqa: E402
from scripts.train_grpo_staged import _run_collect, _run_eval, _run_grpo, _run_warmup  # noqa: E402
from src.training.task_families import (  # noqa: E402
    TaskSplit,
    build_task_split,
    family_requires_openai_judge,
    get_family_task_groups,
    list_task_family_names,
    recommend_task_split,
)
from src.utils.config import DEFAULT_MAX_STEPS, DEFAULT_SEED, get_env_var, get_output_dir  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a multi-task staged curriculum over a scripted task family.")
    parser.add_argument("--family", choices=list_task_family_names(), default="shopping_order")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--warmup-task-count", type=int, default=None)
    parser.add_argument("--holdout-task-count", type=int, default=None)
    parser.add_argument("--split-manifest", default=None)
    parser.add_argument("--model-dir-name", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument(
        "--task-group-max-steps",
        action="append",
        default=[],
        help="Optional per-task-group overrides like site_gitlab=8. Repeat to set multiple groups.",
    )
    parser.add_argument("--warmup-demo-episodes", type=int, default=2)
    parser.add_argument("--warmup-demo-max-steps", type=int, default=4)
    parser.add_argument(
        "--warmup-demo-task-group-episodes",
        action="append",
        default=[],
        help="Optional warmup-demo episode overrides like site_gitlab=4.",
    )
    parser.add_argument(
        "--warmup-demo-task-group-max-steps",
        action="append",
        default=[],
        help="Optional warmup-demo-specific per-task-group overrides like site_gitlab=8.",
    )
    parser.add_argument("--reuse-existing-demos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-existing-stages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-demo-limit-per-task", type=int, default=2)
    parser.add_argument(
        "--warmup-demo-task-group-limit-per-task",
        action="append",
        default=[],
        help="Optional warmup-training demo-limit overrides like site_gitlab=4.",
    )
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--warmup-batch-size", type=int, default=1)
    parser.add_argument("--warmup-gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--warmup-task-group-sample-multiplier",
        action="append",
        default=[],
        help="Optional warmup oversampling multipliers like site_gitlab=2.0.",
    )
    parser.add_argument("--groups-per-task", type=int, default=2)
    parser.add_argument(
        "--task-group-groups-per-task",
        action="append",
        default=[],
        help="Optional rollout group-count overrides like site_gitlab=4.",
    )
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--max-supervised-tokens", type=int, default=512)
    parser.add_argument("--quantization-mode", default=None)
    parser.add_argument("--quant-compute-dtype", default="bfloat16")
    parser.add_argument("--quant-type", default="nf4")
    parser.add_argument("--quant-use-double-quant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--success-bonus", type=float, default=1.0)
    parser.add_argument("--success-step-bonus", type=float, default=0.05)
    parser.add_argument("--invalid-action-penalty", type=float, default=0.15)
    parser.add_argument("--parse-failure-penalty", type=float, default=0.10)
    parser.add_argument("--truncation-penalty", type=float, default=0.05)
    parser.add_argument("--repeat-action-penalty", type=float, default=0.02)
    parser.add_argument("--same-page-repeat-action-penalty", type=float, default=0.03)
    parser.add_argument("--per-step-penalty", type=float, default=0.01)
    parser.add_argument("--success-unique-url-bonus", type=float, default=0.02)
    parser.add_argument("--max-unique-url-bonus-urls", type=int, default=4)
    parser.add_argument("--step-weight-later-step-bonus", type=float, default=0.5)
    parser.add_argument("--step-weight-terminal-success-bonus", type=float, default=1.0)
    parser.add_argument("--step-weight-error-step-multiplier", type=float, default=0.25)
    parser.add_argument("--step-weight-min", type=float, default=0.05)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-training-task-count", type=int, default=20)
    parser.add_argument("--preflight-out", default=None)
    parser.add_argument("--out-dir", default=None)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    family_requirement_status = _validate_family_requirements(
        args.family,
        allow_missing_openai_key=args.dry_run,
    )
    split = _resolve_split(args)
    split_provenance = resolve_split_provenance(args, split)
    preflight = validate_split_preflight(
        split,
        min_training_task_count=args.min_training_task_count,
        family_requirement_status=family_requirement_status,
        split_provenance=split_provenance,
    )
    if not preflight["ok"]:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 1
    if args.dry_run:
        preflight = _write_preflight_artifact_if_requested(args, preflight)
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    out_dir = _resolve_out_dir(args, split)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_manifest_path = out_dir / "split_manifest.json"
    split_manifest_payload = build_split_manifest_payload(
        split,
        split_provenance=preflight["split_provenance"],
        judge_requirements=preflight["judge_requirements"],
        recommended_split_alignment=preflight["recommended_split_alignment"],
    )
    write_json_atomic(split_manifest_path, split_manifest_payload)

    warmup_demo_dir = out_dir / "warmup_demos"
    warmup_demo_summary = _prepare_warmup_demos(args, split, warmup_demo_dir)
    baseline_eval = _run_baseline_eval(args, split, out_dir)

    stage_args = _build_stage_args(args, split, out_dir, warmup_demo_dir)
    if not (args.reuse_existing_stages and (out_dir / "warmup_summary.json").exists()):
        _run_warmup(stage_args, out_dir)
    if not (args.reuse_existing_stages and (out_dir / "rollout_summary.json").exists()):
        _run_collect(stage_args, out_dir)
    if not (args.reuse_existing_stages and (out_dir / "grpo_summary.json").exists()):
        _run_grpo(stage_args, out_dir)
    if not (args.reuse_existing_stages and (out_dir / "eval_compare.json").exists()):
        _run_eval(stage_args, out_dir)

    warmup_training_summary = _load_json(out_dir / "warmup_summary.json")
    eval_compare = _load_json(out_dir / "eval_compare.json")
    family_summary = build_family_run_summary(
        split,
        baseline_eval,
        eval_compare,
        warmup_training_summary=warmup_training_summary,
        run_provenance={
            "split_manifest_path": str(split_manifest_path),
            "split_provenance": split_manifest_payload["split_provenance"],
            "judge_requirements": split_manifest_payload["judge_requirements"],
            "recommended_split_alignment": split_manifest_payload["recommended_split_alignment"],
        },
    )
    family_summary["warmup_demo_summary"] = warmup_demo_summary
    family_summary["split_manifest"] = str(split_manifest_path)
    family_summary["out_dir"] = str(out_dir)
    summary_path = out_dir / "family_summary.json"
    write_json_atomic(summary_path, family_summary)
    audit_paths = _write_training_progress_audits(family_summary, out_dir)

    print(
        json.dumps(
            {
                "family": split.family_name,
                "task_count": len(split.task_ids),
                "training_task_count": len(split.training_task_ids),
                "holdout_task_count": len(split.holdout_task_ids),
                "summary_path": str(summary_path),
                "training_progress_audits": audit_paths,
                "out_dir": str(out_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _get_family_requirement_status(family_name: str) -> dict[str, object]:
    requires_openai_judge = family_requires_openai_judge(family_name)
    openai_api_key_present = bool(get_env_var("OPENAI_API_KEY"))
    return {
        "requires_openai_judge": requires_openai_judge,
        "openai_api_key_present": openai_api_key_present,
        "run_ready": (not requires_openai_judge) or openai_api_key_present,
    }


def _validate_family_requirements(
    family_name: str,
    *,
    allow_missing_openai_key: bool = False,
) -> dict[str, object]:
    status = _get_family_requirement_status(family_name)
    if status["requires_openai_judge"] and not status["openai_api_key_present"] and not allow_missing_openai_key:
        raise RuntimeError(
            f"Family {family_name!r} includes fuzzy-judged shopping tasks and requires OPENAI_API_KEY."
        )
    return status


def _resolve_split(args) -> TaskSplit:
    if args.split_manifest:
        return _load_split_from_manifest(Path(args.split_manifest), expected_family_name=args.family)
    if args.warmup_task_count is None and args.holdout_task_count is None:
        return recommend_task_split(args.family, split_seed=args.split_seed)

    recommended = recommend_task_split(args.family, split_seed=args.split_seed)
    warmup_count = args.warmup_task_count or len(recommended.warmup_task_ids)
    holdout_count = args.holdout_task_count or len(recommended.holdout_task_ids)
    return build_task_split(
        recommended.task_ids,
        family_name=recommended.family_name,
        warmup_count=warmup_count,
        holdout_count=holdout_count,
        split_seed=args.split_seed,
    )


def _load_split_from_manifest(path: Path, *, expected_family_name: str | None = None) -> TaskSplit:
    payload = json.loads(path.read_text(encoding="utf-8"))
    split_payload = payload.get("split_preview", payload)

    split_field_names = {field.name for field in fields(TaskSplit)}
    missing_fields = sorted(field_name for field_name in split_field_names if field_name not in split_payload)
    if missing_fields:
        raise ValueError(
            f"Split manifest {path} is missing required TaskSplit fields: {', '.join(missing_fields)}"
        )

    family_name = split_payload["family_name"]
    if expected_family_name and family_name != expected_family_name:
        raise ValueError(
            f"Split manifest {path} targets family {family_name!r}, but --family was {expected_family_name!r}."
        )

    return TaskSplit(
        family_name=family_name,
        task_ids=tuple(int(task_id) for task_id in split_payload["task_ids"]),
        warmup_task_ids=tuple(int(task_id) for task_id in split_payload["warmup_task_ids"]),
        grpo_task_ids=tuple(int(task_id) for task_id in split_payload["grpo_task_ids"]),
        holdout_task_ids=tuple(int(task_id) for task_id in split_payload["holdout_task_ids"]),
        eval_task_ids=tuple(int(task_id) for task_id in split_payload["eval_task_ids"]),
        split_seed=int(split_payload["split_seed"]),
    )


def _resolve_out_dir(args, split: TaskSplit) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    return get_output_dir(f"{split.family_name}_curriculum_v1", create=True)


def _resolve_preflight_out_path(args) -> Path | None:
    if args.preflight_out:
        return Path(args.preflight_out)
    if args.out_dir:
        return Path(args.out_dir) / "preflight.json"
    return None


def _write_preflight_artifact_if_requested(args, preflight: dict[str, object]) -> dict[str, object]:
    preflight_out_path = _resolve_preflight_out_path(args)
    if preflight_out_path is None:
        return preflight
    payload = dict(preflight)
    payload["preflight_path"] = str(preflight_out_path)
    write_json_atomic(preflight_out_path, payload)
    return payload


def validate_split_preflight(
    split: TaskSplit,
    *,
    min_training_task_count: int,
    family_requirement_status: dict[str, object] | None = None,
    split_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    errors: list[str] = []
    split_task_set = set(split.task_ids)
    warmup_task_set = set(split.warmup_task_ids)
    grpo_task_set = set(split.grpo_task_ids)
    holdout_task_set = set(split.holdout_task_ids)
    eval_task_set = set(split.eval_task_ids)

    if warmup_task_set & grpo_task_set:
        errors.append("Warmup and GRPO task ids must be disjoint.")
    if warmup_task_set & holdout_task_set:
        errors.append("Warmup and holdout task ids must be disjoint.")
    if grpo_task_set & holdout_task_set:
        errors.append("GRPO and holdout task ids must be disjoint.")
    if warmup_task_set | grpo_task_set | holdout_task_set != split_task_set:
        errors.append("Warmup, GRPO, and holdout task ids must partition the family task ids.")
    if eval_task_set != split_task_set:
        errors.append("Eval task ids must match the family task ids for family curriculum runs.")
    if not grpo_task_set:
        errors.append("At least one GRPO task is required.")

    if min_training_task_count > 0 and len(split.training_task_ids) < min_training_task_count and len(split.task_ids) > min_training_task_count:
        errors.append(
            f"Training task count {len(split.training_task_ids)} is below the configured floor "
            f"of {min_training_task_count}."
        )

    task_groups = get_family_task_groups(split.family_name)
    group_counts = {}
    for group_name, group_task_ids in task_groups.items():
        group_task_set = set(group_task_ids) & split_task_set
        if not group_task_set:
            continue
        group_counts[group_name] = {
            "task_count": len(group_task_set),
            "warmup_task_count": len(group_task_set & warmup_task_set),
            "grpo_task_count": len(group_task_set & grpo_task_set),
            "holdout_task_count": len(group_task_set & holdout_task_set),
        }

    judge_requirements = family_requirement_status or _get_family_requirement_status(split.family_name)
    recommended_alignment = _compare_split_to_recommended(split)
    split_provenance_payload = split_provenance or _build_split_provenance(split)

    return {
        "ok": not errors,
        "errors": errors,
        "family_name": split.family_name,
        "task_count": len(split.task_ids),
        "training_task_count": len(split.training_task_ids),
        "holdout_task_count": len(split.holdout_task_ids),
        "launch_ready": (not errors) and bool(judge_requirements["run_ready"]),
        "recommended_large_run_ready": len(split.training_task_ids) >= min_training_task_count,
        "judge_requirements": judge_requirements,
        "recommended_split_alignment": recommended_alignment,
        "split_provenance": split_provenance_payload,
        "split_preview": split.to_dict(),
        "task_group_counts": group_counts,
    }


def _build_split_provenance(
    split: TaskSplit,
    *,
    split_source: str = "recommended",
    source_manifest: str | None = None,
    warmup_task_count: int | None = None,
    holdout_task_count: int | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "split_source": split_source,
        "split_seed": split.split_seed,
    }
    if source_manifest is not None:
        payload["source_manifest"] = source_manifest
    if warmup_task_count is not None or holdout_task_count is not None:
        payload["requested_counts"] = {
            "warmup_task_count": warmup_task_count,
            "holdout_task_count": holdout_task_count,
        }
    return payload


def resolve_split_provenance(args, split: TaskSplit) -> dict[str, object]:
    if args.split_manifest:
        return _build_split_provenance(
            split,
            split_source="manifest",
            source_manifest=str(Path(args.split_manifest).resolve()),
        )
    if args.warmup_task_count is None and args.holdout_task_count is None:
        return _build_split_provenance(split, split_source="recommended")
    return _build_split_provenance(
        split,
        split_source="custom_counts",
        warmup_task_count=args.warmup_task_count,
        holdout_task_count=args.holdout_task_count,
    )


def build_split_manifest_payload(
    split: TaskSplit,
    *,
    split_provenance: dict[str, object],
    judge_requirements: dict[str, object],
    recommended_split_alignment: dict[str, object],
) -> dict[str, object]:
    return {
        "family_name": split.family_name,
        "split_provenance": split_provenance,
        "judge_requirements": judge_requirements,
        "recommended_split_alignment": recommended_split_alignment,
        "split_preview": split.to_dict(),
    }


def _compare_split_to_recommended(split: TaskSplit) -> dict[str, object]:
    recommended = recommend_task_split(split.family_name, split_seed=split.split_seed)
    differences: dict[str, dict[str, list[int]]] = {}
    for field_name in (
        "task_ids",
        "warmup_task_ids",
        "grpo_task_ids",
        "holdout_task_ids",
        "eval_task_ids",
    ):
        actual = tuple(getattr(split, field_name))
        expected = tuple(getattr(recommended, field_name))
        if actual == expected:
            continue
        differences[field_name] = {
            "actual": list(actual),
            "recommended": list(expected),
        }
    return {
        "matches_recommended_split": not differences,
        "recommended_split_seed": recommended.split_seed,
        "differences": differences,
    }


def _prepare_warmup_demos(args, split: TaskSplit, warmup_demo_dir: Path) -> dict[str, object]:
    summary_path = warmup_demo_dir / "summary.json"
    if args.reuse_existing_demos and summary_path.exists():
        return _load_json(summary_path)
    return collect_scripted_warmup_demos(
        task_ids=list(split.warmup_task_ids),
        seed=args.seed,
        episodes=args.warmup_demo_episodes,
        max_steps=args.warmup_demo_max_steps,
        per_task_episodes=_resolve_family_task_group_int_overrides(
            split.family_name,
            split.warmup_task_ids,
            default_value=args.warmup_demo_episodes,
            group_overrides=_get_group_int_overrides(args, "warmup_demo_task_group_episodes"),
        ),
        per_task_max_steps=_resolve_family_task_group_step_overrides(
            split.family_name,
            split.warmup_task_ids,
            default_max_steps=args.warmup_demo_max_steps,
            group_overrides=_get_group_step_overrides(args, "warmup_demo_task_group_max_steps"),
        ),
        out_dir=warmup_demo_dir,
        headed=args.headed,
    )


def _run_baseline_eval(args, split: TaskSplit, out_dir: Path) -> dict[str, object]:
    summary_path = out_dir / "baseline_eval_summary.json"
    eval_root = out_dir / "eval_baseline"
    if args.reuse_existing_stages:
        if summary_path.exists():
            existing_summary = _load_json(summary_path)
            if all(str(task_id) in existing_summary for task_id in split.eval_task_ids):
                return existing_summary
        existing_results = _load_existing_eval_results(eval_root, split.eval_task_ids)
        if existing_results is not None:
            write_json_atomic(summary_path, existing_results)
            return existing_results
    results = _load_partial_eval_results(summary_path, split.eval_task_ids)
    if len(results) < len(split.eval_task_ids):
        for task_id, metrics in _load_partial_eval_results_from_tree(eval_root, split.eval_task_ids).items():
            results.setdefault(task_id, metrics)
    policy = load_policy(
        "qwen",
        model_dir_name=args.model_dir_name,
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.eval_temperature,
        quantization_mode=args.quantization_mode,
        quant_compute_dtype=args.quant_compute_dtype,
        quant_type=args.quant_type,
        quant_use_double_quant=args.quant_use_double_quant,
    )
    task_max_steps = _resolve_family_task_group_step_overrides(
        split.family_name,
        split.eval_task_ids,
        default_max_steps=args.max_steps,
        group_overrides=_get_group_step_overrides(args, "task_group_max_steps"),
    )
    for task_id in split.eval_task_ids:
        if str(task_id) in results:
            continue
        results[str(task_id)] = evaluate_single_task(
            policy=policy,
            task_id=task_id,
            seed=args.seed,
            max_steps=task_max_steps.get(task_id, args.max_steps),
            episodes=args.eval_episodes,
            out_dir=eval_root / f"task_{task_id}",
            headless=not args.headed,
        )
        write_json_atomic(summary_path, results)
    write_json_atomic(summary_path, results)
    return results


def _load_existing_eval_results(eval_root: Path, task_ids: tuple[int, ...]) -> dict[str, object] | None:
    results: dict[str, object] = {}
    for task_id in task_ids:
        metrics_path = eval_root / f"task_{task_id}" / "metrics.json"
        if not metrics_path.exists():
            return None
        results[str(task_id)] = _load_json(metrics_path)
    return results


def _load_partial_eval_results(summary_path: Path, task_ids: tuple[int, ...]) -> dict[str, object]:
    if not summary_path.exists():
        return {}
    payload = _load_json(summary_path)
    return {
        str(task_id): payload[str(task_id)]
        for task_id in task_ids
        if str(task_id) in payload
    }


def _load_partial_eval_results_from_tree(eval_root: Path, task_ids: tuple[int, ...]) -> dict[str, object]:
    results: dict[str, object] = {}
    for task_id in task_ids:
        metrics_path = eval_root / f"task_{task_id}" / "metrics.json"
        if not metrics_path.exists():
            continue
        results[str(task_id)] = _load_json(metrics_path)
    return results


def _build_stage_args(args, split: TaskSplit, out_dir: Path, warmup_demo_dir: Path) -> Namespace:
    return Namespace(
        task_ids=list(split.grpo_task_ids),
        warmup_task_ids=list(split.warmup_task_ids),
        eval_task_ids=list(split.eval_task_ids),
        holdout_task_ids=list(split.holdout_task_ids),
        allow_task_overlap=False,
        seed=args.seed,
        max_steps=args.max_steps,
        task_max_steps=_resolve_family_task_group_step_overrides(
            split.family_name,
            split.task_ids,
            default_max_steps=args.max_steps,
            group_overrides=_get_group_step_overrides(args, "task_group_max_steps"),
        ),
        model_dir_name=args.model_dir_name,
        model_path=args.model_path,
        warmup_demo_dir=[str(warmup_demo_dir)],
        warmup_demo_limit_per_task=_resolve_family_task_group_int_overrides(
            split.family_name,
            split.warmup_task_ids,
            default_value=args.warmup_demo_limit_per_task,
            group_overrides=_get_group_int_overrides(args, "warmup_demo_task_group_limit_per_task"),
        ),
        warmup_epochs=args.warmup_epochs,
        warmup_batch_size=args.warmup_batch_size,
        warmup_gradient_accumulation_steps=args.warmup_gradient_accumulation_steps,
        warmup_sample_multipliers=_resolve_family_task_group_float_overrides(
            split.family_name,
            split.warmup_task_ids,
            default_value=1.0,
            group_overrides=_get_group_float_overrides(args, "warmup_task_group_sample_multiplier"),
        ),
        groups_per_task=_resolve_family_task_group_int_overrides(
            split.family_name,
            split.grpo_task_ids,
            default_value=args.groups_per_task,
            group_overrides=_get_group_int_overrides(args, "task_group_groups_per_task"),
        ),
        group_size=args.group_size,
        iterations=args.iterations,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        score_batch_size=args.score_batch_size,
        max_supervised_tokens=args.max_supervised_tokens,
        quantization_mode=args.quantization_mode,
        quant_compute_dtype=args.quant_compute_dtype,
        quant_type=args.quant_type,
        quant_use_double_quant=args.quant_use_double_quant,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        eval_temperature=args.eval_temperature,
        eval_episodes=args.eval_episodes,
        beta=args.beta,
        clip_range=args.clip_range,
        max_grad_norm=args.max_grad_norm,
        reward_scale=args.reward_scale,
        success_bonus=args.success_bonus,
        success_step_bonus=args.success_step_bonus,
        invalid_action_penalty=args.invalid_action_penalty,
        parse_failure_penalty=args.parse_failure_penalty,
        truncation_penalty=args.truncation_penalty,
        repeat_action_penalty=args.repeat_action_penalty,
        same_page_repeat_action_penalty=args.same_page_repeat_action_penalty,
        per_step_penalty=args.per_step_penalty,
        success_unique_url_bonus=args.success_unique_url_bonus,
        max_unique_url_bonus_urls=args.max_unique_url_bonus_urls,
        step_weight_later_step_bonus=args.step_weight_later_step_bonus,
        step_weight_terminal_success_bonus=args.step_weight_terminal_success_bonus,
        step_weight_error_step_multiplier=args.step_weight_error_step_multiplier,
        step_weight_min=args.step_weight_min,
        headless=not args.headed,
        out_dir=str(out_dir),
    )


def build_family_run_summary(
    split: TaskSplit,
    baseline_eval: dict[str, object],
    eval_compare: dict[str, object],
    warmup_training_summary: dict[str, object] | None = None,
    run_provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    task_groups = get_family_task_groups(split.family_name)
    baseline_per_task = _success_rates_from_eval(baseline_eval)
    warmup_per_task = _success_rates_from_eval(eval_compare["warmup_eval"])
    grpo_per_task = _success_rates_from_eval(eval_compare["grpo_eval"])

    summary = {
        "family_name": split.family_name,
        "task_count": len(split.task_ids),
        "training_task_count": len(split.training_task_ids),
        "holdout_task_count": len(split.holdout_task_ids),
        "family_metadata": _build_family_metadata(split, task_groups),
        "split": split.to_dict(),
        "baseline": _build_stage_summary(baseline_per_task, task_groups),
        "warmup_only": _build_stage_summary(warmup_per_task, task_groups),
        "warmup_plus_grpo": _build_stage_summary(grpo_per_task, task_groups),
        "holdout": _build_holdout_summary(split.holdout_task_ids, baseline_per_task, warmup_per_task, grpo_per_task),
    }
    if run_provenance is not None:
        summary["run_provenance"] = run_provenance
    if warmup_training_summary is not None:
        summary["warmup_training_summary"] = warmup_training_summary
    return summary


def _success_rates_from_eval(source: dict[str, object]) -> dict[str, float]:
    return {
        task_id: float(metrics["success_rate"])
        for task_id, metrics in sorted(source.items(), key=lambda item: int(item[0]))
    }


def _build_stage_summary(
    per_task_success_rate: dict[str, float],
    task_groups: dict[str, tuple[int, ...]],
) -> dict[str, object]:
    summary = {
        "family_success_rate": _mean_success_rate(per_task_success_rate.values()),
        "per_task_success_rate": per_task_success_rate,
    }
    group_summaries = {
        group_name: _build_task_group_summary(task_ids, per_task_success_rate)
        for group_name, task_ids in task_groups.items()
        if task_ids
    }
    if group_summaries:
        summary["task_groups"] = group_summaries
    return summary


def _build_family_metadata(
    split: TaskSplit,
    task_groups: dict[str, tuple[int, ...]],
) -> dict[str, object]:
    judge_free_task_ids = tuple(task_id for task_id in task_groups.get("judge_free", ()) if task_id in split.task_ids)
    judge_gated_task_ids = tuple(task_id for task_id in task_groups.get("judge_gated", ()) if task_id in split.task_ids)
    return {
        "requires_openai_judge": family_requires_openai_judge(split.family_name),
        "judge_free_task_ids": list(judge_free_task_ids),
        "judge_free_task_count": len(judge_free_task_ids),
        "judge_gated_task_ids": list(judge_gated_task_ids),
        "judge_gated_task_count": len(judge_gated_task_ids),
    }


def _build_task_group_summary(
    task_ids: tuple[int, ...],
    per_task_success_rate: dict[str, float],
) -> dict[str, object]:
    group_keys = [str(task_id) for task_id in task_ids if str(task_id) in per_task_success_rate]
    return {
        "task_ids": [int(task_id) for task_id in group_keys],
        "task_count": len(group_keys),
        "success_rate": _mean_success_rate(per_task_success_rate[task_id] for task_id in group_keys),
        "per_task_success_rate": {task_id: per_task_success_rate[task_id] for task_id in group_keys},
    }


def _build_holdout_summary(
    holdout_task_ids: tuple[int, ...],
    baseline_per_task: dict[str, float],
    warmup_per_task: dict[str, float],
    grpo_per_task: dict[str, float],
) -> dict[str, object]:
    if not holdout_task_ids:
        return {
            "task_ids": [],
            "baseline_success_rate": 0.0,
            "warmup_success_rate": 0.0,
            "grpo_success_rate": 0.0,
            "gain_vs_baseline": 0.0,
            "gain_vs_warmup": 0.0,
            "per_task": {},
        }

    holdout_keys = [str(task_id) for task_id in holdout_task_ids]
    baseline = _mean_success_rate(baseline_per_task[task_id] for task_id in holdout_keys)
    warmup = _mean_success_rate(warmup_per_task[task_id] for task_id in holdout_keys)
    grpo = _mean_success_rate(grpo_per_task[task_id] for task_id in holdout_keys)
    return {
        "task_ids": list(holdout_task_ids),
        "baseline_success_rate": baseline,
        "warmup_success_rate": warmup,
        "grpo_success_rate": grpo,
        "gain_vs_baseline": grpo - baseline,
        "gain_vs_warmup": grpo - warmup,
        "per_task": {
            task_id: {
                "baseline_success_rate": baseline_per_task[task_id],
                "warmup_success_rate": warmup_per_task[task_id],
                "grpo_success_rate": grpo_per_task[task_id],
                "gain_vs_baseline": grpo_per_task[task_id] - baseline_per_task[task_id],
                "gain_vs_warmup": grpo_per_task[task_id] - warmup_per_task[task_id],
            }
            for task_id in holdout_keys
        },
    }


def _mean_success_rate(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _write_training_progress_audits(
    family_summary: dict[str, object],
    out_dir: Path,
) -> dict[str, str]:
    from scripts.audit_family_training_progress import audit_family_training_progress  # noqa: E402

    audit_specs = {
        "baseline_vs_warmup": ("baseline", "warmup_only"),
        "baseline_vs_grpo": ("baseline", "warmup_plus_grpo"),
        "warmup_vs_grpo": ("warmup_only", "warmup_plus_grpo"),
    }
    written_paths: dict[str, str] = {}
    for label, (stage_a, stage_b) in audit_specs.items():
        errors, report = audit_family_training_progress(
            family_summary,
            stage_a=stage_a,
            stage_b=stage_b,
        )
        if errors:
            raise RuntimeError(
                f"Unable to build training progress audit {label!r}: {'; '.join(errors)}"
            )
        audit_path = out_dir / f"{label}_audit.json"
        write_json_atomic(audit_path, report)
        written_paths[label] = str(audit_path)
    return written_paths


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_group_step_overrides(args, attr_name: str) -> dict[str, int]:
    return _get_group_int_overrides(args, attr_name)


def _get_group_float_overrides(args, attr_name: str) -> dict[str, float]:
    raw_value = getattr(args, attr_name, None)
    if not raw_value:
        return {}
    if isinstance(raw_value, dict):
        return {str(name): float(value) for name, value in raw_value.items()}
    return _parse_group_float_override_entries(raw_value, subject_name="task group")


def _get_group_int_overrides(args, attr_name: str) -> dict[str, int]:
    raw_value = getattr(args, attr_name, None)
    if not raw_value:
        return {}
    if isinstance(raw_value, dict):
        return dict(raw_value)
    return _parse_group_int_override_entries(raw_value, subject_name="task group")


def _parse_group_int_override_entries(entries: list[str], *, subject_name: str) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid {subject_name} step override {entry!r}; expected NAME=STEPS.")
        name, value_text = entry.split("=", 1)
        name = name.strip()
        value_text = value_text.strip()
        if not name:
            raise ValueError(f"Invalid {subject_name} step override {entry!r}; missing name.")
        try:
            value = int(value_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {subject_name} step override {entry!r}; step count must be an integer."
            ) from exc
        if value <= 0:
            raise ValueError(
                f"Invalid {subject_name} step override {entry!r}; step count must be positive."
            )
        overrides[name] = value
    return overrides


def _parse_group_float_override_entries(entries: list[str], *, subject_name: str) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid {subject_name} multiplier override {entry!r}; expected NAME=MULTIPLIER.")
        name, value_text = entry.split("=", 1)
        name = name.strip()
        value_text = value_text.strip()
        if not name:
            raise ValueError(f"Invalid {subject_name} multiplier override {entry!r}; missing name.")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {subject_name} multiplier override {entry!r}; multiplier must be numeric."
            ) from exc
        if value <= 0.0:
            raise ValueError(
                f"Invalid {subject_name} multiplier override {entry!r}; multiplier must be positive."
            )
        overrides[name] = value
    return overrides


def _resolve_family_task_group_step_overrides(
    family_name: str,
    task_ids: tuple[int, ...] | list[int],
    *,
    default_max_steps: int,
    group_overrides: dict[str, int],
) -> dict[int, int]:
    return _resolve_family_task_group_int_overrides(
        family_name,
        task_ids,
        default_value=default_max_steps,
        group_overrides=group_overrides,
    )


def _resolve_family_task_group_float_overrides(
    family_name: str,
    task_ids: tuple[int, ...] | list[int],
    *,
    default_value: float,
    group_overrides: dict[str, float],
) -> dict[int, float]:
    return _resolve_family_task_group_numeric_overrides(
        family_name,
        task_ids,
        default_value=default_value,
        group_overrides=group_overrides,
    )


def _resolve_family_task_group_int_overrides(
    family_name: str,
    task_ids: tuple[int, ...] | list[int],
    *,
    default_value: int,
    group_overrides: dict[str, int],
) -> dict[int, int]:
    return _resolve_family_task_group_numeric_overrides(
        family_name,
        task_ids,
        default_value=default_value,
        group_overrides=group_overrides,
    )


def _resolve_family_task_group_numeric_overrides(
    family_name: str,
    task_ids: tuple[int, ...] | list[int],
    *,
    default_value: int | float,
    group_overrides: dict[str, int | float],
) -> dict[int, int | float]:
    if not group_overrides:
        return {}
    task_groups = get_family_task_groups(family_name)
    unknown_groups = sorted(group_name for group_name in group_overrides if group_name not in task_groups)
    if unknown_groups:
        raise ValueError(
            f"Unknown task groups for family {family_name!r}: {', '.join(unknown_groups)}"
        )

    task_ids = tuple(int(task_id) for task_id in task_ids)
    overrides: dict[int, int | float] = {}
    for task_id in task_ids:
        resolved_value = default_value
        for group_name, group_value in group_overrides.items():
            if task_id in task_groups[group_name]:
                resolved_value = max(resolved_value, group_value)
        if resolved_value != default_value:
            overrides[task_id] = resolved_value
    return overrides


if __name__ == "__main__":
    raise SystemExit(main())
