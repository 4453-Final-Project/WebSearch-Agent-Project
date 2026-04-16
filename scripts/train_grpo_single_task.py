"""Single-task GRPO training entrypoint built on Tasks 2 and 3 contracts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.checkpointing import CheckpointManager  # noqa: E402
from src.training.collector import RolloutCollector, RolloutConfig  # noqa: E402
from src.training.eval_hooks import EvalConfig, run_eval, write_metrics_table  # noqa: E402
from src.training.grpo import GRPOConfig, GRPOTrainer  # noqa: E402
from src.training.peft_setup import LoRAConfig  # noqa: E402
from src.training.policy_adapter import PolicyAdapterConfig, TrainableQwenPolicy  # noqa: E402
from src.training.rewards import RewardConfig  # noqa: E402
from src.training.warmup import WarmupConfig, load_demo_trajectories, run_supervised_warmup  # noqa: E402
from src.utils.config import (  # noqa: E402
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
    DEFAULT_TASK_ID,
    get_output_dir,
    resolve_model_path,
)
from src.utils.model_profiles import DEFAULT_QWEN_PROFILE, find_model_profile_by_dir_name  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train LoRA adapters with grouped relative policy optimization.")
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--model-dir-name", default=None, help="Optional local model directory name under ../models/.")
    parser.add_argument("--model-path", default=None, help="Optional explicit local model path.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--groups-per-iteration", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--warmup-batch-size", type=int, default=1)
    parser.add_argument("--warmup-gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--warmup-demo-dir",
        action="append",
        default=[],
        help="Runner-format episode directory or parent directory containing warmup demos. Can be repeated.",
    )
    parser.add_argument(
        "--warmup-success-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only use successful demo episodes during supervised warmup.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--max-supervised-tokens", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out-dir",
        default=str(get_output_dir("task4-grpo-single-task", create=True)),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_model_path = resolve_model_path(model_path=args.model_path, model_dir_name=args.model_dir_name)
    profile = find_model_profile_by_dir_name(args.model_dir_name) or DEFAULT_QWEN_PROFILE

    policy = TrainableQwenPolicy(
        PolicyAdapterConfig(
            model_path=str(resolved_model_path),
            policy_name=f"{profile.name}-grpo",
            max_new_tokens=args.max_new_tokens if args.max_new_tokens is not None else profile.max_new_tokens,
            temperature=args.temperature if args.temperature is not None else max(profile.temperature, 0.2),
            top_k=profile.top_k,
            repetition_penalty=profile.repetition_penalty,
            system_prompt=profile.system_prompt,
            use_chat_template=profile.use_chat_template,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            score_batch_size=args.score_batch_size,
            max_supervised_tokens=args.max_supervised_tokens,
            lora=LoRAConfig(
                r=args.lora_r,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            ),
        )
    )

    eval_config = EvalConfig(
        task_id=args.task_id,
        seed=args.seed,
        max_steps=args.max_steps,
        episodes=args.eval_episodes,
        out_dir=str(out_dir / "eval"),
        headless=args.headless,
    )
    before_metrics = run_eval(policy, eval_config, "before")

    warmup_metrics = None
    after_warmup_metrics = None
    if args.warmup_epochs > 0:
        if not args.warmup_demo_dir:
            raise SystemExit("--warmup-epochs requires at least one --warmup-demo-dir.")
        trajectories = load_demo_trajectories(args.warmup_demo_dir, success_only=args.warmup_success_only)
        warmup_metrics = run_supervised_warmup(
            policy,
            trajectories,
            WarmupConfig(
                epochs=args.warmup_epochs,
                batch_size=args.warmup_batch_size,
                gradient_accumulation_steps=args.warmup_gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                success_only=args.warmup_success_only,
                shuffle_seed=args.seed,
            ),
        )
        after_warmup_metrics = run_eval(policy, eval_config, "after_warmup")

    collector = RolloutCollector(
        RolloutConfig(
            task_id=args.task_id,
            seed=args.seed,
            max_steps=args.max_steps,
            groups_per_iteration=args.groups_per_iteration,
            group_size=args.group_size,
            headless=args.headless,
            output_dir=str(out_dir),
        )
    )
    trainer = GRPOTrainer(
        policy=policy,
        collector=collector,
        reward_config=RewardConfig(),
        config=GRPOConfig(
            iterations=args.iterations,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_range=args.clip_range,
            beta=args.beta,
            max_grad_norm=args.max_grad_norm,
        ),
        checkpoint_manager=CheckpointManager(out_dir / "checkpoints"),
    )
    iteration_metrics = trainer.train() if args.iterations > 0 else []

    after_metrics = run_eval(policy, eval_config, "after")
    metrics_table_path = write_metrics_table(
        out_dir,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        iteration_metrics=iteration_metrics,
    )

    summary = {
        "task_id": args.task_id,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "model_path": str(resolved_model_path),
        "model_dir_name": args.model_dir_name,
        "iterations": args.iterations,
        "groups_per_iteration": args.groups_per_iteration,
        "group_size": args.group_size,
        "before_eval": before_metrics,
        "warmup_metrics": warmup_metrics,
        "after_warmup_eval": after_warmup_metrics,
        "after_eval": after_metrics,
        "iteration_metrics": iteration_metrics,
        "metrics_table_path": str(metrics_table_path),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_summary_text(out_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _write_summary_text(out_dir: Path, summary: dict[str, object]) -> Path:
    before = summary.get("before_eval") or {}
    after_warmup = summary.get("after_warmup_eval") or {}
    after = summary.get("after_eval") or {}
    warmup_metrics = summary.get("warmup_metrics") or {}
    lines = [
        "Single-Task Training Summary",
        f"Task ID: {summary.get('task_id', '(unknown)')}",
        f"Seed: {summary.get('seed', '(unknown)')}",
        f"Max Steps: {summary.get('max_steps', '(unknown)')}",
        f"Model Path: {summary.get('model_path', '(unknown)')}",
        f"Iterations: {summary.get('iterations', '(unknown)')}",
        f"Groups Per Iteration: {summary.get('groups_per_iteration', '(unknown)')}",
        f"Group Size: {summary.get('group_size', '(unknown)')}",
        "",
        "Before Eval:",
        f"  success_rate={before.get('success_rate', 0.0):.3f}",
        f"  average_reward={before.get('average_reward', 0.0):.3f}",
        f"  average_steps={before.get('average_steps', 0.0):.3f}",
        "",
        "Warmup:",
    ]
    if warmup_metrics:
        lines.extend(
            [
                f"  episodes_used={warmup_metrics.get('episodes_used', 0)}",
                f"  samples_used={warmup_metrics.get('samples_used', 0)}",
                f"  average_demo_reward={warmup_metrics.get('average_demo_reward', 0.0):.3f}",
                f"  final_loss={warmup_metrics.get('final_loss', 0.0):.6f}",
            ]
        )
    else:
        lines.append("  none")
    lines.extend(
        [
            "",
            "After Warmup Eval:",
            f"  success_rate={after_warmup.get('success_rate', 0.0):.3f}",
            f"  average_reward={after_warmup.get('average_reward', 0.0):.3f}",
            f"  average_steps={after_warmup.get('average_steps', 0.0):.3f}",
            "",
            "After Eval:",
            f"  success_rate={after.get('success_rate', 0.0):.3f}",
            f"  average_reward={after.get('average_reward', 0.0):.3f}",
            f"  average_steps={after.get('average_steps', 0.0):.3f}",
            "",
            f"Metrics Table: {summary.get('metrics_table_path', '(unknown)')}",
        ]
    )
    path = out_dir / "summary.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
