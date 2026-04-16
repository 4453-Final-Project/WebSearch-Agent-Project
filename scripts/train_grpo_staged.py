"""Stage BrowserGym warmup, rollout collection, GRPO, and eval in separate processes."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_single_task import evaluate_single_task  # noqa: E402
from src.env.webarena_runner import run_episode  # noqa: E402
from src.training.checkpointing import CheckpointManager  # noqa: E402
from src.training.grpo import GRPOConfig, GRPOTrainer  # noqa: E402
from src.training.peft_setup import LoRAConfig  # noqa: E402
from src.training.policy_adapter import PolicyAdapterConfig, TrainableQwenPolicy  # noqa: E402
from src.training.rewards import RewardConfig  # noqa: E402
from src.training.trajectory import EpisodeTrajectory, load_episode_trajectory  # noqa: E402
from src.training.warmup import WarmupConfig, load_demo_trajectories, run_supervised_warmup  # noqa: E402
from src.utils.config import DEFAULT_MAX_STEPS, DEFAULT_SEED, get_output_dir, resolve_model_path  # noqa: E402
from src.utils.model_profiles import DEFAULT_QWEN_PROFILE, find_model_profile_by_dir_name  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run staged BrowserGym GRPO in separate processes.")
    parser.add_argument("--stage", choices=("warmup", "collect", "grpo", "eval"), required=True)
    parser.add_argument("--task-id", action="append", type=int, dest="task_ids", required=True)
    parser.add_argument(
        "--warmup-task-id",
        action="append",
        type=int,
        dest="warmup_task_ids",
        default=[],
        help="Optional behavior-cloning task ids. Defaults to the GRPO task ids when omitted.",
    )
    parser.add_argument(
        "--eval-task-id",
        action="append",
        type=int,
        dest="eval_task_ids",
        default=[],
        help="Optional evaluation task ids. Defaults to the GRPO task ids when omitted.",
    )
    parser.add_argument(
        "--allow-task-overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow overlap between explicit warmup task ids and GRPO task ids.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--model-dir-name", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--warmup-demo-dir", action="append", default=[])
    parser.add_argument("--warmup-demo-limit-per-task", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--warmup-batch-size", type=int, default=1)
    parser.add_argument("--warmup-gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--groups-per-task", type=int, default=5)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--score-batch-size", type=int, default=1)
    parser.add_argument("--max-supervised-tokens", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--out-dir",
        default=str(get_output_dir("task4-grpo-staged", create=True)),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    _normalize_stage_task_sets(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "warmup":
        return _run_warmup(args, out_dir)
    if args.stage == "collect":
        return _run_collect(args, out_dir)
    if args.stage == "grpo":
        return _run_grpo(args, out_dir)
    if args.stage == "eval":
        return _run_eval(args, out_dir)
    raise SystemExit(f"Unsupported stage: {args.stage}")


def _run_warmup(args, out_dir: Path) -> int:
    trajectories = _select_warmup_trajectories(args.warmup_demo_dir, args.warmup_task_ids, args.warmup_demo_limit_per_task)
    policy = _build_policy(args, model_path=str(_resolve_stage_model_path(args)))
    warmup_metrics = run_supervised_warmup(
        policy,
        trajectories,
        WarmupConfig(
            epochs=args.warmup_epochs,
            batch_size=args.warmup_batch_size,
            gradient_accumulation_steps=args.warmup_gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            success_only=True,
            shuffle_seed=args.seed,
        ),
    )
    adapter_dir = out_dir / "warmup_adapter"
    policy.save_adapter(adapter_dir)
    summary = {
        "stage": "warmup",
        "warmup_task_ids": args.warmup_task_ids,
        "grpo_task_ids": args.task_ids,
        "eval_task_ids": args.eval_task_ids,
        "selected_demo_count": len(trajectories),
        "selected_demo_task_counts": _count_trajectories_by_task(trajectories),
        "warmup_metrics": warmup_metrics,
        "adapter_dir": str(adapter_dir),
    }
    _write_json(out_dir / "warmup_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _run_collect(args, out_dir: Path) -> int:
    policy = _build_policy(args, model_path=str(out_dir / "warmup_adapter"))
    root_dir = out_dir / "rollouts" / "iteration_0000"
    root_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    base_seed = args.seed
    for task_offset, task_id in enumerate(args.task_ids):
        for group_local_idx in range(args.groups_per_task):
            for member_idx in range(args.group_size):
                group_idx = task_offset * args.groups_per_task + group_local_idx
                episode_seed = base_seed + (group_idx * 10) + member_idx
                episode_dir = root_dir / f"task_{task_id}" / f"group_{group_local_idx:04d}" / f"episode_{member_idx:04d}"
                run_episode(
                    policy,
                    task_id,
                    episode_seed,
                    args.max_steps,
                    str(episode_dir),
                    headless=args.headless,
                )
                trajectory = load_episode_trajectory(episode_dir)
                summaries.append(
                    {
                        "task_id": task_id,
                        "group_idx": group_local_idx,
                        "member_idx": member_idx,
                        "seed": episode_seed,
                        "success": trajectory.success,
                        "reward": trajectory.reward,
                        "steps_taken": trajectory.steps_taken,
                        "invalid_action_count": trajectory.invalid_action_count,
                        "parse_failure_count": trajectory.parse_failure_count,
                    }
                )

    summary = {
        "stage": "collect",
        "warmup_task_ids": args.warmup_task_ids,
        "grpo_task_ids": args.task_ids,
        "episode_count": len(summaries),
        "success_count": sum(1 for item in summaries if item["success"]),
        "task_success_counts": {
            str(task_id): sum(1 for item in summaries if item["task_id"] == task_id and item["success"])
            for task_id in args.task_ids
        },
        "episodes": summaries,
    }
    _write_json(out_dir / "rollout_summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes"}, indent=2))
    return 0


def _run_grpo(args, out_dir: Path) -> int:
    policy = _build_policy(args, model_path=str(out_dir / "warmup_adapter"))
    trainer = GRPOTrainer(
        policy=policy,
        collector=_DiskCollector(out_dir / "rollouts" / "iteration_0000", args.task_ids, args.groups_per_task, args.group_size),
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
    metrics = trainer.train()
    _write_json(out_dir / "grpo_summary.json", metrics)
    print(json.dumps(metrics, indent=2))
    return 0


def _run_eval(args, out_dir: Path) -> int:
    summary = {
        "warmup_task_ids": args.warmup_task_ids,
        "grpo_task_ids": args.task_ids,
        "eval_task_ids": args.eval_task_ids,
        "warmup_eval": _evaluate_adapter(args, out_dir / "warmup_adapter", out_dir / "eval_warmup"),
        "grpo_eval": _evaluate_adapter(args, out_dir / "checkpoints" / f"iter_{args.iterations - 1:04d}" / "adapter", out_dir / "eval_grpo"),
    }
    _write_json(out_dir / "eval_compare.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


def _evaluate_adapter(args, model_path: Path, output_root: Path) -> dict[str, object]:
    policy = _build_policy(args, model_path=str(model_path), temperature=args.eval_temperature)
    results: dict[str, object] = {}
    for task_id in args.eval_task_ids:
        results[str(task_id)] = evaluate_single_task(
            policy=policy,
            task_id=task_id,
            seed=args.seed,
            max_steps=args.max_steps,
            episodes=args.eval_episodes,
            out_dir=output_root / f"task_{task_id}",
            headless=args.headless,
        )
    return results


def _build_policy(args, *, model_path: str, temperature: float | None = None) -> TrainableQwenPolicy:
    profile = find_model_profile_by_dir_name(args.model_dir_name) or DEFAULT_QWEN_PROFILE
    return TrainableQwenPolicy(
        PolicyAdapterConfig(
            model_path=model_path,
            policy_name=f"{profile.name}-staged",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if temperature is None else temperature,
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
                target_modules=("q_proj", "k_proj", "v_proj", "out_proj", "w1", "w2", "w3", "in_proj"),
            ),
        )
    )


def _resolve_stage_model_path(args) -> Path:
    return resolve_model_path(model_path=args.model_path, model_dir_name=args.model_dir_name)


def _normalize_stage_task_sets(args) -> None:
    args.task_ids = list(dict.fromkeys(args.task_ids))
    explicit_warmup_ids = list(dict.fromkeys(args.warmup_task_ids))
    explicit_eval_ids = list(dict.fromkeys(args.eval_task_ids))

    if explicit_warmup_ids and not args.allow_task_overlap:
        overlap = sorted(set(explicit_warmup_ids) & set(args.task_ids))
        if overlap:
            overlap_text = ", ".join(str(task_id) for task_id in overlap)
            raise SystemExit(
                "Explicit warmup task ids must be disjoint from GRPO task ids. "
                f"Overlapping task ids: {overlap_text}. "
                "Pass --allow-task-overlap if you intentionally want the old behavior."
            )

    args.warmup_task_ids = explicit_warmup_ids or list(args.task_ids)
    args.eval_task_ids = explicit_eval_ids or list(args.task_ids)


def _select_warmup_trajectories(paths: list[str], task_ids: list[int], limit_per_task: int) -> list[EpisodeTrajectory]:
    trajectories = load_demo_trajectories(paths, success_only=True)
    selected: list[EpisodeTrajectory] = []
    for task_id in task_ids:
        selected.extend([trajectory for trajectory in trajectories if trajectory.task_id == task_id][:limit_per_task])
    if not selected:
        raise RuntimeError("No warmup trajectories matched the requested task IDs.")
    return selected


def _count_trajectories_by_task(trajectories: list[EpisodeTrajectory]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for trajectory in trajectories:
        counts[str(trajectory.task_id)] += 1
    return dict(sorted(counts.items()))


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


class _DiskCollector:
    def __init__(self, root_dir: Path, task_ids: list[int], groups_per_task: int, group_size: int) -> None:
        self.root_dir = root_dir
        self.task_ids = task_ids
        self.groups_per_task = groups_per_task
        self.group_size = group_size

    def collect_iteration(self, policy, iteration_idx: int) -> list[list[EpisodeTrajectory]]:
        groups: list[list[EpisodeTrajectory]] = []
        for task_id in self.task_ids:
            for group_local_idx in range(self.groups_per_task):
                group: list[EpisodeTrajectory] = []
                for member_idx in range(self.group_size):
                    episode_dir = self.root_dir / f"task_{task_id}" / f"group_{group_local_idx:04d}" / f"episode_{member_idx:04d}"
                    group.append(load_episode_trajectory(episode_dir))
                groups.append(group)
        return groups


if __name__ == "__main__":
    raise SystemExit(main())
