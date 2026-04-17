"""Evaluation scaffold for single-task Task 3 policy runs."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import DEFAULT_MAX_STEPS, DEFAULT_SEED, DEFAULT_TASK_ID, get_output_dir  # noqa: E402
from src.utils.model_profiles import DEFAULT_QWEN_PROFILE, find_model_profile_by_dir_name, find_model_profile_by_path  # noqa: E402


EpisodeRunner = Callable[[Any, int, int, int, str], dict[str, Any]]


class FakeEpisodeRunner:
    """Simple queued runner for tests and local harness validation."""

    def __init__(self, episode_results: list[dict[str, Any]] | None = None) -> None:
        """Initialize the fake runner with optional precomputed episode results."""

        self._episode_results = list(episode_results or [])
        self.calls: list[dict[str, Any]] = []

    def run_episode(
        self,
        policy: Any,
        task_id: int,
        seed: int,
        max_steps: int,
        out_dir: str,
    ) -> dict[str, Any]:
        """Return the next precomputed result or a deterministic default result."""

        call = {
            "policy": policy,
            "task_id": task_id,
            "seed": seed,
            "max_steps": max_steps,
            "out_dir": out_dir,
        }
        self.calls.append(call)
        if self._episode_results:
            return self._episode_results.pop(0)
        return {
            "success": False,
            "reward": 0.0,
            "steps": 0,
            "invalid_action_count": 0,
            "parse_failure_count": 0,
            "failure_reasons": ["fake_runner_default"],
        }


def try_load_task2_runner() -> EpisodeRunner | None:
    """Try to load Task 2's episode runner."""

    try:
        module = import_module("src.env.webarena_runner")
    except ImportError:
        return None

    runner = getattr(module, "run_episode", None)
    if callable(runner):
        return runner
    return None


def resolve_runner(
    runner: EpisodeRunner | Any | None = None,
    *,
    use_fake_runner: bool = False,
) -> EpisodeRunner:
    """Resolve the runner function from injection, fake fallback, or Task 2."""

    if runner is not None:
        if callable(runner):
            return runner
        if hasattr(runner, "run_episode") and callable(runner.run_episode):
            return runner.run_episode
        raise TypeError("Runner must be callable or expose a callable run_episode method.")

    if use_fake_runner:
        return FakeEpisodeRunner().run_episode

    task2_runner = try_load_task2_runner()
    if task2_runner is not None:
        return task2_runner

    raise RuntimeError(
        "No episode runner is available. Task 2 runner was not found and no fake runner was requested."
    )


def normalize_episode_result(raw_result: dict[str, Any]) -> dict[str, Any]:
    """Normalize a runner-produced episode result into a stable metric schema."""

    failure_reasons = raw_result.get("failure_reasons")
    if failure_reasons is None:
        single_reason = raw_result.get("failure_reason")
        failure_reasons = [single_reason] if single_reason else []

    return {
        "success": bool(raw_result.get("success", False)),
        "reward": float(raw_result.get("reward", 0.0)),
        "steps": int(raw_result.get("steps", 0)),
        "invalid_action_count": int(raw_result.get("invalid_action_count", 0)),
        "parse_failure_count": int(raw_result.get("parse_failure_count", 0)),
        "failure_reasons": [str(reason) for reason in failure_reasons if reason],
    }


def aggregate_metrics(episode_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate episode-level results into stable evaluation metrics."""

    normalized_results = [normalize_episode_result(result) for result in episode_results]
    episode_count = len(normalized_results)
    if episode_count == 0:
        return {
            "episode_count": 0,
            "success_rate": 0.0,
            "average_reward": 0.0,
            "average_steps": 0.0,
            "invalid_action_count": 0,
            "parse_failure_count": 0,
            "most_common_failure_reasons": [],
        }

    failure_counter = Counter(
        reason
        for result in normalized_results
        for reason in result["failure_reasons"]
    )

    return {
        "episode_count": episode_count,
        "success_rate": sum(1 for result in normalized_results if result["success"]) / episode_count,
        "average_reward": sum(result["reward"] for result in normalized_results) / episode_count,
        "average_steps": sum(result["steps"] for result in normalized_results) / episode_count,
        "invalid_action_count": sum(result["invalid_action_count"] for result in normalized_results),
        "parse_failure_count": sum(result["parse_failure_count"] for result in normalized_results),
        "most_common_failure_reasons": [
            {"reason": reason, "count": count}
            for reason, count in failure_counter.most_common()
        ],
    }


def write_metrics_json(metrics: dict[str, Any], out_dir: str | Path) -> Path:
    """Write aggregated metrics as JSON to the requested output directory."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics_path


def format_metrics_text(metrics: dict[str, Any]) -> str:
    """Render aggregated evaluation metrics as a readable plain-text summary."""

    lines = [
        "Single-Task Evaluation Summary",
        f"Policy: {metrics.get('policy', '(unknown)')}",
        f"Task ID: {metrics.get('task_id', '(unknown)')}",
        f"Seed: {metrics.get('seed', '(unknown)')}",
        f"Episodes: {metrics.get('episodes', metrics.get('episode_count', 0))}",
        f"Max Steps: {metrics.get('max_steps', '(unknown)')}",
        f"Success Rate: {metrics.get('success_rate', 0.0):.3f}",
        f"Average Reward: {metrics.get('average_reward', 0.0):.3f}",
        f"Average Steps: {metrics.get('average_steps', 0.0):.3f}",
        f"Invalid Action Count: {metrics.get('invalid_action_count', 0)}",
        f"Parse Failure Count: {metrics.get('parse_failure_count', 0)}",
        f"Output Directory: {metrics.get('out_dir', '(unknown)')}",
        "",
        "Most Common Failure Reasons:",
    ]
    failure_reasons = metrics.get("most_common_failure_reasons", [])
    if failure_reasons:
        for item in failure_reasons:
            lines.append(f"- {item['reason']} (count={item['count']})")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def write_metrics_text(metrics: dict[str, Any], out_dir: str | Path) -> Path:
    """Write a human-readable evaluation summary next to metrics.json."""

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_path = output_dir / "metrics.txt"
    text_path.write_text(format_metrics_text(metrics), encoding="utf-8")
    return text_path


def evaluate_single_task(
    policy: Any,
    task_id: int,
    seed: int,
    max_steps: int,
    episodes: int,
    out_dir: str | Path,
    *,
    runner: EpisodeRunner | Any | None = None,
    use_fake_runner: bool = False,
    headless: bool = True,
) -> dict[str, Any]:
    """Run repeated single-task evaluations through a runner interface."""

    resolved_runner = resolve_runner(runner, use_fake_runner=use_fake_runner)
    output_dir = Path(out_dir)
    episode_results: list[dict[str, Any]] = []

    for episode_idx in range(episodes):
        episode_seed = seed + episode_idx
        episode_out_dir = output_dir / f"episode_{episode_idx}"
        runner_kwargs: dict[str, Any] = {}
        try:
            runner_signature = inspect.signature(resolved_runner)
        except (TypeError, ValueError):
            runner_signature = None

        if runner_signature is not None:
            parameters = runner_signature.parameters
            if "headless" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            ):
                runner_kwargs["headless"] = headless

        result = resolved_runner(
            policy,
            task_id,
            episode_seed,
            max_steps,
            str(episode_out_dir),
            **runner_kwargs,
        )
        episode_results.append(normalize_episode_result(result))

    metrics = aggregate_metrics(episode_results)
    metrics.update(
        {
            "task_id": task_id,
            "seed": seed,
            "max_steps": max_steps,
            "episodes": episodes,
            "out_dir": str(output_dir),
            "policy": getattr(policy, "name", None),
        }
    )
    write_metrics_json(metrics, output_dir)
    write_metrics_text(metrics, output_dir)
    return metrics


def load_policy(
    policy_name: str,
    *,
    model_dir_name: str | None = None,
    model_path: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    quantization_mode: str | None = None,
    quant_compute_dtype: str = "bfloat16",
    quant_type: str = "nf4",
    quant_use_double_quant: bool = True,
) -> Any:
    """Load a supported evaluation policy by name."""

    if policy_name == "dummy":
        module = import_module("src.env.webarena_runner")
        policy_cls = getattr(module, "DummyPolicy", None)
        if policy_cls is None:
            raise RuntimeError("src.env.webarena_runner does not define DummyPolicy.")
        return policy_cls()

    if policy_name == "qwen":
        module = import_module("src.agent.qwen_policy")
        policy_cls = getattr(module, "QwenPolicy", None)
        if policy_cls is None:
            raise RuntimeError("src.agent.qwen_policy does not define QwenPolicy.")
        from src.agent.types import PolicyConfig
        from src.utils.config import resolve_model_path

        resolved_model_path = str(resolve_model_path(model_path=model_path, model_dir_name=model_dir_name))
        profile = (
            find_model_profile_by_dir_name(model_dir_name)
            or find_model_profile_by_path(resolved_model_path)
            or DEFAULT_QWEN_PROFILE
        )
        return policy_cls(
            config=PolicyConfig(
                model_path=resolved_model_path,
                policy_name=profile.name,
                max_new_tokens=max_new_tokens if max_new_tokens is not None else profile.max_new_tokens,
                temperature=temperature if temperature is not None else profile.temperature,
                top_k=profile.top_k,
                repetition_penalty=profile.repetition_penalty,
                system_prompt=profile.system_prompt,
                use_chat_template=profile.use_chat_template,
                quantization_mode=quantization_mode,
                quant_compute_dtype=quant_compute_dtype,
                quant_type=quant_type,
                quant_use_double_quant=quant_use_double_quant,
            )
        )

    raise RuntimeError(f"Unsupported policy: {policy_name}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for single-task evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate a single task with a runner-backed policy harness.")
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--policy", choices=("dummy", "qwen"), default="qwen")
    parser.add_argument("--model-dir-name", default=None, help="Optional local model directory name under ../models/.")
    parser.add_argument("--model-path", default=None, help="Optional explicit local model path.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Generation cap override.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature override.")
    parser.add_argument("--quantization-mode", default=None, help="Optional quantized load mode, such as bnb_4bit.")
    parser.add_argument("--quant-compute-dtype", default="bfloat16", help="Quantized compute dtype override.")
    parser.add_argument("--quant-type", default="nf4", help="4-bit quantization type override.")
    parser.add_argument(
        "--quant-use-double-quant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable double quantization for 4-bit bitsandbytes loads.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Launch a visible browser window instead of running headless.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(get_output_dir("eval_single_task")),
    )
    parser.add_argument("--use-fake-runner", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point and print machine-readable aggregated metrics."""

    args = build_arg_parser().parse_args(argv)
    from src.utils.config import resolve_model_path

    policy = load_policy(
        args.policy,
        model_dir_name=args.model_dir_name,
        model_path=str(resolve_model_path(model_path=args.model_path, model_dir_name=args.model_dir_name)),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        quantization_mode=args.quantization_mode,
        quant_compute_dtype=args.quant_compute_dtype,
        quant_type=args.quant_type,
        quant_use_double_quant=args.quant_use_double_quant,
    )
    metrics = evaluate_single_task(
        policy=policy,
        task_id=args.task_id,
        seed=args.seed,
        max_steps=args.max_steps,
        episodes=args.episodes,
        out_dir=args.out_dir,
        use_fake_runner=args.use_fake_runner,
        headless=not args.headed,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
