from __future__ import annotations

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env.webarena_runner import DummyPolicy, run_episode  # noqa: E402
from src.agent.types import PolicyConfig  # noqa: E402
from src.utils.config import DEFAULT_MAX_STEPS, DEFAULT_SEED, DEFAULT_TASK_ID, get_output_dir, resolve_model_path  # noqa: E402
from src.utils.model_profiles import DEFAULT_QWEN_PROFILE, find_model_profile_by_dir_name, find_model_profile_by_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one WebArena episode with a Task 2 runner policy.")
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--policy", choices=("dummy", "qwen"), default="dummy")
    parser.add_argument("--model-dir-name", default=None, help="Optional local model directory name under ../models/.")
    parser.add_argument("--model-path", default=None, help="Optional explicit local model path.")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Generation cap override.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature override.")
    parser.add_argument("--top-k", type=int, default=None, help="Sampling top-k override.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Generation repetition penalty override.")
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Launch a visible browser window instead of running headless.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(get_output_dir("single_task_runs")),
        help="Directory where steps.jsonl, episode.json, and screenshots will be written.",
    )
    return parser.parse_args()


def load_policy(
    policy_name: str,
    *,
    model_dir_name: str | None = None,
    model_path: str | None = None,
    max_new_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
):
    if policy_name == "dummy":
        return DummyPolicy()

    if policy_name == "qwen":
        try:
            module = import_module("src.agent.qwen_policy")
        except ImportError as exc:
            raise RuntimeError(
                "Qwen policy is not available on this branch yet. Implement or merge Task 3 before using --policy qwen."
            ) from exc

        policy_cls = getattr(module, "QwenPolicy", None)
        if policy_cls is None:
            raise RuntimeError("src.agent.qwen_policy does not define QwenPolicy.")
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
                top_k=top_k if top_k is not None else profile.top_k,
                repetition_penalty=(
                    repetition_penalty if repetition_penalty is not None else profile.repetition_penalty
                ),
                system_prompt=profile.system_prompt,
                use_chat_template=profile.use_chat_template,
            )
        )

    raise RuntimeError(f"Unsupported policy: {policy_name}")


def main() -> int:
    args = parse_args()
    policy = load_policy(
        args.policy,
        model_dir_name=args.model_dir_name,
        model_path=str(resolve_model_path(model_path=args.model_path, model_dir_name=args.model_dir_name)),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )
    episode = run_episode(
        policy=policy,
        task_id=args.task_id,
        seed=args.seed,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        headless=not args.headed,
    )
    print(json.dumps(episode, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
