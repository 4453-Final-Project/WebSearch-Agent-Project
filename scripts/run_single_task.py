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
from src.utils.config import DEFAULT_MAX_STEPS, DEFAULT_SEED, DEFAULT_TASK_ID, get_output_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one WebArena episode with a Task 2 runner policy.")
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--policy", choices=("dummy", "qwen"), default="dummy")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(get_output_dir("single_task_runs")),
        help="Directory where steps.jsonl, episode.json, and screenshots will be written.",
    )
    return parser.parse_args()


def load_policy(policy_name: str):
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
        return policy_cls()

    raise RuntimeError(f"Unsupported policy: {policy_name}")


def main() -> int:
    args = parse_args()
    policy = load_policy(args.policy)
    episode = run_episode(
        policy=policy,
        task_id=args.task_id,
        seed=args.seed,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
    )
    print(json.dumps(episode, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
