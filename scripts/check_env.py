from __future__ import annotations

import argparse
import importlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import (  # noqa: E402
    DEFAULT_MAX_STEPS,
    DEFAULT_SEED,
    DEFAULT_TASK_ID,
    OPTIONAL_WA_ENV_VARS,
    REQUIRED_WA_ENV_VARS,
    get_model_id,
    get_required_webarena_env,
    get_webarena_env,
    resolve_model_path,
    sync_webarena_env_aliases,
)


CORE_IMPORT_MODULES = (
    "torch",
    "transformers",
    "browsergym",
    "webarena",
    "playwright",
)

MODEL_CONFIG_NAME = "config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Task 1 local runtime and WebArena setup.")
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID, help="BrowserGym WebArena task id")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed used for the single reset smoke test")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Max episode steps override for the BrowserGym smoke test",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Per-request timeout in seconds for HTTP reachability checks",
    )
    parser.add_argument("--model-dir-name", default=None, help="Optional local model directory name under ../models/.")
    parser.add_argument("--model-path", default=None, help="Optional explicit local model path.")
    return parser.parse_args()


def run_step(label: str, callback) -> None:
    print(f"Status: {label}...", flush=True)
    callback()


def check_python() -> None:
    print(f"Python version: {sys.version.split()[0]}", flush=True)
    print(f"Active interpreter: {Path(sys.executable).resolve()}", flush=True)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Expected Python 3.12.x.")


def check_imports() -> None:
    sync_webarena_env_aliases()
    for module_name in CORE_IMPORT_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised by environment
            raise RuntimeError(f"Import failed for {module_name}: {type(exc).__name__}: {exc}") from exc
        print(f"Import OK: {module_name}", flush=True)

    for optional_module in ("optimum", "gptqmodel"):
        try:
            importlib.import_module(optional_module)
        except Exception:
            print(f"Optional import unavailable: {optional_module}", flush=True)
        else:
            print(f"Optional import OK: {optional_module}", flush=True)


def check_model(model_path_override: str | None = None, model_dir_name: str | None = None) -> None:
    model_id = get_model_id()
    model_path = resolve_model_path(model_path=model_path_override, model_dir_name=model_dir_name)
    config_path = model_path / MODEL_CONFIG_NAME

    print(f"Model path: {model_path}", flush=True)
    print(f"Model id: {model_id}", flush=True)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Local model directory not found: {model_path}\n"
            "Download it with: "
            f"hf download {model_id} --local-dir {model_path}"
        )

    if not config_path.exists():
        raise FileNotFoundError(f"Missing model config: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    quantization_config = data.get("quantization_config")
    if quantization_config is None:
        print("Model config OK: standard Transformers checkpoint", flush=True)
        return

    quant_method = str(quantization_config.get("quant_method", "")).lower()
    if quant_method:
        print(f"Model config OK: quantized checkpoint ({quant_method})", flush=True)
        return

    raise RuntimeError("config.json contains quantization_config but no quant_method.")


def check_webarena_env_vars() -> None:
    env = get_required_webarena_env()
    for name in REQUIRED_WA_ENV_VARS:
        print(f"{name}={env[name]}", flush=True)

    optional_env = get_webarena_env(include_optional=True)
    for name in OPTIONAL_WA_ENV_VARS:
        value = optional_env.get(name)
        printable = "<unset>" if value is None else value
        print(f"{name}={printable}", flush=True)


def fetch_url(url: str, timeout: float) -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "Task1CheckEnv/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return getattr(response, "status", 200)


def check_webarena_urls(timeout: float) -> None:
    env = get_required_webarena_env()
    for name in REQUIRED_WA_ENV_VARS:
        url = env[name]
        try:
            status = fetch_url(url, timeout=timeout)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"{name} returned HTTP {exc.code}: {url}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{name} was not reachable: {url} ({exc.reason})") from exc
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(f"{name} reachability check failed for {url}: {type(exc).__name__}: {exc}") from exc

        if not 200 <= status < 400:
            raise RuntimeError(f"{name} returned HTTP {status}: {url}")

        print(f"Reachable: {name} -> HTTP {status}", flush=True)


def check_playwright() -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, chromium_sandbox=False)
        page = browser.new_page()
        page.goto("about:blank", wait_until="load")
        browser.close()

    print("Playwright Chromium launch OK", flush=True)


def check_browsergym_reset(task_id: int, seed: int, max_steps: int) -> None:
    import gymnasium as gym
    import browsergym.webarena  # noqa: F401

    env_id = f"browsergym/webarena.{task_id}"
    print(f"BrowserGym env id: {env_id}", flush=True)

    env = None
    try:
        env = gym.make(env_id, headless=True, max_episode_steps=max_steps)
        print("gym.make OK", flush=True)
        env.reset(seed=seed)
        print("env.reset OK", flush=True)
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(f"BrowserGym WebArena reset failed: {type(exc).__name__}: {exc}") from exc
    finally:
        if env is not None:
            env.close()


def main() -> int:
    args = parse_args()

    try:
        run_step("checking Python version and active interpreter", check_python)
        run_step("checking required Python imports", check_imports)
        run_step(
            "checking local model path and config",
            lambda: check_model(args.model_path, args.model_dir_name),
        )
        run_step("checking required WebArena environment variables", check_webarena_env_vars)
        run_step("checking HTTP reachability of WebArena URLs", lambda: check_webarena_urls(args.timeout))
        run_step("checking Playwright Chromium launch", check_playwright)
        run_step(
            f"checking BrowserGym WebArena task {args.task_id} reset",
            lambda: check_browsergym_reset(args.task_id, args.seed, args.max_steps),
        )
    except Exception as exc:
        print(f"FAILED: {exc}", flush=True)
        return 1

    print("Verification complete!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
