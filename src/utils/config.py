from __future__ import annotations

import os
from pathlib import Path

DEFAULT_TASK_ID = 310
DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 30

MODEL_DIR_NAME = "Qwen2.5-3B-Instruct-GPTQ-Int4"
OUTPUT_DIR_NAME = "outputs"

REQUIRED_WA_ENV_VARS = (
    "WA_SHOPPING",
    "WA_SHOPPING_ADMIN",
    "WA_REDDIT",
    "WA_GITLAB",
    "WA_WIKIPEDIA",
    "WA_MAP",
    "WA_HOMEPAGE",
)

OPTIONAL_WA_ENV_VARS = ("WA_FULL_RESET",)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_workspace_root() -> Path:
    return get_repo_root().parent


def get_model_path(model_dir_name: str = MODEL_DIR_NAME) -> Path:
    return get_workspace_root() / "models" / model_dir_name


def get_output_dir(*parts: str, create: bool = False) -> Path:
    path = get_workspace_root() / OUTPUT_DIR_NAME
    if parts:
        path = path.joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_env_var(name: str, default: str | None = None, required: bool = False) -> str | None:
    value = os.environ.get(name, default)
    if required and (value is None or value == ""):
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def get_webarena_env(include_optional: bool = True) -> dict[str, str | None]:
    env = {name: get_env_var(name) for name in REQUIRED_WA_ENV_VARS}
    if include_optional:
        env.update({name: get_env_var(name) for name in OPTIONAL_WA_ENV_VARS})
    return env


def get_required_webarena_env() -> dict[str, str]:
    return {name: get_env_var(name, required=True) for name in REQUIRED_WA_ENV_VARS}  # type: ignore[return-value]


def get_missing_webarena_env() -> list[str]:
    return [name for name in REQUIRED_WA_ENV_VARS if not get_env_var(name)]
