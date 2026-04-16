from __future__ import annotations

import os
from pathlib import Path

DEFAULT_TASK_ID = 310
DEFAULT_SEED = 42
DEFAULT_MAX_STEPS = 30

DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"
DEFAULT_MODEL_DIR_NAME = "Qwen3.5-2B"
DEFAULT_OPENAI_JUDGE_MODEL = "gpt-5-mini"
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
WEBARENA_ENV_ALIASES = (
    ("WA_REDDIT", "REDDIT"),
    ("WA_SHOPPING", "SHOPPING"),
    ("WA_SHOPPING_ADMIN", "SHOPPING_ADMIN"),
    ("WA_GITLAB", "GITLAB"),
    ("WA_WIKIPEDIA", "WIKIPEDIA"),
    ("WA_MAP", "MAP"),
    ("WA_HOMEPAGE", "HOMEPAGE"),
)


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_workspace_root() -> Path:
    return get_repo_root().parent


def get_model_id() -> str:
    return get_env_var("MODEL_ID", DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID


def get_model_dir_name(default: str = DEFAULT_MODEL_DIR_NAME) -> str:
    return get_env_var("MODEL_DIR_NAME", default) or default


def get_openai_judge_model() -> str:
    return get_env_var("OPENAI_JUDGE_MODEL", DEFAULT_OPENAI_JUDGE_MODEL) or DEFAULT_OPENAI_JUDGE_MODEL


def get_model_path(model_dir_name: str | None = None) -> Path:
    resolved_dir_name = model_dir_name or get_model_dir_name()
    task3_override = get_env_var("TASK3_MODEL_PATH")
    if task3_override:
        return Path(task3_override).expanduser()
    return get_workspace_root() / "models" / resolved_dir_name


def resolve_model_path(
    *,
    model_path: str | None = None,
    model_dir_name: str | None = None,
) -> Path:
    """Resolve an explicit model override or fall back to the default sibling-model layout."""

    if model_path:
        return Path(model_path).expanduser()
    if model_dir_name:
        return get_workspace_root() / "models" / model_dir_name
    return get_model_path()


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


def sync_webarena_env_aliases() -> None:
    for source_name, target_name in WEBARENA_ENV_ALIASES:
        value = get_env_var(source_name)
        if value and not os.environ.get(target_name):
            os.environ[target_name] = value
