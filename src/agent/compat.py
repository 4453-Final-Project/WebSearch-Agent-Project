"""Compatibility helpers for unfinished Task 1 and Task 2 interfaces."""

from __future__ import annotations

import os
from importlib import import_module
from typing import Any, Iterable

from .types import NormalizedObservation, OpenTab, PolicyConfig


def normalize_observation(raw: dict[str, Any]) -> NormalizedObservation:
    """Normalize partially-stable Task 2 observation payloads.

    The current project is still converging on a final observation schema.
    This helper accepts multiple alias keys so Task 3 can be tested against
    temporary fixtures and swapped to the final Task 2 output later.
    """

    goal = _coerce_text(_pick_alias(raw, "goal", "task_goal"))
    current_url = _coerce_text(_pick_alias(raw, "current_url", "url"))
    open_tabs = _normalize_open_tabs(_pick_alias(raw, "open_tabs", "tabs", default=[]))
    visible_page_summary = _coerce_text(
        _pick_alias(raw, "visible_page_summary", "page_summary", "visible_text")
    )
    dom_or_ax_snippet = _coerce_text(
        _pick_alias(raw, "dom_or_ax_snippet", "dom_snippet", "ax_tree")
    )
    previous_actions = _normalize_string_list(
        _pick_alias(raw, "previous_actions", "action_history", default=[])
    )
    previous_errors = _normalize_string_list(
        _pick_alias(raw, "previous_errors", "error_history", default=[])
    )
    last_action_error = _coerce_optional_text(_pick_alias(raw, "last_action_error", default=None))

    return NormalizedObservation(
        goal=goal,
        current_url=current_url,
        open_tabs=open_tabs,
        visible_page_summary=visible_page_summary,
        dom_or_ax_snippet=dom_or_ax_snippet,
        previous_actions=previous_actions,
        previous_errors=previous_errors,
        last_action_error=last_action_error,
    )


def load_policy_config() -> PolicyConfig:
    """Load Task 3 policy configuration from Task 1 config or environment.

    The loader first attempts to read optional values from ``src.utils.config``.
    Missing modules or attributes are tolerated. Environment variables then
    provide overrides, and safe defaults keep the contract usable in tests.
    """

    config_module = _load_optional_config_module()

    model_path = _get_config_value(
        config_module,
        env_name="TASK3_MODEL_PATH",
        attr_names=("TASK3_MODEL_PATH", "MODEL_PATH"),
        default="",
    )
    max_new_tokens = _get_int_config_value(
        config_module,
        env_name="TASK3_MAX_NEW_TOKENS",
        attr_names=("TASK3_MAX_NEW_TOKENS", "MAX_NEW_TOKENS"),
        default=128,
    )
    temperature = _get_float_config_value(
        config_module,
        env_name="TASK3_TEMPERATURE",
        attr_names=("TASK3_TEMPERATURE", "TEMPERATURE"),
        default=0.0,
    )
    device = _get_optional_str_config_value(
        config_module,
        env_name="TASK3_DEVICE",
        attr_names=("TASK3_DEVICE", "DEVICE"),
        default=None,
    )

    return PolicyConfig(
        model_path=model_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device=device,
    )


def _pick_alias(raw: dict[str, Any], *keys: str, default: Any = "") -> Any:
    """Return the first present alias value from a raw observation dictionary."""

    for key in keys:
        if key in raw:
            return raw[key]
    return default


def _normalize_open_tabs(value: Any) -> list[OpenTab]:
    """Convert raw tab payloads into ``OpenTab`` instances."""

    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [OpenTab(title=str(value), url=str(value))]
    if not isinstance(value, Iterable):
        return []

    normalized_tabs: list[OpenTab] = []
    for item in value:
        if isinstance(item, OpenTab):
            normalized_tabs.append(item)
            continue

        if isinstance(item, dict):
            title = _coerce_text(_pick_alias(item, "title", "name"))
            url = _coerce_text(_pick_alias(item, "url", "href"))
            normalized_tabs.append(OpenTab(title=title, url=url))
            continue

        title = _coerce_text(getattr(item, "title", ""))
        url = _coerce_text(getattr(item, "url", ""))
        if title or url:
            normalized_tabs.append(OpenTab(title=title, url=url))
            continue

        text = _coerce_text(item)
        normalized_tabs.append(OpenTab(title=text, url=text))

    return normalized_tabs


def _normalize_string_list(value: Any) -> list[str]:
    """Normalize a single string or iterable payload into a list of strings."""

    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)]
    if not isinstance(value, Iterable):
        return [str(value)]
    return [str(item) for item in value]


def _coerce_text(value: Any) -> str:
    """Convert a raw value into a non-``None`` string."""

    if value is None:
        return ""
    return str(value)


def _coerce_optional_text(value: Any) -> str | None:
    """Convert a raw value into an optional string."""

    if value is None:
        return None
    text = str(value)
    return text


def _load_optional_config_module() -> Any:
    """Import ``src.utils.config`` if it is available, otherwise return ``None``."""

    try:
        return import_module("src.utils.config")
    except ImportError:
        return None


def _get_config_attr(config_module: Any, attr_names: tuple[str, ...]) -> Any:
    """Read the first present configuration attribute from an optional module."""

    if config_module is None:
        return None
    for attr_name in attr_names:
        if hasattr(config_module, attr_name):
            return getattr(config_module, attr_name)
    return None


def _get_config_value(
    config_module: Any,
    env_name: str,
    attr_names: tuple[str, ...],
    default: str,
) -> str:
    """Resolve a string config value from module attributes and environment."""

    value = _get_config_attr(config_module, attr_names)
    if value is None:
        value = os.getenv(env_name)
    if value is None:
        return default
    return str(value)


def _get_int_config_value(
    config_module: Any,
    env_name: str,
    attr_names: tuple[str, ...],
    default: int,
) -> int:
    """Resolve an integer config value with forgiving parsing."""

    value = _get_config_attr(config_module, attr_names)
    if value is None:
        value = os.getenv(env_name)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _get_float_config_value(
    config_module: Any,
    env_name: str,
    attr_names: tuple[str, ...],
    default: float,
) -> float:
    """Resolve a float config value with forgiving parsing."""

    value = _get_config_attr(config_module, attr_names)
    if value is None:
        value = os.getenv(env_name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_optional_str_config_value(
    config_module: Any,
    env_name: str,
    attr_names: tuple[str, ...],
    default: str | None,
) -> str | None:
    """Resolve an optional string config value."""

    value = _get_config_attr(config_module, attr_names)
    if value is None:
        value = os.getenv(env_name)
    if value is None or value == "":
        return default
    return str(value)
