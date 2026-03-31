from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Protocol

import browsergym.webarena  # noqa: F401
import gymnasium as gym
import numpy as np
from PIL import Image

from src.utils.config import DEFAULT_MAX_STEPS


MAX_VISIBLE_SUMMARY_ITEMS = 12
MAX_AX_SNIPPET_ITEMS = 40
MAX_TEXT_CHARS = 4000


class Policy(Protocol):
    def act(self, observation: dict[str, Any], step_idx: int) -> Any:
        """Produce the next BrowserGym action or a structured decision."""


class DummyPolicy:
    """Deterministic stop-equivalent policy for runner validation."""

    name = "dummy"

    def __init__(self, action_text: str = 'send_msg_to_user("N/A")') -> None:
        self.action_text = action_text

    def act(self, observation: dict[str, Any], step_idx: int) -> dict[str, Any]:
        return {
            "raw_text": self.action_text,
            "action_text": self.action_text,
            "parse_error": None,
            "should_retry": False,
        }


def make_env(
    task_id: int,
    headless: bool,
    record_video_dir: str | Path | None = None,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
):
    env_id = f"browsergym/webarena.{task_id}"
    env_kwargs: dict[str, Any] = {
        "headless": headless,
        "max_episode_steps": max_steps,
    }
    if record_video_dir is not None:
        env_kwargs["record_video_dir"] = str(Path(record_video_dir))
    return gym.make(env_id, **env_kwargs)


def serialize_observation(
    obs: Mapping[str, Any],
    *,
    previous_actions: list[str] | None = None,
    previous_errors: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "goal": _as_text(obs.get("goal")),
        "current_url": _as_text(obs.get("url")),
        "open_tabs": _serialize_open_tabs(obs),
        "visible_page_summary": _truncate_text(_build_visible_page_summary(obs)),
        "dom_or_ax_snippet": _truncate_text(_build_dom_or_ax_snippet(obs)),
        "previous_actions": list(previous_actions or []),
        "previous_errors": list(previous_errors or []),
        "last_action_error": _as_optional_text(obs.get("last_action_error")),
    }


def run_episode(
    policy: Policy | Any,
    task_id: int,
    seed: int,
    max_steps: int,
    out_dir: str | Path,
    *,
    headless: bool = True,
) -> dict[str, Any]:
    output_dir = Path(out_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    steps_path = output_dir / "steps.jsonl"
    episode_path = output_dir / "episode.json"
    record_video_dir = output_dir / "videos"

    previous_actions: list[str] = []
    previous_errors: list[str] = []
    failure_reasons: list[str] = []
    invalid_action_count = 0
    parse_failure_count = 0
    total_reward = 0.0
    raw_goal = ""
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
    episode_start = time.time()

    env = make_env(
        task_id=task_id,
        headless=headless,
        record_video_dir=record_video_dir,
        max_steps=max_steps,
    )

    try:
        obs, reset_info = env.reset(seed=seed)
        raw_goal = _as_text(obs.get("goal"))

        with steps_path.open("w", encoding="utf-8") as steps_file:
            for step_idx in range(max_steps):
                serialized_observation = serialize_observation(
                    obs,
                    previous_actions=previous_actions,
                    previous_errors=previous_errors,
                )
                screenshot_relpath = _save_screenshot(
                    obs.get("screenshot"),
                    screenshots_dir / f"step_{step_idx:04d}.png",
                    output_dir,
                )
                observation_summary = _build_observation_summary(serialized_observation)
                decision = _normalize_policy_output(_call_policy(policy, serialized_observation, step_idx))

                if decision["parse_error"]:
                    parse_failure_count += 1

                next_obs, reward, terminated, truncated, step_info = env.step(decision["action_text"])
                total_reward += float(reward)

                last_action_error = _as_optional_text(next_obs.get("last_action_error"))
                if last_action_error:
                    invalid_action_count += 1
                    previous_errors.append(last_action_error)
                    if last_action_error not in failure_reasons:
                        failure_reasons.append(last_action_error)

                step_record = {
                    "step_idx": step_idx,
                    "observation": serialized_observation,
                    "observation_summary": observation_summary,
                    "screenshot_path": screenshot_relpath,
                    "raw_agent_output": decision["raw_text"],
                    "parsed_action": decision["action_text"],
                    "parse_error": decision["parse_error"],
                    "should_retry": decision["should_retry"],
                    "last_action_error": last_action_error,
                    "reward": float(reward),
                    "done": bool(terminated or truncated),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": _make_json_safe(step_info),
                }
                steps_file.write(json.dumps(step_record, sort_keys=True) + "\n")
                steps_file.flush()

                previous_actions.append(decision["action_text"])
                final_info = _make_json_safe(step_info)
                obs = next_obs

                if terminated or truncated:
                    break

        if truncated and "episode_truncated" not in failure_reasons:
            failure_reasons.append("episode_truncated")
        if total_reward <= 0.0 and not failure_reasons:
            failure_reasons.append("terminated_without_positive_reward")

        episode_data = {
            "task_id": task_id,
            "env_id": f"browsergym/webarena.{task_id}",
            "seed": seed,
            "max_steps": max_steps,
            "policy_name": _policy_name(policy),
            "raw_goal": raw_goal,
            "steps": len(previous_actions),
            "reward": float(total_reward),
            "success": bool(total_reward > 0.0),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "invalid_action_count": invalid_action_count,
            "parse_failure_count": parse_failure_count,
            "failure_reasons": failure_reasons,
            "headless": headless,
            "started_at": episode_start,
            "finished_at": time.time(),
            "duration_sec": time.time() - episode_start,
            "output_dir": str(output_dir),
            "steps_path": str(steps_path),
            "episode_path": str(episode_path),
            "reset_info": _make_json_safe(reset_info),
            "final_info": final_info,
        }
        episode_path.write_text(
            json.dumps(episode_data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return episode_data
    finally:
        env.close()


def _call_policy(policy: Policy | Any, observation: dict[str, Any], step_idx: int) -> Any:
    if hasattr(policy, "act"):
        return policy.act(observation, step_idx)
    if callable(policy):
        return policy(observation, step_idx)
    raise TypeError("Policy must be callable or expose an act(observation, step_idx) method.")


def _normalize_policy_output(result: Any) -> dict[str, Any]:
    if isinstance(result, str):
        return {
            "raw_text": result,
            "action_text": result,
            "parse_error": None,
            "should_retry": False,
        }

    if isinstance(result, Mapping):
        raw_text = _as_text(result.get("raw_text") or result.get("action_text"))
        action_text = _as_text(result.get("action_text"))
        return {
            "raw_text": raw_text,
            "action_text": action_text,
            "parse_error": _as_optional_text(result.get("parse_error")),
            "should_retry": bool(result.get("should_retry", False)),
        }

    raw_text = _as_optional_text(getattr(result, "raw_text", None))
    action_text = _as_optional_text(getattr(result, "action_text", None))
    if action_text is None:
        raise TypeError("Policy output must provide an action_text field or be a raw action string.")

    return {
        "raw_text": raw_text or action_text,
        "action_text": action_text,
        "parse_error": _as_optional_text(getattr(result, "parse_error", None)),
        "should_retry": bool(getattr(result, "should_retry", False)),
    }


def _serialize_open_tabs(obs: Mapping[str, Any]) -> list[dict[str, str]]:
    titles = list(obs.get("open_pages_titles", ()))
    urls = list(obs.get("open_pages_urls", ()))
    tab_count = max(len(titles), len(urls))
    tabs: list[dict[str, str]] = []
    for index in range(tab_count):
        title = _as_text(titles[index] if index < len(titles) else "")
        url = _as_text(urls[index] if index < len(urls) else "")
        tabs.append({"title": title, "url": url})
    return tabs


def _build_observation_summary(serialized_observation: Mapping[str, Any]) -> dict[str, Any]:
    visible_page_summary = _as_text(serialized_observation.get("visible_page_summary"))
    dom_or_ax_snippet = _as_text(serialized_observation.get("dom_or_ax_snippet"))
    return {
        "current_url": _as_text(serialized_observation.get("current_url")),
        "open_tabs": serialized_observation.get("open_tabs", []),
        "visible_page_summary_preview": visible_page_summary[:400],
        "dom_or_ax_snippet_preview": dom_or_ax_snippet[:600],
        "last_action_error": serialized_observation.get("last_action_error"),
    }


def _build_visible_page_summary(obs: Mapping[str, Any]) -> str:
    ax_items = _extract_ax_items(obs, max_items=MAX_VISIBLE_SUMMARY_ITEMS)
    if not ax_items:
        return "(empty)"

    summary_lines: list[str] = []
    for item in ax_items:
        line = item["label"]
        if item["clickable"]:
            line += " (clickable)"
        if item["focused"]:
            line += " (focused)"
        summary_lines.append(line)
    return "\n".join(summary_lines)


def _build_dom_or_ax_snippet(obs: Mapping[str, Any]) -> str:
    ax_items = _extract_ax_items(obs, max_items=MAX_AX_SNIPPET_ITEMS)
    if not ax_items:
        return "(empty)"

    snippet_lines = []
    for item in ax_items:
        line = f"[{item['bid']}] role={item['role']}"
        if item["name"]:
            line += f' name="{item["name"]}"'
        if item["clickable"]:
            line += " clickable"
        if item["focused"]:
            line += " focused"
        snippet_lines.append(line)
    return "\n".join(snippet_lines)


def _extract_ax_items(obs: Mapping[str, Any], *, max_items: int) -> list[dict[str, Any]]:
    axtree = obs.get("axtree_object")
    if not isinstance(axtree, Mapping):
        return []

    nodes = axtree.get("nodes")
    if not isinstance(nodes, list):
        return []

    extra_props = obs.get("extra_element_properties")
    if not isinstance(extra_props, Mapping):
        extra_props = {}

    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for node in nodes:
        if not isinstance(node, Mapping):
            continue

        bid = _as_optional_text(node.get("browsergym_id"))
        if not bid:
            continue

        props = extra_props.get(bid, {})
        if not isinstance(props, Mapping):
            props = {}

        bbox = props.get("bbox")
        clickable = bool(props.get("clickable", False))
        focused = _node_property_flag(node, "focused")
        role = _extract_nested_value(node.get("role")) or "unknown"
        name = _normalize_inline_text(_extract_nested_value(node.get("name")) or "")

        if not (name or clickable or focused):
            continue
        if bbox is None and not clickable and not focused:
            continue
        if role in {"none", "generic", "StaticText", "InlineTextBox"} and not (clickable or focused) and not name:
            continue

        label = name or f"[{bid}] {role}"
        key = (bid, role, label)
        if key in seen:
            continue
        seen.add(key)

        items.append(
            {
                "bid": bid,
                "role": role,
                "name": name,
                "label": label,
                "clickable": clickable,
                "focused": focused,
            }
        )
        if len(items) >= max_items:
            break

    return items


def _node_property_flag(node: Mapping[str, Any], name: str) -> bool:
    properties = node.get("properties")
    if not isinstance(properties, list):
        return False

    for prop in properties:
        if not isinstance(prop, Mapping):
            continue
        if prop.get("name") != name:
            continue
        value = prop.get("value")
        if isinstance(value, Mapping):
            return bool(value.get("value"))
    return False


def _extract_nested_value(value: Any) -> str | None:
    if isinstance(value, Mapping):
        nested = value.get("value")
        if nested is not None:
            return _as_optional_text(nested)
    return _as_optional_text(value)


def _normalize_inline_text(value: str) -> str:
    normalized = " ".join(value.split())
    return normalized[:300]


def _truncate_text(value: str) -> str:
    if len(value) <= MAX_TEXT_CHARS:
        return value
    return f"{value[:MAX_TEXT_CHARS]}..."


def _save_screenshot(screenshot: Any, target_path: Path, output_dir: Path) -> str | None:
    if screenshot is None:
        return None

    array = np.asarray(screenshot)
    if array.size == 0:
        return None

    image = Image.fromarray(array.astype(np.uint8))
    image.save(target_path)
    return str(target_path.relative_to(output_dir))


def _policy_name(policy: Any) -> str:
    return _as_optional_text(getattr(policy, "name", None)) or policy.__class__.__name__


def _make_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _as_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
