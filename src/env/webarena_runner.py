from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import browsergym.webarena  # noqa: F401
import gymnasium as gym
import numpy as np
import openai
from PIL import Image
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from src.agent.prompting import derive_shopping_category_labels
from src.utils.config import DEFAULT_MAX_STEPS, get_openai_judge_model, sync_webarena_env_aliases


MAX_VISIBLE_SUMMARY_ITEMS = 12
MAX_AX_SNIPPET_ITEMS = 40
MAX_TEXT_CHARS = 4000
DEFAULT_BROWSER_TIMEOUT_MS = 30000
SHOPPING_LOW_VALUE_LABELS = {
    "privacy and cookie policy",
    "search terms",
    "advanced search",
    "orders and returns",
    "contact us",
    "report all bugs",
    "my account",
    "my wish list",
    "sign in",
    "create an account",
    "store logo",
    "one stop market",
}
SHOPPING_GOAL_STOPWORDS = {
    "the",
    "by",
    "from",
    "show",
    "me",
    "list",
    "products",
    "product",
    "category",
    "items",
    "item",
    "ascending",
    "descending",
    "price",
    "prices",
    "sort",
    "sorted",
}


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
    sync_webarena_env_aliases()
    _patch_webarena_openai_judges()
    _patch_playwright_shopping_navigation()
    _patch_webarena_shopping_login()
    _patch_webarena_shopping_task_urls()
    env_id = f"browsergym/webarena.{task_id}"
    env_kwargs: dict[str, Any] = {
        "headless": headless,
        "max_episode_steps": max_steps,
        "timeout": _resolve_browser_timeout_ms(),
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
    step_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    output_dir = Path(out_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    screenshots_dir = output_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    steps_path = output_dir / "steps.jsonl"
    steps_text_path = output_dir / "steps.txt"
    episode_path = output_dir / "episode.json"
    episode_text_path = output_dir / "episode.txt"
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
    step_text_blocks: list[str] = []

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
                step_text_blocks.append(_format_step_record_text(step_record))
                if step_callback is not None:
                    step_callback(step_record)

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
            "steps_text_path": str(steps_text_path),
            "episode_text_path": str(episode_text_path),
            "reset_info": _make_json_safe(reset_info),
            "final_info": final_info,
        }
        episode_path.write_text(
            json.dumps(episode_data, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        steps_text_path.write_text("\n\n".join(step_text_blocks) + ("\n" if step_text_blocks else ""), encoding="utf-8")
        episode_text_path.write_text(_format_episode_data_text(episode_data), encoding="utf-8")
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


def _format_step_record_text(step_record: Mapping[str, Any]) -> str:
    summary = step_record.get("observation_summary", {})
    lines = [
        f"Step {step_record.get('step_idx', '(unknown)')}",
        f"URL: {summary.get('current_url', '(unknown)')}",
        f"Action: {step_record.get('parsed_action', '')}",
        f"Reward: {float(step_record.get('reward', 0.0)):.3f}",
        f"Done: {bool(step_record.get('done', False))}",
        f"Terminated: {bool(step_record.get('terminated', False))}",
        f"Truncated: {bool(step_record.get('truncated', False))}",
        f"Parse Error: {step_record.get('parse_error') or 'none'}",
        f"Last Action Error: {step_record.get('last_action_error') or 'none'}",
        "Visible Page Summary Preview:",
        str(summary.get("visible_page_summary_preview", "(empty)")) or "(empty)",
        "DOM/AX Snippet Preview:",
        str(summary.get("dom_or_ax_snippet_preview", "(empty)")) or "(empty)",
    ]
    return "\n".join(lines)


def _format_episode_data_text(episode_data: Mapping[str, Any]) -> str:
    failure_reasons = episode_data.get("failure_reasons", [])
    lines = [
        "Episode Summary",
        f"Policy: {episode_data.get('policy_name', '(unknown)')}",
        f"Task ID: {episode_data.get('task_id', '(unknown)')}",
        f"Seed: {episode_data.get('seed', '(unknown)')}",
        f"Max Steps: {episode_data.get('max_steps', '(unknown)')}",
        f"Steps Taken: {episode_data.get('steps', 0)}",
        f"Reward: {float(episode_data.get('reward', 0.0)):.3f}",
        f"Success: {bool(episode_data.get('success', False))}",
        f"Invalid Action Count: {episode_data.get('invalid_action_count', 0)}",
        f"Parse Failure Count: {episode_data.get('parse_failure_count', 0)}",
        f"Duration (sec): {float(episode_data.get('duration_sec', 0.0)):.3f}",
        "Failure Reasons:",
    ]
    if failure_reasons:
        lines.extend(f"- {reason}" for reason in failure_reasons)
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


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
    summary_lines: list[str] = []
    seen_labels: set[str] = set()

    current_url = _as_text(obs.get("url"))
    goal = _as_text(obs.get("goal"))
    if _is_shopping_domain(current_url):
        goal_keywords = _shopping_goal_keywords(goal)
        summary_lines.extend(
            _extract_shopping_order_extra_summary_lines(
                obs,
                goal_keywords=goal_keywords,
                seen_labels=seen_labels,
                max_items=MAX_VISIBLE_SUMMARY_ITEMS,
            )
        )
        seen_labels.update(summary_lines)

    for item in ax_items:
        line = item["label"]
        if line in seen_labels:
            continue
        if item["clickable"]:
            line += " (clickable)"
        if item["focused"]:
            line += " (focused)"
        summary_lines.append(line)
        seen_labels.add(item["label"])
        if len(summary_lines) >= MAX_VISIBLE_SUMMARY_ITEMS:
            break

    if not summary_lines:
        return "(empty)"
    return "\n".join(summary_lines[:MAX_VISIBLE_SUMMARY_ITEMS])


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
    goal = _as_text(obs.get("goal"))
    current_url = _as_text(obs.get("url"))
    shopping_goal_keywords = _shopping_goal_keywords(goal) if _is_shopping_domain(current_url) else set()
    shopping_category_labels = derive_shopping_category_labels(goal) if _is_shopping_domain(current_url) else []

    for index, node in enumerate(nodes):
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
        if bbox is None and not clickable and not focused and not _allow_bboxless_shopping_order_text(name, role, shopping_goal_keywords):
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
                "index": index,
                "bid": bid,
                "role": role,
                "name": name,
                "label": label,
                "clickable": clickable,
                "focused": focused,
            }
        )

    shopping_next_category_target = ""
    if _is_shopping_domain(current_url):
        shopping_next_category_target = _shopping_next_category_target(
            shopping_category_labels,
            current_url,
            items,
        )
        items.sort(
            key=lambda item: (
                -_shopping_item_priority_score(item, shopping_goal_keywords, shopping_next_category_target),
                item["index"],
            )
        )

    if len(items) > max_items:
        items = items[:max_items]

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


def _shopping_goal_keywords(goal: str) -> set[str]:
    tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'/-]*", goal)}
    return {token for token in tokens if len(token) >= 3 and token not in SHOPPING_GOAL_STOPWORDS}


def _shopping_item_priority_score(
    item: Mapping[str, Any],
    goal_keywords: set[str],
    next_category_target: str,
) -> int:
    name = _as_text(item.get("name")).lower()
    role = _as_text(item.get("role")).lower()
    score = 0

    if name in SHOPPING_LOW_VALUE_LABELS:
        score -= 20

    if next_category_target:
        category_score = _shopping_category_match_score(name, next_category_target)
        if category_score > 0:
            score += 50 + category_score
            if role in {"link", "menuitem"}:
                score += 15

    keyword_hits = sum(1 for keyword in goal_keywords if keyword in name)
    score += keyword_hits * 25

    if _is_shopping_order_goal_keywords(goal_keywords):
        if re.search(r"\b0*\d{3,}\b", name):
            score += 35
        if "$" in name:
            score += 25
            if "names" in goal_keywords:
                score -= 14
            if "billing" in goal_keywords or "address" in goal_keywords or "date" in goal_keywords:
                score -= 24
        if any(term in name for term in ("canceled", "cancelled", "pending", "complete", "closed")):
            score += 26
        if any(
            term in name
            for term in (
                "my orders",
                "recent orders",
                "recently ordered",
                "order total",
                "order #",
                "status",
                "date",
                "ship to",
                "shipping method",
            )
        ):
            score += 18
        if name == "view order":
            score -= 8
        if "date" in goal_keywords:
            if "order date" in name:
                score += 60
            elif re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", name):
                score += 48
            elif re.search(
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
                name,
            ):
                score += 45
        if "names" in goal_keywords or "name" in goal_keywords:
            if "items ordered" in name or "product name" in name:
                score += 55
            if role in {"link", "gridcell"} and len(name) >= 30 and "$" not in name and "ordered:" not in name:
                score += 42
        if "shipping" in goal_keywords and "method" in goal_keywords:
            if "shipping method" in name:
                score += 65
        if "billing" in goal_keywords or "address" in goal_keywords:
            if "billing address" in name:
                score += 120
            elif "shipping address" in name:
                score += 95
            elif "address book" in name:
                score += 32
            elif any(term in name for term in ("san mateo", "california", "united states")):
                score += 90
            elif re.search(r"\b\d{3,5}\b", name) and ("," in name or "dr" in name or "ave" in name or "st" in name):
                score += 88
            if "ordered:" in name:
                score -= 18

    if any(term in name for term in ("sort by", "set ascending direction", "set descending direction")):
        score += 18
    if "search" in name:
        score += 12
    if role in {"combobox", "searchbox", "textbox", "option"}:
        score += 10
    if role in {"button", "heading", "link", "menuitem"}:
        score += 4
    if item.get("focused"):
        score += 8
    if item.get("clickable"):
        score += 3

    return score


def _allow_bboxless_shopping_order_text(name: str, role: str, goal_keywords: set[str]) -> bool:
    lowered_name = (name or "").lower()
    lowered_role = (role or "").lower()
    if not lowered_name or not _is_shopping_order_goal_keywords(goal_keywords):
        return False
    if lowered_role not in {"statictext", "text", "labeltext", "generic"}:
        return False
    if "date" in goal_keywords and ("order date" in lowered_name or re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", lowered_name)):
        return True
    if "names" in goal_keywords and ("items ordered" in lowered_name or "product name" in lowered_name or len(lowered_name) >= 30):
        return True
    if "shipping" in goal_keywords and "method" in goal_keywords and "shipping method" in lowered_name:
        return True
    if "billing" in goal_keywords or "address" in goal_keywords:
        if any(term in lowered_name for term in ("billing address", "shipping address", "san mateo", "california", "united states")):
            return True
        if re.search(r"\b\d{3,5}\b", lowered_name) and any(token in lowered_name for token in ("dr", "ave", "street", "st", ",")):
            return True
    return False


def _extract_shopping_order_extra_summary_lines(
    obs: Mapping[str, Any],
    *,
    goal_keywords: set[str],
    seen_labels: set[str],
    max_items: int,
) -> list[str]:
    if max_items <= 0 or not _is_shopping_order_goal_keywords(goal_keywords):
        return []
    axtree = obs.get("axtree_object")
    if not isinstance(axtree, Mapping):
        return []
    nodes = axtree.get("nodes")
    if not isinstance(nodes, list):
        return []

    candidates: list[tuple[int, str]] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        name = _normalize_inline_text(_extract_nested_value(node.get("name")) or "")
        role = _extract_nested_value(node.get("role")) or "unknown"
        if not name or name in seen_labels:
            continue
        browsergym_id = _as_optional_text(node.get("browsergym_id"))
        if browsergym_id:
            continue
        if not _allow_bboxless_shopping_order_text(name, role, goal_keywords):
            continue
        score = _shopping_item_priority_score(
            {
                "name": name,
                "role": role,
                "clickable": False,
                "focused": False,
            },
            goal_keywords,
            "",
        )
        candidates.append((score, name))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    output: list[str] = []
    seen_output: set[str] = set()
    for _, name in candidates:
        if name in seen_output:
            continue
        output.append(name)
        seen_output.add(name)
        if len(output) >= max_items:
            break
    return output


def _shopping_next_category_target(
    category_labels: list[str],
    current_url: str,
    items: list[dict[str, Any]],
) -> str:
    if not category_labels:
        return ""
    clickable_names = [
        _shopping_normalize_label(_as_text(item.get("name")))
        for item in items
        if _as_text(item.get("role")).lower() in {"link", "menuitem"}
    ]
    parsed = urlparse(current_url or "")
    visible_targets = [
        _shopping_normalize_label(label)
        for label in category_labels
        if _shopping_normalize_label(label)
        and any(
            _shopping_category_match_score(name, _shopping_normalize_label(label)) > 0
            for name in clickable_names
        )
    ]
    if "cat=" in (parsed.query or ""):
        top_level = _shopping_normalize_label(category_labels[0]) if category_labels else ""
        for target in reversed(visible_targets):
            if target != top_level:
                return target
        return ""
    for target in reversed(visible_targets):
        return target
    normalized_path = _shopping_normalize_label(urlparse(current_url or "").path.replace("/", " "))
    for label in category_labels:
        normalized_label = _shopping_normalize_label(label)
        if normalized_label and normalized_label not in normalized_path:
            return normalized_label
    return ""


def _shopping_category_match_score(name: str, target: str) -> int:
    normalized_name = _shopping_normalize_label(name)
    if not normalized_name or not target:
        return 0
    target_word_count = len(target.split())
    name_word_count = len(normalized_name.split())
    if normalized_name == target:
        return 20
    if normalized_name.startswith(f"{target} "):
        return 12
    if target in normalized_name and name_word_count <= target_word_count + 2:
        return 6
    return 0


def _shopping_normalize_label(value: str) -> str:
    lowered = value.lower().replace("-", " ")
    lowered = re.sub(r"[^a-z0-9&' ]+", " ", lowered)
    return " ".join(lowered.split())


def _is_shopping_order_goal_keywords(goal_keywords: set[str]) -> bool:
    trigger_keywords = {
        "order",
        "orders",
        "cancelled",
        "canceled",
        "shipping",
        "refund",
        "purchase",
        "purchases",
        "spent",
    }
    return any(keyword in goal_keywords for keyword in trigger_keywords)


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


def _resolve_browser_timeout_ms() -> int:
    raw_value = os.getenv("BROWSERGYM_TIMEOUT_MS")
    if not raw_value:
        return DEFAULT_BROWSER_TIMEOUT_MS
    try:
        return int(raw_value)
    except ValueError:
        return DEFAULT_BROWSER_TIMEOUT_MS


def _patch_webarena_shopping_login() -> None:
    import browsergym.webarena.instance as webarena_instance

    current = webarena_instance.WebArenaInstance.ui_login
    if getattr(current, "_patched_for_shopping_domcontentloaded", False):
        return

    def patched_ui_login(self, site: str, page):
        """
        The shopping login page often never reaches Playwright's full `load` event
        on the AMI, even though the page is interactive. Using
        `domcontentloaded` here keeps BrowserGym task reset/login reliable without
        changing the rest of the environment behavior.
        """

        url = self.urls[site]
        page = page.context.new_page()

        match site:
            case "reddit":
                username = self.credentials[site]["username"]
                password = self.credentials[site]["password"]

                page.goto(f"{url}")
                page.get_by_role("link", name="Log in").click()
                page.get_by_label("Username").fill(username)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Log in").click()

            case "gitlab":
                username = self.credentials[site]["username"]
                password = self.credentials[site]["password"]

                page.goto(f"{url}/users/sign_in")
                page.get_by_label("Username or email").fill(username)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Sign in").click()

            case "shopping":
                username = self.credentials[site]["username"]
                password = self.credentials[site]["password"]
                base_url = url.rstrip("/")

                page.goto(f"{base_url}/customer/account/login/", wait_until="domcontentloaded")
                page.locator("form#login-form").evaluate(
                    "(form, action) => form.setAttribute('action', action)",
                    f"{base_url}/customer/account/loginPost/",
                )
                page.get_by_label("Email", exact=True).fill(username)
                page.get_by_label("Password", exact=True).fill(password)
                page.get_by_role("button", name="Sign In").click(no_wait_after=True)
                try:
                    page.wait_for_timeout(1500)
                    page.goto(f"{base_url}/customer/account/", wait_until="domcontentloaded", timeout=15000)
                except Exception:
                    page.wait_for_timeout(3000)

            case "shopping_admin":
                username = self.credentials[site]["username"]
                password = self.credentials[site]["password"]

                page.goto(url)
                page.get_by_label("Username").fill(username)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Sign in").click()

            case "wikipedia":
                page.goto(url)

            case "map":
                page.goto(url)

            case _:
                raise ValueError

        page.close()

    patched_ui_login._patched_for_shopping_domcontentloaded = True  # type: ignore[attr-defined]
    webarena_instance.WebArenaInstance.ui_login = patched_ui_login


def _patch_playwright_shopping_navigation() -> None:
    from playwright.sync_api import _generated as sync_generated

    current = sync_generated.Page.goto
    if getattr(current, "_patched_for_shopping_domcontentloaded", False):
        return

    original_page_goto = current

    def patched_page_goto(self, url: str, *args, **kwargs):
        normalized_kwargs = _normalize_shopping_navigation_kwargs(url, kwargs)
        return original_page_goto(self, url, *args, **normalized_kwargs)

    patched_page_goto._patched_for_shopping_domcontentloaded = True  # type: ignore[attr-defined]
    sync_generated.Page.goto = patched_page_goto


def _patch_webarena_openai_judges() -> None:
    import webarena.evaluation_harness.evaluators as evaluators
    import webarena.evaluation_harness.helper_functions as helper_functions

    current = helper_functions.llm_fuzzy_match
    if getattr(current, "_patched_for_configurable_openai_judge", False):
        return

    def patched_llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
        messages = _build_webarena_fuzzy_judge_messages(pred, reference, question)
        response = _generate_openai_judge_response(helper_functions, messages).lower()
        if "partially correct" in response or "incorrect" in response:
            return 0.0
        assert "correct" in response
        return 1.0

    def patched_llm_ua_match(pred: str, reference: str, question: str) -> float:
        messages = _build_webarena_ua_judge_messages(pred, reference, question)
        response = _generate_openai_judge_response(helper_functions, messages).lower()
        if "different" in response:
            return 0.0
        assert "same" in response
        return 1.0

    patched_llm_fuzzy_match._patched_for_configurable_openai_judge = True  # type: ignore[attr-defined]
    patched_llm_ua_match._patched_for_configurable_openai_judge = True  # type: ignore[attr-defined]
    helper_functions.llm_fuzzy_match = patched_llm_fuzzy_match
    helper_functions.llm_ua_match = patched_llm_ua_match
    evaluators.llm_fuzzy_match = patched_llm_fuzzy_match
    evaluators.llm_ua_match = patched_llm_ua_match


def _generate_openai_judge_response(
    helper_functions: Any,
    messages: list[dict[str, str]],
) -> str:
    requested_model = get_openai_judge_model()
    try:
        return _call_openai_judge_model(requested_model, messages)
    except openai.NotFoundError:
        if requested_model == "gpt-5-mini":
            raise
        return _call_openai_judge_model("gpt-5-mini", messages)


def _call_openai_judge_model(model: str, messages: list[dict[str, str]]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable must be set when using OpenAI judge tasks.")
    client = openai.OpenAI(
        api_key=api_key,
        organization=os.environ.get("OPENAI_ORGANIZATION", ""),
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=768,
    )
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"OpenAI judge model returned empty content for model {model}.")
    return content


def _patch_webarena_shopping_task_urls() -> None:
    import browsergym.webarena.task as webarena_task

    current = webarena_task.GenericWebArenaTask.__init__
    if getattr(current, "_patched_for_shopping_url_normalization", False):
        return

    original_init = current

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.task_configs = [_normalize_shopping_task_config(task_config) for task_config in self.task_configs]

    patched_init._patched_for_shopping_url_normalization = True  # type: ignore[attr-defined]
    webarena_task.GenericWebArenaTask.__init__ = patched_init


def _normalize_shopping_navigation_kwargs(url: str, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(kwargs)
    if _is_shopping_domain(url) and updated.get("wait_until", "load") in {"load", "networkidle"}:
        updated["wait_until"] = "domcontentloaded"
    return updated


def _normalize_shopping_task_config(task_config: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(task_config)
    sites = normalized.get("sites", [])
    if "shopping" not in sites:
        return normalized

    start_url = normalized.get("start_url")
    if isinstance(start_url, str) and start_url:
        normalized["start_url"] = " |AND| ".join(
            _normalize_shopping_task_url(part.strip())
            for part in start_url.split(" |AND| ")
        )

    eval_config = normalized.get("eval")
    if isinstance(eval_config, Mapping):
        normalized_eval = dict(eval_config)
        reference_url = normalized_eval.get("reference_url")
        if isinstance(reference_url, str) and reference_url:
            normalized_eval["reference_url"] = " |OR| ".join(
                _normalize_shopping_task_url(part.strip())
                for part in reference_url.split(" |OR| ")
            )
        normalized["eval"] = normalized_eval

    return normalized


def _normalize_shopping_task_url(url: str) -> str:
    parsed = urlparse(url)
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if not path.startswith("/"):
        path = "/" + path

    query_pairs: list[tuple[str, str]] = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        cleaned_value = value.strip() if key == "q" else value
        query_pairs.append((key, cleaned_value))

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            urlencode(query_pairs),
            parsed.fragment,
        )
    )


def _is_shopping_domain(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc.endswith(":7770") or parsed.netloc.endswith(":7780")


def _build_webarena_fuzzy_judge_messages(
    pred: str,
    reference: str,
    question: str,
) -> list[dict[str, str]]:
    message = (
        "Help a teacher to grade the answer of a student given a question. "
        "Keep in mind that the student may use different phrasing or wording to answer the question. "
        "The goal is to evaluate whether the answer is semantically equivalent to the reference answer.\n"
    )
    message += f"question: {question}\n"
    message += f"reference answer: {reference}\n"
    message += "all the string 'N/A' that you see is a special sequence that means 'not achievable'\n"
    message += f"student answer: {pred}\n"
    message += "Conclude the judgement by correct/incorrect/partially correct."
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]


def _build_webarena_ua_judge_messages(
    pred: str,
    reference: str,
    question: str,
) -> list[dict[str, str]]:
    message = ""
    message += f"task: {question}\n"
    message += f"actual unachievable reason: {reference}\n"
    message += f"reported unachievable reason: {pred}\n"
    message += (
        "The task described above is inherently unachievable due to the reason specified under "
        "'actual unachievable reason'. An individual previously attempted this task and was unable "
        "to complete it. They provided a reason for their failure, which is listed under "
        "'reported unachievable reason'. Your role is to review both the actual and reported reasons. "
        "Determine if the reported reason aligns with the actual reason, even if implicitly. "
        "If the stated reason is in line with the actual reason, respond with 'same'. "
        "Otherwise, respond with 'different'."
    )
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": message},
    ]


def _as_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None
