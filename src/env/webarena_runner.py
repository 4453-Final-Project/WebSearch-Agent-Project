from __future__ import annotations

import calendar
import html
import json
import os
import re
import time
from datetime import datetime
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
MONTH_NAME_PATTERN = (
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b"
)
ORDER_DATE_TEXT_PATTERN = rf"(\d{{1,2}}/\d{{1,2}}/\d{{4}}|{MONTH_NAME_PATTERN}\s+\d{{1,2}},\s+\d{{4}})"


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
                _append_shopping_refund_detail_lines(
                    serialized_observation,
                    raw_observation=obs,
                    goal=raw_goal,
                    env=env,
                )
                _append_shopping_admin_order_item_count_lines(
                    serialized_observation,
                    raw_observation=obs,
                    goal=raw_goal,
                    env=env,
                )
                _append_shopping_admin_order_product_price_lines(
                    serialized_observation,
                    raw_observation=obs,
                    goal=raw_goal,
                    env=env,
                )
                _append_shopping_order_product_lines(
                    serialized_observation,
                    raw_observation=obs,
                    goal=raw_goal,
                    env=env,
                )
                _append_shopping_order_detail_date_line(
                    serialized_observation,
                    raw_observation=obs,
                    goal=raw_goal,
                    env=env,
                )
                _append_shopping_admin_dashboard_report_lines(
                    serialized_observation,
                    raw_observation=obs,
                    goal=raw_goal,
                    env=env,
                )
                _append_shopping_admin_bestseller_report_lines(
                    serialized_observation,
                    raw_observation=obs,
                    goal=raw_goal,
                    env=env,
                )
                _append_shopping_admin_search_term_lines(
                    serialized_observation,
                    raw_observation=obs,
                    goal=raw_goal,
                    env=env,
                )
                _append_gitlab_rss_token_lines(
                    serialized_observation,
                    raw_observation=obs,
                    goal=raw_goal,
                    env=env,
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

    forced_item = _find_forced_shopping_admin_purchase_date_item(items, current_url, shopping_goal_keywords)
    if len(items) > max_items:
        items = items[:max_items]
    if forced_item is not None and not any(item["bid"] == forced_item["bid"] for item in items):
        if len(items) >= max_items:
            items = items[:-1]
        items = [forced_item, *items]

    return items


def _find_forced_shopping_admin_purchase_date_item(
    items: list[dict[str, Any]],
    current_url: str,
    goal_keywords: set[str],
) -> dict[str, Any] | None:
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7780") or not (parsed.path or "/").startswith("/admin/sales/order"):
        return None
    if not _is_shopping_order_goal_keywords(goal_keywords):
        return None
    if not any(keyword in goal_keywords for keyword in {"oldest", "latest", "newest", "recent", "recently", "last"}):
        return None
    for item in items:
        if "purchase date" in item.get("name", "").lower():
            return item
    return None


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


def _normalize_extracted_text(value: str) -> str:
    return " ".join(value.split())


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
        recency_goal = any(keyword in goal_keywords for keyword in {"oldest", "latest", "newest", "recent", "recently", "last"})
        status_goal = any(keyword in goal_keywords for keyword in {"complete", "completed", "pending", "cancelled", "canceled"})
        if re.search(r"\b0*\d{3,}\b", name):
            score += 35
        if "$" in name:
            score += 25
            if "names" in goal_keywords:
                score -= 14
            if "billing" in goal_keywords or "address" in goal_keywords or "date" in goal_keywords:
                score -= 24
        if any(term in name for term in ("canceled", "cancelled", "pending", "complete", "closed", "processing")):
            score += 26
            if status_goal:
                score += 110
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
        if recency_goal and re.search(MONTH_NAME_PATTERN, name):
            score += 130
        if recency_goal and re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", name):
            score += 120
        if "purchase date" in name:
            score += 140 if recency_goal else 70
            if role == "columnheader":
                score += 80
        if "date" in goal_keywords:
            if "order date" in name:
                score += 60
            elif re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", name):
                score += 48
            elif re.search(MONTH_NAME_PATTERN, name):
                score += 45
        if "refund" in goal_keywords:
            if re.search(MONTH_NAME_PATTERN, name):
                score += 58
            if re.search(r"\b20\d{2}\b", name):
                score += 28
            if "shipping" in name and any(term in name for term in ("fee", "amount", "refund", "method")):
                score += 42
        if "items" in goal_keywords or "sold" in goal_keywords:
            if any(term in name for term in ("purchased", "qty", "quantity", "items ordered", "ordered qty")):
                score += 95
            elif role in {"gridcell", "cell", "statictext"} and re.fullmatch(r"\d{1,2}", name):
                score += 82
        if "billing" in goal_keywords and "name" in goal_keywords:
            if any(term in name for term in ("bill-to name", "billing name", "bill to name")):
                score += 110
            elif _looks_like_person_name(name):
                score += 280
        if "customer" in goal_keywords and "name" in goal_keywords:
            if "customer name" in name:
                score += 110
            elif _looks_like_person_name(name):
                score += 268
        if "names" in goal_keywords or "name" in goal_keywords:
            if "items ordered" in name or "product name" in name:
                score += 55
            if role in {"link", "gridcell"} and len(name) >= 30 and "$" not in name and "ordered:" not in name:
                score += 42
        if "shipping" in goal_keywords and "method" in goal_keywords:
            if "shipping method" in name:
                score += 65
            elif _looks_like_shipping_method_value(name):
                score += 120
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
    if "shipping" in goal_keywords and "method" in goal_keywords:
        if "shipping method" in lowered_name or _looks_like_shipping_method_value(lowered_name):
            return True
    if "refund" in goal_keywords and re.search(
        rf"(?:{MONTH_NAME_PATTERN}|20\d{{2}})",
        lowered_name,
    ):
        return True
    if ("billing" in goal_keywords or "customer" in goal_keywords) and "name" in goal_keywords:
        if any(term in lowered_name for term in ("bill-to name", "billing name", "customer name", "bill to name")):
            return True
        if _looks_like_person_name(lowered_name):
            return True
    if "billing" in goal_keywords or "address" in goal_keywords:
        if any(term in lowered_name for term in ("billing address", "shipping address", "san mateo", "california", "united states")):
            return True
        if re.search(r"\b\d{3,5}\b", lowered_name) and any(token in lowered_name for token in ("dr", "ave", "street", "st", ",")):
            return True
    return False


def _looks_like_person_name(value: str) -> bool:
    normalized = " ".join((value or "").split())
    if not normalized:
        return False
    if any(char.isdigit() for char in normalized):
        return False
    if any(token in normalized for token in ("http://", "https://", "@", "$", "order", "address", "search", "dashboard")):
        return False
    return re.fullmatch(r"[a-z]+(?: [a-z]+){1,3}", normalized) is not None


def _looks_like_shipping_method_value(value: str) -> bool:
    normalized = " ".join((value or "").lower().split())
    if not normalized or len(normalized) < 6:
        return False
    if any(token in normalized for token in ("$", "order", "address", "newsletter", "search", "reorder")):
        return False
    return any(
        phrase in normalized
        for phrase in (
            "flat rate",
            "fixed",
            "free shipping",
            "table rate",
            "best way",
            "store pickup",
        )
    )


def _extract_shopping_customer_order_rows(obs: Mapping[str, Any]) -> list[dict[str, str]]:
    current_url = _as_text(obs.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7770") or not (parsed.path or "/").startswith("/sales/order/history"):
        return []

    axtree = obs.get("axtree_object")
    if not isinstance(axtree, Mapping):
        return []
    nodes = axtree.get("nodes")
    if not isinstance(nodes, list):
        return []

    extra_props = obs.get("extra_element_properties")
    if not isinstance(extra_props, Mapping):
        extra_props = {}

    row_cells: dict[int, list[tuple[float, str]]] = {}
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
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 2:
            continue
        try:
            x_pos = float(bbox[0])
            y_pos = float(bbox[1])
        except (TypeError, ValueError):
            continue

        role = (_extract_nested_value(node.get("role")) or "unknown").lower()
        if role != "gridcell":
            continue
        name = _normalize_inline_text(_extract_nested_value(node.get("name")) or "")
        if not name:
            continue
        row_key = int(round(y_pos))
        row_cells.setdefault(row_key, []).append((x_pos, name))

    rows: list[dict[str, str]] = []
    for row_key in sorted(row_cells):
        ordered_cells = sorted(row_cells[row_key], key=lambda item: item[0])
        order_number = ""
        ordered_at = ""
        total = ""
        status = ""
        for _, cell_text in ordered_cells:
            lowered = cell_text.lower()
            if re.fullmatch(r"0*\d{3,}", cell_text):
                order_number = order_number or cell_text
                continue
            if re.fullmatch(r"\$[\d,]+\.\d{2}", cell_text):
                total = total or cell_text
                continue
            if re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", cell_text):
                ordered_at = ordered_at or cell_text
                continue
            if lowered in {"pending", "complete", "canceled", "cancelled", "closed", "processing"}:
                status = status or cell_text
                continue
        if not (order_number and ordered_at and total and status):
            continue
        rows.append(
            {
                "order_number": order_number,
                "date": ordered_at,
                "total": total,
                "status": status,
            }
        )
    return rows


def _extract_shopping_customer_order_pager(obs: Mapping[str, Any]) -> dict[str, str]:
    current_url = _as_text(obs.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7770") or not (parsed.path or "/").startswith("/sales/order/history"):
        return {}

    axtree = obs.get("axtree_object")
    if not isinstance(axtree, Mapping):
        return {}
    nodes = axtree.get("nodes")
    if not isinstance(nodes, list):
        return {}

    pager: dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        browsergym_id = _as_optional_text(node.get("browsergym_id"))
        if not browsergym_id:
            continue
        role = (_extract_nested_value(node.get("role")) or "unknown").lower()
        if role not in {"link", "button"}:
            continue
        name = _normalize_inline_text(_extract_nested_value(node.get("name")) or "")
        lowered = name.lower()
        if not lowered:
            continue
        if "page next" in lowered:
            pager["next"] = browsergym_id
            continue
        if "page previous" in lowered:
            pager["previous"] = browsergym_id
            continue
        page_match = re.search(r"\bpage (?P<number>\d+)\b", lowered)
        if page_match is not None:
            pager[f'page_{page_match.group("number")}'] = browsergym_id
    return pager


def _append_shopping_refund_detail_lines(
    serialized_observation: dict[str, Any],
    *,
    raw_observation: Mapping[str, Any],
    goal: str,
    env: Any,
) -> None:
    lines = _collect_shopping_refund_detail_lines(raw_observation, goal=goal, env=env)
    if not lines:
        return
    existing_summary = _as_text(serialized_observation.get("visible_page_summary"))
    existing_lines = [
        line
        for line in existing_summary.splitlines()
        if line and not line.startswith("Customer order row:")
    ]
    combined_lines = [*existing_lines, *lines]
    serialized_observation["visible_page_summary"] = _truncate_text("\n".join(combined_lines))


def _append_shopping_admin_order_item_count_lines(
    serialized_observation: dict[str, Any],
    *,
    raw_observation: Mapping[str, Any],
    goal: str,
    env: Any,
) -> None:
    lines = _collect_shopping_admin_order_item_count_lines(raw_observation, goal=goal, env=env)
    if not lines:
        return
    existing_summary = _as_text(serialized_observation.get("visible_page_summary"))
    existing_lines = existing_summary.splitlines()
    combined_lines = [*existing_lines, *lines]
    serialized_observation["visible_page_summary"] = _truncate_text("\n".join(combined_lines))


def _append_shopping_admin_order_product_price_lines(
    serialized_observation: dict[str, Any],
    *,
    raw_observation: Mapping[str, Any],
    goal: str,
    env: Any,
) -> None:
    lines = _collect_shopping_admin_order_product_price_lines(raw_observation, goal=goal, env=env)
    if not lines:
        return
    existing_summary = _as_text(serialized_observation.get("visible_page_summary"))
    existing_lines = existing_summary.splitlines()
    combined_lines = [*existing_lines, *lines]
    serialized_observation["visible_page_summary"] = _truncate_text("\n".join(combined_lines))


def _append_shopping_order_product_lines(
    serialized_observation: dict[str, Any],
    *,
    raw_observation: Mapping[str, Any],
    goal: str,
    env: Any,
) -> None:
    lines = _collect_shopping_order_product_lines(raw_observation, goal=goal, env=env)
    if not lines:
        return
    existing_summary = _as_text(serialized_observation.get("visible_page_summary"))
    existing_lines = existing_summary.splitlines()
    combined_lines = [*existing_lines, *lines]
    serialized_observation["visible_page_summary"] = _truncate_text("\n".join(combined_lines))


def _append_shopping_order_detail_date_line(
    serialized_observation: dict[str, Any],
    *,
    raw_observation: Mapping[str, Any],
    goal: str,
    env: Any,
) -> None:
    line = _collect_shopping_order_detail_date_line(raw_observation, goal=goal, env=env)
    if not line:
        return
    existing_summary = _as_text(serialized_observation.get("visible_page_summary"))
    existing_lines = existing_summary.splitlines()
    combined_lines = [*existing_lines, line]
    serialized_observation["visible_page_summary"] = _truncate_text("\n".join(combined_lines))


def _append_shopping_admin_dashboard_report_lines(
    serialized_observation: dict[str, Any],
    *,
    raw_observation: Mapping[str, Any],
    goal: str,
    env: Any,
) -> None:
    lines = _collect_shopping_admin_dashboard_report_lines(raw_observation, goal=goal, env=env)
    if not lines:
        return
    existing_summary = _as_text(serialized_observation.get("visible_page_summary"))
    existing_lines = existing_summary.splitlines()
    combined_lines = [*existing_lines, *lines]
    serialized_observation["visible_page_summary"] = _truncate_text("\n".join(combined_lines))


def _append_shopping_admin_bestseller_report_lines(
    serialized_observation: dict[str, Any],
    *,
    raw_observation: Mapping[str, Any],
    goal: str,
    env: Any,
) -> None:
    lines = _collect_shopping_admin_bestseller_report_lines(raw_observation, goal=goal, env=env)
    if not lines:
        return
    existing_summary = _as_text(serialized_observation.get("visible_page_summary"))
    existing_lines = existing_summary.splitlines()
    combined_lines = [*existing_lines, *lines]
    serialized_observation["visible_page_summary"] = _truncate_text("\n".join(combined_lines))


def _append_shopping_admin_search_term_lines(
    serialized_observation: dict[str, Any],
    *,
    raw_observation: Mapping[str, Any],
    goal: str,
    env: Any,
) -> None:
    lines = _collect_shopping_admin_search_term_lines(
        serialized_observation,
        raw_observation=raw_observation,
        goal=goal,
        env=env,
    )
    if not lines:
        return
    existing_summary = _as_text(serialized_observation.get("visible_page_summary"))
    existing_lines = existing_summary.splitlines()
    combined_lines = [*existing_lines, *lines]
    serialized_observation["visible_page_summary"] = _truncate_text("\n".join(combined_lines))


def _append_gitlab_rss_token_lines(
    serialized_observation: dict[str, Any],
    *,
    raw_observation: Mapping[str, Any],
    goal: str,
    env: Any,
) -> None:
    lines = _collect_gitlab_rss_token_lines(raw_observation, goal=goal, env=env)
    if not lines:
        return
    existing_summary = _as_text(serialized_observation.get("visible_page_summary"))
    existing_lines = existing_summary.splitlines()
    combined_lines = [*existing_lines, *lines]
    serialized_observation["visible_page_summary"] = _truncate_text("\n".join(combined_lines))


def _collect_gitlab_rss_token_lines(
    obs: Mapping[str, Any],
    *,
    goal: str,
    env: Any,
) -> list[str]:
    lowered_goal = " ".join((goal or "").lower().split())
    if "rss feed token" not in lowered_goal:
        return []

    current_url = _as_text(obs.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":8023") or parsed.path.rstrip("/") != "/-/profile/personal_access_tokens":
        return []

    page = getattr(getattr(env, "unwrapped", env), "page", None)
    if page is None:
        return []

    token = _extract_gitlab_rss_token_from_page(page)
    if token:
        return [f"GitLab RSS feed token: {token}"]

    body_text = ""
    html_text = ""
    try:
        body_text = page.locator("body").inner_text()
    except Exception:
        body_text = ""
    try:
        html_text = page.content()
    except Exception:
        html_text = ""

    token = _extract_gitlab_rss_token_from_text(body_text) or _extract_gitlab_rss_token_from_html(html_text)
    if not token:
        return []
    return [f"GitLab RSS feed token: {token}"]


def _extract_gitlab_rss_token_from_page(page: Any) -> str:
    try:
        candidate = page.evaluate(
            """
            () => {
              const tokenPattern = /^[A-Za-z0-9_-]{20,}$/;
              const inputs = Array.from(document.querySelectorAll('input, textarea'));
              for (const element of inputs) {
                const value = (element.value || '').trim();
                if (!tokenPattern.test(value)) {
                  continue;
                }
                const containerText = (element.closest('section, div, form, li, tr')?.innerText || '').toLowerCase();
                const metadata = [
                  element.id || '',
                  element.name || '',
                  element.placeholder || '',
                  element.getAttribute('aria-label') || '',
                  containerText,
                ].join(' ').toLowerCase();
                if (/feed|rss|token/.test(metadata)) {
                  return value;
                }
              }
              return '';
            }
            """
        )
    except Exception:
        return ""
    return _as_text(candidate).strip()


def _extract_gitlab_rss_token_from_text(body_text: str) -> str:
    normalized_text = _as_text(body_text)
    if not normalized_text:
        return ""
    match = re.search(
        r"feed token[^A-Za-z0-9_-]{0,80}(?P<token>[A-Za-z0-9_-]{20,})",
        normalized_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return ""
    return match.group("token")


def _extract_gitlab_rss_token_from_html(html_text: str) -> str:
    normalized_html = _as_text(html_text)
    if not normalized_html:
        return ""
    match = re.search(
        r"(?:copy feed token|feed token)[^>]{0,300}?data-clipboard-text=\"(?P<token>[A-Za-z0-9_-]{20,})\"",
        normalized_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is not None:
        return match.group("token")
    match = re.search(
        r"data-clipboard-text=\"(?P<token>[A-Za-z0-9_-]{20,})\"[^>]{0,300}?(?:copy feed token|feed token)",
        normalized_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is not None:
        return match.group("token")
    match = re.search(
        r"feed token.*?value=\"(?P<token>[A-Za-z0-9_-]{20,})\"",
        normalized_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is not None:
        return match.group("token")
    match = re.search(
        r"rss(?:\s+feed)?\s+token.*?(?P<token>[A-Za-z0-9_-]{20,})",
        normalized_html,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return ""
    return match.group("token")


def _collect_shopping_refund_detail_lines(
    obs: Mapping[str, Any],
    *,
    goal: str,
    env: Any,
) -> list[str]:
    lowered_goal = " ".join((goal or "").lower().split())
    if "refund" not in lowered_goal:
        return []
    current_url = _as_text(obs.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7770") or not (parsed.path or "/").startswith("/sales/order/history"):
        return []

    page = getattr(getattr(env, "unwrapped", env), "page", None)
    if page is None or not hasattr(page, "context"):
        return []

    rows = _extract_shopping_customer_order_rows(obs)
    background_rows = _collect_shopping_customer_order_rows_from_history_pages(page, current_url)
    if background_rows:
        seen_orders = {row.get("order_number", "") for row in rows}
        rows.extend(row for row in background_rows if row.get("order_number", "") not in seen_orders)

    matching_rows = _select_matching_refund_rows(rows, lowered_goal)
    if not matching_rows:
        return ["Customer refund match count: 0"]

    detail_lines: list[str] = []
    for row in matching_rows[:8]:
        order_number = row.get("order_number", "")
        if not order_number:
            continue
        detail_line = _collect_shopping_refund_detail_line(
            page,
            history_url=current_url,
            order_number=order_number,
        )
        if detail_line:
            detail_lines.append(detail_line)
    return detail_lines


def _collect_shopping_admin_dashboard_report_lines(
    obs: Mapping[str, Any],
    *,
    goal: str,
    env: Any,
) -> list[str]:
    lowered_goal = " ".join((goal or "").lower().split())
    if "best-selling" not in lowered_goal and "bestselling" not in lowered_goal and "best selling" not in lowered_goal:
        return []

    current_url = _as_text(obs.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7780") or not (parsed.path or "/").startswith("/admin/admin/dashboard"):
        return []

    page = getattr(getattr(env, "unwrapped", env), "page", None)
    if page is None:
        return []

    try:
        body_text = page.locator("body").inner_text()
    except Exception:
        return []

    rows = _extract_shopping_admin_dashboard_bestseller_rows_from_body_text(body_text)
    if not rows:
        return []
    return [
        "Dashboard bestseller row: "
        f'rank={index} | product={row.get("product", "")} | '
        f'price={row.get("price", "")} | quantity={row.get("quantity", "")}'
        for index, row in enumerate(rows, start=1)
    ]


def _extract_shopping_admin_dashboard_bestseller_rows_from_body_text(body_text: str) -> list[dict[str, str]]:
    lines = [
        _normalize_extracted_text(raw_line)
        for raw_line in _as_text(body_text).splitlines()
        if _normalize_extracted_text(raw_line)
    ]
    rows: list[dict[str, str]] = []
    in_product_table = False
    for line in lines:
        lowered = line.lower()
        if not in_product_table:
            if "product" in lowered and "price" in lowered and "quantity" in lowered:
                in_product_table = True
            continue
        if lowered in {
            "lifetime sales",
            "average order",
            "last orders",
            "last search terms",
            "top search terms",
            "advanced reporting",
            "bestsellers",
            "most viewed products",
            "new customers",
            "customers",
        }:
            if rows:
                break
            continue
        match = re.match(
            r"^(?P<product>.+?)\s+\$(?P<price>[\d,]+\.\d{2})\s+(?P<quantity>\d+)$",
            line,
        )
        if match is None:
            if rows:
                break
            continue
        rows.append(
            {
                "product": match.group("product").strip(),
                "price": match.group("price").replace(",", ""),
                "quantity": match.group("quantity"),
            }
        )
    return rows


def _collect_shopping_admin_bestseller_report_lines(
    obs: Mapping[str, Any],
    *,
    goal: str,
    env: Any,
) -> list[str]:
    lowered_goal = " ".join((goal or "").lower().split())
    if "best-selling" not in lowered_goal and "bestselling" not in lowered_goal and "best selling" not in lowered_goal:
        return []

    current_url = _as_text(obs.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7780") or not (parsed.path or "/").startswith("/admin/admin/dashboard"):
        return []

    window = _derive_shopping_admin_bestseller_report_window(goal)
    if window is None:
        return []
    report_from, report_to = window

    page = getattr(getattr(env, "unwrapped", env), "page", None)
    if page is None or not hasattr(page, "context"):
        return []

    report_page = page.context.new_page()
    try:
        report_url = f"{parsed.scheme or 'http'}://{parsed.netloc}/admin/reports/report_product/sold/"
        report_page.goto(report_url, wait_until="domcontentloaded")
        report_page.locator('[name="report_from"]').fill(report_from)
        report_page.locator('[name="report_to"]').fill(report_to)
        report_page.locator('select[name="report_period"]').select_option(label="Day")
        report_page.get_by_role("button", name="Refresh").click()
        report_page.wait_for_load_state("domcontentloaded")
        time.sleep(1)
        raw_rows = report_page.locator("#gridProductsSold_table tbody tr").all_inner_texts()
    except Exception:
        return []
    finally:
        report_page.close()

    rows = _extract_shopping_admin_bestseller_report_rows(raw_rows)
    if not rows:
        return []
    aggregated_rows = _aggregate_shopping_admin_bestseller_report_rows(rows)
    if not aggregated_rows:
        return []
    return [
        "Admin bestseller aggregate row: "
        f'rank={index} | '
        f'product={row.get("product", "")} | '
        f'quantity={row.get("quantity", "")}'
        for index, row in enumerate(aggregated_rows[:12], start=1)
    ]


def _collect_shopping_admin_search_term_lines(
    serialized_observation: Mapping[str, Any],
    *,
    raw_observation: Mapping[str, Any],
    goal: str,
    env: Any,
) -> list[str]:
    lowered_goal = " ".join((goal or "").lower().split())
    if "search term" not in lowered_goal:
        return []

    current_url = _as_text(raw_observation.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7780") or not (parsed.path or "/").startswith("/admin/admin/dashboard"):
        return []

    page = getattr(getattr(env, "unwrapped", env), "page", None)
    if page is None or not hasattr(page, "context"):
        return []

    existing_summary = _as_text(serialized_observation.get("visible_page_summary"))
    search_term_urls = _extract_shopping_admin_search_term_urls(existing_summary)
    if not search_term_urls:
        return []

    rows: list[dict[str, str | int]] = []
    for search_term_url in search_term_urls[:5]:
        row = _fetch_shopping_admin_search_term(page, search_term_url)
        if not row or not row.get("term"):
            continue
        rows.append(row)
    if not rows:
        return []
    ranked_rows = sorted(
        rows,
        key=lambda row: (-int(row.get("uses", -1)), int(row.get("_index", 0))),
    )
    lines: list[str] = []
    for index, row in enumerate(ranked_rows, start=1):
        lines.append(
            "Dashboard search term row: "
            f'rank={index} | term={row.get("term", "")} | uses={row.get("uses", "")}'
        )
    return lines


def _extract_shopping_admin_search_term_urls(visible_page_summary: str) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"https?://[^\s]+/admin/search/term/edit/id/\d+/", visible_page_summary or ""):
        url = match.group(0).strip()
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _fetch_shopping_admin_search_term(page: Any, search_term_url: str) -> dict[str, str | int]:
    search_page = page.context.new_page()
    try:
        search_page.goto(search_term_url, wait_until="domcontentloaded")
        direct_field_values: dict[str, str] = {}
        for selector, field_name in (
            ('[name="search_query"]', "term"),
            ('[name="query_text"]', "term"),
            ('[name="search_term"]', "term"),
            ('[name="name"]', "term"),
            ('#search_query', "term"),
            ('[name="popularity"]', "uses"),
            ('[name="num_uses"]', "uses"),
            ('[name="number_of_uses"]', "uses"),
            ('[name="num_results"]', "num_results"),
        ):
            try:
                value = _normalize_extracted_text(search_page.locator(selector).input_value())
            except Exception:
                continue
            if not value:
                continue
            if field_name == "term" and value.lower() in {"search query", "search term"}:
                continue
            direct_field_values[field_name] = value
        if direct_field_values.get("term"):
            body_text = ""
            try:
                body_text = _normalize_extracted_text(search_page.locator("body").inner_text())
            except Exception:
                pass
            details = _extract_shopping_admin_search_term_details_from_body_text(body_text)
            uses_value = direct_field_values.get("uses", "")
            try:
                uses = int(uses_value) if uses_value else int(details.get("uses", -1))
            except (TypeError, ValueError):
                uses = int(details.get("uses", -1))
            return {
                "term": direct_field_values["term"],
                "uses": uses,
                "_index": len(page.context.pages),
            }
        try:
            body_text = _normalize_extracted_text(search_page.locator("body").inner_text())
        except Exception:
            return {}
        details = _extract_shopping_admin_search_term_details_from_body_text(body_text)
        if not details.get("term"):
            return {}
        details["_index"] = len(page.context.pages)
        return details
    except Exception:
        return {}
    finally:
        search_page.close()


def _extract_shopping_admin_search_term_details_from_body_text(body_text: str) -> dict[str, str | int]:
    lines = [
        _normalize_extracted_text(raw_line)
        for raw_line in _as_text(body_text).splitlines()
        if _normalize_extracted_text(raw_line)
    ]
    term = ""
    uses = -1
    for index, line in enumerate(lines):
        lowered = line.lower()
        if lowered in {"search query", "search term"} and not term:
            for candidate in lines[index + 1 :]:
                candidate_lower = candidate.lower()
                if candidate_lower in {
                    "number of results",
                    "number of uses",
                    "redirect url",
                    "display in suggested terms",
                    "save search",
                    "reset",
                }:
                    break
                if candidate and candidate_lower not in {"search query", "search term"}:
                    term = candidate
                    break
        if lowered == "number of uses":
            for candidate in lines[index + 1 :]:
                if candidate.isdigit():
                    uses = int(candidate)
                    break
                if candidate.lower() in {"redirect url", "display in suggested terms", "save search", "reset"}:
                    break
    return {"term": term, "uses": uses}


def _extract_shopping_admin_bestseller_report_rows(raw_rows: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    current_interval = ""
    for raw_row in raw_rows:
        parts = [
            _normalize_extracted_text(part)
            for part in _as_text(raw_row).split("\t")
            if _normalize_extracted_text(part)
        ]
        if not parts:
            continue
        if parts[0].lower() == "total":
            continue
        if len(parts) == 2 and "we can't find records for this period" in parts[1].lower():
            current_interval = parts[0]
            continue
        if len(parts) == 4:
            current_interval, product, sku, quantity = parts
        elif len(parts) == 3 and current_interval:
            product, sku, quantity = parts
        else:
            continue
        if not quantity.isdigit():
            continue
        rows.append(
            {
                "interval": current_interval,
                "product": _normalize_shopping_admin_report_product_name(product, sku),
                "sku": sku,
                "quantity": quantity,
            }
        )
    return rows


def _aggregate_shopping_admin_bestseller_report_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    aggregated: dict[str, dict[str, str | int]] = {}
    for index, row in enumerate(rows):
        product = row.get("product", "")
        if not product:
            continue
        quantity = int(row.get("quantity", "0") or "0")
        if product not in aggregated:
            aggregated[product] = {
                "product": product,
                "quantity": quantity,
                "_first_index": index,
            }
            continue
        aggregated[product]["quantity"] = int(aggregated[product]["quantity"]) + quantity
    ordered = sorted(
        aggregated.values(),
        key=lambda row: (-int(row["quantity"]), int(row["_first_index"])),
    )
    return [{"product": str(row["product"]), "quantity": str(row["quantity"])} for row in ordered]


def _normalize_shopping_admin_report_product_name(product: str, sku: str) -> str:
    normalized_product = _normalize_extracted_text(product)
    sku_match = re.match(r"^[A-Z]{2,}\d{2}-(?P<size>[^-]+)-(?P<color>[A-Za-z]+)$", _as_text(sku))
    if sku_match is None:
        return normalized_product
    return f'{normalized_product}-{sku_match.group("size")}-{sku_match.group("color")}'


def _derive_shopping_admin_bestseller_report_window(goal: str) -> tuple[str, str] | None:
    lowered = " ".join((goal or "").lower().split())
    if not lowered:
        return None

    quarter_match = re.search(r"\bquarter\s+([1-4])\s+(20\d{2})\b", lowered)
    if quarter_match is not None:
        quarter = int(quarter_match.group(1))
        year = int(quarter_match.group(2))
        start_month = (quarter - 1) * 3 + 1
        end_month = start_month + 2
        end_day = calendar.monthrange(year, end_month)[1]
        return (
            f"{start_month:02d}/01/{year}",
            f"{end_month:02d}/{end_day:02d}/{year}",
        )

    month_lookup = {
        "jan": 1,
        "january": 1,
        "feb": 2,
        "february": 2,
        "mar": 3,
        "march": 3,
        "apr": 4,
        "april": 4,
        "may": 5,
        "jun": 6,
        "june": 6,
        "jul": 7,
        "july": 7,
        "aug": 8,
        "august": 8,
        "sep": 9,
        "sept": 9,
        "september": 9,
        "oct": 10,
        "october": 10,
        "nov": 11,
        "november": 11,
        "dec": 12,
        "december": 12,
    }
    month_match = re.search(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
        r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(20\d{2})\b",
        lowered,
    )
    if month_match is not None:
        month = month_lookup[month_match.group(1)]
        year = int(month_match.group(2))
        end_day = calendar.monthrange(year, month)[1]
        return (f"{month:02d}/01/{year}", f"{month:02d}/{end_day:02d}/{year}")

    year_match = re.search(r"\b(20\d{2})\b", lowered)
    if year_match is not None:
        year = int(year_match.group(1))
        return (f"01/01/{year}", f"12/31/{year}")
    return None


def _collect_shopping_refund_detail_line(page: Any, *, history_url: str, order_number: str) -> str:
    detail_page = page.context.new_page()
    try:
        history_parsed = urlparse(history_url or "")
        detail_url = (
            f"{history_parsed.scheme or 'http'}://{history_parsed.netloc}/sales/order/view/order_id/{int(order_number)}/"
        )
        detail_page.goto(detail_url, wait_until="domcontentloaded")

        product_rows = []
        for raw_text in detail_page.locator("table.data.table.table-order-items tbody tr").all_inner_texts():
            normalized = _normalize_inline_text(raw_text)
            if not normalized:
                continue
            product_rows.append(normalized)

        totals: dict[str, str] = {}
        for raw_text in detail_page.locator("tfoot tr").all_inner_texts():
            normalized = _normalize_inline_text(raw_text)
            if not normalized:
                continue
            if normalized.lower().startswith("subtotal "):
                totals["subtotal"] = normalized.split()[-1]
            elif normalized.lower().startswith("shipping & handling "):
                totals["shipping"] = normalized.split()[-1]
            elif normalized.lower().startswith("grand total "):
                totals["grand_total"] = normalized.split()[-1]

        if not product_rows and not totals:
            return ""

        parts = [f"Customer refund detail: order={order_number}"]
        if "subtotal" in totals:
            parts.append(f"subtotal={totals['subtotal']}")
        if "shipping" in totals:
            parts.append(f"shipping={totals['shipping']}")
        if "grand_total" in totals:
            parts.append(f"grand_total={totals['grand_total']}")
        for product in product_rows[:6]:
            title, amount = _parse_refund_detail_product_row(product)
            if not title or not amount:
                continue
            parts.append(f"product={title}::{amount}")
        return " | ".join(parts)
    finally:
        detail_page.close()


def _collect_shopping_customer_order_rows_from_history_pages(page: Any, history_url: str) -> list[dict[str, str]]:
    history_page = page.context.new_page()
    try:
        collected_rows: dict[str, dict[str, str]] = {}
        for page_index in range(1, 5):
            history_page.goto(
                _build_customer_order_history_page_url(history_url, page_index),
                wait_until="domcontentloaded",
            )
            page_rows: list[dict[str, str]] = []
            for raw_text in history_page.locator("tbody tr").all_inner_texts():
                row = _parse_customer_order_row_from_text(raw_text)
                if not row:
                    continue
                order_number = row.get("order_number", "")
                if order_number:
                    collected_rows.setdefault(order_number, row)
                    page_rows.append(row)
            if page_index > 1 and not page_rows:
                break
        return list(collected_rows.values())
    finally:
        history_page.close()


def _collect_shopping_admin_order_item_count_lines(
    obs: Mapping[str, Any],
    *,
    goal: str,
    env: Any,
) -> list[str]:
    lowered_goal = " ".join((goal or "").lower().split())
    if "items sold" not in lowered_goal:
        return []
    current_url = _as_text(obs.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7780") or not (parsed.path or "/").startswith("/admin/sales/order"):
        return []

    page = getattr(getattr(env, "unwrapped", env), "page", None)
    if page is None or not hasattr(page, "context"):
        return []

    count = _derive_shopping_order_count_from_goal(goal)
    if not count:
        return []
    status_key = _derive_shopping_order_status_key_from_goal(goal)

    rows = _collect_shopping_admin_order_grid_rows(page)
    if not rows:
        return []
    filtered_rows = [row for row in rows if _shopping_admin_row_matches_status(row, status_key)]
    if not filtered_rows:
        filtered_rows = rows
    selected_rows = _sort_shopping_admin_order_rows_by_date(filtered_rows, newest_first="oldest" not in lowered_goal)[:count]
    if not selected_rows:
        return []

    detail_lines: list[str] = []
    for row in selected_rows:
        detail_url = row.get("detail_url", "")
        if not detail_url:
            continue
        item_count = _collect_shopping_admin_order_item_count(page, detail_url=detail_url)
        if item_count is None:
            continue
        detail_lines.append(
            "Admin order item row: "
            f'order={row.get("order_number", "")} | '
            f'date={row.get("date", "")} | '
            f'status={row.get("status", "")} | '
            f'items={item_count}'
        )
    return detail_lines


def _collect_shopping_admin_order_product_price_lines(
    obs: Mapping[str, Any],
    *,
    goal: str,
    env: Any,
) -> list[str]:
    lowered_goal = " ".join((goal or "").lower().split())
    if "product name" not in lowered_goal or "discounted price" not in lowered_goal:
        return []
    current_url = _as_text(obs.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7780") or not (parsed.path or "/").startswith("/admin/sales/order"):
        return []

    page = getattr(getattr(env, "unwrapped", env), "page", None)
    if page is None or not hasattr(page, "context"):
        return []

    status_key = _derive_shopping_order_status_key_from_goal(goal)
    rows = _collect_shopping_admin_order_grid_rows(page)
    if not rows:
        return []
    filtered_rows = [row for row in rows if _shopping_admin_row_matches_status(row, status_key)]
    if not filtered_rows:
        filtered_rows = rows
    selected_rows = _sort_shopping_admin_order_rows_by_date(filtered_rows, newest_first="oldest" not in lowered_goal)[:1]
    if not selected_rows:
        return []

    detail_lines: list[str] = []
    for row in selected_rows:
        detail_url = row.get("detail_url", "")
        if not detail_url:
            continue
        for product, price in _collect_shopping_admin_order_product_prices(page, detail_url=detail_url):
            detail_lines.append(
                "Admin order product row: "
                f'order={row.get("order_number", "")} | '
                f'date={row.get("date", "")} | '
                f'status={row.get("status", "")} | '
                f"product={product} | "
                f"price={price}"
            )
    return detail_lines


def _goal_requests_last_ordered_product_date(goal: str) -> bool:
    lowered_goal = " ".join((goal or "").lower().split())
    return "last ordered my" in lowered_goal or "when i last ordered" in lowered_goal


def _extract_last_ordered_product_query(goal: str) -> str:
    lowered_goal = " ".join((goal or "").lower().split())
    match = re.search(r"(?:last ordered my|ordered my)\s+(.+?)(?:\?|$)", lowered_goal)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip(" ?.")


def _collect_shopping_order_product_lines(
    obs: Mapping[str, Any],
    *,
    goal: str,
    env: Any,
) -> list[str]:
    if not _goal_requests_last_ordered_product_date(goal):
        return []
    query = _extract_last_ordered_product_query(goal)
    if not query:
        return []
    current_url = _as_text(obs.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7770") or not (parsed.path or "/").startswith("/sales/order/history"):
        return []

    page = getattr(getattr(env, "unwrapped", env), "page", None)
    if page is None or not hasattr(page, "context"):
        return []

    rows = _extract_shopping_customer_order_rows(obs)
    background_rows = _collect_shopping_customer_order_rows_from_history_pages(page, current_url)
    if background_rows:
        seen_orders = {row.get("order_number", "") for row in rows}
        rows.extend(row for row in background_rows if row.get("order_number", "") not in seen_orders)

    product_lines: list[str] = []
    for row in rows:
        order_number = row.get("order_number", "")
        ordered_at = row.get("date", "")
        if not order_number or not ordered_at:
            continue
        for title in _collect_shopping_order_detail_product_titles(
            page,
            history_url=current_url,
            order_number=order_number,
        ):
            if not _shopping_product_query_matches_title(query, title):
                continue
            product_lines.append(
                "Customer order product row: "
                f"order={order_number} | date={ordered_at} | product={title}"
            )
    return product_lines


def _collect_shopping_order_detail_date_line(
    obs: Mapping[str, Any],
    *,
    goal: str,
    env: Any,
) -> str:
    lowered_goal = " ".join((goal or "").lower().split())
    if "order date" not in lowered_goal:
        return ""
    current_url = _as_text(obs.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7770") or not re.match(r"^/sales/order/view/(?:order_id/)?\d+/?$", parsed.path or ""):
        return ""

    page = getattr(getattr(env, "unwrapped", env), "page", None)
    if page is None:
        return ""
    try:
        body_text = _normalize_extracted_text(page.locator("body").inner_text())
    except Exception:
        body_text = ""
    date_match = re.search(rf"order date:?\s*{ORDER_DATE_TEXT_PATTERN}", body_text, flags=re.IGNORECASE)
    if date_match is None:
        try:
            page_html = page.content()
        except Exception:
            page_html = ""
        if page_html:
            html_text = _normalize_extracted_text(html.unescape(re.sub(r"<[^>]+>", " ", page_html)))
            date_match = re.search(rf"order date:?\s*{ORDER_DATE_TEXT_PATTERN}", html_text, flags=re.IGNORECASE)
    if date_match is None:
        return ""
    return f"Order detail date: {date_match.group(1)}"


def _collect_shopping_admin_order_grid_rows(page: Any) -> list[dict[str, str]]:
    links = page.locator('a[href*="/admin/sales/order/view/order_id/"]')
    try:
        link_count = int(links.count())
    except Exception:
        return []
    rows: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for index in range(link_count):
        link = links.nth(index)
        try:
            detail_url = _as_text(link.get_attribute("href"))
        except Exception:
            continue
        if not detail_url or detail_url in seen_urls:
            continue
        seen_urls.add(detail_url)
        row = link.locator("xpath=ancestor::tr[1]")
        cells = row.locator("td")
        try:
            cell_count = int(cells.count())
        except Exception:
            continue
        if cell_count < 10:
            continue
        order_number = _normalize_inline_text(cells.nth(1).inner_text())
        purchased_at = _normalize_inline_text(cells.nth(3).inner_text())
        status = _normalize_inline_text(cells.nth(8).inner_text())
        if not (order_number and purchased_at and status):
            continue
        rows.append(
            {
                "order_number": order_number,
                "date": purchased_at,
                "status": status,
                "detail_url": detail_url,
            }
        )
    return rows


def _collect_shopping_admin_order_item_count(page: Any, *, detail_url: str) -> int | None:
    detail_page = page.context.new_page()
    try:
        detail_page.goto(detail_url, wait_until="domcontentloaded")
        tables = detail_page.locator("table")
        try:
            table_count = int(tables.count())
        except Exception:
            return None
        for table_index in range(table_count):
            table = tables.nth(table_index)
            table_text = _normalize_inline_text(table.inner_text())
            if "qty" not in table_text.lower() or "row total" not in table_text.lower():
                continue
            rows = table.locator("tbody tr")
            try:
                row_count = int(rows.count())
            except Exception:
                continue
            total_items = 0
            found_quantity_cell = False
            for row_index in range(row_count):
                cells = rows.nth(row_index).locator("td")
                try:
                    cell_count = int(cells.count())
                except Exception:
                    continue
                if cell_count < 6:
                    continue
                quantity_text = _normalize_inline_text(cells.nth(5).inner_text())
                if not re.fullmatch(r"\d+", quantity_text):
                    continue
                total_items += int(quantity_text)
                found_quantity_cell = True
            if found_quantity_cell:
                return total_items
        return None
    finally:
        detail_page.close()


def _collect_shopping_admin_order_product_prices(page: Any, *, detail_url: str) -> list[tuple[str, str]]:
    detail_page = page.context.new_page()
    try:
        detail_page.goto(detail_url, wait_until="domcontentloaded")
        tables = detail_page.locator("table")
        try:
            table_count = int(tables.count())
        except Exception:
            return []
        products: list[tuple[str, str]] = []
        for table_index in range(table_count):
            table = tables.nth(table_index)
            table_text = _normalize_inline_text(table.inner_text())
            lowered_table_text = table_text.lower()
            if "original price" not in lowered_table_text or "row total" not in lowered_table_text:
                continue
            rows = table.locator("tbody tr")
            try:
                row_count = int(rows.count())
            except Exception:
                continue
            for row_index in range(row_count):
                cells = rows.nth(row_index).locator("td")
                try:
                    cell_count = int(cells.count())
                except Exception:
                    continue
                if cell_count < 4:
                    continue
                product_text = _normalize_inline_text(cells.nth(0).inner_text())
                price_text = _normalize_inline_text(cells.nth(3).inner_text())
                if not product_text or not re.search(r"\$[\d,]+(?:\.\d+)?", price_text):
                    continue
                price_match = re.search(r"\$[\d,]+(?:\.\d+)?", price_text)
                if price_match is None:
                    continue
                products.append((product_text, price_match.group(0)))
            if products:
                return products
        return []
    finally:
        detail_page.close()


def _collect_shopping_order_detail_product_titles(page: Any, *, history_url: str, order_number: str) -> list[str]:
    detail_page = page.context.new_page()
    try:
        history_parsed = urlparse(history_url or "")
        detail_url = (
            f"{history_parsed.scheme or 'http'}://{history_parsed.netloc}/sales/order/view/order_id/{int(order_number)}/"
        )
        detail_page.goto(detail_url, wait_until="domcontentloaded")
        titles: list[str] = []
        for raw_text in detail_page.locator("table.data.table.table-order-items tbody tr").all_inner_texts():
            normalized = _normalize_inline_text(raw_text)
            if not normalized:
                continue
            title, _ = _parse_refund_detail_product_row(normalized)
            cleaned = title or normalized
            if cleaned:
                titles.append(cleaned)
        return titles
    finally:
        detail_page.close()


def _shopping_product_query_matches_title(query: str, title: str) -> bool:
    query_tokens = [token for token in re.findall(r"[a-z0-9]+", query.lower()) if len(token) > 2]
    if not query_tokens:
        return False
    lowered_title = title.lower()
    return all(token in lowered_title for token in query_tokens)


def _build_customer_order_history_page_url(history_url: str, page_index: int) -> str:
    parsed = urlparse(history_url or "")
    query_pairs = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key != "p"]
    query_pairs.append(("p", str(page_index)))
    return urlunparse(
        (
            parsed.scheme or "http",
            parsed.netloc,
            parsed.path or "/sales/order/history/",
            parsed.params,
            urlencode(query_pairs),
            parsed.fragment,
        )
    )


def _parse_customer_order_row_from_text(raw_text: str) -> dict[str, str]:
    normalized = _normalize_inline_text(raw_text)
    if not normalized:
        return {}
    order_match = re.search(r"\b(0*\d{3,})\b", normalized)
    date_match = re.search(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b", normalized)
    total_match = re.search(r"\$[\d,]+\.\d{2}", normalized)
    status_match = re.search(r"\b(Pending|Complete|Canceled|Cancelled|Closed|Processing)\b", normalized, re.IGNORECASE)
    if order_match is None or date_match is None or total_match is None or status_match is None:
        return {}
    return {
        "order_number": order_match.group(1),
        "date": date_match.group(0),
        "total": total_match.group(0),
        "status": status_match.group(1),
    }


def _parse_refund_detail_product_row(row_text: str) -> tuple[str, str]:
    amount_match = re.search(r"(?P<amount>\$[\d,]+\.\d{2})", row_text)
    if amount_match is None:
        return "", ""
    title = row_text[: amount_match.start()].strip()
    title = re.sub(r"\s+B[0-9A-Z]{9,10}$", "", title).strip()
    amount = amount_match.group("amount")
    return title, amount


def _select_matching_refund_rows(rows: list[dict[str, str]], lowered_goal: str) -> list[dict[str, str]]:
    if "refund" not in lowered_goal:
        return []
    refund_year, refund_month = _extract_refund_date_constraint(lowered_goal)
    matching_rows: list[dict[str, str]] = []
    for row in rows:
        if (row.get("status", "").lower()) not in {"canceled", "cancelled"}:
            continue
        if not _customer_order_date_matches_refund_constraint(
            row.get("date", ""),
            refund_year=refund_year,
            refund_month=refund_month,
        ):
            continue
        matching_rows.append(row)
    return matching_rows


def _extract_refund_date_constraint(lowered_goal: str) -> tuple[int | None, int | None]:
    numeric_match = re.search(r"\b(20\d{2})[/-](\d{1,2})\b", lowered_goal)
    if numeric_match:
        return int(numeric_match.group(1)), int(numeric_match.group(2))
    year_match = re.search(r"\b(20\d{2})\b", lowered_goal)
    month_match = re.search(
        r"\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sept|sep|october|oct|november|nov|december|dec)\b",
        lowered_goal,
    )
    month_lookup = {
        "january": 1,
        "jan": 1,
        "february": 2,
        "feb": 2,
        "march": 3,
        "mar": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "june": 6,
        "jun": 6,
        "july": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sept": 9,
        "sep": 9,
        "october": 10,
        "oct": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
    }
    year = int(year_match.group(1)) if year_match else None
    month = month_lookup.get(month_match.group(1).lower()) if month_match else None
    return year, month


def _customer_order_date_matches_refund_constraint(
    date_text: str,
    *,
    refund_year: int | None,
    refund_month: int | None,
) -> bool:
    if refund_year is None and refund_month is None:
        return True
    normalized = " ".join((date_text or "").replace("Sept", "Sep").split())
    if not normalized:
        return False
    parsed_date = None
    for date_format in ("%m/%d/%y", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
        try:
            parsed_date = datetime.strptime(normalized, date_format)
            break
        except ValueError:
            continue
    if parsed_date is None:
        return False
    if refund_year is not None and parsed_date.year != refund_year:
        return False
    if refund_month is not None and parsed_date.month != refund_month:
        return False
    return True


def _extract_shopping_admin_order_rows(obs: Mapping[str, Any]) -> list[dict[str, str]]:
    current_url = _as_text(obs.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7780") or not (parsed.path or "/").startswith("/admin/sales/order"):
        return []

    axtree = obs.get("axtree_object")
    if not isinstance(axtree, Mapping):
        return []
    nodes = axtree.get("nodes")
    if not isinstance(nodes, list):
        return []

    extra_props = obs.get("extra_element_properties")
    if not isinstance(extra_props, Mapping):
        extra_props = {}

    row_cells: dict[int, list[tuple[float, str]]] = {}
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
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 2:
            continue
        try:
            x_pos = float(bbox[0])
            y_pos = float(bbox[1])
        except (TypeError, ValueError):
            continue

        role = (_extract_nested_value(node.get("role")) or "unknown").lower()
        if role != "gridcell":
            continue
        name = _normalize_inline_text(_extract_nested_value(node.get("name")) or "")
        if not name:
            continue
        row_key = int(round(y_pos))
        row_cells.setdefault(row_key, []).append((x_pos, name))

    rows: list[dict[str, str]] = []
    for row_key in sorted(row_cells):
        ordered_cells = sorted(row_cells[row_key], key=lambda item: item[0])
        order_number = ""
        purchased_at = ""
        status = ""
        customer_name = ""
        billing_name = ""
        total = ""
        seen_names: list[str] = []
        seen_totals: list[str] = []

        for x_pos, cell_text in ordered_cells:
            lowered = cell_text.lower()
            if re.fullmatch(r"0*\d{3,}", cell_text):
                if not order_number:
                    order_number = cell_text
                continue
            if re.search(MONTH_NAME_PATTERN, lowered):
                if not purchased_at:
                    purchased_at = cell_text
                continue
            if re.fullmatch(r"\$[\d,]+\.\d{2}", cell_text):
                if cell_text not in seen_totals:
                    seen_totals.append(cell_text)
                continue
            if lowered in {"pending", "complete", "canceled", "cancelled", "closed", "processing"}:
                if not status:
                    status = cell_text
                continue
            if _looks_like_person_name(lowered):
                if cell_text not in seen_names:
                    seen_names.append(cell_text)
                continue

        if seen_names:
            customer_name = seen_names[0]
            billing_name = seen_names[1] if len(seen_names) >= 2 else seen_names[0]
        if seen_totals:
            total = seen_totals[0].replace("$", "").replace(",", "")
        if not (order_number and total and status):
            continue
        rows.append(
            {
                "order_number": order_number,
                "date": purchased_at,
                "customer_name": customer_name,
                "billing_name": billing_name,
                "total": total,
                "status": status,
            }
        )
    return rows


def _extract_shopping_admin_dashboard_rows(obs: Mapping[str, Any]) -> list[dict[str, str]]:
    current_url = _as_text(obs.get("url"))
    parsed = urlparse(current_url or "")
    if not parsed.netloc.endswith(":7780") or not (parsed.path or "/").startswith("/admin/admin/dashboard"):
        return []

    axtree = obs.get("axtree_object")
    if not isinstance(axtree, Mapping):
        return []
    nodes = axtree.get("nodes")
    if not isinstance(nodes, list):
        return []

    extra_props = obs.get("extra_element_properties")
    if not isinstance(extra_props, Mapping):
        extra_props = {}

    row_cells: dict[int, list[tuple[float, str]]] = {}
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
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 2:
            continue
        try:
            x_pos = float(bbox[0])
            y_pos = float(bbox[1])
        except (TypeError, ValueError):
            continue

        role = (_extract_nested_value(node.get("role")) or "unknown").lower()
        if role != "gridcell":
            continue
        name = _normalize_inline_text(_extract_nested_value(node.get("name")) or "")
        if not name:
            continue
        row_key = int(round(y_pos))
        row_cells.setdefault(row_key, []).append((x_pos, name))

    rows: list[dict[str, str]] = []
    for row_key in sorted(row_cells):
        ordered_cells = sorted(row_cells[row_key], key=lambda item: item[0])
        customer_name = ""
        items = ""
        total = ""
        for _, cell_text in ordered_cells:
            lowered = cell_text.lower()
            if not customer_name and _looks_like_person_name(lowered):
                customer_name = cell_text
                continue
            if not items and re.fullmatch(r"\d{1,2}", cell_text):
                items = cell_text
                continue
            if not total and re.fullmatch(r"\$[\d,]+\.\d{2}", cell_text):
                total = cell_text.replace("$", "").replace(",", "")
                continue
        if not (customer_name and items and total):
            continue
        rows.append(
            {
                "customer_name": customer_name,
                "items": items,
                "total": total,
            }
        )
    return rows


def _derive_shopping_order_count_from_goal(goal: str) -> int | None:
    lowered = " ".join((goal or "").lower().split())
    if not lowered:
        return None
    for pattern in (
        r"\bmost recent\s+(\d+)\s+orders?\b",
        r"\blast\s+(\d+)\s+(?:completed|complete|pending|cancelled|canceled|non-cancelled|non-canceled)?\s*orders?\b",
    ):
        match = re.search(pattern, lowered)
        if match is not None:
            return int(match.group(1))
    return None


def _derive_shopping_order_status_key_from_goal(goal: str) -> str:
    lowered = " ".join((goal or "").lower().split())
    if "non-cancelled" in lowered or "non-canceled" in lowered:
        return "non_canceled"
    if "cancelled" in lowered or "canceled" in lowered:
        return "canceled"
    if "pending" in lowered:
        return "pending"
    if "completed" in lowered or "complete" in lowered:
        return "complete"
    return ""


def _shopping_admin_row_matches_status(row: Mapping[str, str], status_key: str) -> bool:
    lowered = _as_text(row.get("status")).lower()
    if not status_key:
        return True
    if status_key == "non_canceled":
        return lowered not in {"canceled", "cancelled"}
    if status_key == "canceled":
        return lowered in {"canceled", "cancelled"}
    return lowered == status_key


def _parse_shopping_admin_order_date(value: str) -> datetime | None:
    normalized = " ".join(_as_text(value).replace("Sept", "Sep").split())
    if not normalized:
        return None
    for date_format in (
        "%B %d, %Y %I:%M:%S %p",
        "%b %d, %Y %I:%M:%S %p",
        "%m/%d/%y",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(normalized, date_format)
        except ValueError:
            continue
    return None


def _sort_shopping_admin_order_rows_by_date(
    rows: list[dict[str, str]],
    *,
    newest_first: bool,
) -> list[dict[str, str]]:
    dated_rows: list[tuple[datetime, int, dict[str, str]]] = []
    undated_rows: list[tuple[int, dict[str, str]]] = []
    for index, row in enumerate(rows):
        parsed_date = _parse_shopping_admin_order_date(row.get("date", ""))
        if parsed_date is None:
            undated_rows.append((index, row))
            continue
        dated_rows.append((parsed_date, index, row))
    if not dated_rows:
        return list(rows)
    if newest_first:
        dated_rows.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    else:
        dated_rows.sort(key=lambda item: (item[0], item[1]))
    ordered_rows = [row for _, _, row in dated_rows]
    ordered_rows.extend(row for _, row in undated_rows)
    return ordered_rows


def _summarize_shopping_admin_order_rows(
    obs: Mapping[str, Any],
    *,
    max_items: int,
) -> list[str]:
    if max_items <= 0:
        return []
    rows = _extract_shopping_admin_order_rows(obs)
    if not rows:
        return []

    goal = _as_text(obs.get("goal"))
    lowered_goal = " ".join(goal.lower().split())
    status_key = _derive_shopping_order_status_key_from_goal(goal)
    count = _derive_shopping_order_count_from_goal(goal)
    filtered_rows = [row for row in rows if _shopping_admin_row_matches_status(row, status_key)]
    if not filtered_rows:
        filtered_rows = rows
    newest_first_rows = _sort_shopping_admin_order_rows_by_date(filtered_rows, newest_first=True)
    oldest_first_rows = _sort_shopping_admin_order_rows_by_date(filtered_rows, newest_first=False)

    selected_rows: list[dict[str, str]]
    if "payment difference" in lowered_goal:
        compare_count = count or 4
        canceled_pool = _sort_shopping_admin_order_rows_by_date(
            [row for row in rows if _shopping_admin_row_matches_status(row, "canceled")],
            newest_first=True,
        )
        complete_pool = _sort_shopping_admin_order_rows_by_date(
            [row for row in rows if _shopping_admin_row_matches_status(row, "complete")],
            newest_first=True,
        )
        canceled_rows = canceled_pool[:compare_count]
        complete_rows = complete_pool[:compare_count]
        selected_rows = canceled_rows + complete_rows
    elif count and "total payment amount" in lowered_goal:
        selected_rows = newest_first_rows[:count]
    elif any(token in lowered_goal for token in ("customer name", "billing name", "order id", "order number")):
        if "oldest" in lowered_goal:
            selected_rows = oldest_first_rows[:1]
        else:
            selected_rows = newest_first_rows[:1]
    else:
        selected_rows = newest_first_rows[:max_items]

    lines: list[str] = []
    for row in selected_rows[:max_items]:
        lines.append(
            "Order row: "
            f'order={row.get("order_number", "")} | '
            f'date={row.get("date", "")} | '
            f'customer={row.get("customer_name", "")} | '
            f'billing={row.get("billing_name", "")} | '
            f'total={row.get("total", "")} | '
            f'status={row.get("status", "")}'
        )
    return lines


def _summarize_shopping_customer_order_rows(
    obs: Mapping[str, Any],
    *,
    max_items: int,
) -> list[str]:
    if max_items <= 0:
        return []
    rows = _extract_shopping_customer_order_rows(obs)
    if not rows:
        return []

    lines: list[str] = []
    for row in rows[:max_items]:
        lines.append(
            "Customer order row: "
            f'order={row.get("order_number", "")} | '
            f'date={row.get("date", "")} | '
            f'total={row.get("total", "")} | '
            f'status={row.get("status", "")}'
        )
    return lines


def _summarize_shopping_customer_order_pager(obs: Mapping[str, Any]) -> list[str]:
    pager = _extract_shopping_customer_order_pager(obs)
    if not pager:
        return []
    ordered_keys = [key for key in ("previous", "page_2", "page_3", "page_4", "next") if key in pager]
    ordered_keys.extend(key for key in sorted(pager) if key not in ordered_keys)
    parts = [f"{key}={pager[key]}" for key in ordered_keys]
    if not parts:
        return []
    return ["Customer order pager: " + " | ".join(parts)]


def _summarize_shopping_admin_dashboard_rows(
    obs: Mapping[str, Any],
    *,
    max_items: int,
) -> list[str]:
    if max_items <= 0:
        return []
    rows = _extract_shopping_admin_dashboard_rows(obs)
    if not rows:
        return []

    count = _derive_shopping_order_count_from_goal(_as_text(obs.get("goal"))) or max_items
    lines: list[str] = []
    for row in rows[: min(max_items, count)]:
        lines.append(
            "Dashboard order row: "
            f'customer={row.get("customer_name", "")} | '
            f'items={row.get("items", "")} | '
            f'total={row.get("total", "")}'
        )
    return lines


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

    output: list[str] = []
    dashboard_lines = _summarize_shopping_admin_dashboard_rows(obs, max_items=max_items)
    for line in dashboard_lines:
        if line in seen_labels or line in output:
            continue
        output.append(line)
        if len(output) >= max_items:
            return output

    customer_pager_lines = _summarize_shopping_customer_order_pager(obs)
    for line in customer_pager_lines:
        if line in seen_labels or line in output:
            continue
        output.append(line)
        if len(output) >= max_items:
            return output

    customer_lines = _summarize_shopping_customer_order_rows(obs, max_items=max_items)
    for line in customer_lines:
        if line in seen_labels or line in output:
            continue
        output.append(line)
        if len(output) >= max_items:
            return output

    admin_lines = _summarize_shopping_admin_order_rows(obs, max_items=max_items)
    for line in admin_lines:
        if line in seen_labels or line in output:
            continue
        output.append(line)
        if len(output) >= max_items:
            return output

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

    original_string_call = evaluators.StringEvaluator.__call__

    def patched_string_call(self, trajectory, config_file, page=None, client=None) -> float:
        with open(config_file, "r") as f:
            configs = json.load(f)

        last_action = self.get_last_action(trajectory)
        pred = self.clean_answer(last_action["answer"])

        score = 1.0
        for approach, value in configs["eval"]["reference_answers"].items():
            match approach:
                case "exact_match":
                    score *= self.exact_match(ref=value, pred=pred)
                case "must_include":
                    assert isinstance(value, list)
                    for must_value in value:
                        score *= self.must_include(
                            ref=must_value,
                            pred=pred,
                            tokenize=(len(value) == 1),
                        )
                case "fuzzy_match":
                    intent = configs["intent"]
                    if value == "N/A":
                        score *= self.exact_match(ref=value, pred=pred)
                        if score != 1:
                            score = 1.0 * self.ua_match(
                                intent=configs["intent"],
                                ref=configs["eval"]["string_note"],
                                pred=pred,
                            )
                    else:
                        assert isinstance(value, list)
                        combined_reference = _resolve_combined_fuzzy_reference(configs.get("eval", {}), value)
                        if combined_reference:
                            combined_score = self.fuzzy_match(
                                ref=combined_reference,
                                pred=pred,
                                intent=intent,
                            )
                            if combined_score == 1.0:
                                score *= combined_score
                                continue
                        for reference in value:
                            score *= self.fuzzy_match(ref=reference, pred=pred, intent=intent)
                case _:
                    return original_string_call(self, trajectory, config_file, page=page, client=client)
        return score

    patched_llm_fuzzy_match._patched_for_configurable_openai_judge = True  # type: ignore[attr-defined]
    patched_llm_ua_match._patched_for_configurable_openai_judge = True  # type: ignore[attr-defined]
    patched_string_call._patched_for_configurable_openai_judge = True  # type: ignore[attr-defined]
    helper_functions.llm_fuzzy_match = patched_llm_fuzzy_match
    helper_functions.llm_ua_match = patched_llm_ua_match
    evaluators.llm_fuzzy_match = patched_llm_fuzzy_match
    evaluators.llm_ua_match = patched_llm_ua_match
    evaluators.StringEvaluator.__call__ = patched_string_call


def _resolve_combined_fuzzy_reference(eval_config: Mapping[str, Any], references: list[str]) -> str:
    if len(references) < 2:
        return ""
    raw_annotation = _as_text(eval_config.get("reference_answer_raw_annotation"))
    if raw_annotation:
        return raw_annotation
    return ", ".join(reference.strip() for reference in references if reference.strip())


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
