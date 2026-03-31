"""Prompt construction helpers for the Task 3 policy wrapper."""

from __future__ import annotations

from .types import NormalizedObservation


def build_system_prompt() -> str:
    """Build the fixed system prompt for the browser policy model."""

    return (
        "You are a browser automation policy.\n"
        "Respond with exactly one line.\n"
        "Use this exact format and nothing else:\n"
        "ACTION: <browser action string>\n"
        'Examples:\n'
        'ACTION: click("Log in")\n'
        'ACTION: type("Search", "gaming laptop")\n'
        'ACTION: stop("N/A")'
    )


def build_user_prompt(observation: NormalizedObservation, step_idx: int) -> str:
    """Build the deterministic user prompt from a normalized observation."""

    sections = [
        f"Step: {step_idx}",
        f"Task Goal: {observation.goal}",
        f"Current URL: {observation.current_url}",
        "Open Tabs:",
        _render_open_tabs(observation),
        "Visible Page Summary:",
        _render_text_block(observation.visible_page_summary),
        "Relevant DOM or AX-Tree Snippet:",
        _render_text_block(observation.dom_or_ax_snippet),
        "Previous Actions:",
        _render_history(observation.previous_actions),
        "Previous Errors:",
        _render_history(observation.previous_errors),
    ]

    if observation.last_action_error is not None:
        sections.extend(
            [
                "Last Action Error:",
                _render_text_block(observation.last_action_error),
            ]
        )

    sections.extend(
        [
            "Output Contract:",
            "Respond with exactly one line.",
            "Use this exact format and nothing else:",
            "ACTION: <browser action string>",
        ]
    )

    return "\n".join(sections)


def build_retry_prompt(
    observation: NormalizedObservation,
    step_idx: int,
    previous_raw_text: str,
    parse_error: str,
) -> str:
    """Build a retry prompt that asks the model to correct invalid output."""

    sections = [
        build_user_prompt(observation, step_idx),
        "Previous Model Output:",
        _render_text_block(previous_raw_text),
        "Parse Error:",
        _render_text_block(parse_error),
        "Correction:",
        "Your previous response could not be parsed.",
        "Correct it and respond again with exactly one line.",
        "Use this exact format and nothing else:",
        "ACTION: <browser action string>",
    ]
    return "\n".join(sections)


def build_full_prompt(observation: NormalizedObservation, step_idx: int) -> str:
    """Build the full prompt text used for a normal policy generation step."""

    return "\n\n".join([build_system_prompt(), build_user_prompt(observation, step_idx)])


def _render_open_tabs(observation: NormalizedObservation) -> str:
    """Render open tabs in a deterministic numbered format."""

    if not observation.open_tabs:
        return "(none)"
    return "\n".join(
        f"{index}. {tab.title} | {tab.url}"
        for index, tab in enumerate(observation.open_tabs, start=1)
    )


def _render_history(items: list[str]) -> str:
    """Render action or error history with a stable empty state."""

    if not items:
        return "(none)"
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items, start=1))


def _render_text_block(value: str) -> str:
    """Render a text field with a stable empty state."""

    return value if value else "(empty)"
