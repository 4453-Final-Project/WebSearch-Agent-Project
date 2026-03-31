"""Parsing helpers for Task 3 policy model outputs."""

from __future__ import annotations

from .types import AgentDecision


def extract_action_text(raw_text: str) -> tuple[str | None, str | None]:
    """Extract the first valid ``ACTION:`` payload from raw model output."""

    for line in raw_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if stripped_line.startswith("ACTION:"):
            action_text = stripped_line[len("ACTION:") :].strip()
            if action_text:
                return action_text, None
            return None, "ACTION line is present but has an empty payload."
    return None, "No ACTION line found in model output."


def make_decision(raw_text: str) -> AgentDecision:
    """Parse model output into an ``AgentDecision`` with retry guidance."""

    action_text, parse_error = extract_action_text(raw_text)
    return AgentDecision(
        raw_text=raw_text,
        action_text=action_text or "",
        parse_error=parse_error,
        should_retry=should_retry_once(parse_error, retry_count=0),
    )


def make_fallback_decision(raw_text: str = "", reason: str = "") -> AgentDecision:
    """Create a safe fallback decision that stops execution."""

    parse_error = "Falling back to safe stop action."
    if reason:
        parse_error = f"{parse_error} Reason: {reason}"
    return AgentDecision(
        raw_text=raw_text,
        action_text='stop("N/A")',
        parse_error=parse_error,
        should_retry=False,
    )


def should_retry_once(parse_error: str | None, retry_count: int) -> bool:
    """Return ``True`` only for the first parse failure."""

    return parse_error is not None and retry_count == 0
