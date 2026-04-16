"""Parsing helpers for Task 3 policy model outputs."""

from __future__ import annotations

import re

from .types import AgentDecision

SUPPORTED_ACTIONS = {
    "click",
    "fill",
    "hover",
    "scroll",
    "press",
    "goto",
    "go_back",
    "go_forward",
    "select_option",
    "send_msg_to_user",
    "report_infeasible",
    "noop",
}

ACTION_ALIASES = {
    "type": "fill",
    "wait": "noop",
    "stop": "send_msg_to_user",
    "keyboard_press": "press",
}

NO_ARG_ACTIONS = {
    "go_back",
    "go_forward",
    "report_infeasible",
}

ACTION_NAME_PATTERN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
NUMERIC_BID_PATTERN = re.compile(r'^\s*["\']\d+["\']\s*$')
QUOTED_STRING_PATTERN = re.compile(r'^\s*(["\']).*\1\s*$')


def extract_action_text(raw_text: str) -> tuple[str | None, str | None]:
    """Extract, normalize, and validate exactly one BrowserGym action."""

    action_payload = _extract_action_payload(raw_text)
    if action_payload is None:
        return None, "No ACTION line found in model output."
    if action_payload == "":
        return None, "ACTION line is present but has an empty payload."

    normalized_action, parse_error = normalize_action_text(action_payload)
    if parse_error is not None:
        return None, parse_error
    return normalized_action, None


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

    parse_error = "Falling back to safe send_msg_to_user action."
    if reason:
        parse_error = f"{parse_error} Reason: {reason}"
    return AgentDecision(
        raw_text=raw_text,
        action_text='send_msg_to_user("N/A")',
        parse_error=parse_error,
        should_retry=False,
    )


def should_retry_once(parse_error: str | None, retry_count: int) -> bool:
    """Return ``True`` only for the first parse failure."""

    return parse_error is not None and retry_count == 0


def normalize_action_text(action_text: str) -> tuple[str | None, str | None]:
    """Normalize aliases and validate a single BrowserGym action expression."""

    stripped = action_text.strip()
    canonical_bare = ACTION_ALIASES.get(stripped, stripped)
    if canonical_bare in NO_ARG_ACTIONS:
        return f"{canonical_bare}()", None

    extracted_call, extract_error = _extract_first_call(action_text)
    if extract_error is not None:
        return None, extract_error

    action_name, remainder = _split_action_name(extracted_call)
    canonical_name = ACTION_ALIASES.get(action_name, action_name)
    if canonical_name not in SUPPORTED_ACTIONS:
        return (
            None,
            f"Unsupported action type '{action_name}'. Allowed actions: {', '.join(sorted(SUPPORTED_ACTIONS))}.",
        )
    validation_error = _validate_action_signature(canonical_name, remainder)
    if validation_error is not None:
        return None, validation_error
    return f"{canonical_name}{remainder}", None


def _extract_action_payload(raw_text: str) -> str | None:
    for line in raw_text.splitlines():
        if "ACTION:" not in line:
            continue
        _, payload = line.split("ACTION:", 1)
        return payload.strip()
    return None


def _extract_first_call(action_text: str) -> tuple[str | None, str | None]:
    text = action_text.strip()
    if not text:
        return None, "ACTION line is present but has an empty payload."

    name_match = ACTION_NAME_PATTERN.match(text)
    if name_match is None:
        token = text.split("(", 1)[0].split(None, 1)[0] if text else ""
        if token:
            return (
                None,
                f"Unsupported action format '{token}'. Expected exactly one function-style BrowserGym action.",
            )
        return None, "Unsupported action format. Expected exactly one function-style BrowserGym action."

    depth = 0
    quote_char: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote_char is not None:
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote_char:
                quote_char = None
            continue

        if char in {'"', "'"}:
            quote_char = char
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth -= 1
            if depth == 0:
                return text[: index + 1].strip(), None

    return None, "ACTION payload is missing a complete function call."


def _split_action_name(action_text: str) -> tuple[str, str]:
    match = ACTION_NAME_PATTERN.match(action_text)
    if match is None:
        return "", action_text
    action_name = match.group(1)
    remainder = action_text[match.end(1) :]
    return action_name, remainder


def _validate_action_signature(action_name: str, remainder: str) -> str | None:
    arguments, error = _parse_arguments(remainder)
    if error is not None:
        return error

    if action_name in {"click", "hover"}:
        if len(arguments) != 1 or not _is_numeric_bid(arguments[0]):
            return f"{action_name}() requires exactly one numeric quoted bid like {action_name}(\"58\")."
        return None

    if action_name in {"fill", "select_option", "press"}:
        if len(arguments) != 2:
            return f"{action_name}() requires exactly two arguments."
        if not _is_numeric_bid(arguments[0]):
            return (
                f"{action_name}() requires the first argument to be a numeric quoted bid like "
                f'{action_name}("130", "...").'
            )
        return None

    if action_name == "goto":
        if len(arguments) != 1 or not _is_quoted_string(arguments[0]):
            return 'goto() requires exactly one quoted URL like goto("http://3.14.148.71:8023/explore").'
        return None

    if action_name == "send_msg_to_user":
        if len(arguments) != 1 or not _is_quoted_string(arguments[0]):
            return 'send_msg_to_user() requires exactly one quoted message like send_msg_to_user("N/A").'
        return None

    if action_name == "noop":
        if len(arguments) != 1:
            return "noop() requires exactly one duration argument like noop(500)."
        return None

    if action_name in NO_ARG_ACTIONS:
        if arguments:
            return f"{action_name}() does not take any arguments."
        return None

    return None


def _parse_arguments(remainder: str) -> tuple[list[str], str | None]:
    text = remainder.strip()
    if not text.startswith("(") or not text.endswith(")"):
        return [], "ACTION payload is missing a complete function call."

    inner = text[1:-1].strip()
    if inner == "":
        return [], None

    arguments: list[str] = []
    current: list[str] = []
    depth = 0
    quote_char: str | None = None
    escaped = False

    for char in inner:
        if quote_char is not None:
            current.append(char)
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == quote_char:
                quote_char = None
            continue

        if char in {'"', "'"}:
            quote_char = char
            current.append(char)
            continue
        if char in {"(", "[", "{"}:
            depth += 1
            current.append(char)
            continue
        if char in {")", "]", "}"}:
            depth = max(0, depth - 1)
            current.append(char)
            continue
        if char == "," and depth == 0:
            arguments.append("".join(current).strip())
            current = []
            continue
        current.append(char)

    if quote_char is not None:
        return [], "ACTION payload contains an unterminated quoted argument."

    final_argument = "".join(current).strip()
    if final_argument:
        arguments.append(final_argument)
    return arguments, None


def _is_numeric_bid(value: str) -> bool:
    return NUMERIC_BID_PATTERN.fullmatch(value) is not None


def _is_quoted_string(value: str) -> bool:
    return QUOTED_STRING_PATTERN.fullmatch(value) is not None
