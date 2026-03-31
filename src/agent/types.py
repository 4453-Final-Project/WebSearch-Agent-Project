"""Shared data contracts for the Task 3 policy layer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class OpenTab:
    """Represents an open browser tab in a normalized observation."""

    title: str
    url: str


@dataclass(slots=True)
class NormalizedObservation:
    """Normalized browser state consumed by the policy wrapper."""

    goal: str
    current_url: str
    open_tabs: list[OpenTab] = field(default_factory=list)
    visible_page_summary: str = ""
    dom_or_ax_snippet: str = ""
    previous_actions: list[str] = field(default_factory=list)
    previous_errors: list[str] = field(default_factory=list)
    last_action_error: str | None = None


@dataclass(slots=True)
class AgentDecision:
    """Structured result from policy output parsing."""

    raw_text: str
    action_text: str
    parse_error: str | None
    should_retry: bool


@dataclass(slots=True)
class PolicyConfig:
    """Configuration used to load and run the Task 3 policy."""

    model_path: str
    max_new_tokens: int = 128
    temperature: float = 0.0
    device: str | None = None
