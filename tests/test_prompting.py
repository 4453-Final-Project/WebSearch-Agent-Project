"""Unit tests for Task 3 prompt construction helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.prompting import (
    build_full_prompt,
    build_retry_prompt,
    build_system_prompt,
    build_user_prompt,
)
from src.agent.types import NormalizedObservation, OpenTab


class PromptingTests(unittest.TestCase):
    """Tests for deterministic Task 3 prompt construction."""

    def setUp(self) -> None:
        self.observation = NormalizedObservation(
            goal="Find the pricing page",
            current_url="https://example.com/home",
            open_tabs=[
                OpenTab(title="Home", url="https://example.com/home"),
                OpenTab(title="Pricing", url="https://example.com/pricing"),
            ],
            visible_page_summary="Homepage with a top navigation bar.",
            dom_or_ax_snippet='button "Pricing"\nlink "Contact"',
            previous_actions=['click("Menu")', 'click("Products")'],
            previous_errors=["timeout on prior page"],
            last_action_error='Element not found: "Pricing"',
        )

    def test_user_prompt_contains_all_required_sections(self) -> None:
        prompt = build_user_prompt(self.observation, step_idx=3)

        expected_sections = [
            "Step: 3",
            "Task Goal: Find the pricing page",
            "Current URL: https://example.com/home",
            "Open Tabs:",
            "Visible Page Summary:",
            "Relevant DOM or AX-Tree Snippet:",
            "Previous Actions:",
            "Previous Errors:",
            "Last Action Error:",
            "Output Contract:",
        ]

        for section in expected_sections:
            self.assertIn(section, prompt)

    def test_retry_prompt_includes_previous_output_and_parse_error(self) -> None:
        prompt = build_retry_prompt(
            self.observation,
            step_idx=4,
            previous_raw_text="I think you should click pricing",
            parse_error="Missing ACTION prefix",
        )

        self.assertIn("Previous Model Output:", prompt)
        self.assertIn("I think you should click pricing", prompt)
        self.assertIn("Parse Error:", prompt)
        self.assertIn("Missing ACTION prefix", prompt)
        self.assertIn("Correct it and respond again with exactly one line.", prompt)

    def test_prompts_enforce_action_contract(self) -> None:
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(self.observation, step_idx=1)
        full_prompt = build_full_prompt(self.observation, step_idx=1)

        expected_lines = [
            "Respond with exactly one line.",
            "Use this exact format and nothing else:",
            "ACTION: <browser action string>",
        ]

        for line in expected_lines:
            self.assertIn(line, system_prompt)
            self.assertIn(line, user_prompt)
            self.assertIn(line, full_prompt)

    def test_open_tabs_render_deterministically(self) -> None:
        prompt = build_user_prompt(self.observation, step_idx=2)

        self.assertIn("1. Home | https://example.com/home", prompt)
        self.assertIn("2. Pricing | https://example.com/pricing", prompt)
        self.assertLess(
            prompt.index("1. Home | https://example.com/home"),
            prompt.index("2. Pricing | https://example.com/pricing"),
        )

    def test_empty_histories_and_tabs_are_handled_cleanly(self) -> None:
        observation = NormalizedObservation(
            goal="Stop when done",
            current_url="https://example.com",
            open_tabs=[],
            visible_page_summary="",
            dom_or_ax_snippet="",
            previous_actions=[],
            previous_errors=[],
            last_action_error=None,
        )

        prompt = build_user_prompt(observation, step_idx=0)

        self.assertIn("Open Tabs:\n(none)", prompt)
        self.assertIn("Visible Page Summary:\n(empty)", prompt)
        self.assertIn("Relevant DOM or AX-Tree Snippet:\n(empty)", prompt)
        self.assertIn("Previous Actions:\n(none)", prompt)
        self.assertIn("Previous Errors:\n(none)", prompt)
        self.assertNotIn("Last Action Error:", prompt)


if __name__ == "__main__":
    unittest.main()
