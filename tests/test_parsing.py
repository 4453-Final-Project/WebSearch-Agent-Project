"""Unit tests for Task 3 action parsing and retry behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.parsing import (
    extract_action_text,
    make_decision,
    make_fallback_decision,
    should_retry_once,
)


class ParsingTests(unittest.TestCase):
    """Tests for strict ACTION-line parsing."""

    def test_extract_action_text_from_valid_action_line(self) -> None:
        action_text, parse_error = extract_action_text('ACTION: click("Log in")')

        self.assertEqual(action_text, 'click("Log in")')
        self.assertIsNone(parse_error)

    def test_extract_action_text_reports_missing_action_line(self) -> None:
        action_text, parse_error = extract_action_text("click the login button")

        self.assertIsNone(action_text)
        self.assertEqual(parse_error, "No ACTION line found in model output.")

    def test_extract_action_text_reports_empty_action_payload(self) -> None:
        action_text, parse_error = extract_action_text("ACTION:   ")

        self.assertIsNone(action_text)
        self.assertEqual(parse_error, "ACTION line is present but has an empty payload.")

    def test_extract_action_text_prefers_first_non_empty_action_line(self) -> None:
        raw_text = "\nReasoning here\nACTION: click(\"Search\")\nACTION: stop(\"N/A\")"

        action_text, parse_error = extract_action_text(raw_text)

        self.assertEqual(action_text, 'click("Search")')
        self.assertIsNone(parse_error)

    def test_make_decision_preserves_raw_text_and_sets_retry_on_failure(self) -> None:
        raw_text = "This output does not follow the contract."

        decision = make_decision(raw_text)

        self.assertEqual(decision.raw_text, raw_text)
        self.assertEqual(decision.action_text, "")
        self.assertEqual(decision.parse_error, "No ACTION line found in model output.")
        self.assertTrue(decision.should_retry)

    def test_make_decision_succeeds_without_retry(self) -> None:
        raw_text = 'ACTION: type("Search", "gaming laptop")'

        decision = make_decision(raw_text)

        self.assertEqual(decision.raw_text, raw_text)
        self.assertEqual(decision.action_text, 'type("Search", "gaming laptop")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_make_fallback_decision_uses_safe_stop_action(self) -> None:
        decision = make_fallback_decision(raw_text="bad output", reason="parse failed twice")

        self.assertEqual(decision.raw_text, "bad output")
        self.assertEqual(decision.action_text, 'stop("N/A")')
        self.assertIn("parse failed twice", decision.parse_error or "")
        self.assertFalse(decision.should_retry)

    def test_should_retry_once_only_allows_first_failure(self) -> None:
        self.assertTrue(should_retry_once("bad parse", retry_count=0))
        self.assertFalse(should_retry_once("bad parse", retry_count=1))
        self.assertFalse(should_retry_once(None, retry_count=0))


if __name__ == "__main__":
    unittest.main()
