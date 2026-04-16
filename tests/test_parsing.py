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
    normalize_action_text,
    should_retry_once,
)


class ParsingTests(unittest.TestCase):
    """Tests for strict ACTION-line parsing."""

    def test_extract_action_text_from_valid_action_line(self) -> None:
        action_text, parse_error = extract_action_text('ACTION: click("58")')

        self.assertEqual(action_text, 'click("58")')
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
        raw_text = "\nReasoning here\nACTION: click(\"Search\")\nACTION: send_msg_to_user(\"N/A\")"

        action_text, parse_error = extract_action_text(raw_text)

        self.assertIsNone(action_text)
        self.assertIn("numeric quoted bid", parse_error or "")

    def test_extract_action_text_truncates_multiple_actions_on_same_line(self) -> None:
        raw_text = 'ACTION: click("Search") ACTION: send_msg_to_user("N/A")'

        action_text, parse_error = extract_action_text(raw_text)

        self.assertIsNone(action_text)
        self.assertIn("numeric quoted bid", parse_error or "")

    def test_normalize_action_text_maps_type_alias_to_fill(self) -> None:
        action_text, parse_error = normalize_action_text('type("130", "administrate")')

        self.assertEqual(action_text, 'fill("130", "administrate")')
        self.assertIsNone(parse_error)

    def test_normalize_action_text_maps_wait_alias_to_noop(self) -> None:
        action_text, parse_error = normalize_action_text("wait(500)")

        self.assertEqual(action_text, "noop(500)")
        self.assertIsNone(parse_error)

    def test_normalize_action_text_maps_keyboard_press_alias_to_press(self) -> None:
        action_text, parse_error = normalize_action_text('keyboard_press("Enter")')

        self.assertIsNone(action_text)
        self.assertIn("requires exactly two arguments", parse_error or "")

    def test_normalize_action_text_rejects_fill_with_non_numeric_bid(self) -> None:
        action_text, parse_error = normalize_action_text('fill("filter_by_name", "administrate")')

        self.assertIsNone(action_text)
        self.assertIn("first argument to be a numeric quoted bid", parse_error or "")

    def test_normalize_action_text_rejects_click_with_url_argument(self) -> None:
        action_text, parse_error = normalize_action_text('click("https://3.14.148.71:8023/explore?filter=administrate")')

        self.assertIsNone(action_text)
        self.assertIn("numeric quoted bid", parse_error or "")

    def test_normalize_action_text_rejects_goto_with_extra_arguments(self) -> None:
        action_text, parse_error = normalize_action_text('goto("http://3.14.148.71:8023/explore", "filterbyname")')

        self.assertIsNone(action_text)
        self.assertIn("exactly one quoted URL", parse_error or "")

    def test_normalize_action_text_rejects_unsupported_action_name(self) -> None:
        action_text, parse_error = normalize_action_text('click_all("A")')

        self.assertIsNone(action_text)
        self.assertIn("Unsupported action type 'click_all'", parse_error or "")

    def test_normalize_action_text_rejects_tab_focus(self) -> None:
        action_text, parse_error = normalize_action_text('tab_focus("Projects")')

        self.assertIsNone(action_text)
        self.assertIn("Unsupported action type 'tab_focus'", parse_error or "")

    def test_make_decision_preserves_raw_text_and_sets_retry_on_failure(self) -> None:
        raw_text = "This output does not follow the contract."

        decision = make_decision(raw_text)

        self.assertEqual(decision.raw_text, raw_text)
        self.assertEqual(decision.action_text, "")
        self.assertEqual(decision.parse_error, "No ACTION line found in model output.")
        self.assertTrue(decision.should_retry)

    def test_make_decision_succeeds_without_retry(self) -> None:
        raw_text = 'ACTION: type("130", "administrate")'

        decision = make_decision(raw_text)

        self.assertEqual(decision.raw_text, raw_text)
        self.assertEqual(decision.action_text, 'fill("130", "administrate")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)

    def test_make_fallback_decision_uses_safe_send_message_action(self) -> None:
        decision = make_fallback_decision(raw_text="bad output", reason="parse failed twice")

        self.assertEqual(decision.raw_text, "bad output")
        self.assertEqual(decision.action_text, 'send_msg_to_user("N/A")')
        self.assertIn("parse failed twice", decision.parse_error or "")
        self.assertFalse(decision.should_retry)

    def test_should_retry_once_only_allows_first_failure(self) -> None:
        self.assertTrue(should_retry_once("bad parse", retry_count=0))
        self.assertFalse(should_retry_once("bad parse", retry_count=1))
        self.assertFalse(should_retry_once(None, retry_count=0))


if __name__ == "__main__":
    unittest.main()
