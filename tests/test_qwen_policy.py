"""Unit tests for the Task 3 Qwen policy flow."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.fake_backend import FakeBackend
from src.agent.qwen_policy import QwenPolicy
from src.agent.types import NormalizedObservation, OpenTab, PolicyConfig


class QwenPolicyTests(unittest.TestCase):
    """Tests for the backend-driven Qwen policy skeleton."""

    def setUp(self) -> None:
        self.config = PolicyConfig(model_path="fake-model")
        self.raw_observation = {
            "task_goal": "Open the pricing page",
            "url": "https://example.com/home",
            "tabs": [
                {"title": "Home", "url": "https://example.com/home"},
                {"title": "Pricing", "url": "https://example.com/pricing"},
            ],
            "page_summary": "Homepage with product navigation.",
            "dom_snippet": 'link "Pricing"',
            "action_history": ['click("Menu")'],
            "error_history": [],
        }
        self.normalized_observation = NormalizedObservation(
            goal="Open the pricing page",
            current_url="https://example.com/home",
            open_tabs=[
                OpenTab(title="Home", url="https://example.com/home"),
                OpenTab(title="Pricing", url="https://example.com/pricing"),
            ],
            visible_page_summary="Homepage with product navigation.",
            dom_or_ax_snippet='link "Pricing"',
            previous_actions=['click("Menu")'],
            previous_errors=[],
        )

    def test_successful_first_pass_action_generation(self) -> None:
        backend = FakeBackend(['ACTION: click("Pricing")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(self.raw_observation, step_idx=1)

        self.assertEqual(decision.action_text, 'click("Pricing")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)
        self.assertEqual(len(backend.prompts), 1)

    def test_first_pass_failure_then_successful_retry(self) -> None:
        backend = FakeBackend(
            [
                "I should click the pricing link.",
                'ACTION: click("Pricing")',
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(self.normalized_observation, step_idx=2)

        self.assertEqual(decision.action_text, 'click("Pricing")')
        self.assertIsNone(decision.parse_error)
        self.assertFalse(decision.should_retry)
        self.assertEqual(len(backend.prompts), 2)
        self.assertIn("Previous Model Output:", backend.prompts[1])
        self.assertIn("Parse Error:", backend.prompts[1])

    def test_two_failures_lead_to_safe_stop_action(self) -> None:
        backend = FakeBackend(
            [
                "Malformed answer",
                "Still malformed",
            ]
        )
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(self.raw_observation, step_idx=3)

        self.assertEqual(decision.action_text, 'stop("N/A")')
        self.assertFalse(decision.should_retry)
        self.assertIsNotNone(decision.parse_error)
        self.assertIn("No ACTION line found in model output.", decision.parse_error or "")
        self.assertEqual(decision.raw_text, "Still malformed")

    def test_act_accepts_raw_dict_observation_input(self) -> None:
        backend = FakeBackend(['ACTION: click("Pricing")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(self.raw_observation, step_idx=4)

        self.assertEqual(decision.action_text, 'click("Pricing")')
        self.assertIn("Task Goal: Open the pricing page", backend.prompts[0])
        self.assertIn("Current URL: https://example.com/home", backend.prompts[0])

    def test_act_accepts_normalized_observation_input(self) -> None:
        backend = FakeBackend(['ACTION: stop("N/A")'])
        policy = QwenPolicy(backend=backend, config=self.config)

        decision = policy.act(self.normalized_observation, step_idx=5)

        self.assertEqual(decision.action_text, 'stop("N/A")')
        self.assertEqual(len(backend.prompts), 1)
        self.assertIn("Open Tabs:", backend.prompts[0])


if __name__ == "__main__":
    unittest.main()
