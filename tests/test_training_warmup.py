"""Unit tests for behavior-cloning warmup helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.warmup import WarmupConfig, build_warmup_samples, run_supervised_warmup
from src.training.trajectory import SampleWeightConfig
from tests.test_training_grpo import FakePolicy, make_episode


class WarmupHelpersTests(unittest.TestCase):
    def test_build_warmup_samples_collects_prompt_response_pairs(self) -> None:
        trajectory = make_episode(seed=1, reward=1.0, success=True)
        samples = build_warmup_samples([trajectory])
        self.assertEqual(len(samples), 1)
        self.assertIn("You are a browser automation policy.", samples[0].prompt)
        self.assertIn("Step: 0", samples[0].prompt)
        self.assertIn("Output Contract:", samples[0].prompt)
        self.assertEqual(samples[0].response_text, 'ACTION: click("Buy")')
        self.assertAlmostEqual(samples[0].sample_weight, 1.0, places=5)

    def test_build_warmup_samples_upweight_terminal_steps(self) -> None:
        trajectory = make_episode(seed=2, reward=1.0, success=True, step_count=3)
        trajectory.steps[-1].action_text = 'send_msg_to_user("done")'
        trajectory.steps[-1].response_text = 'ACTION: send_msg_to_user("done")'
        trajectory.steps[-1].raw_text = 'ACTION: send_msg_to_user("done")'

        samples = build_warmup_samples([trajectory])

        self.assertEqual(len(samples), 3)
        self.assertAlmostEqual(sum(sample.sample_weight for sample in samples), 1.0, places=5)
        self.assertGreater(samples[-1].sample_weight, samples[0].sample_weight)

    def test_build_warmup_samples_respect_custom_step_weight_config(self) -> None:
        trajectory = make_episode(seed=3, reward=1.0, success=True, step_count=3)
        trajectory.steps[-1].action_text = 'send_msg_to_user("done")'
        trajectory.steps[-1].response_text = 'ACTION: send_msg_to_user("done")'
        trajectory.steps[-1].raw_text = 'ACTION: send_msg_to_user("done")'

        default_samples = build_warmup_samples([trajectory])
        tuned_samples = build_warmup_samples(
            [trajectory],
            sample_weight_config=SampleWeightConfig(
                later_step_bonus=0.0,
                terminal_success_bonus=3.0,
                error_step_multiplier=0.5,
                min_step_weight=0.01,
            ),
        )

        self.assertGreater(tuned_samples[-1].sample_weight, default_samples[-1].sample_weight)
        self.assertAlmostEqual(sum(sample.sample_weight for sample in tuned_samples), 1.0, places=5)

    def test_supervised_warmup_updates_policy(self) -> None:
        policy = FakePolicy()
        trajectory = make_episode(seed=1, reward=1.0, success=True)
        initial_value = float(policy.parameter.detach().item())

        metrics = run_supervised_warmup(
            policy,
            [trajectory],
            WarmupConfig(epochs=2, batch_size=1, max_grad_norm=1.0, success_only=True, shuffle_seed=0),
        )

        self.assertGreater(metrics["samples_used"], 0)
        self.assertGreater(float(policy.parameter.detach().item()), initial_value)


if __name__ == "__main__":
    unittest.main()
