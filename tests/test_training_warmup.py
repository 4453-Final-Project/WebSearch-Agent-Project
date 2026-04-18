"""Unit tests for behavior-cloning warmup helpers."""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.warmup import (
    WarmupConfig,
    _sample_epoch_warmup_samples,
    build_warmup_samples,
    run_supervised_warmup,
)
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

    def test_build_warmup_samples_assign_task_sampling_weights(self) -> None:
        shopping = make_episode(seed=4, reward=1.0, success=True)
        gitlab = make_episode(seed=5, reward=1.0, success=True)
        shopping.task_id = 21
        gitlab.task_id = 132

        samples = build_warmup_samples(
            [shopping, gitlab],
            task_sample_multipliers={132: 3.0},
        )

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].sampling_weight, 1.0)
        self.assertEqual(samples[1].sampling_weight, 3.0)

    def test_sample_epoch_warmup_samples_oversamples_weighted_tasks(self) -> None:
        shopping = make_episode(seed=6, reward=1.0, success=True)
        gitlab = make_episode(seed=7, reward=1.0, success=True)
        shopping.task_id = 21
        gitlab.task_id = 132
        samples = build_warmup_samples(
            [shopping, gitlab],
            task_sample_multipliers={132: 4.0},
        )

        epoch_samples = _sample_epoch_warmup_samples(samples, random.Random(0))

        self.assertEqual(len(epoch_samples), len(samples))
        self.assertGreaterEqual(sum(1 for sample in epoch_samples if sample.sampling_weight == 4.0), 1)
        self.assertGreater(
            sum(sample.sampling_weight for sample in epoch_samples),
            sum(sample.sampling_weight for sample in samples),
        )

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

    def test_supervised_warmup_reports_task_sampling_mix(self) -> None:
        policy = FakePolicy()
        shopping = make_episode(seed=8, reward=1.0, success=True)
        gitlab = make_episode(seed=9, reward=1.0, success=True)
        shopping.task_id = 21
        gitlab.task_id = 132

        metrics = run_supervised_warmup(
            policy,
            [shopping, gitlab],
            WarmupConfig(
                epochs=12,
                batch_size=1,
                max_grad_norm=1.0,
                success_only=True,
                shuffle_seed=0,
                task_sample_multipliers={132: 4.0},
            ),
        )

        self.assertEqual(metrics["source_task_sample_counts"], {"21": 1, "132": 1})
        self.assertEqual(metrics["task_sample_multipliers"], {"21": 1.0, "132": 4.0})
        self.assertGreater(
            metrics["average_epoch_task_sample_counts"]["132"],
            metrics["average_epoch_task_sample_counts"]["21"],
        )


if __name__ == "__main__":
    unittest.main()
