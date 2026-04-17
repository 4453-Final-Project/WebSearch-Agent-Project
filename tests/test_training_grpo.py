"""Unit tests for Task 4 training helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.checkpointing import CheckpointManager
from src.training.grpo import GRPOConfig, GRPOTrainer, compute_group_advantages
from src.training.ppo import compute_clipped_policy_objective
from src.training.rewards import (
    RewardConfig,
    build_optimization_samples,
    compute_episode_reward_breakdown,
    compute_episode_score,
)
from src.training.trajectory import EpisodeTrajectory, SampleWeightConfig, TrajectoryStep


class FakePolicy:
    name = "fake-grpo"

    def __init__(self) -> None:
        self.parameter = torch.nn.Parameter(torch.tensor(0.0))
        self.optimizer = torch.optim.SGD([self.parameter], lr=0.1)

    def score_responses(self, prompts, responses, *, use_reference: bool = False, requires_grad: bool = True):
        base = self.parameter.expand(len(prompts))
        if use_reference:
            return (base + 0.05).detach()
        return base

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step_optimizer(self) -> None:
        self.optimizer.step()

    def clip_grad_norm_(self, max_grad_norm: float) -> float:
        return float(torch.nn.utils.clip_grad_norm_([self.parameter], max_grad_norm))

    def save_adapter(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "adapter.txt").write_text("fake-adapter", encoding="utf-8")

class FakeCollector:
    def __init__(self, grouped_episodes):
        self.grouped_episodes = grouped_episodes
        self.calls = 0

    def collect_iteration(self, policy, iteration_idx):
        self.calls += 1
        return self.grouped_episodes


def make_episode(
    seed: int,
    reward: float,
    *,
    success: bool,
    invalid_actions: int = 0,
    step_count: int = 1,
) -> EpisodeTrajectory:
    return EpisodeTrajectory(
        task_id=310,
        seed=seed,
        raw_goal="goal",
        success=success,
        reward=reward,
        steps_taken=step_count,
        invalid_action_count=invalid_actions,
        parse_failure_count=0,
        terminated=True,
        truncated=False,
        failure_reasons=[],
        output_dir="episode",
        steps_path="steps.jsonl",
        episode_path="episode.json",
        policy_name="fake",
        steps=[
            TrajectoryStep(
                step_idx=step_idx,
                prompt="prompt",
                response_text='ACTION: click("Buy")',
                action_text='click("Buy")',
                raw_text='ACTION: click("Buy")',
                parse_error=None,
                last_action_error=None,
                reward=reward,
                done=step_idx == step_count - 1,
                terminated=step_idx == step_count - 1,
                truncated=False,
                observation={"current_url": "http://example.com/page"},
                info={},
            )
            for step_idx in range(step_count)
        ],
    )


class TrainingHelpersTests(unittest.TestCase):
    def test_group_advantages_center_scores(self) -> None:
        advantages = compute_group_advantages([1.0, 3.0])
        self.assertEqual(len(advantages), 2)
        self.assertAlmostEqual(sum(advantages), 0.0, places=5)

    def test_reward_shaping_penalizes_invalid_actions(self) -> None:
        trajectory = make_episode(seed=1, reward=0.0, success=False, invalid_actions=2)
        score = compute_episode_score(trajectory, RewardConfig())
        self.assertLess(score, 0.0)

    def test_reward_shaping_penalizes_repeated_actions(self) -> None:
        trajectory = make_episode(seed=2, reward=0.0, success=False, step_count=3)
        for step in trajectory.steps:
            step.action_text = 'goto("http://3.14.148.71:8023/explore")'
            step.response_text = 'ACTION: goto("http://3.14.148.71:8023/explore")'
            step.raw_text = 'ACTION: goto("http://3.14.148.71:8023/explore")'
        score = compute_episode_score(trajectory, RewardConfig(repeat_action_penalty=0.05))
        self.assertLessEqual(score, -0.10)

    def test_reward_shaping_prefers_faster_successes(self) -> None:
        fast_success = make_episode(seed=3, reward=1.0, success=True, step_count=1)
        slow_success = make_episode(seed=4, reward=1.0, success=True, step_count=4)

        fast_score = compute_episode_score(fast_success, RewardConfig())
        slow_score = compute_episode_score(slow_success, RewardConfig())

        self.assertGreater(fast_score, slow_score)

    def test_reward_shaping_penalizes_extra_steps_on_failures(self) -> None:
        short_failure = make_episode(seed=5, reward=0.0, success=False, step_count=1)
        long_failure = make_episode(seed=6, reward=0.0, success=False, step_count=4)

        short_score = compute_episode_score(short_failure, RewardConfig())
        long_score = compute_episode_score(long_failure, RewardConfig())

        self.assertGreater(short_score, long_score)

    def test_reward_shaping_penalizes_same_page_repeated_actions_more_than_generic_repeat(self) -> None:
        trajectory = make_episode(seed=8, reward=0.0, success=False, step_count=3)
        for step in trajectory.steps:
            step.action_text = 'click("123")'
            step.response_text = 'ACTION: click("123")'
            step.raw_text = 'ACTION: click("123")'
            step.observation = {"current_url": "http://example.com/orders"}

        breakdown = compute_episode_reward_breakdown(trajectory, RewardConfig())

        self.assertEqual(breakdown["repeat_action_count"], 2)
        self.assertEqual(breakdown["same_page_repeat_action_count"], 2)
        self.assertGreater(breakdown["same_page_repeat_action_penalty"], 0.0)

    def test_reward_shaping_does_not_count_terminal_send_msg_as_same_page_stall(self) -> None:
        trajectory = make_episode(seed=9, reward=1.0, success=True, step_count=2)
        for step in trajectory.steps:
            step.action_text = 'send_msg_to_user("done")'
            step.response_text = 'ACTION: send_msg_to_user("done")'
            step.raw_text = 'ACTION: send_msg_to_user("done")'
            step.observation = {"current_url": "http://example.com/orders"}

        breakdown = compute_episode_reward_breakdown(trajectory, RewardConfig())

        self.assertEqual(breakdown["same_page_repeat_action_count"], 0)

    def test_reward_shaping_rewards_successful_multi_page_navigation(self) -> None:
        single_page = make_episode(seed=10, reward=1.0, success=True, step_count=2)
        multi_page = make_episode(seed=11, reward=1.0, success=True, step_count=2)

        for step in single_page.steps:
            step.observation = {"current_url": "http://example.com/account"}

        multi_page.steps[0].observation = {"current_url": "http://example.com/account"}
        multi_page.steps[1].observation = {"current_url": "http://example.com/orders"}

        single_score = compute_episode_score(single_page, RewardConfig())
        multi_score = compute_episode_score(multi_page, RewardConfig())
        multi_breakdown = compute_episode_reward_breakdown(multi_page, RewardConfig())

        self.assertGreater(multi_score, single_score)
        self.assertEqual(multi_breakdown["unique_url_count"], 2)
        self.assertGreater(multi_breakdown["success_unique_url_bonus"], 0.0)

    def test_clipped_policy_objective_returns_scalars(self) -> None:
        stats = compute_clipped_policy_objective(
            current_logprobs=torch.tensor([0.2, 0.1]),
            old_logprobs=torch.tensor([0.0, 0.0]),
            advantages=torch.tensor([1.0, -1.0]),
            clip_range=0.2,
        )
        self.assertTrue(isinstance(stats.loss, torch.Tensor))
        self.assertGreater(stats.ratio_mean, 0.0)

    def test_optimization_samples_split_weight_across_steps(self) -> None:
        episode = make_episode(seed=7, reward=0.0, success=False, step_count=3)
        samples = build_optimization_samples(
            episode,
            group_id=0,
            episode_score=0.0,
            advantage=-1.0,
        )
        self.assertEqual(len(samples), 3)
        self.assertAlmostEqual(sum(sample.sample_weight for sample in samples), 1.0, places=5)
        self.assertLess(samples[0].sample_weight, samples[1].sample_weight)
        self.assertLess(samples[1].sample_weight, samples[2].sample_weight)

    def test_optimization_samples_upweight_successful_terminal_step(self) -> None:
        episode = make_episode(seed=12, reward=1.0, success=True, step_count=3)
        episode.steps[-1].action_text = 'send_msg_to_user("done")'
        episode.steps[-1].response_text = 'ACTION: send_msg_to_user("done")'
        episode.steps[-1].raw_text = 'ACTION: send_msg_to_user("done")'

        samples = build_optimization_samples(
            episode,
            group_id=0,
            episode_score=1.0,
            advantage=1.0,
        )

        self.assertEqual(len(samples), 3)
        self.assertGreater(samples[-1].sample_weight, 0.5)
        self.assertAlmostEqual(sum(sample.sample_weight for sample in samples), 1.0, places=5)

    def test_optimization_samples_downweight_invalid_and_parse_failed_steps(self) -> None:
        episode = make_episode(seed=13, reward=0.0, success=False, step_count=3)
        episode.steps[1].last_action_error = "not clickable"
        episode.steps[2].parse_error = "bad parse"

        samples = build_optimization_samples(
            episode,
            group_id=0,
            episode_score=-0.5,
            advantage=-1.0,
        )

        self.assertEqual(len(samples), 3)
        self.assertLess(samples[1].sample_weight, samples[0].sample_weight)
        self.assertLess(samples[2].sample_weight, samples[0].sample_weight)
        self.assertAlmostEqual(sum(sample.sample_weight for sample in samples), 1.0, places=5)

    def test_optimization_samples_respect_custom_step_weight_config(self) -> None:
        episode = make_episode(seed=14, reward=1.0, success=True, step_count=3)
        episode.steps[-1].action_text = 'send_msg_to_user("done")'
        episode.steps[-1].response_text = 'ACTION: send_msg_to_user("done")'
        episode.steps[-1].raw_text = 'ACTION: send_msg_to_user("done")'

        default_samples = build_optimization_samples(
            episode,
            group_id=0,
            episode_score=1.0,
            advantage=1.0,
        )
        tuned_samples = build_optimization_samples(
            episode,
            group_id=0,
            episode_score=1.0,
            advantage=1.0,
            sample_weight_config=SampleWeightConfig(
                later_step_bonus=0.0,
                terminal_success_bonus=3.0,
                error_step_multiplier=0.5,
                min_step_weight=0.01,
            ),
        )

        self.assertGreater(tuned_samples[-1].sample_weight, default_samples[-1].sample_weight)
        self.assertAlmostEqual(sum(sample.sample_weight for sample in tuned_samples), 1.0, places=5)

    def test_weighted_policy_objective_respects_sample_weights(self) -> None:
        stats = compute_clipped_policy_objective(
            current_logprobs=torch.tensor([-1.0, -1.0, 0.0]),
            old_logprobs=torch.tensor([0.0, 0.0, 0.0]),
            advantages=torch.tensor([-1.0, -1.0, 1.0]),
            clip_range=0.2,
            sample_weights=torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0]),
        )
        self.assertLess(stats.loss.item(), 0.0)

    def test_grpo_trainer_saves_checkpoint_and_metrics(self) -> None:
        policy = FakePolicy()
        collector = FakeCollector([[make_episode(seed=1, reward=1.0, success=True), make_episode(seed=2, reward=0.0, success=False)]])

        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = GRPOTrainer(
                policy=policy,
                collector=collector,
                reward_config=RewardConfig(),
                config=GRPOConfig(iterations=1, ppo_epochs=1, batch_size=2),
                checkpoint_manager=CheckpointManager(temp_dir),
            )
            metrics = trainer.train()

            self.assertEqual(len(metrics), 1)
            self.assertTrue(Path(metrics[0]["checkpoint_path"]).exists())
            self.assertEqual(collector.calls, 1)
            self.assertIn("repeat_actions", metrics[0])
            self.assertIn("same_page_repeat_actions", metrics[0])
            self.assertIn("unique_urls", metrics[0])


if __name__ == "__main__":
    unittest.main()
