"""Unit tests for the single-task evaluation scaffold."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_single_task import (
    FakeEpisodeRunner,
    aggregate_metrics,
    evaluate_single_task,
    try_load_task2_runner,
    write_metrics_json,
)


class EvalSingleTaskTests(unittest.TestCase):
    """Tests for evaluation metric aggregation and fake-runner execution."""

    def test_metric_aggregation_correctness(self) -> None:
        metrics = aggregate_metrics(
            [
                {
                    "success": True,
                    "reward": 1.0,
                    "steps": 5,
                    "invalid_action_count": 1,
                    "parse_failure_count": 0,
                    "failure_reasons": [],
                },
                {
                    "success": False,
                    "reward": 0.5,
                    "steps": 7,
                    "invalid_action_count": 2,
                    "parse_failure_count": 1,
                    "failure_reasons": ["parse_error", "timeout"],
                },
                {
                    "success": False,
                    "reward": 0.0,
                    "steps": 3,
                    "invalid_action_count": 1,
                    "parse_failure_count": 2,
                    "failure_reason": "parse_error",
                },
            ]
        )

        self.assertEqual(metrics["episode_count"], 3)
        self.assertAlmostEqual(metrics["success_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["average_reward"], 0.5)
        self.assertAlmostEqual(metrics["average_steps"], 5.0)
        self.assertEqual(metrics["invalid_action_count"], 4)
        self.assertEqual(metrics["parse_failure_count"], 3)
        self.assertEqual(
            metrics["most_common_failure_reasons"],
            [
                {"reason": "parse_error", "count": 2},
                {"reason": "timeout", "count": 1},
            ],
        )

    def test_fake_runner_path_runs_without_task2(self) -> None:
        runner = FakeEpisodeRunner(
            [
                {
                    "success": True,
                    "reward": 1.0,
                    "steps": 4,
                    "invalid_action_count": 0,
                    "parse_failure_count": 0,
                    "failure_reasons": [],
                },
                {
                    "success": False,
                    "reward": 0.0,
                    "steps": 2,
                    "invalid_action_count": 1,
                    "parse_failure_count": 1,
                    "failure_reasons": ["bad_action"],
                },
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = evaluate_single_task(
                policy=object(),
                task_id=9,
                seed=100,
                max_steps=15,
                episodes=2,
                out_dir=temp_dir,
                runner=runner,
            )

            self.assertEqual(metrics["episodes"], 2)
            self.assertAlmostEqual(metrics["success_rate"], 0.5)
            self.assertEqual(len(runner.calls), 2)
            self.assertEqual(runner.calls[0]["seed"], 100)
            self.assertEqual(runner.calls[1]["seed"], 101)

    def test_json_metrics_output_is_written(self) -> None:
        metrics = {
            "episode_count": 1,
            "success_rate": 1.0,
            "average_reward": 1.0,
            "average_steps": 3.0,
            "invalid_action_count": 0,
            "parse_failure_count": 0,
            "most_common_failure_reasons": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = write_metrics_json(metrics, temp_dir)

            self.assertTrue(metrics_path.exists())
            loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, metrics)

    def test_script_functions_do_not_require_task2_module(self) -> None:
        self.assertIsNone(try_load_task2_runner())

        with tempfile.TemporaryDirectory() as temp_dir:
            metrics = evaluate_single_task(
                policy=None,
                task_id=3,
                seed=7,
                max_steps=10,
                episodes=1,
                out_dir=temp_dir,
                use_fake_runner=True,
            )

            self.assertEqual(metrics["episodes"], 1)
            self.assertEqual(metrics["task_id"], 3)
            self.assertIn("metrics.json", str(Path(temp_dir) / "metrics.json"))


if __name__ == "__main__":
    unittest.main()
