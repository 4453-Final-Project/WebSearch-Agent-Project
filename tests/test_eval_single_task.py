"""Unit tests for the single-task evaluation scaffold."""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval_single_task import (
    FakeEpisodeRunner,
    aggregate_metrics,
    evaluate_single_task,
    format_metrics_text,
    load_policy,
    try_load_task2_runner,
    write_metrics_json,
    write_metrics_text,
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

    def test_text_metrics_output_is_written(self) -> None:
        metrics = {
            "episode_count": 1,
            "success_rate": 1.0,
            "average_reward": 1.0,
            "average_steps": 3.0,
            "invalid_action_count": 0,
            "parse_failure_count": 0,
            "most_common_failure_reasons": [],
            "task_id": 310,
            "seed": 42,
            "episodes": 1,
            "max_steps": 4,
            "policy": "qwen",
            "out_dir": "tmp",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = write_metrics_text(metrics, temp_dir)
            self.assertTrue(text_path.exists())
            content = text_path.read_text(encoding="utf-8")
            self.assertIn("Single-Task Evaluation Summary", content)
            self.assertIn("Success Rate: 1.000", content)

    def test_task2_runner_is_available_on_merged_main(self) -> None:
        self.assertTrue(callable(try_load_task2_runner()))

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

    def test_load_policy_supports_dummy(self) -> None:
        policy = load_policy("dummy")
        self.assertTrue(hasattr(policy, "act"))

    def test_load_policy_uses_model_dir_name_to_resolve_path_and_profile(self) -> None:
        captured: dict[str, object] = {}

        class FakePolicy:
            def __init__(self, config) -> None:
                captured["config"] = config

        module = importlib.import_module("scripts.eval_single_task")
        with patch.object(module, "import_module", return_value=SimpleNamespace(QwenPolicy=FakePolicy)):
            policy = load_policy("qwen", model_dir_name="LFM2.5-350M")

        self.assertIsInstance(policy, FakePolicy)
        config = captured["config"]
        self.assertEqual(Path(config.model_path).name, "LFM2.5-350M")
        self.assertEqual(config.policy_name, "lfm2.5-350m")

    def test_load_policy_infers_profile_from_adapter_path(self) -> None:
        captured: dict[str, object] = {}

        class FakePolicy:
            def __init__(self, config) -> None:
                captured["config"] = config

        with tempfile.TemporaryDirectory() as temp_dir:
            adapter_dir = Path(temp_dir) / "adapter"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "/tmp/models/LFM2.5-350M"}),
                encoding="utf-8",
            )
            module = importlib.import_module("scripts.eval_single_task")
            with patch.object(module, "import_module", return_value=SimpleNamespace(QwenPolicy=FakePolicy)):
                policy = load_policy("qwen", model_path=str(adapter_dir))

        self.assertIsInstance(policy, FakePolicy)
        config = captured["config"]
        self.assertEqual(config.model_path, str(adapter_dir))
        self.assertEqual(config.policy_name, "lfm2.5-350m")


if __name__ == "__main__":
    unittest.main()
