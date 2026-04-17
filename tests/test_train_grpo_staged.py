from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_grpo_staged import _evaluate_adapter, _normalize_stage_task_sets


class TrainGrpoStagedTests(unittest.TestCase):
    def test_explicit_warmup_tasks_must_be_disjoint_by_default(self) -> None:
        args = Namespace(
            task_ids=[325, 326],
            warmup_task_ids=[324, 325],
            eval_task_ids=[],
            holdout_task_ids=[],
            allow_task_overlap=False,
        )

        with self.assertRaises(SystemExit):
            _normalize_stage_task_sets(args)

    def test_task_sets_default_to_grpo_tasks_when_omitted(self) -> None:
        args = Namespace(
            task_ids=[325, 326, 326],
            warmup_task_ids=[],
            eval_task_ids=[],
            holdout_task_ids=[],
            allow_task_overlap=False,
        )

        _normalize_stage_task_sets(args)

        self.assertEqual(args.task_ids, [325, 326])
        self.assertEqual(args.warmup_task_ids, [325, 326])
        self.assertEqual(args.eval_task_ids, [325, 326])
        self.assertEqual(args.holdout_task_ids, [])

    def test_explicit_disjoint_warmup_and_eval_sets_are_preserved(self) -> None:
        args = Namespace(
            task_ids=[327, 328],
            warmup_task_ids=[325, 326],
            eval_task_ids=[324, 329],
            holdout_task_ids=[],
            allow_task_overlap=False,
        )

        _normalize_stage_task_sets(args)

        self.assertEqual(args.task_ids, [327, 328])
        self.assertEqual(args.warmup_task_ids, [325, 326])
        self.assertEqual(args.eval_task_ids, [324, 329])
        self.assertEqual(args.holdout_task_ids, [324, 329])

    def test_explicit_holdout_tasks_must_be_disjoint_from_training_sets(self) -> None:
        args = Namespace(
            task_ids=[327, 328],
            warmup_task_ids=[325, 326],
            eval_task_ids=[324, 327],
            holdout_task_ids=[327],
            allow_task_overlap=False,
        )

        with self.assertRaises(SystemExit):
            _normalize_stage_task_sets(args)

    def test_explicit_holdout_tasks_must_be_included_in_eval_set(self) -> None:
        args = Namespace(
            task_ids=[327, 328],
            warmup_task_ids=[325, 326],
            eval_task_ids=[325, 326, 327, 328],
            holdout_task_ids=[324],
            allow_task_overlap=False,
        )

        with self.assertRaises(SystemExit):
            _normalize_stage_task_sets(args)

    def test_evaluate_adapter_reuses_existing_metrics_and_only_runs_missing_tasks(self) -> None:
        args = Namespace(
            eval_task_ids=[101, 102],
            seed=7,
            max_steps=4,
            eval_episodes=2,
            headless=True,
            eval_temperature=0.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            task_101_dir = output_root / "task_101"
            task_101_dir.mkdir(parents=True)
            (task_101_dir / "metrics.json").write_text('{"success_rate": 0.5}', encoding="utf-8")

            with (
                patch("scripts.train_grpo_staged._build_policy", return_value="policy") as build_policy,
                patch(
                    "scripts.train_grpo_staged.evaluate_single_task",
                    return_value={"success_rate": 1.0},
                ) as evaluate_single_task_mock,
            ):
                results = _evaluate_adapter(args, Path("adapter"), output_root)

        self.assertEqual(results, {"101": {"success_rate": 0.5}, "102": {"success_rate": 1.0}})
        build_policy.assert_called_once_with(args, model_path="adapter", temperature=0.0)
        evaluate_single_task_mock.assert_called_once()
        self.assertEqual(evaluate_single_task_mock.call_args.kwargs["task_id"], 102)


if __name__ == "__main__":
    unittest.main()
