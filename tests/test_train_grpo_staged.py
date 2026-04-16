from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_grpo_staged import _normalize_stage_task_sets


class TrainGrpoStagedTests(unittest.TestCase):
    def test_explicit_warmup_tasks_must_be_disjoint_by_default(self) -> None:
        args = Namespace(
            task_ids=[325, 326],
            warmup_task_ids=[324, 325],
            eval_task_ids=[],
            allow_task_overlap=False,
        )

        with self.assertRaises(SystemExit):
            _normalize_stage_task_sets(args)

    def test_task_sets_default_to_grpo_tasks_when_omitted(self) -> None:
        args = Namespace(
            task_ids=[325, 326, 326],
            warmup_task_ids=[],
            eval_task_ids=[],
            allow_task_overlap=False,
        )

        _normalize_stage_task_sets(args)

        self.assertEqual(args.task_ids, [325, 326])
        self.assertEqual(args.warmup_task_ids, [325, 326])
        self.assertEqual(args.eval_task_ids, [325, 326])

    def test_explicit_disjoint_warmup_and_eval_sets_are_preserved(self) -> None:
        args = Namespace(
            task_ids=[327, 328],
            warmup_task_ids=[325, 326],
            eval_task_ids=[324, 329],
            allow_task_overlap=False,
        )

        _normalize_stage_task_sets(args)

        self.assertEqual(args.task_ids, [327, 328])
        self.assertEqual(args.warmup_task_ids, [325, 326])
        self.assertEqual(args.eval_task_ids, [324, 329])


if __name__ == "__main__":
    unittest.main()
