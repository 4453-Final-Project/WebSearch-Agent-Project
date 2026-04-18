from __future__ import annotations

import unittest


from scripts.run_family_curriculum import _build_stage_summary
from src.training.task_families import get_family_task_groups
from scripts.audit_subset_family_stage_scorecard import audit_subset_family_stage_scorecard


class AuditSubsetFamilyStageScorecardTests(unittest.TestCase):
    def test_audit_reports_omitted_source_tasks_and_unresolved_subset_tasks(self) -> None:
        stage_summary = _build_stage_summary({"128": 1.0, "131": 0.0}, get_family_task_groups("shopping_order"))
        scorecard = {
            "scorecard_type": "subset_family_stage_scorecard",
            "family_name": "shopping_order",
            "task_count": 2,
            "training_task_count": 1,
            "holdout_task_count": 1,
            "split": {
                "family_name": "shopping_order",
                "task_ids": [128, 131],
                "warmup_task_ids": [128],
                "grpo_task_ids": [],
                "holdout_task_ids": [131],
                "eval_task_ids": [128, 131],
                "split_seed": 42,
            },
            "current_stack": stage_summary,
            "current_stack_meta": {
                "source_summary": "C:\\source.json",
                "source_stage": "current_stack",
                "source_family_name": "shopping_exact",
                "matched_task_count": 2,
                "matched_task_ids": [128, 131],
                "omitted_source_task_count": 1,
                "omitted_source_task_ids": [324],
            },
        }

        errors, audit = audit_subset_family_stage_scorecard(scorecard, benchmark_blockers=["131"])

        self.assertEqual(errors, [])
        self.assertEqual(audit["omitted_source_task_ids"], [324])
        self.assertEqual(audit["unresolved_task_ids"], [131])


if __name__ == "__main__":
    unittest.main()
