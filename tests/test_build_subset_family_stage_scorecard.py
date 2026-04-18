from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


from scripts.build_subset_family_stage_scorecard import build_subset_family_stage_scorecard


class BuildSubsetFamilyStageScorecardTests(unittest.TestCase):
    def test_builds_subset_scorecard_from_source_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path = tmp_path / "shopping_order_manifest.json"
            source_summary_path = tmp_path / "shopping_exact_current_stack_summary.json"

            manifest_path.write_text(
                json.dumps(
                    {
                        "family_name": "shopping_order",
                        "split_preview": {
                            "family_name": "shopping_order",
                            "task_ids": [128, 131],
                            "warmup_task_ids": [128],
                            "grpo_task_ids": [],
                            "holdout_task_ids": [131],
                            "eval_task_ids": [128, 131],
                            "split_seed": 42,
                        },
                    }
                ),
                encoding="utf-8",
            )
            source_summary_path.write_text(
                json.dumps(
                    {
                        "family_name": "shopping_exact",
                        "current_stack": {
                            "per_task_success_rate": {
                                "128": 1.0,
                                "131": 0.0,
                                "324": 1.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            scorecard = build_subset_family_stage_scorecard(
                subset_split_manifest_path=manifest_path,
                source_summary_path=source_summary_path,
                source_stage="current_stack",
                label="current_stack",
                benchmark_blockers=("131",),
            )

        self.assertEqual(scorecard["scorecard_type"], "subset_family_stage_scorecard")
        self.assertEqual(scorecard["family_name"], "shopping_order")
        self.assertEqual(scorecard["current_stack"]["family_success_rate"], 0.5)
        self.assertEqual(scorecard["current_stack"]["per_task_success_rate"], {"128": 1.0, "131": 0.0})
        self.assertEqual(scorecard["current_stack_meta"]["matched_task_ids"], [128, 131])
        self.assertEqual(scorecard["current_stack_meta"]["omitted_source_task_ids"], [324])


if __name__ == "__main__":
    unittest.main()
