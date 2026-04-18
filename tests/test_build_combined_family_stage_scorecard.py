from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_combined_family_stage_scorecard import (
    build_combined_family_stage_scorecard,
    validate_combined_family_stage_scorecard,
)


class BuildCombinedFamilyStageScorecardTests(unittest.TestCase):
    def test_builds_union_scorecard_with_consistent_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            split_manifest = tmp_path / "split.json"
            split_manifest.write_text(
                json.dumps(
                    {
                        "family_name": "bootstrap44",
                        "split_preview": {
                            "family_name": "bootstrap44",
                            "task_ids": [0, 1, 12, 13],
                            "warmup_task_ids": [0, 1],
                            "grpo_task_ids": [12],
                            "holdout_task_ids": [13],
                            "eval_task_ids": [0, 1, 12, 13],
                            "split_seed": 42,
                        },
                    }
                ),
                encoding="utf-8",
            )
            source_a = tmp_path / "source_a.json"
            source_b = tmp_path / "source_b.json"
            source_a.write_text(
                json.dumps(
                    {
                        "family_name": "bootstrap44",
                        "current_stack": {
                            "family_success_rate": 1.0,
                            "per_task_success_rate": {"0": 1.0, "1": 1.0, "12": 1.0},
                        },
                    }
                ),
                encoding="utf-8",
            )
            source_b.write_text(
                json.dumps(
                    {
                        "family_name": "bootstrap44",
                        "current_stack": {
                            "family_success_rate": 0.5,
                            "per_task_success_rate": {"12": 1.0, "13": 0.0},
                        },
                    }
                ),
                encoding="utf-8",
            )

            scorecard, audit = build_combined_family_stage_scorecard(
                combined_split_manifest_path=split_manifest,
                label="current_stack",
                source_specs={
                    "a": (source_a, "current_stack"),
                    "b": (source_b, "current_stack"),
                },
            )

            self.assertEqual(scorecard["family_name"], "bootstrap44")
            self.assertAlmostEqual(scorecard["current_stack"]["family_success_rate"], 0.75)
            self.assertEqual(scorecard["current_stack"]["per_task_success_rate"]["13"], 0.0)
            self.assertEqual(scorecard["current_stack_meta"]["sources"]["a"]["matched_task_ids"], [0, 1, 12])
            self.assertEqual(scorecard["current_stack_meta"]["overlap_task_ids"], {"12": ["a", "b"]})
            self.assertEqual(audit["consistent_overlap_task_ids"], [12])

            errors, derived = validate_combined_family_stage_scorecard(scorecard)
            self.assertEqual(errors, [])
            self.assertAlmostEqual(derived["stage_metrics"]["current_stack"]["family_success_rate"], 0.75)
