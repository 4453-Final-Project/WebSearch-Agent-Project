from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_combined_family_stage_scorecard import audit_combined_family_stage_scorecard
from scripts.build_combined_family_stage_scorecard import build_combined_family_stage_scorecard


class AuditCombinedFamilyStageScorecardTests(unittest.TestCase):
    def test_reports_overlap_consistency_and_unique_contributions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            scorecard = self._build_scorecard(Path(tmp_dir))

            errors, report = audit_combined_family_stage_scorecard(scorecard)

            self.assertEqual(errors, [])
            self.assertEqual(report["family_name"], "bootstrap44")
            self.assertAlmostEqual(report["scorecard_success_rate"], 0.75)
            self.assertEqual(report["consistent_overlap_task_ids"], [12])
            self.assertEqual(report["source_contributions"]["a"]["unique_task_ids"], [0, 1])
            self.assertEqual(report["source_contributions"]["b"]["unique_task_ids"], [13])
            self.assertTrue(report["overlap_consistency"]["12"]["consistent_with_merged_score"])

    def _build_scorecard(self, tmp_path: Path) -> dict[str, object]:
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
        scorecard, _ = build_combined_family_stage_scorecard(
            combined_split_manifest_path=split_manifest,
            label="current_stack",
            source_specs={
                "a": (source_a, "current_stack"),
                "b": (source_b, "current_stack"),
            },
        )
        return scorecard


if __name__ == "__main__":
    unittest.main()
