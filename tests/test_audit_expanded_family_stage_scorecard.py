from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_expanded_family_stage_scorecard import audit_expanded_family_stage_scorecard
from scripts.build_expanded_family_stage_scorecard import build_expanded_family_stage_scorecard


class AuditExpandedFamilyStageScorecardTests(unittest.TestCase):
    def test_reports_added_task_lift_without_inherited_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorecard = self._build_scorecard(Path(tmpdir))

            errors, report = audit_expanded_family_stage_scorecard(scorecard)

            self.assertEqual(errors, [])
            self.assertEqual(report["family_name"], "bootstrap44")
            self.assertAlmostEqual(report["expanded_family_success_rate"], 4 / 44)
            self.assertAlmostEqual(report["inherited_slice_success_rate"], 1 / 41)
            self.assertAlmostEqual(report["added_task_success_rate"], 1.0)
            self.assertEqual(report["added_success_task_ids"], [12, 13, 144])
            self.assertEqual(report["added_unresolved_task_ids"], [])
            self.assertEqual(report["inherited_progress"]["improved_task_ids"], [])
            self.assertEqual(report["inherited_progress"]["regressed_task_ids"], [])
            self.assertEqual(
                report["added_task_group_success"]["site_shopping_admin"]["added_task_ids"],
                [12, 13],
            )
            self.assertEqual(
                report["added_task_group_success"]["site_shopping"]["added_task_ids"],
                [144],
            )

    def test_reports_inherited_drift_when_scorecard_changes_base_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorecard = self._build_scorecard(Path(tmpdir))
            scorecard["current_stack"]["per_task_success_rate"]["0"] = 0.0
            scorecard["current_stack"]["family_success_rate"] = 3 / 44
            scorecard["current_stack"]["task_groups"]["judge_free"]["per_task_success_rate"]["0"] = 0.0
            judge_free_values = list(scorecard["current_stack"]["task_groups"]["judge_free"]["per_task_success_rate"].values())
            scorecard["current_stack"]["task_groups"]["judge_free"]["success_rate"] = sum(judge_free_values) / len(judge_free_values)
            scorecard["current_stack"]["task_groups"]["site_shopping_admin"]["per_task_success_rate"]["0"] = 0.0
            admin_values = list(
                scorecard["current_stack"]["task_groups"]["site_shopping_admin"]["per_task_success_rate"].values()
            )
            scorecard["current_stack"]["task_groups"]["site_shopping_admin"]["success_rate"] = (
                sum(admin_values) / len(admin_values)
            )

            errors, report = audit_expanded_family_stage_scorecard(scorecard)

            self.assertEqual(errors, [])
            self.assertEqual(report["inherited_progress"]["regressed_task_ids"], [0])
            self.assertEqual(report["expansion_summary"]["changed_inherited_task_ids"], [0])

    def _build_scorecard(self, tmp_path: Path) -> dict[str, object]:
        base_summary_path = tmp_path / "bootstrap41_current_stack_summary.json"
        bootstrap41_task_ids = [
            0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 21, 23, 25, 26, 27, 28, 29, 30, 31, 36,
            41, 66, 67, 68, 69, 70, 71, 72, 77, 124, 125, 126, 132, 133, 134, 135,
            136, 141, 188, 259, 293,
        ]
        base_summary_path.write_text(
            json.dumps(
                {
                    "family_name": "bootstrap41",
                    "run_provenance": {
                        "split_manifest_path": "bootstrap41_manifest.json",
                        "split_provenance": {"split_source": "recommended", "split_seed": 42},
                        "judge_requirements": {
                            "requires_openai_judge": False,
                            "openai_api_key_present": None,
                            "run_ready": True,
                        },
                        "recommended_split_alignment": {
                            "matches_recommended_split": True,
                            "recommended_split_seed": 42,
                            "differences": {},
                        },
                    },
                    "current_stack": {
                        "family_success_rate": 1 / 41,
                        "per_task_success_rate": {
                            str(task_id): (1.0 if task_id == 0 else 0.0)
                            for task_id in bootstrap41_task_ids
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

        overrides = {}
        for task_id, avg_steps in ((12, 1.0), (13, 1.0), (144, 2.0)):
            metrics_path = tmp_path / f"task{task_id}_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "success_rate": 1.0,
                        "episodes": 1,
                        "average_steps": avg_steps,
                    }
                ),
                encoding="utf-8",
            )
            overrides[str(task_id)] = metrics_path

        return build_expanded_family_stage_scorecard(
            expanded_split_manifest_path=PROJECT_ROOT / "scripts" / "local" / "bootstrap44_curriculum_manifest.json",
            base_summary_path=base_summary_path,
            base_stage="current_stack",
            label="current_stack",
            overrides=overrides,
        )


if __name__ == "__main__":
    unittest.main()
