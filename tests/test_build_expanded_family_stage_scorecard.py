from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_expanded_family_stage_scorecard import build_expanded_family_stage_scorecard
from scripts.build_expanded_family_stage_scorecard import (
    validate_and_audit_expanded_family_stage_scorecard,
)


class BuildExpandedFamilyStageScorecardTests(unittest.TestCase):
    def test_builds_bootstrap44_scorecard_from_bootstrap41_stage_and_added_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
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

            override_task_12 = tmp_path / "task12_metrics.json"
            override_task_12.write_text(
                json.dumps({"task_id": 12, "success_rate": 1.0, "episodes": 1, "average_steps": 1.0}),
                encoding="utf-8",
            )
            override_task_13 = tmp_path / "task13_metrics.json"
            override_task_13.write_text(
                json.dumps({"task_id": 13, "success_rate": 1.0, "episodes": 1, "average_steps": 1.0}),
                encoding="utf-8",
            )
            override_task_144 = tmp_path / "task144_metrics.json"
            override_task_144.write_text(
                json.dumps({"task_id": 144, "success_rate": 1.0, "episodes": 1, "average_steps": 2.0}),
                encoding="utf-8",
            )

            scorecard = build_expanded_family_stage_scorecard(
                expanded_split_manifest_path=PROJECT_ROOT / "scripts" / "local" / "bootstrap44_curriculum_manifest.json",
                base_summary_path=base_summary_path,
                base_stage="current_stack",
                label="current_stack",
                overrides={"12": override_task_12, "13": override_task_13, "144": override_task_144},
            )

            self.assertEqual(scorecard["family_name"], "bootstrap44")
            self.assertAlmostEqual(scorecard["current_stack"]["family_success_rate"], 4 / 44)
            self.assertEqual(scorecard["current_stack"]["per_task_success_rate"]["0"], 1.0)
            self.assertEqual(scorecard["current_stack"]["per_task_success_rate"]["12"], 1.0)
            self.assertEqual(scorecard["current_stack"]["per_task_success_rate"]["13"], 1.0)
            self.assertEqual(scorecard["current_stack"]["per_task_success_rate"]["144"], 1.0)
            self.assertEqual(scorecard["current_stack_meta"]["added_task_ids"], [12, 13, 144])
            self.assertEqual(scorecard["current_stack_meta"]["inherited_task_count"], 41)
            self.assertEqual(scorecard["current_stack"]["task_groups"]["site_shopping_admin"]["per_task_success_rate"]["12"], 1.0)
            self.assertEqual(scorecard["current_stack"]["task_groups"]["site_shopping_admin"]["per_task_success_rate"]["13"], 1.0)
            self.assertEqual(scorecard["current_stack"]["task_groups"]["site_shopping"]["per_task_success_rate"]["144"], 1.0)

    def test_builder_can_validate_and_emit_matching_audit_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
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

            scorecard = build_expanded_family_stage_scorecard(
                expanded_split_manifest_path=PROJECT_ROOT / "scripts" / "local" / "bootstrap44_curriculum_manifest.json",
                base_summary_path=base_summary_path,
                base_stage="current_stack",
                label="current_stack",
                overrides=overrides,
            )
            errors, derived, audit_report = validate_and_audit_expanded_family_stage_scorecard(scorecard)

            self.assertEqual(errors, [])
            self.assertEqual(derived["family_name"], "bootstrap44")
            self.assertIsNotNone(audit_report)
            assert audit_report is not None
            self.assertEqual(audit_report["added_success_task_ids"], [12, 13, 144])
            self.assertEqual(audit_report["expansion_summary"]["changed_inherited_task_ids"], [])
            self.assertAlmostEqual(audit_report["added_task_success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
