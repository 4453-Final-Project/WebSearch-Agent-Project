from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.build_family_override_summary import build_family_override_summary
from scripts.validate_family_summary import validate_family_summary


class BuildFamilyOverrideSummaryTests(unittest.TestCase):
    def test_builds_valid_family_summary_with_current_stack_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            split_manifest_path = tmp_path / "split_manifest.json"
            split_manifest_path.write_text(
                json.dumps(
                    {
                        "family_name": "bootstrap41",
                        "split_provenance": {
                            "split_source": "manifest",
                            "split_seed": 42,
                            "source_manifest": str(split_manifest_path),
                        },
                        "judge_requirements": {
                            "requires_openai_judge": False,
                            "openai_api_key_present": False,
                            "run_ready": True,
                        },
                        "recommended_split_alignment": {
                            "matches_recommended_split": True,
                            "recommended_split_seed": 42,
                            "differences": {},
                        },
                        "split_preview": {
                            "family_name": "bootstrap41",
                            "task_ids": [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 21, 23, 25, 26, 27, 28, 29, 30, 31, 36, 41, 66, 67, 68, 69, 70, 71, 72, 77, 124, 125, 126, 132, 133, 134, 135, 136, 141, 188, 259, 293],
                            "warmup_task_ids": [0, 1, 11, 21, 23, 124, 27, 28, 66, 132, 133, 293, 7, 9, 70],
                            "grpo_task_ids": [2, 3, 4, 5, 25, 26, 125, 126, 29, 30, 31, 67, 134, 135, 136, 10, 36, 72],
                            "holdout_task_ids": [41, 77, 141, 188, 68, 69, 259, 71],
                            "eval_task_ids": [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 21, 23, 25, 26, 27, 28, 29, 30, 31, 36, 41, 66, 67, 68, 69, 70, 71, 72, 77, 124, 125, 126, 132, 133, 134, 135, 136, 141, 188, 259, 293],
                            "split_seed": 42,
                        },
                    }
                ),
                encoding="utf-8",
            )
            baseline_path = tmp_path / "baseline_eval_summary.json"
            baseline_payload = {
                str(task_id): {
                    "task_id": task_id,
                    "success_rate": 1.0 if task_id in {1, 2, 4, 5, 7, 9, 10, 36, 70, 71, 72, 188} else 0.0
                }
                for task_id in [0, 1, 2, 3, 4, 5, 7, 9, 10, 11, 21, 23, 25, 26, 27, 28, 29, 30, 31, 36, 41, 66, 67, 68, 69, 70, 71, 72, 77, 124, 125, 126, 132, 133, 134, 135, 136, 141, 188, 259, 293]
            }
            baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

            override_task_21 = tmp_path / "task21_metrics.json"
            override_task_21.write_text(json.dumps({"task_id": 21, "success_rate": 1.0, "episodes": 1, "average_steps": 1.0}), encoding="utf-8")
            override_task_132 = tmp_path / "task132_metrics.json"
            override_task_132.write_text(json.dumps({"task_id": 132, "success_rate": 1.0, "episodes": 1, "average_steps": 5.0}), encoding="utf-8")

            summary = build_family_override_summary(
                split_manifest_path=split_manifest_path,
                baseline_eval_summary_path=baseline_path,
                label="current_stack",
                overrides={"21": override_task_21, "132": override_task_132},
            )

            self.assertAlmostEqual(summary["baseline"]["family_success_rate"], 12 / 41)
            self.assertAlmostEqual(summary["current_stack"]["family_success_rate"], 14 / 41)
            self.assertEqual(summary["current_stack"]["per_task_success_rate"]["21"], 1.0)
            self.assertEqual(summary["current_stack"]["per_task_success_rate"]["132"], 1.0)
            self.assertEqual(summary["current_stack_meta"]["override_task_ids"], [21, 132])
            self.assertEqual(summary["current_stack"]["task_groups"]["site_shopping"]["per_task_success_rate"]["21"], 1.0)
            self.assertEqual(summary["current_stack"]["task_groups"]["site_gitlab"]["per_task_success_rate"]["132"], 1.0)

            errors, derived = validate_family_summary(summary)
            self.assertEqual(errors, [])
            self.assertAlmostEqual(derived["stage_metrics"]["current_stack"]["family_success_rate"], 14 / 41)


if __name__ == "__main__":
    unittest.main()
