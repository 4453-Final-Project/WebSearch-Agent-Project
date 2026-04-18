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
from scripts.validate_expanded_family_stage_scorecard import validate_expanded_family_stage_scorecard


class ValidateExpandedFamilyStageScorecardTests(unittest.TestCase):
    def test_accepts_valid_bootstrap44_scorecard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorecard = self._build_scorecard(Path(tmpdir))
            errors, derived = validate_expanded_family_stage_scorecard(scorecard)

            self.assertEqual(errors, [])
            self.assertEqual(derived["family_name"], "bootstrap44")
            self.assertEqual(derived["label"], "current_stack")
            self.assertAlmostEqual(derived["stage_metrics"]["current_stack"]["family_success_rate"], 4 / 44)
            self.assertEqual(
                derived["stage_meta"]["current_stack_meta"]["added_task_ids"],
                [12, 13, 144],
            )

    def test_rejects_overlap_between_inherited_and_added_task_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scorecard = self._build_scorecard(Path(tmpdir))
            scorecard["current_stack_meta"]["added_task_ids"] = [0, 12, 13, 144]

            errors, _ = validate_expanded_family_stage_scorecard(scorecard)

            self.assertTrue(
                any("inherited and added task ids overlap" in error for error in errors),
                errors,
            )

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
        for task_id in (12, 13, 144):
            metrics_path = tmp_path / f"task{task_id}_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "task_id": task_id,
                        "success_rate": 1.0,
                        "episodes": 1,
                        "average_steps": 1.0,
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
