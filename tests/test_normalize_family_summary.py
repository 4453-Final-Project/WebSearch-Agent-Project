from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.normalize_family_summary import normalize_family_summary  # noqa: E402


def _build_legacy_exact_summary() -> dict[str, object]:
    return {
        "family_name": "shopping_exact",
        "task_count": 3,
        "training_task_count": 2,
        "holdout_task_count": 1,
        "split": {
            "family_name": "shopping_exact",
            "task_ids": [128, 324, 325],
            "warmup_task_ids": [325],
            "grpo_task_ids": [128],
            "holdout_task_ids": [324],
            "eval_task_ids": [128, 324, 325],
            "split_seed": 42,
        },
        "baseline": {
            "family_success_rate": 2 / 3,
            "per_task_success_rate": {"128": 1.0, "324": 0.0, "325": 1.0},
        },
        "warmup_only": {
            "family_success_rate": 2 / 3,
            "per_task_success_rate": {"128": 1.0, "324": 0.0, "325": 1.0},
        },
        "warmup_plus_grpo": {
            "family_success_rate": 1.0,
            "per_task_success_rate": {"128": 1.0, "324": 1.0, "325": 1.0},
        },
        "holdout": {
            "task_ids": [324],
            "baseline_success_rate": 0.0,
            "warmup_success_rate": 0.0,
            "grpo_success_rate": 1.0,
            "gain_vs_baseline": 1.0,
            "gain_vs_warmup": 1.0,
            "per_task": {
                "324": {
                    "baseline_success_rate": 0.0,
                    "warmup_success_rate": 0.0,
                    "grpo_success_rate": 1.0,
                    "gain_vs_baseline": 1.0,
                    "gain_vs_warmup": 1.0,
                }
            },
        },
    }


class NormalizeFamilySummaryTests(unittest.TestCase):
    def test_normalize_family_summary_adds_family_metadata_and_task_groups(self) -> None:
        normalized = normalize_family_summary(_build_legacy_exact_summary())

        self.assertEqual(normalized["family_metadata"]["judge_free_task_count"], 3)
        self.assertEqual(normalized["family_metadata"]["judge_gated_task_count"], 0)
        self.assertIn("task_groups", normalized["baseline"])
        self.assertEqual(normalized["baseline"]["task_groups"]["judge_free"]["task_ids"], [128, 324, 325])

    def test_normalize_family_summary_recomputes_custom_stage_holdout(self) -> None:
        summary = _build_legacy_exact_summary()
        summary["current_stack"] = {
            "family_success_rate": 1.0,
            "per_task_success_rate": {"128": 1.0, "324": 1.0, "325": 1.0},
            "applied_overrides": {
                "324": {"metrics_path": "fake.json", "success_rate": 1.0, "episodes": 1, "average_steps": 2.0}
            },
        }

        normalized = normalize_family_summary(summary)

        self.assertEqual(normalized["current_stack"]["applied_overrides"]["324"]["metrics_path"], "fake.json")
        self.assertEqual(normalized["current_stack_holdout"]["grpo_success_rate"], 1.0)
        self.assertEqual(normalized["current_stack_holdout"]["gain_vs_baseline"], 1.0)

    def test_normalize_family_summary_backfills_run_provenance_from_legacy_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            summary = _build_legacy_exact_summary()
            manifest_path = tmp_path / "split_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "family_name": "shopping_exact",
                        "task_ids": [128, 324, 325],
                        "warmup_task_ids": [325],
                        "grpo_task_ids": [128],
                        "holdout_task_ids": [324],
                        "eval_task_ids": [128, 324, 325],
                        "split_seed": 42,
                    }
                ),
                encoding="utf-8",
            )
            summary["split_manifest"] = str(manifest_path)

            normalized = normalize_family_summary(summary)

        self.assertEqual(normalized["run_provenance"]["split_manifest_path"], str(manifest_path))
        self.assertEqual(normalized["run_provenance"]["split_provenance"]["split_source"], "legacy_manifest")
        self.assertTrue(normalized["run_provenance"]["split_provenance"]["inferred_from_legacy_artifact"])
        self.assertFalse(normalized["run_provenance"]["recommended_split_alignment"]["matches_recommended_split"])

    def test_normalize_family_summary_resolves_mnt_split_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            summary = _build_legacy_exact_summary()
            manifest_path = tmp_path / "split_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "family_name": "shopping_exact",
                        "split_provenance": {"split_source": "manifest", "split_seed": 42},
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
                            "family_name": "shopping_exact",
                            "task_ids": [128, 324, 325],
                            "warmup_task_ids": [325],
                            "grpo_task_ids": [128],
                            "holdout_task_ids": [324],
                            "eval_task_ids": [128, 324, 325],
                            "split_seed": 42,
                        },
                    }
                ),
                encoding="utf-8",
            )
            drive_letter = manifest_path.drive.rstrip(":\\").lower()
            relative_parts = manifest_path.relative_to(manifest_path.anchor).parts
            summary["split_manifest"] = "/" + "/".join(["mnt", drive_letter, *relative_parts])

            normalized = normalize_family_summary(summary)

        self.assertEqual(normalized["run_provenance"]["split_provenance"]["split_source"], "manifest")
        self.assertEqual(normalized["run_provenance"]["judge_requirements"]["run_ready"], True)


if __name__ == "__main__":
    unittest.main()
