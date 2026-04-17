from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_family_reevals import (
    build_label_meta,
    build_merged_stage,
    build_stage_holdout_summary,
)


class MergeFamilyReevalsTests(unittest.TestCase):
    def test_build_label_meta_records_override_and_base_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metrics_path = tmp_path / "task_200_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "success_rate": 1.0,
                        "episodes": 1,
                        "average_steps": 3.0,
                    }
                ),
                encoding="utf-8",
            )
            base_summary = {
                "run_provenance": {
                    "split_provenance": {
                        "source": "split_manifest",
                        "path": "scripts/local/shopping_exact_curriculum_manifest.json",
                    }
                },
                "warmup_plus_grpo": {
                    "family_success_rate": 0.5,
                    "per_task_success_rate": {
                        "131": 0.0,
                        "200": 0.5,
                        "324": 1.0,
                    },
                },
            }
            merged_stage = build_merged_stage(
                base_summary,
                base_stage="warmup_plus_grpo",
                overrides={"200": metrics_path},
            )

            meta = build_label_meta(
                base_summary,
                base_summary_path=Path("base_summary.json"),
                base_stage="warmup_plus_grpo",
                benchmark_blockers=["131"],
                overrides={"200": metrics_path},
                merged_stage=merged_stage,
            )

        self.assertEqual(meta["override_count"], 1)
        self.assertEqual(meta["override_task_ids"], [200])
        self.assertEqual(meta["override_metrics_paths"]["200"], str(metrics_path))
        self.assertEqual(
            meta["base_run_provenance"]["split_provenance"]["path"],
            "scripts/local/shopping_exact_curriculum_manifest.json",
        )
        self.assertEqual(meta["applied_override_metrics"]["200"]["success_rate"], 1.0)
        self.assertEqual(meta["applied_override_metrics"]["200"]["average_steps"], 3.0)

    def test_build_merged_stage_overrides_selected_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metrics_path = tmp_path / "task_200_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "success_rate": 1.0,
                        "episodes": 1,
                        "average_steps": 3.0,
                    }
                ),
                encoding="utf-8",
            )
            base_summary = {
                "warmup_plus_grpo": {
                    "family_success_rate": 0.5,
                    "per_task_success_rate": {
                        "131": 0.0,
                        "200": 0.5,
                        "324": 1.0,
                    },
                }
            }

            merged = build_merged_stage(
                base_summary,
                base_stage="warmup_plus_grpo",
                overrides={"200": metrics_path},
            )

            self.assertAlmostEqual(merged["family_success_rate"], (0.0 + 1.0 + 1.0) / 3)
            self.assertEqual(
                merged["per_task_success_rate"],
                {"131": 0.0, "200": 1.0, "324": 1.0},
            )
            self.assertEqual(
                merged["applied_overrides"]["200"]["metrics_path"],
                str(metrics_path),
            )
            self.assertEqual(merged["applied_overrides"]["200"]["average_steps"], 3.0)

    def test_build_merged_stage_recomputes_task_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            metrics_path = tmp_path / "task_96_metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "success_rate": 1.0,
                        "episodes": 1,
                        "average_steps": 2.0,
                    }
                ),
                encoding="utf-8",
            )
            base_summary = {
                "family_metadata": {
                    "judge_free_task_ids": [128, 324],
                    "judge_gated_task_ids": [96],
                },
                "warmup_plus_grpo": {
                    "family_success_rate": 2 / 3,
                    "per_task_success_rate": {
                        "96": 0.0,
                        "128": 1.0,
                        "324": 1.0,
                    },
                    "task_groups": {
                        "judge_free": {
                            "task_ids": [128, 324],
                            "task_count": 2,
                            "success_rate": 1.0,
                            "per_task_success_rate": {"128": 1.0, "324": 1.0},
                        },
                        "judge_gated": {
                            "task_ids": [96],
                            "task_count": 1,
                            "success_rate": 0.0,
                            "per_task_success_rate": {"96": 0.0},
                        },
                    },
                },
            }

            merged = build_merged_stage(
                base_summary,
                base_stage="warmup_plus_grpo",
                overrides={"96": metrics_path},
            )

            self.assertEqual(merged["task_groups"]["judge_gated"]["success_rate"], 1.0)
            self.assertEqual(merged["task_groups"]["judge_gated"]["per_task_success_rate"], {"96": 1.0})

    def test_build_stage_holdout_summary_recomputes_holdout_after_override(self) -> None:
        base_summary = {
            "split": {
                "holdout_task_ids": [200, 324],
            },
            "baseline": {
                "per_task_success_rate": {
                    "200": 0.5,
                    "324": 0.5,
                }
            },
            "warmup_only": {
                "per_task_success_rate": {
                    "200": 0.5,
                    "324": 1.0,
                }
            },
        }

        holdout = build_stage_holdout_summary(
            base_summary,
            stage_per_task_success_rate={"200": 1.0, "324": 1.0},
        )

        self.assertEqual(holdout["task_ids"], [200, 324])
        self.assertEqual(holdout["grpo_success_rate"], 1.0)
        self.assertEqual(holdout["gain_vs_baseline"], 0.5)
        self.assertEqual(holdout["gain_vs_warmup"], 0.25)
        self.assertEqual(holdout["per_task"]["200"]["gain_vs_baseline"], 0.5)


if __name__ == "__main__":
    unittest.main()
