from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_family_training_progress import audit_family_training_progress  # noqa: E402


def _build_summary() -> dict[str, object]:
    baseline = {"96": 0.0, "128": 1.0, "324": 0.0}
    warmup = {"96": 1.0, "128": 1.0, "324": 0.0}
    grpo = {"96": 1.0, "128": 0.0, "324": 1.0}
    summary = {
        "family_name": "shopping_full",
        "task_count": 3,
        "training_task_count": 2,
        "holdout_task_count": 1,
        "split": {
            "eval_task_ids": [96, 128, 324],
            "warmup_task_ids": [128],
            "grpo_task_ids": [96],
            "holdout_task_ids": [324],
            "split_seed": 42,
        },
        "baseline": {
            "family_success_rate": 1 / 3,
            "per_task_success_rate": copy.deepcopy(baseline),
        },
        "warmup_only": {
            "family_success_rate": 2 / 3,
            "per_task_success_rate": copy.deepcopy(warmup),
        },
        "warmup_plus_grpo": {
            "family_success_rate": 2 / 3,
            "per_task_success_rate": copy.deepcopy(grpo),
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
    for stage_name in ("baseline", "warmup_only", "warmup_plus_grpo"):
        per_task = summary[stage_name]["per_task_success_rate"]
        summary[stage_name]["task_groups"] = {
            "judge_free": {
                "task_ids": [128, 324],
                "task_count": 2,
                "success_rate": (float(per_task["128"]) + float(per_task["324"])) / 2,
                "per_task_success_rate": {"128": per_task["128"], "324": per_task["324"]},
            },
            "judge_gated": {
                "task_ids": [96],
                "task_count": 1,
                "success_rate": float(per_task["96"]),
                "per_task_success_rate": {"96": per_task["96"]},
            },
        }
    return summary


class AuditFamilyTrainingProgressTests(unittest.TestCase):
    def test_reports_improvements_and_regressions_between_stages(self) -> None:
        errors, report = audit_family_training_progress(
            _build_summary(),
            stage_a="baseline",
            stage_b="warmup_plus_grpo",
        )

        self.assertEqual(errors, [])
        self.assertEqual(report["family_name"], "shopping_full")
        self.assertAlmostEqual(report["family_success_rate_delta"], 1 / 3)
        self.assertEqual(report["task_progress"]["improved_task_ids"], [96, 324])
        self.assertEqual(report["task_progress"]["regressed_task_ids"], [128])
        self.assertEqual(report["task_group_progress"]["judge_gated"]["improved_task_ids"], [96])
        self.assertEqual(report["task_group_progress"]["judge_free"]["improved_task_ids"], [324])
        self.assertEqual(report["task_group_progress"]["judge_free"]["regressed_task_ids"], [128])

    def test_reports_regression_when_later_stage_drops_task(self) -> None:
        errors, report = audit_family_training_progress(
            _build_summary(),
            stage_a="warmup_only",
            stage_b="warmup_plus_grpo",
        )

        self.assertEqual(errors, [])
        self.assertEqual(report["task_progress"]["improved_task_ids"], [324])
        self.assertEqual(report["task_progress"]["regressed_task_ids"], [128])
        self.assertEqual(report["task_group_progress"]["judge_free"]["regressed_task_ids"], [128])

    def test_errors_when_stage_is_missing(self) -> None:
        errors, report = audit_family_training_progress(
            _build_summary(),
            stage_a="baseline",
            stage_b="not_a_stage",
        )

        self.assertEqual(report, {})
        self.assertEqual(errors, ["stage_b 'not_a_stage' not found in summary"])


if __name__ == "__main__":
    unittest.main()
