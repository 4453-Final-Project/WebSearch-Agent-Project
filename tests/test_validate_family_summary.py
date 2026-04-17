from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.validate_family_summary import validate_family_summary  # noqa: E402


def _build_exact_summary() -> dict[str, object]:
    per_task = {"128": 1.0, "131": 0.0, "324": 1.0}
    return {
        "family_name": "shopping_exact",
        "task_count": 3,
        "training_task_count": 2,
        "holdout_task_count": 1,
        "split": {
            "family_name": "shopping_exact",
            "task_ids": [128, 131, 324],
            "eval_task_ids": [128, 131, 324],
            "warmup_task_ids": [128],
            "grpo_task_ids": [131],
            "holdout_task_ids": [324],
            "split_seed": 42,
        },
        "baseline": {
            "family_success_rate": 2 / 3,
            "per_task_success_rate": copy.deepcopy(per_task),
        },
        "warmup_only": {
            "family_success_rate": 2 / 3,
            "per_task_success_rate": copy.deepcopy(per_task),
        },
        "warmup_plus_grpo": {
            "family_success_rate": 2 / 3,
            "per_task_success_rate": copy.deepcopy(per_task),
        },
        "holdout": {
            "task_ids": [324],
            "baseline_success_rate": 1.0,
            "warmup_success_rate": 1.0,
            "grpo_success_rate": 1.0,
            "gain_vs_baseline": 0.0,
            "gain_vs_warmup": 0.0,
            "per_task": {
                "324": {
                    "baseline_success_rate": 1.0,
                    "warmup_success_rate": 1.0,
                    "grpo_success_rate": 1.0,
                    "gain_vs_baseline": 0.0,
                    "gain_vs_warmup": 0.0,
                }
            },
        },
        "current_stack": {
            "family_success_rate": 1.0,
            "per_task_success_rate": {"128": 1.0, "131": 1.0, "324": 1.0},
            "applied_overrides": {
                "131": {
                    "metrics_path": "C:\\runs\\shopping_exact\\eval_task_131\\metrics.json",
                    "success_rate": 1.0,
                    "episodes": 1,
                    "average_steps": 2.0,
                }
            },
        },
        "current_stack_holdout": {
            "task_ids": [324],
            "baseline_success_rate": 1.0,
            "warmup_success_rate": 1.0,
            "grpo_success_rate": 1.0,
            "gain_vs_baseline": 0.0,
            "gain_vs_warmup": 0.0,
            "per_task": {
                "324": {
                    "baseline_success_rate": 1.0,
                    "warmup_success_rate": 1.0,
                    "grpo_success_rate": 1.0,
                    "gain_vs_baseline": 0.0,
                    "gain_vs_warmup": 0.0,
                }
            },
        },
        "current_stack_meta": {
            "base_summary": "C:\\runs\\shopping_exact\\family_summary.json",
            "base_stage": "warmup_plus_grpo",
            "benchmark_blockers": ["131"],
            "override_count": 1,
            "override_task_ids": [131],
            "override_metrics_paths": {
                "131": "C:\\runs\\shopping_exact\\eval_task_131\\metrics.json",
            },
            "base_run_provenance": {
                "split_manifest_path": "C:\\runs\\shopping_exact\\split_manifest.json",
            },
            "applied_override_metrics": {
                "131": {
                    "metrics_path": "C:\\runs\\shopping_exact\\eval_task_131\\metrics.json",
                    "success_rate": 1.0,
                    "episodes": 1,
                    "average_steps": 2.0,
                }
            },
        },
        "run_provenance": {
            "split_manifest_path": "C:\\runs\\shopping_exact\\split_manifest.json",
            "split_provenance": {
                "split_source": "recommended",
                "split_seed": 42,
            },
            "judge_requirements": {
                "requires_openai_judge": False,
                "openai_api_key_present": None,
                "run_ready": None,
            },
        },
    }


def _build_full_summary() -> dict[str, object]:
    return {
        "family_name": "shopping_full",
        "task_count": 3,
        "training_task_count": 2,
        "holdout_task_count": 1,
        "split": {
            "eval_task_ids": [96, 128, 324],
            "warmup_task_ids": [128],
            "grpo_task_ids": [96],
            "holdout_task_ids": [324],
        },
        "baseline": {
            "family_success_rate": 1 / 3,
            "per_task_success_rate": {"96": 0.0, "128": 1.0, "324": 0.0},
            "task_groups": {
                "judge_free": {
                    "task_ids": [128, 324],
                    "task_count": 2,
                    "success_rate": 0.5,
                    "per_task_success_rate": {"128": 1.0, "324": 0.0},
                },
                "judge_gated": {
                    "task_ids": [96],
                    "task_count": 1,
                    "success_rate": 0.0,
                    "per_task_success_rate": {"96": 0.0},
                },
            },
        },
        "warmup_only": {
            "family_success_rate": 2 / 3,
            "per_task_success_rate": {"96": 0.0, "128": 1.0, "324": 1.0},
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
        "warmup_plus_grpo": {
            "family_success_rate": 1.0,
            "per_task_success_rate": {"96": 1.0, "128": 1.0, "324": 1.0},
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
                    "success_rate": 1.0,
                    "per_task_success_rate": {"96": 1.0},
                },
            },
        },
        "holdout": {
            "task_ids": [324],
            "baseline_success_rate": 0.0,
            "warmup_success_rate": 1.0,
            "grpo_success_rate": 1.0,
            "gain_vs_baseline": 1.0,
            "gain_vs_warmup": 0.0,
            "per_task": {
                "324": {
                    "baseline_success_rate": 0.0,
                    "warmup_success_rate": 1.0,
                    "grpo_success_rate": 1.0,
                    "gain_vs_baseline": 1.0,
                    "gain_vs_warmup": 0.0,
                }
            },
        },
    }


def _build_bootstrap_summary_with_overlapping_groups() -> dict[str, object]:
    per_task = {
        "0": 0.0,
        "1": 1.0,
        "7": 1.0,
        "21": 0.0,
    }
    return {
        "family_name": "bootstrap41",
        "task_count": 4,
        "training_task_count": 3,
        "holdout_task_count": 1,
        "split": {
            "family_name": "bootstrap41",
            "task_ids": [0, 1, 7, 21],
            "eval_task_ids": [0, 1, 7, 21],
            "warmup_task_ids": [0, 1],
            "grpo_task_ids": [7],
            "holdout_task_ids": [21],
            "split_seed": 42,
        },
        "baseline": {
            "family_success_rate": 0.5,
            "per_task_success_rate": copy.deepcopy(per_task),
            "task_groups": {
                "judge_free": {
                    "task_ids": [0, 1, 7, 21],
                    "task_count": 4,
                    "success_rate": 0.5,
                    "per_task_success_rate": copy.deepcopy(per_task),
                },
                "site_shopping_admin": {
                    "task_ids": [0, 1],
                    "task_count": 2,
                    "success_rate": 0.5,
                    "per_task_success_rate": {"0": 0.0, "1": 1.0},
                },
                "site_map": {
                    "task_ids": [7],
                    "task_count": 1,
                    "success_rate": 1.0,
                    "per_task_success_rate": {"7": 1.0},
                },
                "site_shopping": {
                    "task_ids": [21],
                    "task_count": 1,
                    "success_rate": 0.0,
                    "per_task_success_rate": {"21": 0.0},
                },
            },
        },
    }


class ValidateFamilySummaryTests(unittest.TestCase):
    def test_validate_family_summary_accepts_exact_summary_and_computes_blocker_aware_rate(self) -> None:
        errors, derived = validate_family_summary(_build_exact_summary(), benchmark_blockers=["131"])

        self.assertEqual(errors, [])
        stage_metrics = derived["stage_metrics"]["warmup_plus_grpo"]
        self.assertEqual(stage_metrics["family_success_rate"], 2 / 3)
        self.assertEqual(stage_metrics["effective_family_success_rate_excluding_blockers"], 1.0)
        self.assertEqual(stage_metrics["unresolved_task_ids"], [131])
        self.assertEqual(stage_metrics["unresolved_task_ids_excluding_blockers"], [])
        self.assertEqual(derived["holdout_metrics"]["current_stack_holdout"]["stage_name"], "current_stack")
        self.assertEqual(derived["holdout_metrics"]["current_stack_holdout"]["grpo_success_rate"], 1.0)
        self.assertEqual(derived["stage_meta"]["current_stack_meta"]["stage_name"], "current_stack")
        self.assertEqual(
            derived["stage_meta"]["current_stack_meta"]["base_summary"],
            "C:\\runs\\shopping_exact\\family_summary.json",
        )
        self.assertEqual(derived["stage_meta"]["current_stack_meta"]["override_task_ids"], [131])
        self.assertTrue(derived["stage_meta"]["current_stack_meta"]["has_base_run_provenance"])
        self.assertEqual(
            derived["stage_meta"]["current_stack_meta"]["base_run_provenance_split_source"],
            None,
        )
        self.assertTrue(derived["run_provenance"]["present"])
        self.assertEqual(derived["run_provenance"]["split_source"], "recommended")

    def test_validate_family_summary_accepts_grouped_full_summary(self) -> None:
        errors, derived = validate_family_summary(_build_full_summary())

        self.assertEqual(errors, [])
        stage_metrics = derived["stage_metrics"]["baseline"]
        self.assertEqual(stage_metrics["family_success_rate"], 1 / 3)
        self.assertEqual(stage_metrics["unresolved_task_ids"], [96, 324])

    def test_validate_family_summary_reports_group_partition_mismatch(self) -> None:
        summary = _build_full_summary()
        summary["baseline"]["task_groups"]["judge_free"]["task_ids"] = [128]

        errors, _ = validate_family_summary(summary)

        self.assertTrue(any("summary.baseline.task_groups" in error for error in errors))

    def test_validate_family_summary_allows_overlapping_task_groups(self) -> None:
        errors, derived = validate_family_summary(_build_bootstrap_summary_with_overlapping_groups())

        self.assertEqual(errors, [])
        self.assertEqual(derived["stage_metrics"]["baseline"]["family_success_rate"], 0.5)

    def test_validate_family_summary_reports_stage_specific_holdout_mismatch(self) -> None:
        summary = _build_exact_summary()
        summary["current_stack_holdout"]["grpo_success_rate"] = 0.5

        errors, _ = validate_family_summary(summary)

        self.assertTrue(any("summary.current_stack_holdout.grpo_success_rate" in error for error in errors))

    def test_validate_family_summary_reports_run_provenance_split_seed_mismatch(self) -> None:
        summary = _build_exact_summary()
        summary["run_provenance"]["split_provenance"]["split_seed"] = 7

        errors, _ = validate_family_summary(summary)

        self.assertTrue(
            any("summary.run_provenance.split_provenance.split_seed" in error for error in errors)
        )

    def test_validate_family_summary_reports_run_provenance_alignment_mismatch(self) -> None:
        summary = _build_exact_summary()
        summary["run_provenance"]["recommended_split_alignment"] = {
            "matches_recommended_split": True,
            "recommended_split_seed": 42,
            "differences": {},
        }

        errors, _ = validate_family_summary(summary)

        self.assertTrue(
            any(
                "summary.run_provenance.recommended_split_alignment.matches_recommended_split" in error
                for error in errors
            )
        )

    def test_validate_family_summary_reports_stage_meta_override_mismatch(self) -> None:
        summary = _build_exact_summary()
        summary["current_stack_meta"]["override_count"] = 2

        errors, _ = validate_family_summary(summary)

        self.assertTrue(any("summary.current_stack_meta.override_count" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
