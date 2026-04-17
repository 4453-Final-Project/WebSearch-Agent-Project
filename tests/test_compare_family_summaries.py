from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_family_summaries import build_arg_parser, compare_family_summaries  # noqa: E402


def _build_summary(per_task_stage_map: dict[str, dict[str, float]]) -> dict[str, object]:
    summary = {
        "family_name": "shopping_exact",
        "task_count": 3,
        "training_task_count": 2,
        "holdout_task_count": 1,
        "split": {
            "family_name": "shopping_exact",
            "task_ids": [128, 131, 324],
            "warmup_task_ids": [128],
            "grpo_task_ids": [131],
            "holdout_task_ids": [324],
            "eval_task_ids": [128, 131, 324],
            "split_seed": 42,
        },
    }
    for stage_name, per_task in per_task_stage_map.items():
        summary[stage_name] = {
            "family_success_rate": sum(per_task.values()) / len(per_task),
            "per_task_success_rate": copy.deepcopy(per_task),
        }
    if "warmup_plus_grpo" in per_task_stage_map:
        summary["holdout"] = {
            "task_ids": [324],
            "baseline_success_rate": per_task_stage_map["baseline"]["324"],
            "warmup_success_rate": per_task_stage_map["warmup_only"]["324"],
            "grpo_success_rate": per_task_stage_map["warmup_plus_grpo"]["324"],
            "gain_vs_baseline": per_task_stage_map["warmup_plus_grpo"]["324"] - per_task_stage_map["baseline"]["324"],
            "gain_vs_warmup": per_task_stage_map["warmup_plus_grpo"]["324"] - per_task_stage_map["warmup_only"]["324"],
            "per_task": {
                "324": {
                    "baseline_success_rate": per_task_stage_map["baseline"]["324"],
                    "warmup_success_rate": per_task_stage_map["warmup_only"]["324"],
                    "grpo_success_rate": per_task_stage_map["warmup_plus_grpo"]["324"],
                    "gain_vs_baseline": per_task_stage_map["warmup_plus_grpo"]["324"] - per_task_stage_map["baseline"]["324"],
                    "gain_vs_warmup": per_task_stage_map["warmup_plus_grpo"]["324"] - per_task_stage_map["warmup_only"]["324"],
                }
            },
        }
    return summary


def _attach_run_provenance(
    summary: dict[str, object],
    *,
    split_source: str,
    split_seed: int = 42,
    split_manifest_path: str | None = None,
    matches_recommended_split: bool | None = None,
    requires_openai_judge: bool = False,
    run_ready: bool | None = True,
    inferred_from_legacy_artifact: bool = False,
) -> dict[str, object]:
    run_provenance = {
        "split_manifest_path": split_manifest_path,
        "split_provenance": {
            "split_source": split_source,
            "split_seed": split_seed,
            "inferred_from_legacy_artifact": inferred_from_legacy_artifact,
        },
        "judge_requirements": {
            "requires_openai_judge": requires_openai_judge,
            "openai_api_key_present": None,
            "run_ready": run_ready,
            "inferred_from_legacy_artifact": inferred_from_legacy_artifact,
        },
    }
    if matches_recommended_split is not None:
        run_provenance["recommended_split_alignment"] = {
            "matches_recommended_split": matches_recommended_split,
            "recommended_split_seed": split_seed,
            "differences": {},
        }
    summary["run_provenance"] = run_provenance
    return summary


def _attach_task_groups(
    summary: dict[str, object],
    group_task_ids: dict[str, list[int]],
) -> dict[str, object]:
    for stage_name, stage_summary in summary.items():
        if not isinstance(stage_summary, dict) or "per_task_success_rate" not in stage_summary:
            continue
        per_task = {str(task_id): float(value) for task_id, value in stage_summary["per_task_success_rate"].items()}
        stage_summary["task_groups"] = {}
        for group_name, task_ids in group_task_ids.items():
            task_keys = [str(task_id) for task_id in task_ids]
            group_per_task = {task_id: per_task[task_id] for task_id in task_keys}
            stage_summary["task_groups"][group_name] = {
                "task_ids": task_ids,
                "task_count": len(task_ids),
                "success_rate": sum(group_per_task.values()) / len(group_per_task),
                "per_task_success_rate": group_per_task,
            }
    return summary


def _attach_stage_meta(
    summary: dict[str, object],
    *,
    stage_name: str,
    base_stage: str,
    override_task_ids: list[int],
    benchmark_blockers: list[int] | None = None,
) -> dict[str, object]:
    stage_summary = summary[stage_name]
    summary[f"{stage_name}_meta"] = {
        "base_summary": "C:\\runs\\shopping_exact\\family_summary.json",
        "base_stage": base_stage,
        "benchmark_blockers": [str(task_id) for task_id in (benchmark_blockers or [])],
        "override_count": len(override_task_ids),
        "override_task_ids": override_task_ids,
        "override_metrics_paths": {
            str(task_id): f"C:\\runs\\shopping_exact\\eval_task_{task_id}\\metrics.json"
            for task_id in override_task_ids
        },
        "base_run_provenance": {
            "split_manifest_path": "C:\\runs\\shopping_exact\\split_manifest.json",
        },
        "applied_override_metrics": {
            str(task_id): {
                "metrics_path": f"C:\\runs\\shopping_exact\\eval_task_{task_id}\\metrics.json",
                "success_rate": float(stage_summary["per_task_success_rate"][str(task_id)]),
                "episodes": 1,
                "average_steps": 2.0,
            }
            for task_id in override_task_ids
        },
    }
    stage_summary["applied_overrides"] = copy.deepcopy(summary[f"{stage_name}_meta"]["applied_override_metrics"])
    return summary


class CompareFamilySummariesTests(unittest.TestCase):
    def test_arg_parser_accepts_out_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "report.json"
            args = build_arg_parser().parse_args(
                [
                    "--summary-a",
                    "a.json",
                    "--summary-b",
                    "b.json",
                    "--out",
                    str(out_path),
                ]
            )

        self.assertEqual(args.out, out_path)

    def test_compare_family_summaries_reports_stage_deltas(self) -> None:
        summary_a = _build_summary(
            {
                "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
            }
        )
        summary_b = _build_summary(
            {
                "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_only": {"128": 1.0, "131": 0.0, "324": 1.0},
                "warmup_plus_grpo": {"128": 1.0, "131": 1.0, "324": 1.0},
                "current_stack": {"128": 1.0, "131": 1.0, "324": 1.0},
            }
        )
        summary_b["current_stack_holdout"] = {
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
        }

        errors, comparison = compare_family_summaries(
            summary_a,
            summary_b,
            label_a="base",
            label_b="improved",
            benchmark_blockers=["131"],
        )

        self.assertEqual(errors, [])
        self.assertEqual(comparison["common_stage_names"], ["baseline", "warmup_only", "warmup_plus_grpo"])
        self.assertEqual(comparison["stage_names_only_in_b"], ["current_stack"])
        warmup_plus_grpo = comparison["stage_comparison"]["warmup_plus_grpo"]
        self.assertEqual(warmup_plus_grpo["improved_task_ids_b_over_a"], [131, 324])
        self.assertEqual(warmup_plus_grpo["success_task_count_delta_b_minus_a"], 2)
        self.assertEqual(
            warmup_plus_grpo["effective_family_success_rate_excluding_blockers_delta_b_minus_a"],
            0.5,
        )

    def test_compare_family_summaries_supports_custom_stage_mapping(self) -> None:
        summary_a = _build_summary(
            {
                "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
            }
        )
        summary_b = _build_summary(
            {
                "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
                "current_stack": {"128": 1.0, "131": 1.0, "324": 1.0},
            }
        )
        summary_b["current_stack_holdout"] = {
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
        }

        errors, comparison = compare_family_summaries(
            summary_a,
            summary_b,
            label_a="base",
            label_b="current",
            benchmark_blockers=["131"],
            compare_stage_specs=["warmup_plus_grpo=current_stack"],
        )

        self.assertEqual(errors, [])
        custom_stage = comparison["custom_stage_comparison"]["warmup_plus_grpo__vs__current_stack"]
        self.assertEqual(custom_stage["improved_task_ids_b_over_a"], [131, 324])
        self.assertEqual(custom_stage["effective_family_success_rate_excluding_blockers_delta_b_minus_a"], 0.5)
        custom_holdout = comparison["custom_holdout_comparison"]["holdout__vs__current_stack_holdout"]
        self.assertEqual(custom_holdout["grpo_success_rate_delta_b_minus_a"], 1.0)

    def test_compare_family_summaries_reports_validation_errors(self) -> None:
        summary_a = {"family_name": "shopping_exact"}
        summary_b = {"family_name": "shopping_exact"}

        errors, comparison = compare_family_summaries(summary_a, summary_b, label_a="a", label_b="b")

        self.assertTrue(errors)
        self.assertEqual(comparison, {})

    def test_compare_family_summaries_reports_task_group_deltas(self) -> None:
        summary_a = _attach_task_groups(
            _build_summary(
                {
                    "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
                }
            ),
            {"judge_free": [128, 131], "judge_gated": [324]},
        )
        summary_b = _attach_task_groups(
            _build_summary(
                {
                    "baseline": {"128": 1.0, "131": 1.0, "324": 0.0},
                    "warmup_only": {"128": 1.0, "131": 1.0, "324": 1.0},
                    "warmup_plus_grpo": {"128": 1.0, "131": 1.0, "324": 1.0},
                }
            ),
            {"judge_free": [128, 131], "judge_gated": [324]},
        )

        errors, comparison = compare_family_summaries(summary_a, summary_b, label_a="a", label_b="b")

        self.assertEqual(errors, [])
        group_comparison = comparison["stage_comparison"]["warmup_plus_grpo"]["task_group_comparison"]
        self.assertEqual(group_comparison["common_group_names"], ["judge_free", "judge_gated"])
        self.assertEqual(group_comparison["group_names_only_in_a"], [])
        self.assertEqual(group_comparison["group_names_only_in_b"], [])
        self.assertEqual(group_comparison["groups"]["judge_free"]["improved_task_ids_b_over_a"], [131])
        self.assertEqual(group_comparison["groups"]["judge_free"]["success_rate_delta_b_minus_a"], 0.5)
        self.assertEqual(group_comparison["groups"]["judge_gated"]["improved_task_ids_b_over_a"], [324])
        self.assertEqual(group_comparison["groups"]["judge_gated"]["success_rate_delta_b_minus_a"], 1.0)

    def test_compare_family_summaries_reports_provenance_differences(self) -> None:
        summary_a = _attach_run_provenance(
            _build_summary(
                {
                    "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
                }
            ),
            split_source="recommended",
            split_manifest_path="C:\\runs\\a\\split_manifest.json",
            inferred_from_legacy_artifact=True,
        )
        summary_b = _attach_run_provenance(
            _build_summary(
                {
                    "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_only": {"128": 1.0, "131": 0.0, "324": 1.0},
                    "warmup_plus_grpo": {"128": 1.0, "131": 1.0, "324": 1.0},
                }
            ),
            split_source="manifest",
            split_manifest_path="C:\\runs\\b\\split_manifest.json",
            inferred_from_legacy_artifact=False,
        )

        errors, comparison = compare_family_summaries(summary_a, summary_b, label_a="a", label_b="b")

        self.assertEqual(errors, [])
        self.assertEqual(comparison["artifact_provenance"]["a"]["split_source"], "recommended")
        self.assertEqual(comparison["artifact_provenance"]["b"]["split_source"], "manifest")
        self.assertFalse(comparison["provenance_comparison"]["same_split_source"])
        self.assertFalse(comparison["provenance_comparison"]["same_split_manifest_path"])
        self.assertFalse(comparison["provenance_comparison"]["same_legacy_inference_state"])

    def test_compare_family_summaries_reports_stage_meta_differences(self) -> None:
        summary_a = _attach_stage_meta(
            _build_summary(
                {
                    "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "current_stack": {"128": 1.0, "131": 1.0, "324": 1.0},
                }
            ),
            stage_name="current_stack",
            base_stage="warmup_plus_grpo",
            override_task_ids=[131],
            benchmark_blockers=[131],
        )
        summary_b = _attach_stage_meta(
            _build_summary(
                {
                    "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "current_stack": {"128": 1.0, "131": 1.0, "324": 1.0},
                }
            ),
            stage_name="current_stack",
            base_stage="warmup_plus_grpo",
            override_task_ids=[131, 324],
            benchmark_blockers=[131],
        )

        errors, comparison = compare_family_summaries(summary_a, summary_b, label_a="a", label_b="b")

        self.assertEqual(errors, [])
        self.assertEqual(comparison["common_stage_meta_names"], ["current_stack_meta"])
        self.assertEqual(comparison["artifact_stage_meta"]["a"]["current_stack_meta"]["override_task_ids"], [131])
        self.assertEqual(
            comparison["artifact_stage_meta"]["b"]["current_stack_meta"]["override_task_ids"],
            [131, 324],
        )
        meta_comparison = comparison["stage_meta_comparison"]["current_stack_meta"]
        self.assertEqual(
            meta_comparison["base_summary_a"],
            "C:\\runs\\shopping_exact\\family_summary.json",
        )
        self.assertTrue(meta_comparison["same_base_summary"])
        self.assertFalse(meta_comparison["same_override_task_ids"])
        self.assertEqual(meta_comparison["override_count_delta_b_minus_a"], 1)

    def test_compare_family_summaries_reports_one_sided_stage_meta_lineage(self) -> None:
        summary_a = _build_summary(
            {
                "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
            }
        )
        summary_b = _attach_stage_meta(
            _build_summary(
                {
                    "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "current_stack": {"128": 1.0, "131": 1.0, "324": 1.0},
                }
            ),
            stage_name="current_stack",
            base_stage="warmup_plus_grpo",
            override_task_ids=[131, 324],
            benchmark_blockers=[131],
        )

        errors, comparison = compare_family_summaries(summary_a, summary_b, label_a="a", label_b="b")

        self.assertEqual(errors, [])
        self.assertEqual(comparison["stage_meta_names_only_in_b"], ["current_stack_meta"])
        lineage = comparison["stage_meta_lineage_only_in_b"]["current_stack_meta"]
        self.assertEqual(lineage["base_stage"], "warmup_plus_grpo")
        self.assertEqual(lineage["override_count"], 2)
        self.assertEqual(lineage["override_task_ids"], [131, 324])

    def test_compare_family_summaries_reports_one_sided_stage_meta_base_match(self) -> None:
        summary_a = _build_summary(
            {
                "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
            }
        )
        summary_b = _attach_stage_meta(
            _build_summary(
                {
                    "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "current_stack": {"128": 1.0, "131": 1.0, "324": 1.0},
                }
            ),
            stage_name="current_stack",
            base_stage="warmup_plus_grpo",
            override_task_ids=[131],
            benchmark_blockers=[131],
        )

        errors, comparison = compare_family_summaries(
            summary_a,
            summary_b,
            label_a="a",
            label_b="b",
            summary_a_path="C:\\runs\\shopping_exact\\family_summary.json",
        )

        self.assertEqual(errors, [])
        lineage = comparison["stage_meta_lineage_only_in_b"]["current_stack_meta"]
        self.assertEqual(lineage["counterpart_summary_path"], "C:\\runs\\shopping_exact\\family_summary.json")
        self.assertTrue(lineage["base_summary_matches_counterpart_summary_path"])

    def test_compare_family_summaries_normalizes_windows_and_mnt_paths_for_base_match(self) -> None:
        summary_a = _build_summary(
            {
                "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
            }
        )
        summary_b = _attach_stage_meta(
            _build_summary(
                {
                    "baseline": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_only": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "warmup_plus_grpo": {"128": 1.0, "131": 0.0, "324": 0.0},
                    "current_stack": {"128": 1.0, "131": 1.0, "324": 1.0},
                }
            ),
            stage_name="current_stack",
            base_stage="warmup_plus_grpo",
            override_task_ids=[131],
            benchmark_blockers=[131],
        )
        summary_b["current_stack_meta"]["base_summary"] = "/mnt/c/runs/shopping_exact/family_summary.json"

        errors, comparison = compare_family_summaries(
            summary_a,
            summary_b,
            label_a="a",
            label_b="b",
            summary_a_path="C:\\runs\\shopping_exact\\family_summary.json",
        )

        self.assertEqual(errors, [])
        lineage = comparison["stage_meta_lineage_only_in_b"]["current_stack_meta"]
        self.assertEqual(lineage["normalized_base_summary"], "c:/runs/shopping_exact/family_summary.json")
        self.assertEqual(
            lineage["normalized_counterpart_summary_path"],
            "c:/runs/shopping_exact/family_summary.json",
        )
        self.assertTrue(lineage["base_summary_matches_counterpart_summary_path"])


if __name__ == "__main__":
    unittest.main()
