from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.refresh_family_current_stack import (
    _merge_manifest_args,
    build_arg_parser,
    build_refreshed_summary,
    refresh_family_current_stack,
    validate_refresh_inputs,
)


def _build_summary() -> dict[str, object]:
    return {
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
        "family_metadata": {
            "judge_free_task_ids": [128, 131, 324],
            "judge_gated_task_ids": [],
        },
        "run_provenance": {
            "split_provenance": {
                "source": "split_manifest",
                "path": "scripts/local/shopping_exact_curriculum_manifest.json",
            },
            "recommended_split_alignment": {
                "matches_recommended_split": False,
            },
        },
        "baseline": {
            "family_success_rate": 1 / 3,
            "per_task_success_rate": {"128": 1.0, "131": 0.0, "324": 0.0},
            "task_groups": {
                "judge_free": {
                    "task_ids": [128, 131, 324],
                    "task_count": 3,
                    "success_rate": 1 / 3,
                    "per_task_success_rate": {"128": 1.0, "131": 0.0, "324": 0.0},
                }
            },
        },
        "warmup_only": {
            "family_success_rate": 2 / 3,
            "per_task_success_rate": {"128": 1.0, "131": 0.0, "324": 1.0},
            "task_groups": {
                "judge_free": {
                    "task_ids": [128, 131, 324],
                    "task_count": 3,
                    "success_rate": 2 / 3,
                    "per_task_success_rate": {"128": 1.0, "131": 0.0, "324": 1.0},
                }
            },
        },
        "warmup_plus_grpo": {
            "family_success_rate": 2 / 3,
            "per_task_success_rate": {"128": 1.0, "131": 0.0, "324": 1.0},
            "task_groups": {
                "judge_free": {
                    "task_ids": [128, 131, 324],
                    "task_count": 3,
                    "success_rate": 2 / 3,
                    "per_task_success_rate": {"128": 1.0, "131": 0.0, "324": 1.0},
                }
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


class RefreshFamilyCurrentStackTests(unittest.TestCase):
    def test_validate_refresh_inputs_reports_missing_metrics_file(self) -> None:
        missing_path = Path("C:/definitely/missing/metrics.json")

        errors, preview = validate_refresh_inputs(
            _build_summary(),
            base_stage="warmup_plus_grpo",
            overrides={"131": missing_path},
        )

        self.assertEqual(
            errors,
            [f"Override metrics file is missing for task 131: {missing_path}"],
        )
        self.assertEqual(preview["override_task_ids"], [131])

    def test_validate_refresh_inputs_reports_unknown_override_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "task_999.json"
            metrics_path.write_text(
                json.dumps({"success_rate": 1.0, "episodes": 1, "average_steps": 2.0}),
                encoding="utf-8",
            )

            errors, preview = validate_refresh_inputs(
                _build_summary(),
                base_stage="warmup_plus_grpo",
                overrides={"999": metrics_path},
            )

        self.assertEqual(
            errors,
            ["Override task 999 is not present in base stage 'warmup_plus_grpo'"],
        )
        self.assertEqual(preview["override_task_ids"], [999])

    def test_manifest_populates_paths_and_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path = tmp_path / "manifest.json"
            metrics_path = tmp_path / "task_131.json"
            metrics_path.write_text(
                json.dumps({"success_rate": 1.0, "episodes": 1, "average_steps": 2.0}),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "base_summary": "base.json",
                        "summary_out": "summary.json",
                        "compare_report_out": "compare.json",
                        "benchmark_blockers": ["131"],
                        "overrides": {"131": "task_131.json"},
                    }
                ),
                encoding="utf-8",
            )

            args = build_arg_parser().parse_args(["--manifest", str(manifest_path)])
            for key in (
                "base_summary",
                "summary_out",
                "compare_report_out",
                "base_stage",
                "label",
                "compare_base_stage",
                "compare_label_a",
                "compare_label_b",
                "benchmark_blocker",
            ):
                setattr(args, f"_{key}_explicit", False)
            merged = _merge_manifest_args(args)

        self.assertEqual(merged.base_summary, (tmp_path / "base.json").resolve())
        self.assertEqual(merged.summary_out, (tmp_path / "summary.json").resolve())
        self.assertEqual(merged.compare_report_out, (tmp_path / "compare.json").resolve())
        self.assertEqual(merged.benchmark_blocker, ["131"])
        self.assertEqual(merged.override, [f"131={(tmp_path / 'task_131.json').resolve()}"])

    def test_cli_override_wins_over_manifest_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path = tmp_path / "manifest.json"
            metrics_path_a = tmp_path / "task_131_a.json"
            metrics_path_b = tmp_path / "task_131_b.json"
            metrics_path_a.write_text(
                json.dumps({"success_rate": 0.0, "episodes": 1, "average_steps": 2.0}),
                encoding="utf-8",
            )
            metrics_path_b.write_text(
                json.dumps({"success_rate": 1.0, "episodes": 1, "average_steps": 2.0}),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "base_summary": "base.json",
                        "summary_out": "summary.json",
                        "overrides": {"131": "task_131_a.json"},
                    }
                ),
                encoding="utf-8",
            )

            args = build_arg_parser().parse_args(
                ["--manifest", str(manifest_path), "--override", f"131={metrics_path_b}"]
            )
            for key in (
                "base_summary",
                "summary_out",
                "compare_report_out",
                "base_stage",
                "label",
                "compare_base_stage",
                "compare_label_a",
                "compare_label_b",
                "benchmark_blocker",
            ):
                setattr(args, f"_{key}_explicit", False)
            merged = _merge_manifest_args(args)

        self.assertEqual(merged.override, [f"131={metrics_path_b}"])

    def test_build_refreshed_summary_adds_label_stage_and_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "task_131.json"
            metrics_path.write_text(
                json.dumps({"success_rate": 1.0, "episodes": 1, "average_steps": 2.0}),
                encoding="utf-8",
            )
            refreshed = build_refreshed_summary(
                _build_summary(),
                base_summary_path=Path("base.json"),
                base_stage="warmup_plus_grpo",
                label="current_stack",
                overrides={"131": metrics_path},
                benchmark_blockers=["131"],
            )

        self.assertEqual(refreshed["current_stack"]["per_task_success_rate"]["131"], 1.0)
        self.assertEqual(refreshed["current_stack"]["task_groups"]["judge_free"]["success_rate"], 1.0)
        self.assertEqual(refreshed["current_stack_holdout"]["grpo_success_rate"], 1.0)
        self.assertEqual(refreshed["current_stack_meta"]["benchmark_blockers"], ["131"])
        self.assertEqual(refreshed["current_stack_meta"]["override_count"], 1)
        self.assertEqual(refreshed["current_stack_meta"]["override_task_ids"], [131])
        self.assertEqual(refreshed["current_stack_meta"]["override_metrics_paths"]["131"], str(metrics_path))
        self.assertEqual(
            refreshed["current_stack_meta"]["base_run_provenance"]["split_provenance"]["path"],
            "scripts/local/shopping_exact_curriculum_manifest.json",
        )
        self.assertEqual(refreshed["current_stack_meta"]["applied_override_metrics"]["131"]["success_rate"], 1.0)

    def test_refresh_family_current_stack_returns_validation_and_compare_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_path = Path(tmp_dir) / "task_131.json"
            metrics_path.write_text(
                json.dumps({"success_rate": 1.0, "episodes": 1, "average_steps": 2.0}),
                encoding="utf-8",
            )
            refreshed, validation_errors, derived, compare_errors, compare_payload = refresh_family_current_stack(
                base_summary=_build_summary(),
                base_summary_path=Path("base.json"),
                base_stage="warmup_plus_grpo",
                label="current_stack",
                overrides={"131": metrics_path},
                benchmark_blockers=["131"],
                compare_base_stage="warmup_plus_grpo",
                compare_label_a="base",
                compare_label_b="current_stack",
            )

        self.assertEqual(validation_errors, [])
        self.assertEqual(compare_errors, [])
        self.assertEqual(derived["stage_metrics"]["current_stack"]["success_task_count"], 3)
        custom_stage = compare_payload["custom_stage_comparison"]["warmup_plus_grpo__vs__current_stack"]
        self.assertEqual(custom_stage["improved_task_ids_b_over_a"], [131])
        self.assertAlmostEqual(
            custom_stage["task_group_comparison"]["groups"]["judge_free"]["success_rate_delta_b_minus_a"],
            1 / 3,
        )
        self.assertEqual(refreshed["current_stack_holdout"]["task_ids"], [324])


if __name__ == "__main__":
    unittest.main()
