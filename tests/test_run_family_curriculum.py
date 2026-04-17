from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_family_curriculum import (
    _get_family_requirement_status,
    _load_existing_eval_results,
    _load_partial_eval_results,
    _load_partial_eval_results_from_tree,
    _load_split_from_manifest,
    _run_baseline_eval,
    _resolve_preflight_out_path,
    _validate_family_requirements,
    _write_preflight_artifact_if_requested,
    build_arg_parser,
    build_split_manifest_payload,
    build_family_run_summary,
    resolve_split_provenance,
    validate_split_preflight,
)
from src.training.task_families import TaskSplit


class RunFamilyCurriculumTests(unittest.TestCase):
    def test_arg_parser_reuses_existing_stages_by_default(self) -> None:
        args = build_arg_parser().parse_args([])
        self.assertTrue(args.reuse_existing_stages)
        self.assertEqual(args.min_training_task_count, 20)
        self.assertIsNone(args.preflight_out)
        self.assertIsNone(args.quantization_mode)

    def test_arg_parser_accepts_quantization_overrides(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--family",
                "shopping_exact",
                "--quantization-mode",
                "bnb_4bit",
                "--quant-compute-dtype",
                "float16",
                "--quant-type",
                "nf4",
                "--no-quant-use-double-quant",
            ]
        )

        self.assertEqual(args.quantization_mode, "bnb_4bit")
        self.assertEqual(args.quant_compute_dtype, "float16")
        self.assertEqual(args.quant_type, "nf4")
        self.assertFalse(args.quant_use_double_quant)

    def test_resolve_preflight_out_path_uses_explicit_path(self) -> None:
        args = build_arg_parser().parse_args(["--preflight-out", "C:\\tmp\\preflight.json"])

        self.assertEqual(str(_resolve_preflight_out_path(args)), "C:\\tmp\\preflight.json")

    def test_resolve_preflight_out_path_defaults_under_out_dir(self) -> None:
        args = build_arg_parser().parse_args(["--out-dir", "C:\\tmp\\family_run", "--dry-run"])

        self.assertEqual(_resolve_preflight_out_path(args), Path("C:\\tmp\\family_run") / "preflight.json")

    def test_load_split_from_preflight_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "preflight.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "family_name": "shopping_full",
                        "split_preview": {
                            "family_name": "shopping_full",
                            "task_ids": [96, 128, 189, 324],
                            "warmup_task_ids": [128],
                            "grpo_task_ids": [96, 189],
                            "holdout_task_ids": [324],
                            "eval_task_ids": [96, 128, 189, 324],
                            "split_seed": 42,
                        },
                    }
                ),
                encoding="utf-8",
            )

            split = _load_split_from_manifest(manifest_path, expected_family_name="shopping_full")

            self.assertEqual(split.family_name, "shopping_full")
            self.assertEqual(split.training_task_ids, (128, 96, 189))

    def test_load_split_from_manifest_rejects_family_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "split_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "family_name": "shopping_exact",
                        "task_ids": [128, 189, 324],
                        "warmup_task_ids": [128],
                        "grpo_task_ids": [189],
                        "holdout_task_ids": [324],
                        "eval_task_ids": [128, 189, 324],
                        "split_seed": 42,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                _load_split_from_manifest(manifest_path, expected_family_name="shopping_full")

    def test_resolve_split_provenance_reports_manifest_source(self) -> None:
        args = build_arg_parser().parse_args(
            ["--family", "shopping_exact", "--split-manifest", "C:\\tmp\\shopping_exact_manifest.json"]
        )
        split = TaskSplit(
            family_name="shopping_exact",
            task_ids=(128, 189, 324),
            warmup_task_ids=(128,),
            grpo_task_ids=(189,),
            holdout_task_ids=(324,),
            eval_task_ids=(128, 189, 324),
            split_seed=42,
        )

        provenance = resolve_split_provenance(args, split)

        self.assertEqual(provenance["split_source"], "manifest")
        self.assertTrue(provenance["source_manifest"].endswith("shopping_exact_manifest.json"))

    def test_resolve_split_provenance_reports_custom_counts(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--family",
                "shopping_exact",
                "--warmup-task-count",
                "10",
                "--holdout-task-count",
                "6",
            ]
        )
        split = TaskSplit(
            family_name="shopping_exact",
            task_ids=(128, 189, 324),
            warmup_task_ids=(128,),
            grpo_task_ids=(189,),
            holdout_task_ids=(324,),
            eval_task_ids=(128, 189, 324),
            split_seed=42,
        )

        provenance = resolve_split_provenance(args, split)

        self.assertEqual(provenance["split_source"], "custom_counts")
        self.assertEqual(
            provenance["requested_counts"],
            {"warmup_task_count": 10, "holdout_task_count": 6},
        )

    def test_build_split_manifest_payload_wraps_split_preview_with_provenance(self) -> None:
        split = TaskSplit(
            family_name="shopping_exact",
            task_ids=(128, 189, 324),
            warmup_task_ids=(128,),
            grpo_task_ids=(189,),
            holdout_task_ids=(324,),
            eval_task_ids=(128, 189, 324),
            split_seed=42,
        )

        payload = build_split_manifest_payload(
            split,
            split_provenance={"split_source": "recommended", "split_seed": 42},
            judge_requirements={
                "requires_openai_judge": False,
                "openai_api_key_present": False,
                "run_ready": True,
            },
            recommended_split_alignment={
                "matches_recommended_split": True,
                "recommended_split_seed": 42,
                "differences": {},
            },
        )

        self.assertEqual(payload["family_name"], "shopping_exact")
        self.assertEqual(payload["split_provenance"]["split_source"], "recommended")
        self.assertEqual(payload["split_preview"]["holdout_task_ids"], (324,))

    def test_write_preflight_artifact_if_requested_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            preflight_path = Path(tmp_dir) / "preflight.json"
            args = build_arg_parser().parse_args(["--preflight-out", str(preflight_path), "--dry-run"])

            payload = _write_preflight_artifact_if_requested(
                args,
                {"ok": True, "family_name": "shopping_full", "split_preview": {"training_task_count": 40}},
            )

            self.assertEqual(payload["preflight_path"], str(preflight_path))
            written = json.loads(preflight_path.read_text(encoding="utf-8"))
            self.assertEqual(written["preflight_path"], str(preflight_path))
            self.assertEqual(written["split_preview"]["training_task_count"], 40)

    def test_validate_split_preflight_accepts_large_exact_split(self) -> None:
        split = TaskSplit(
            family_name="shopping_exact",
            task_ids=tuple(range(100, 132)),
            warmup_task_ids=tuple(range(100, 114)),
            grpo_task_ids=tuple(range(114, 126)),
            holdout_task_ids=tuple(range(126, 132)),
            eval_task_ids=tuple(range(100, 132)),
            split_seed=42,
        )

        preflight = validate_split_preflight(split, min_training_task_count=20)

        self.assertTrue(preflight["ok"])
        self.assertTrue(preflight["launch_ready"])
        self.assertTrue(preflight["recommended_large_run_ready"])
        self.assertFalse(preflight["recommended_split_alignment"]["matches_recommended_split"])
        self.assertEqual(preflight["split_provenance"]["split_source"], "recommended")
        self.assertEqual(preflight["training_task_count"], 26)

    def test_validate_split_preflight_rejects_small_training_pool_for_large_family(self) -> None:
        split = TaskSplit(
            family_name="shopping_exact",
            task_ids=tuple(range(100, 132)),
            warmup_task_ids=(100, 101, 102, 103),
            grpo_task_ids=(104, 105, 106, 107),
            holdout_task_ids=tuple(range(108, 132)),
            eval_task_ids=tuple(range(100, 132)),
            split_seed=42,
        )

        preflight = validate_split_preflight(split, min_training_task_count=20)

        self.assertFalse(preflight["ok"])
        self.assertIn("below the configured floor", preflight["errors"][0])
        self.assertFalse(preflight["recommended_large_run_ready"])
        self.assertFalse(preflight["recommended_split_alignment"]["matches_recommended_split"])

    def test_validate_split_preflight_allows_small_family_recipe(self) -> None:
        split = TaskSplit(
            family_name="shopping_search_sort",
            task_ids=(324, 325, 326, 327, 328),
            warmup_task_ids=(325, 326),
            grpo_task_ids=(327, 328),
            holdout_task_ids=(324,),
            eval_task_ids=(324, 325, 326, 327, 328),
            split_seed=42,
        )

        preflight = validate_split_preflight(split, min_training_task_count=20)

        self.assertTrue(preflight["ok"])
        self.assertTrue(preflight["launch_ready"])
        self.assertFalse(preflight["recommended_large_run_ready"])
        self.assertTrue(preflight["recommended_split_alignment"]["matches_recommended_split"])

    def test_validate_split_preflight_rejects_overlap(self) -> None:
        split = TaskSplit(
            family_name="shopping_exact",
            task_ids=(101, 102, 103, 104),
            warmup_task_ids=(101, 102),
            grpo_task_ids=(102,),
            holdout_task_ids=(103, 104),
            eval_task_ids=(101, 102, 103, 104),
            split_seed=42,
        )

        preflight = validate_split_preflight(split, min_training_task_count=0)

        self.assertFalse(preflight["ok"])
        self.assertIn("Warmup and GRPO task ids must be disjoint.", preflight["errors"])

    def test_validate_split_preflight_reports_task_group_counts(self) -> None:
        split = TaskSplit(
            family_name="shopping_full",
            task_ids=(96, 128, 189, 324),
            warmup_task_ids=(128,),
            grpo_task_ids=(96, 189),
            holdout_task_ids=(324,),
            eval_task_ids=(96, 128, 189, 324),
            split_seed=42,
        )

        preflight = validate_split_preflight(split, min_training_task_count=0)

        self.assertTrue(preflight["ok"])
        self.assertEqual(
            preflight["task_group_counts"]["judge_free"],
            {
                "task_count": 3,
                "warmup_task_count": 1,
                "grpo_task_count": 1,
                "holdout_task_count": 1,
            },
        )
        self.assertEqual(
            preflight["task_group_counts"]["judge_gated"],
            {
                "task_count": 1,
                "warmup_task_count": 0,
                "grpo_task_count": 1,
                "holdout_task_count": 0,
            },
        )

    def test_validate_split_preflight_reports_not_launch_ready_without_judge_key(self) -> None:
        split = TaskSplit(
            family_name="shopping_full",
            task_ids=(96, 128, 189, 324),
            warmup_task_ids=(128,),
            grpo_task_ids=(96, 189),
            holdout_task_ids=(324,),
            eval_task_ids=(96, 128, 189, 324),
            split_seed=42,
        )

        preflight = validate_split_preflight(
            split,
            min_training_task_count=0,
            family_requirement_status={
                "requires_openai_judge": True,
                "openai_api_key_present": False,
                "run_ready": False,
            },
        )

        self.assertTrue(preflight["ok"])
        self.assertFalse(preflight["launch_ready"])
        self.assertEqual(
            preflight["judge_requirements"],
            {
                "requires_openai_judge": True,
                "openai_api_key_present": False,
                "run_ready": False,
            },
        )

    def test_validate_split_preflight_reports_recommended_alignment_for_checked_in_exact_manifest(self) -> None:
        split = TaskSplit(
            family_name="shopping_exact",
            task_ids=(
                128, 129, 130, 131, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198, 199, 200,
                231, 232, 233, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 358, 360, 362,
            ),
            warmup_task_ids=(188, 190, 194, 197, 198, 199, 231, 319, 320, 322, 325, 326, 327, 328),
            grpo_task_ids=(128, 129, 130, 131, 192, 193, 196, 232, 233, 321, 323, 362),
            holdout_task_ids=(189, 195, 200, 324, 358, 360),
            eval_task_ids=(
                128, 129, 130, 131, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198, 199, 200,
                231, 232, 233, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 358, 360, 362,
            ),
            split_seed=42,
        )

        preflight = validate_split_preflight(split, min_training_task_count=20)

        self.assertTrue(preflight["recommended_split_alignment"]["matches_recommended_split"])
        self.assertEqual(preflight["recommended_split_alignment"]["differences"], {})

    def test_load_existing_eval_results_reads_per_task_metrics_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_101_dir = root / "task_101"
            task_102_dir = root / "task_102"
            task_101_dir.mkdir(parents=True)
            task_102_dir.mkdir(parents=True)
            (task_101_dir / "metrics.json").write_text('{"success_rate": 0.5}', encoding="utf-8")
            (task_102_dir / "metrics.json").write_text('{"success_rate": 1.0}', encoding="utf-8")

            results = _load_existing_eval_results(root, (101, 102))

        self.assertEqual(results, {"101": {"success_rate": 0.5}, "102": {"success_rate": 1.0}})

    def test_load_partial_eval_results_reads_completed_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_path = Path(tmp_dir) / "baseline_eval_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "101": {"success_rate": 0.5},
                        "102": {"success_rate": 1.0},
                        "999": {"success_rate": 0.0},
                    }
                ),
                encoding="utf-8",
            )

            results = _load_partial_eval_results(summary_path, (101, 102, 103))

        self.assertEqual(results, {"101": {"success_rate": 0.5}, "102": {"success_rate": 1.0}})

    def test_load_partial_eval_results_from_tree_reads_completed_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_101_dir = root / "task_101"
            task_102_dir = root / "task_102"
            task_101_dir.mkdir(parents=True)
            task_102_dir.mkdir(parents=True)
            (task_101_dir / "metrics.json").write_text('{"success_rate": 0.5}', encoding="utf-8")
            (task_102_dir / "metrics.json").write_text('{"success_rate": 1.0}', encoding="utf-8")

            results = _load_partial_eval_results_from_tree(root, (101, 102, 103))

        self.assertEqual(results, {"101": {"success_rate": 0.5}, "102": {"success_rate": 1.0}})

    def test_run_baseline_eval_reuses_partial_summary_and_writes_incrementally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            summary_path = out_dir / "baseline_eval_summary.json"
            summary_path.write_text(
                json.dumps({"101": {"success_rate": 0.5}}),
                encoding="utf-8",
            )
            split = TaskSplit(
                family_name="shopping_exact",
                task_ids=(101, 102),
                warmup_task_ids=(101,),
                grpo_task_ids=(),
                holdout_task_ids=(102,),
                eval_task_ids=(101, 102),
                split_seed=42,
            )
            args = mock.Mock(
                reuse_existing_stages=False,
                model_dir_name="Qwen3.5-2B",
                model_path=None,
                max_new_tokens=64,
                eval_temperature=0.0,
                seed=42,
                max_steps=4,
                eval_episodes=2,
                headed=False,
            )

            with mock.patch("scripts.run_family_curriculum.load_policy", return_value="policy") as load_policy_mock:
                with mock.patch(
                    "scripts.run_family_curriculum.evaluate_single_task",
                    return_value={"success_rate": 1.0},
                ) as eval_mock:
                    results = _run_baseline_eval(args, split, out_dir)

            self.assertEqual(load_policy_mock.call_count, 1)
            self.assertEqual(eval_mock.call_count, 1)
            self.assertEqual(eval_mock.call_args.kwargs["task_id"], 102)
            self.assertEqual(
                results,
                {
                    "101": {"success_rate": 0.5},
                    "102": {"success_rate": 1.0},
                },
            )
            written = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(written, results)

    def test_run_baseline_eval_reuses_partial_metrics_tree_without_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            eval_root = out_dir / "eval_baseline" / "task_101"
            eval_root.mkdir(parents=True)
            (eval_root / "metrics.json").write_text(
                json.dumps({"success_rate": 0.5}),
                encoding="utf-8",
            )
            split = TaskSplit(
                family_name="shopping_exact",
                task_ids=(101, 102),
                warmup_task_ids=(101,),
                grpo_task_ids=(),
                holdout_task_ids=(102,),
                eval_task_ids=(101, 102),
                split_seed=42,
            )
            args = mock.Mock(
                reuse_existing_stages=False,
                model_dir_name="Qwen3.5-2B",
                model_path=None,
                max_new_tokens=64,
                eval_temperature=0.0,
                seed=42,
                max_steps=4,
                eval_episodes=2,
                headed=False,
            )

            with mock.patch("scripts.run_family_curriculum.load_policy", return_value="policy") as load_policy_mock:
                with mock.patch(
                    "scripts.run_family_curriculum.evaluate_single_task",
                    return_value={"success_rate": 1.0},
                ) as eval_mock:
                    results = _run_baseline_eval(args, split, out_dir)

            self.assertEqual(load_policy_mock.call_count, 1)
            self.assertEqual(eval_mock.call_count, 1)
            self.assertEqual(eval_mock.call_args.kwargs["task_id"], 102)
            self.assertEqual(
                results,
                {
                    "101": {"success_rate": 0.5},
                    "102": {"success_rate": 1.0},
                },
            )

    def test_build_family_run_summary_computes_holdout_transfer(self) -> None:
        split = TaskSplit(
            family_name="shopping_exact",
            task_ids=(101, 102, 103, 104),
            warmup_task_ids=(101,),
            grpo_task_ids=(102, 103),
            holdout_task_ids=(104,),
            eval_task_ids=(101, 102, 103, 104),
            split_seed=42,
        )
        baseline_eval = {
            "101": {"success_rate": 0.0},
            "102": {"success_rate": 0.5},
            "103": {"success_rate": 0.5},
            "104": {"success_rate": 0.0},
        }
        eval_compare = {
            "warmup_eval": {
                "101": {"success_rate": 1.0},
                "102": {"success_rate": 0.5},
                "103": {"success_rate": 0.5},
                "104": {"success_rate": 0.0},
            },
            "grpo_eval": {
                "101": {"success_rate": 1.0},
                "102": {"success_rate": 1.0},
                "103": {"success_rate": 1.0},
                "104": {"success_rate": 1.0},
            },
        }

        summary = build_family_run_summary(split, baseline_eval, eval_compare)

        self.assertEqual(summary["training_task_count"], 3)
        self.assertEqual(summary["baseline"]["family_success_rate"], 0.25)
        self.assertEqual(summary["warmup_only"]["family_success_rate"], 0.5)
        self.assertEqual(summary["warmup_plus_grpo"]["family_success_rate"], 1.0)
        self.assertEqual(summary["holdout"]["gain_vs_baseline"], 1.0)
        self.assertEqual(summary["holdout"]["gain_vs_warmup"], 1.0)

    def test_validate_family_requirements_rejects_full_family_without_openai_key(self) -> None:
        with mock.patch("scripts.run_family_curriculum.get_env_var", return_value=None):
            with self.assertRaises(RuntimeError):
                _validate_family_requirements("shopping_full")

    def test_validate_family_requirements_allows_dry_run_without_openai_key(self) -> None:
        with mock.patch("scripts.run_family_curriculum.get_env_var", return_value=None):
            status = _validate_family_requirements("shopping_full", allow_missing_openai_key=True)

        self.assertEqual(
            status,
            {
                "requires_openai_judge": True,
                "openai_api_key_present": False,
                "run_ready": False,
            },
        )

    def test_validate_family_requirements_allows_exact_family_without_openai_key(self) -> None:
        with mock.patch("scripts.run_family_curriculum.get_env_var", return_value=None):
            _validate_family_requirements("shopping_exact")

    def test_get_family_requirement_status_reports_exact_family_ready_without_key(self) -> None:
        with mock.patch("scripts.run_family_curriculum.get_env_var", return_value=None):
            status = _get_family_requirement_status("shopping_exact")

        self.assertEqual(
            status,
            {
                "requires_openai_judge": False,
                "openai_api_key_present": False,
                "run_ready": True,
            },
        )

    def test_build_family_run_summary_includes_judge_partition_metadata(self) -> None:
        split = TaskSplit(
            family_name="shopping_full",
            task_ids=(96, 128, 324),
            warmup_task_ids=(128,),
            grpo_task_ids=(96,),
            holdout_task_ids=(324,),
            eval_task_ids=(96, 128, 324),
            split_seed=42,
        )
        baseline_eval = {
            "96": {"success_rate": 0.0},
            "128": {"success_rate": 1.0},
            "324": {"success_rate": 0.0},
        }
        eval_compare = {
            "warmup_eval": {
                "96": {"success_rate": 0.0},
                "128": {"success_rate": 1.0},
                "324": {"success_rate": 1.0},
            },
            "grpo_eval": {
                "96": {"success_rate": 1.0},
                "128": {"success_rate": 1.0},
                "324": {"success_rate": 1.0},
            },
        }

        summary = build_family_run_summary(split, baseline_eval, eval_compare)

        self.assertTrue(summary["family_metadata"]["requires_openai_judge"])
        self.assertEqual(summary["family_metadata"]["judge_free_task_ids"], [128, 324])
        self.assertEqual(summary["family_metadata"]["judge_gated_task_ids"], [96])
        self.assertEqual(summary["baseline"]["task_groups"]["judge_free"]["success_rate"], 0.5)
        self.assertEqual(summary["baseline"]["task_groups"]["judge_gated"]["success_rate"], 0.0)
        self.assertEqual(summary["warmup_plus_grpo"]["task_groups"]["judge_gated"]["success_rate"], 1.0)

    def test_build_family_run_summary_includes_run_provenance(self) -> None:
        split = TaskSplit(
            family_name="shopping_exact",
            task_ids=(101, 102, 103),
            warmup_task_ids=(101,),
            grpo_task_ids=(102,),
            holdout_task_ids=(103,),
            eval_task_ids=(101, 102, 103),
            split_seed=42,
        )

        summary = build_family_run_summary(
            split,
            baseline_eval={
                "101": {"success_rate": 0.0},
                "102": {"success_rate": 0.0},
                "103": {"success_rate": 0.0},
            },
            eval_compare={
                "warmup_eval": {
                    "101": {"success_rate": 1.0},
                    "102": {"success_rate": 0.0},
                    "103": {"success_rate": 0.0},
                },
                "grpo_eval": {
                    "101": {"success_rate": 1.0},
                    "102": {"success_rate": 1.0},
                    "103": {"success_rate": 0.0},
                },
            },
            run_provenance={
                "split_manifest_path": "C:\\tmp\\split_manifest.json",
                "split_provenance": {"split_source": "manifest", "split_seed": 42},
            },
        )

        self.assertEqual(summary["run_provenance"]["split_manifest_path"], "C:\\tmp\\split_manifest.json")
        self.assertEqual(summary["run_provenance"]["split_provenance"]["split_source"], "manifest")


if __name__ == "__main__":
    unittest.main()
