from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_grpo_staged import (
    _build_reward_config,
    _build_sample_weight_config,
    _evaluate_adapter,
    _normalize_stage_task_sets,
    _resolve_task_groups_per_task,
    _resolve_task_max_steps,
    _resolve_warmup_demo_limits,
    _resolve_warmup_sample_multipliers,
    _run_collect,
    _select_warmup_trajectories,
)


class TrainGrpoStagedTests(unittest.TestCase):
    def test_explicit_warmup_tasks_must_be_disjoint_by_default(self) -> None:
        args = Namespace(
            task_ids=[325, 326],
            warmup_task_ids=[324, 325],
            eval_task_ids=[],
            holdout_task_ids=[],
            allow_task_overlap=False,
        )

        with self.assertRaises(SystemExit):
            _normalize_stage_task_sets(args)

    def test_task_sets_default_to_grpo_tasks_when_omitted(self) -> None:
        args = Namespace(
            task_ids=[325, 326, 326],
            warmup_task_ids=[],
            eval_task_ids=[],
            holdout_task_ids=[],
            allow_task_overlap=False,
        )

        _normalize_stage_task_sets(args)

        self.assertEqual(args.task_ids, [325, 326])
        self.assertEqual(args.warmup_task_ids, [325, 326])
        self.assertEqual(args.eval_task_ids, [325, 326])
        self.assertEqual(args.holdout_task_ids, [])

    def test_explicit_disjoint_warmup_and_eval_sets_are_preserved(self) -> None:
        args = Namespace(
            task_ids=[327, 328],
            warmup_task_ids=[325, 326],
            eval_task_ids=[324, 329],
            holdout_task_ids=[],
            allow_task_overlap=False,
        )

        _normalize_stage_task_sets(args)

        self.assertEqual(args.task_ids, [327, 328])
        self.assertEqual(args.warmup_task_ids, [325, 326])
        self.assertEqual(args.eval_task_ids, [324, 329])
        self.assertEqual(args.holdout_task_ids, [324, 329])

    def test_explicit_holdout_tasks_must_be_disjoint_from_training_sets(self) -> None:
        args = Namespace(
            task_ids=[327, 328],
            warmup_task_ids=[325, 326],
            eval_task_ids=[324, 327],
            holdout_task_ids=[327],
            allow_task_overlap=False,
        )

        with self.assertRaises(SystemExit):
            _normalize_stage_task_sets(args)

    def test_explicit_holdout_tasks_must_be_included_in_eval_set(self) -> None:
        args = Namespace(
            task_ids=[327, 328],
            warmup_task_ids=[325, 326],
            eval_task_ids=[325, 326, 327, 328],
            holdout_task_ids=[324],
            allow_task_overlap=False,
        )

        with self.assertRaises(SystemExit):
            _normalize_stage_task_sets(args)

    def test_evaluate_adapter_reuses_existing_metrics_and_only_runs_missing_tasks(self) -> None:
        args = Namespace(
            eval_task_ids=[101, 102],
            seed=7,
            max_steps=4,
            task_max_steps={102: 8},
            eval_episodes=2,
            headless=True,
            eval_temperature=0.0,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_root = Path(tmp_dir)
            task_101_dir = output_root / "task_101"
            task_101_dir.mkdir(parents=True)
            (task_101_dir / "metrics.json").write_text('{"success_rate": 0.5}', encoding="utf-8")

            with (
                patch("scripts.train_grpo_staged._build_policy", return_value="policy") as build_policy,
                patch(
                    "scripts.train_grpo_staged.evaluate_single_task",
                    return_value={"success_rate": 1.0},
                ) as evaluate_single_task_mock,
            ):
                results = _evaluate_adapter(args, Path("adapter"), output_root)

        self.assertEqual(results, {"101": {"success_rate": 0.5}, "102": {"success_rate": 1.0}})
        build_policy.assert_called_once_with(args, model_path="adapter", temperature=0.0)
        evaluate_single_task_mock.assert_called_once()
        self.assertEqual(evaluate_single_task_mock.call_args.kwargs["task_id"], 102)
        self.assertEqual(evaluate_single_task_mock.call_args.kwargs["max_steps"], 8)

    def test_resolve_task_max_steps_parses_cli_entries(self) -> None:
        args = Namespace(task_max_steps=["133=8", "293=6"])

        overrides = _resolve_task_max_steps(args)

        self.assertEqual(overrides, {133: 8, 293: 6})

    def test_resolve_warmup_demo_limits_parses_cli_entries(self) -> None:
        args = Namespace(warmup_demo_limit_override=["132=4", "133=5"])

        overrides = _resolve_warmup_demo_limits(args)

        self.assertEqual(overrides, {132: 4, 133: 5})

    def test_resolve_warmup_demo_limits_accepts_dict_overrides(self) -> None:
        args = Namespace(warmup_demo_limit_override={132: 4, 133: 5})

        overrides = _resolve_warmup_demo_limits(args)

        self.assertEqual(overrides, {132: 4, 133: 5})

    def test_resolve_task_groups_per_task_parses_cli_entries(self) -> None:
        args = Namespace(task_groups_per_task=["134=4", "135=5"])

        overrides = _resolve_task_groups_per_task(args)

        self.assertEqual(overrides, {134: 4, 135: 5})

    def test_resolve_task_groups_per_task_accepts_dict_overrides(self) -> None:
        args = Namespace(task_groups_per_task={134: 4, 135: 5})

        overrides = _resolve_task_groups_per_task(args)

        self.assertEqual(overrides, {134: 4, 135: 5})

    def test_resolve_warmup_sample_multipliers_parses_cli_entries(self) -> None:
        args = Namespace(warmup_sample_multiplier_override=["132=2.5", "133=3.0"])

        overrides = _resolve_warmup_sample_multipliers(args)

        self.assertEqual(overrides, {132: 2.5, 133: 3.0})

    def test_resolve_warmup_sample_multipliers_accepts_dict_overrides(self) -> None:
        args = Namespace(warmup_sample_multiplier_override={132: 2.5, 133: 3.0})

        overrides = _resolve_warmup_sample_multipliers(args)

        self.assertEqual(overrides, {132: 2.5, 133: 3.0})

    def test_select_warmup_trajectories_uses_per_task_limits(self) -> None:
        trajectories = [
            Namespace(task_id=132),
            Namespace(task_id=132),
            Namespace(task_id=132),
            Namespace(task_id=21),
            Namespace(task_id=21),
        ]

        with patch("scripts.train_grpo_staged.load_demo_trajectories", return_value=trajectories):
            selected = _select_warmup_trajectories(
                ["demo_dir"],
                [132, 21],
                2,
                per_task_limits={132: 3},
            )

        self.assertEqual([trajectory.task_id for trajectory in selected], [132, 132, 132, 21, 21])

    def test_run_collect_uses_task_specific_max_steps(self) -> None:
        args = Namespace(
            task_ids=[133, 293],
            warmup_task_ids=[133, 293],
            holdout_task_ids=[],
            groups_per_task=1,
            task_groups_per_task={133: 2},
            group_size=1,
            seed=7,
            max_steps=4,
            task_max_steps={133: 8},
            headless=True,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            with (
                patch("scripts.train_grpo_staged._build_policy", return_value="policy"),
                patch("scripts.train_grpo_staged.run_episode") as run_episode_mock,
                patch(
                    "scripts.train_grpo_staged.load_episode_trajectory",
                    side_effect=[
                        Namespace(
                            success=True,
                            reward=1.0,
                            steps_taken=4,
                            invalid_action_count=0,
                            parse_failure_count=0,
                        ),
                        Namespace(
                            success=True,
                            reward=1.0,
                            steps_taken=4,
                            invalid_action_count=0,
                            parse_failure_count=0,
                        ),
                        Namespace(
                            success=False,
                            reward=0.0,
                            steps_taken=4,
                            invalid_action_count=0,
                            parse_failure_count=0,
                        ),
                    ],
                ),
            ):
                _run_collect(args, out_dir)

        self.assertEqual(run_episode_mock.call_count, 3)
        self.assertEqual(run_episode_mock.call_args_list[0].args[3], 8)
        self.assertEqual(run_episode_mock.call_args_list[1].args[3], 8)
        self.assertEqual(run_episode_mock.call_args_list[2].args[3], 4)

    def test_build_reward_config_uses_cli_overrides(self) -> None:
        args = Namespace(
            reward_scale=2.0,
            success_bonus=1.5,
            success_step_bonus=0.2,
            invalid_action_penalty=0.3,
            parse_failure_penalty=0.25,
            truncation_penalty=0.15,
            repeat_action_penalty=0.07,
            same_page_repeat_action_penalty=0.11,
            per_step_penalty=0.04,
            success_unique_url_bonus=0.08,
            max_unique_url_bonus_urls=6,
            step_weight_later_step_bonus=0.9,
            step_weight_terminal_success_bonus=1.7,
            step_weight_error_step_multiplier=0.4,
            step_weight_min=0.02,
        )

        reward_config = _build_reward_config(args)

        self.assertEqual(reward_config.reward_scale, 2.0)
        self.assertEqual(reward_config.success_bonus, 1.5)
        self.assertEqual(reward_config.same_page_repeat_action_penalty, 0.11)
        self.assertEqual(reward_config.max_unique_url_bonus_urls, 6)
        self.assertEqual(reward_config.sample_weight_config.later_step_bonus, 0.9)
        self.assertEqual(reward_config.sample_weight_config.terminal_success_bonus, 1.7)

    def test_build_sample_weight_config_uses_cli_overrides(self) -> None:
        args = Namespace(
            step_weight_later_step_bonus=0.8,
            step_weight_terminal_success_bonus=1.4,
            step_weight_error_step_multiplier=0.3,
            step_weight_min=0.07,
        )

        sample_weight_config = _build_sample_weight_config(args)

        self.assertEqual(sample_weight_config.later_step_bonus, 0.8)
        self.assertEqual(sample_weight_config.terminal_success_bonus, 1.4)
        self.assertEqual(sample_weight_config.error_step_multiplier, 0.3)
        self.assertEqual(sample_weight_config.min_step_weight, 0.07)

    def test_arg_parser_exposes_quantization_overrides(self) -> None:
        from scripts.train_grpo_staged import build_arg_parser

        args = build_arg_parser().parse_args(
            [
                "--stage",
                "warmup",
                "--task-id",
                "325",
                "--task-max-steps",
                "325=8",
                "--warmup-demo-limit-override",
                "325=4",
                "--task-groups-per-task",
                "325=3",
                "--warmup-sample-multiplier-override",
                "325=2.5",
                "--quantization-mode",
                "bnb_4bit",
                "--quant-compute-dtype",
                "float16",
                "--quant-type",
                "nf4",
                "--no-quant-use-double-quant",
                "--step-weight-later-step-bonus",
                "0.8",
                "--step-weight-terminal-success-bonus",
                "1.4",
                "--step-weight-error-step-multiplier",
                "0.3",
                "--step-weight-min",
                "0.07",
            ]
        )

        self.assertEqual(args.quantization_mode, "bnb_4bit")
        self.assertEqual(args.task_max_steps, ["325=8"])
        self.assertEqual(args.warmup_demo_limit_override, ["325=4"])
        self.assertEqual(args.task_groups_per_task, ["325=3"])
        self.assertEqual(args.warmup_sample_multiplier_override, ["325=2.5"])
        self.assertEqual(args.quant_compute_dtype, "float16")
        self.assertEqual(args.quant_type, "nf4")
        self.assertFalse(args.quant_use_double_quant)
        self.assertEqual(args.step_weight_later_step_bonus, 0.8)
        self.assertEqual(args.step_weight_terminal_success_bonus, 1.4)
        self.assertEqual(args.step_weight_error_step_multiplier, 0.3)
        self.assertEqual(args.step_weight_min, 0.07)


if __name__ == "__main__":
    unittest.main()
