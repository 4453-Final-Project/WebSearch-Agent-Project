from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plot_shopping_searchsort_results import (  # noqa: E402
    FamilyStage,
    build_family_stages,
    build_arg_parser,
    write_run_card,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class PlotShoppingSearchsortResultsTests(unittest.TestCase):
    def test_build_arg_parser_accepts_custom_run_metadata(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "--run-dir",
                "/tmp/run",
                "--out-dir",
                "/tmp/out",
                "--run-name",
                "qwen_shopping_disjoint_v1",
                "--model-name",
                "Qwen/Qwen3.5-2B",
                "--title-suffix",
                "Qwen Disjoint Run",
            ]
        )

        self.assertEqual(args.run_dir, "/tmp/run")
        self.assertEqual(args.out_dir, "/tmp/out")
        self.assertEqual(args.run_name, "qwen_shopping_disjoint_v1")
        self.assertEqual(args.model_name, "Qwen/Qwen3.5-2B")
        self.assertEqual(args.title_suffix, "Qwen Disjoint Run")

    def test_build_family_stages_and_run_card_use_selected_run_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "qwen_shopping_disjoint_v1"
            _write_json(run_dir / "eval_baseline" / "task_324" / "metrics.json", {"success_rate": 0.0})
            _write_json(run_dir / "eval_baseline" / "task_325" / "metrics.json", {"success_rate": 1.0})
            _write_json(run_dir / "eval_baseline" / "task_326" / "metrics.json", {"success_rate": 1.0})
            _write_json(run_dir / "eval_baseline" / "task_327" / "metrics.json", {"success_rate": 0.0})
            _write_json(run_dir / "eval_baseline" / "task_328" / "metrics.json", {"success_rate": 1.0})
            _write_json(
                run_dir / "eval_compare.json",
                {
                    "warmup_eval": {
                        "324": {"success_rate": 0.0, "average_steps": 4.0},
                        "325": {"success_rate": 1.0, "average_steps": 3.0},
                        "326": {"success_rate": 1.0, "average_steps": 2.0},
                        "327": {"success_rate": 1.0, "average_steps": 3.0},
                        "328": {"success_rate": 1.0, "average_steps": 3.0},
                    },
                    "grpo_eval": {
                        "324": {"success_rate": 1.0, "average_steps": 4.0},
                        "325": {"success_rate": 1.0, "average_steps": 3.0},
                        "326": {"success_rate": 1.0, "average_steps": 2.0},
                        "327": {"success_rate": 1.0, "average_steps": 3.0},
                        "328": {"success_rate": 1.0, "average_steps": 3.0},
                    },
                },
            )
            _write_json(
                run_dir / "warmup_summary.json",
                {
                    "warmup_task_ids": [325, 326],
                    "grpo_task_ids": [327, 328],
                    "eval_task_ids": [324, 325, 326, 327, 328],
                    "holdout_task_ids": [324],
                    "selected_demo_count": 6,
                    "selected_demo_task_counts": {"325": 3, "326": 3},
                    "warmup_metrics": {"epochs": 1, "final_loss": 3.0, "success_demo_count": 6},
                },
            )
            _write_json(
                run_dir / "rollout_summary.json",
                {
                    "episode_count": 20,
                    "success_count": 19,
                    "task_success_counts": {"327": 9, "328": 10},
                    "episodes": [{"steps_taken": 3.0}] * 20,
                },
            )
            _write_json(
                run_dir / "grpo_summary.json",
                [
                    {
                        "iteration": 0,
                        "success_rate": 0.95,
                    }
                ],
            )

            stages = build_family_stages(run_dir)
            self.assertEqual([stage.label for stage in stages], ["baseline", "warmup_only", "warmup_plus_grpo"])
            self.assertEqual(stages[0].per_task["327"], 0.0)
            self.assertEqual(stages[2].per_task["324"], 1.0)

            out_dir = Path(temp_dir) / "report"
            run_card_path = write_run_card(
                out_dir,
                stages,
                disjoint_dir=run_dir,
                run_name="qwen_shopping_disjoint_v1",
                model_name="Qwen/Qwen3.5-2B",
            )
            run_card = json.loads(run_card_path.read_text(encoding="utf-8"))

            self.assertEqual(run_card["run_name"], "qwen_shopping_disjoint_v1")
            self.assertEqual(run_card["model"], "Qwen/Qwen3.5-2B")
            self.assertEqual(run_card["task_split"]["grpo_task_ids"], [327, 328])
            self.assertEqual(run_card["task_split"]["holdout_task_ids"], [324])
            self.assertEqual(run_card["holdout"]["gain_vs_baseline"], 1.0)
            self.assertEqual(run_card["holdout"]["gain_vs_warmup"], 1.0)


if __name__ == "__main__":
    unittest.main()
