from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.live_family_run_status import build_live_family_run_status  # noqa: E402


def _to_wsl_path(path: Path) -> str:
    path_str = str(path)
    if len(path_str) >= 3 and path_str[1:3] == ":\\":
        rest = path_str[3:].replace("\\", "/")
        return f"/mnt/{path_str[0].lower()}/{rest}"
    return path_str.replace("\\", "/")


class LiveFamilyRunStatusTests(unittest.TestCase):
    def test_reports_warmup_stage_before_baseline_starts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            (out_dir / "preflight.json").write_text(
                json.dumps(
                    {
                        "family_name": "shopping_full",
                        "split_preview": {"training_task_count": 40},
                    }
                ),
                encoding="utf-8",
            )
            warmup_root = out_dir / "warmup_demos" / "task_0188"
            warmup_root.mkdir(parents=True)
            (warmup_root / "summary.json").write_text("{}", encoding="utf-8")
            (out_dir / "warmup_demos" / "summary.json").write_text(
                json.dumps({"task_count": 16}),
                encoding="utf-8",
            )

            status = build_live_family_run_status(out_dir)

        self.assertEqual(status["family_name"], "shopping_full")
        self.assertEqual(status["current_stage"], "warmup_demos")
        self.assertIsNotNone(status["latest_activity"]["path"])
        self.assertIsNotNone(status["latest_non_monitoring_activity"]["path"])
        self.assertGreaterEqual(status["latest_activity"]["age_seconds"], 0.0)
        self.assertEqual(status["warmup_demos"]["completed_task_count"], 1)
        self.assertEqual(status["warmup_in_progress_activity"], {})
        self.assertEqual(status["warmup_task_groups"], {})
        self.assertEqual(status["baseline_eval"]["completed_task_count"], 0)
        self.assertEqual(status["baseline_in_progress_activity"], {})
        self.assertFalse(status["run_stdout_log"]["exists"])
        self.assertFalse(status["run_stderr_log"]["exists"])
        self.assertEqual(status["baseline_task_groups"], {})
        self.assertEqual(status["baseline_progress_task_groups"], {})
        self.assertEqual(status["warmup_training"], {})
        self.assertEqual(status["rollout_collection"]["started_episode_count"], 0)
        self.assertEqual(status["grpo_training"]["checkpoint_count"], 0)
        self.assertEqual(
            status["run_process"],
            {
                "run_pid_present": False,
                "saved_run_pid": None,
                "run_pid": None,
                "run_pid_alive": None,
                "run_pid_source": "missing",
                "probe_backend": None,
                "probe_output": None,
            },
        )
        self.assertEqual(
            status["tmux_session"],
            {
                "session_name_present": False,
                "session_name": None,
                "session_alive": None,
                "probe_backend": None,
                "probe_output": None,
                "pane_pid": None,
                "pane_command": None,
            },
        )

    def test_reports_baseline_progress_from_partial_eval_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            (out_dir / "split_manifest.json").write_text(
                json.dumps(
                    {
                        "family_name": "shopping_full",
                        "split_preview": {
                            "task_ids": [96, 128, 324],
                            "training_task_count": 40,
                            "holdout_task_count": 8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            warmup_root = out_dir / "warmup_demos"
            warmup_root.mkdir(parents=True)
            (warmup_root / "summary.json").write_text(json.dumps({"task_count": 16}), encoding="utf-8")
            completed = out_dir / "eval_baseline" / "task_0096"
            completed.mkdir(parents=True)
            (completed / "metrics.json").write_text(json.dumps({"success_rate": 0.0}), encoding="utf-8")
            completed_two = out_dir / "eval_baseline" / "task_0128"
            completed_two.mkdir(parents=True)
            (completed_two / "metrics.json").write_text(json.dumps({"success_rate": 1.0}), encoding="utf-8")
            started = out_dir / "eval_baseline" / "task_0117"
            started.mkdir(parents=True)
            latest = started / "episode_0" / "screenshots"
            latest.mkdir(parents=True)
            (latest / "step_0000.png").write_text("png", encoding="utf-8")

            status = build_live_family_run_status(out_dir)

        self.assertEqual(status["current_stage"], "baseline_evaluation")
        self.assertEqual(status["baseline_eval"]["started_task_count"], 3)
        self.assertEqual(status["baseline_eval"]["completed_task_count"], 2)
        self.assertEqual(status["baseline_eval"]["completed_task_ids"], [96, 128])
        self.assertEqual(status["baseline_eval"]["in_progress_task_ids"], [117])
        self.assertEqual(status["baseline_metrics"]["completed_task_count"], 2)
        self.assertEqual(status["baseline_metrics"]["success_task_count"], 1)
        self.assertEqual(status["baseline_metrics"]["average_success_rate"], 0.5)
        self.assertEqual(
            status["baseline_metrics"]["per_task_success_rate"],
            {"96": 0.0, "128": 1.0},
        )
        self.assertEqual(
            status["baseline_task_groups"]["judge_free"],
            {
                "completed_task_count": 1,
                "average_success_rate": 1.0,
                "success_task_count": 1,
                "completed_task_ids": [128],
            },
        )
        self.assertEqual(
            status["baseline_task_groups"]["judge_gated"],
            {
                "completed_task_count": 1,
                "average_success_rate": 0.0,
                "success_task_count": 0,
                "completed_task_ids": [96],
            },
        )
        self.assertEqual(
            status["baseline_progress_task_groups"]["judge_gated"],
            {
                "started_task_count": 1,
                "completed_task_count": 1,
                "in_progress_task_count": 0,
                "started_task_ids": [96],
                "completed_task_ids": [96],
                "in_progress_task_ids": [],
            },
        )
        self.assertEqual(
            status["baseline_progress_task_groups"]["judge_free"],
            {
                "started_task_count": 1,
                "completed_task_count": 1,
                "in_progress_task_count": 0,
                "started_task_ids": [128],
                "completed_task_ids": [128],
                "in_progress_task_ids": [],
            },
        )
        self.assertEqual(status["warmup_eval"]["eval"]["started_task_count"], 0)
        self.assertEqual(status["grpo_eval"]["eval"]["started_task_count"], 0)
        self.assertTrue(str(status["baseline_in_progress_activity"]["117"]["path"]).endswith("step_0000.png"))

    def test_reports_warmup_in_progress_activity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            (out_dir / "split_manifest.json").write_text(
                json.dumps(
                    {
                        "family_name": "bootstrap41",
                        "split_preview": {
                            "task_ids": [0],
                            "training_task_count": 33,
                            "holdout_task_count": 8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            started = out_dir / "warmup_demos" / "task_0000" / "episode_0000" / "screenshots"
            started.mkdir(parents=True)
            (started / "step_0000.png").write_text("png", encoding="utf-8")

            status = build_live_family_run_status(out_dir)

        self.assertEqual(status["current_stage"], "warmup_demos")
        self.assertEqual(status["warmup_demos"]["in_progress_task_ids"], [0])
        self.assertTrue(str(status["warmup_in_progress_activity"]["0"]["path"]).endswith("step_0000.png"))

    def test_prefers_baseline_stage_while_partial_summary_and_in_progress_task_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            (out_dir / "split_manifest.json").write_text(
                json.dumps(
                    {
                        "family_name": "shopping_full",
                        "split_preview": {
                            "task_ids": [96, 117, 128],
                            "training_task_count": 40,
                            "holdout_task_count": 8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "baseline_eval_summary.json").write_text(
                json.dumps({"96": {"success_rate": 1.0}}),
                encoding="utf-8",
            )
            completed = out_dir / "eval_baseline" / "task_0096"
            completed.mkdir(parents=True)
            (completed / "metrics.json").write_text(json.dumps({"success_rate": 1.0}), encoding="utf-8")
            started = out_dir / "eval_baseline" / "task_0117"
            started.mkdir(parents=True)

            status = build_live_family_run_status(out_dir)

        self.assertEqual(status["current_stage"], "baseline_evaluation")
        self.assertEqual(status["baseline_eval"]["in_progress_task_ids"], [117])

    def test_reports_later_stage_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            (out_dir / "split_manifest.json").write_text(
                json.dumps(
                    {
                        "family_name": "shopping_full",
                        "split_preview": {
                            "task_ids": [96, 128, 324],
                            "training_task_count": 40,
                            "holdout_task_count": 8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "warmup_summary.json").write_text(
                json.dumps(
                    {
                        "selected_demo_count": 16,
                        "selected_demo_task_counts": {"128": 2},
                        "warmup_metrics": {
                            "average_demo_reward": 1.0,
                            "final_loss": 0.02,
                            "epochs": 1,
                            "success_demo_count": 16,
                        },
                    }
                ),
                encoding="utf-8",
            )
            rollout_episode = out_dir / "rollouts" / "iteration_0000" / "task_128" / "group_0000" / "episode_0000"
            rollout_episode.mkdir(parents=True)
            (rollout_episode / "episode.json").write_text("{}", encoding="utf-8")
            (out_dir / "rollout_summary.json").write_text(
                json.dumps(
                    {
                        "episode_count": 4,
                        "success_count": 3,
                        "task_success_counts": {"128": 3},
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_dir = out_dir / "checkpoints" / "iter_0000"
            checkpoint_dir.mkdir(parents=True)
            (out_dir / "grpo_summary.json").write_text(
                json.dumps(
                    [
                        {
                            "iteration": 0,
                            "success_rate": 0.75,
                            "episode_reward": 0.9,
                            "loss": 0.01,
                            "approx_kl": 0.02,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            warmup_eval_task = out_dir / "eval_warmup" / "task_0128"
            warmup_eval_task.mkdir(parents=True)
            (warmup_eval_task / "metrics.json").write_text(json.dumps({"success_rate": 1.0}), encoding="utf-8")
            grpo_eval_task = out_dir / "eval_grpo" / "task_0096"
            grpo_eval_task.mkdir(parents=True)
            (grpo_eval_task / "metrics.json").write_text(json.dumps({"success_rate": 0.0}), encoding="utf-8")

            status = build_live_family_run_status(out_dir)

        self.assertEqual(status["current_stage"], "final_evaluation")
        self.assertEqual(
            status["warmup_training"],
            {
                "selected_demo_count": 16,
                "selected_demo_task_counts": {"128": 2},
                "average_demo_reward": 1.0,
                "final_loss": 0.02,
                "epochs": 1,
                "success_demo_count": 16,
            },
        )
        self.assertEqual(status["rollout_collection"]["episode_count"], 4)
        self.assertEqual(status["rollout_collection"]["success_count"], 3)
        self.assertEqual(status["rollout_collection"]["success_rate"], 0.75)
        self.assertEqual(status["rollout_collection"]["started_episode_count"], 1)
        self.assertEqual(status["grpo_training"]["iteration_count"], 1)
        self.assertEqual(status["grpo_training"]["checkpoint_count"], 1)
        self.assertEqual(status["warmup_eval"]["eval"]["completed_task_ids"], [128])
        self.assertEqual(status["warmup_eval"]["metrics"]["average_success_rate"], 1.0)
        self.assertEqual(status["grpo_eval"]["eval"]["completed_task_ids"], [96])
        self.assertEqual(status["grpo_eval"]["task_groups"]["judge_gated"]["completed_task_ids"], [96])

    def test_reports_complete_when_family_summary_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            (out_dir / "family_summary.json").write_text(json.dumps({"family_name": "shopping_full"}), encoding="utf-8")

            status = build_live_family_run_status(out_dir)

        self.assertEqual(status["current_stage"], "complete")
        self.assertTrue(status["artifact_presence"]["family_summary"])

    def test_reports_latest_activity_for_newest_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            older = out_dir / "preflight.json"
            newer = out_dir / "split_manifest.json"
            older.write_text("{}", encoding="utf-8")
            newer.write_text("{}", encoding="utf-8")
            older_stat = older.stat()
            os.utime(older, (older_stat.st_atime, older_stat.st_mtime - 10))

            status = build_live_family_run_status(out_dir)

        self.assertTrue(str(status["latest_activity"]["path"]).endswith("split_manifest.json"))
        self.assertGreaterEqual(status["latest_activity"]["age_seconds"], 0.0)

    def test_reports_run_log_tails_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            stdout_log = out_dir / "run.stdout.log"
            stderr_log = out_dir / "run.stderr.log"
            stdout_log.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")
            stderr_log.write_text("warn 1\n\nwarn 2\n", encoding="utf-8")

            status = build_live_family_run_status(out_dir)

        self.assertTrue(status["run_stdout_log"]["exists"])
        self.assertEqual(status["run_stdout_log"]["last_nonempty_line"], "line 3")
        self.assertEqual(status["run_stdout_log"]["last_lines"], ["line 1", "line 2", "line 3"])
        self.assertTrue(status["run_stderr_log"]["exists"])
        self.assertEqual(status["run_stderr_log"]["last_nonempty_line"], "warn 2")
        self.assertEqual(status["run_stderr_log"]["last_lines"], ["warn 1", "", "warn 2"])

    def test_falls_back_to_legacy_run_log_when_stdout_log_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            legacy_stdout_log = out_dir / "run.log"
            legacy_stdout_log.write_text("boot\nstep\n", encoding="utf-8")

            status = build_live_family_run_status(out_dir)

        self.assertTrue(status["run_stdout_log"]["exists"])
        self.assertTrue(str(status["run_stdout_log"]["path"]).endswith("run.log"))
        self.assertEqual(status["run_stdout_log"]["last_nonempty_line"], "step")
        self.assertEqual(status["run_stdout_log"]["last_lines"], ["boot", "step"])

    def test_reports_latest_non_monitoring_activity_excluding_live_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            artifact = out_dir / "split_manifest.json"
            monitor = out_dir / "live_status.json"
            artifact.write_text("{}", encoding="utf-8")
            monitor.write_text("{}", encoding="utf-8")
            artifact_stat = artifact.stat()
            os.utime(artifact, (artifact_stat.st_atime, artifact_stat.st_mtime - 10))

            status = build_live_family_run_status(out_dir)

        self.assertTrue(str(status["latest_activity"]["path"]).endswith("live_status.json"))
        self.assertTrue(str(status["latest_non_monitoring_activity"]["path"]).endswith("split_manifest.json"))

    def test_reports_latest_non_monitoring_activity_excluding_metadata_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            artifact = out_dir / "warmup_demos" / "task_0188" / "summary.json"
            artifact.parent.mkdir(parents=True)
            metadata_paths = [
                out_dir / "live_status.json",
                out_dir / "run.pid",
                out_dir / "tmux_session.txt",
            ]
            artifact.write_text("{}", encoding="utf-8")
            artifact_stat = artifact.stat()
            os.utime(artifact, (artifact_stat.st_atime, artifact_stat.st_mtime - 10))
            for metadata_path in metadata_paths:
                metadata_path.write_text("{}", encoding="utf-8")

            status = build_live_family_run_status(out_dir)

        latest_activity_path = str(status["latest_activity"]["path"])
        self.assertTrue(
            latest_activity_path.endswith("live_status.json")
            or latest_activity_path.endswith("run.pid")
            or latest_activity_path.endswith("tmux_session.txt")
        )
        self.assertTrue(str(status["latest_non_monitoring_activity"]["path"]).endswith("summary.json"))

    @mock.patch("scripts.live_family_run_status.subprocess.run")
    def test_reports_wsl_run_pid_probe_when_pid_exists(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = mock.Mock(returncode=0, stdout="390934 S python -u scripts/run_family_curriculum.py\n")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            (out_dir / "run.pid").write_text("390934\n", encoding="utf-8")

            status = build_live_family_run_status(out_dir)

        self.assertTrue(status["run_process"]["run_pid_present"])
        self.assertEqual(status["run_process"]["saved_run_pid"], 390934)
        self.assertEqual(status["run_process"]["run_pid"], 390934)
        self.assertTrue(status["run_process"]["run_pid_alive"])
        self.assertEqual(status["run_process"]["run_pid_source"], "pid_file")
        self.assertIn("python -u scripts/run_family_curriculum.py", status["run_process"]["probe_output"])

    @mock.patch("scripts.live_family_run_status.subprocess.run")
    def test_reports_dead_run_pid_when_probe_finds_nothing(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = mock.Mock(returncode=1, stdout="")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            (out_dir / "run.pid").write_text("390934\n", encoding="utf-8")

            status = build_live_family_run_status(out_dir)

        self.assertTrue(status["run_process"]["run_pid_present"])
        self.assertEqual(status["run_process"]["saved_run_pid"], 390934)
        self.assertEqual(status["run_process"]["run_pid"], 390934)
        self.assertFalse(status["run_process"]["run_pid_alive"])
        self.assertEqual(status["run_process"]["run_pid_source"], "pid_file")

    @mock.patch("scripts.live_family_run_status.subprocess.run")
    def test_falls_back_to_process_scan_when_saved_pid_is_stale(self, mock_run: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            mock_run.side_effect = [
                mock.Mock(returncode=1, stdout=""),
                mock.Mock(
                    returncode=0,
                    stdout=(
                        "tyler 401234 1 0 00:00 ? "
                        "python -u scripts/run_family_curriculum.py --family shopping_full "
                        f"--out-dir {_to_wsl_path(out_dir)}\n"
                    ),
                ),
            ]
            (out_dir / "run.pid").write_text("390934\n", encoding="utf-8")
            (out_dir / "split_manifest.json").write_text(
                json.dumps(
                    {
                        "family_name": "shopping_full",
                        "split_preview": {
                            "task_ids": [96],
                            "training_task_count": 40,
                            "holdout_task_count": 8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            status = build_live_family_run_status(out_dir)

        self.assertTrue(status["run_process"]["run_pid_present"])
        self.assertEqual(status["run_process"]["saved_run_pid"], 390934)
        self.assertEqual(status["run_process"]["run_pid"], 401234)
        self.assertTrue(status["run_process"]["run_pid_alive"])
        self.assertEqual(status["run_process"]["run_pid_source"], "scan")

    @mock.patch("scripts.live_family_run_status.subprocess.run")
    def test_can_discover_run_process_without_pid_file(self, mock_run: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            mock_run.return_value = mock.Mock(
                returncode=0,
                stdout=(
                    "tyler 402345 1 0 00:00 ? "
                    "python -u scripts/run_family_curriculum.py --family shopping_full "
                    f"--out-dir {_to_wsl_path(out_dir)}\n"
                ),
            )
            (out_dir / "split_manifest.json").write_text(
                json.dumps(
                    {
                        "family_name": "shopping_full",
                        "split_preview": {
                            "task_ids": [96],
                            "training_task_count": 40,
                            "holdout_task_count": 8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            status = build_live_family_run_status(out_dir)

        self.assertFalse(status["run_process"]["run_pid_present"])
        self.assertEqual(status["run_process"]["saved_run_pid"], None)
        self.assertEqual(status["run_process"]["run_pid"], 402345)
        self.assertTrue(status["run_process"]["run_pid_alive"])
        self.assertEqual(status["run_process"]["run_pid_source"], "scan")

    @mock.patch("scripts.live_family_run_status.subprocess.run")
    def test_reports_tmux_session_when_present(self, mock_run: mock.Mock) -> None:
        mock_run.return_value = mock.Mock(returncode=0, stdout="399790\tpython3\n")
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            (out_dir / "tmux_session.txt").write_text("qwen_shopping_full_v3_balanced\n", encoding="utf-8")

            status = build_live_family_run_status(out_dir)

        self.assertTrue(status["tmux_session"]["session_name_present"])
        self.assertEqual(status["tmux_session"]["session_name"], "qwen_shopping_full_v3_balanced")
        self.assertTrue(status["tmux_session"]["session_alive"])
        self.assertEqual(status["tmux_session"]["pane_pid"], 399790)
        self.assertEqual(status["tmux_session"]["pane_command"], "python3")


if __name__ == "__main__":
    unittest.main()
