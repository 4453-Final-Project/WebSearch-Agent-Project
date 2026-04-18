from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SCRIPTS = PROJECT_ROOT / "scripts" / "local"


def _read_local_script(name: str) -> str:
    return (LOCAL_SCRIPTS / name).read_text(encoding="utf-8")


class LocalLaunchersTests(unittest.TestCase):
    def test_exact_family_launchers_use_checked_in_manifest(self) -> None:
        qwen_script = _read_local_script("run_qwen_shopping_exact_curriculum.sh")
        liquid_script = _read_local_script("run_liquid_shopping_exact_curriculum.sh")

        self.assertIn("shopping_exact_curriculum_manifest.json", qwen_script)
        self.assertIn("shopping_exact_curriculum_manifest.json", liquid_script)
        self.assertIn("--split-manifest", qwen_script)
        self.assertIn("--split-manifest", liquid_script)

    def test_full_family_launchers_use_shared_checked_in_manifest(self) -> None:
        qwen_script = _read_local_script("run_qwen_shopping_full_curriculum.sh")
        liquid_script = _read_local_script("run_liquid_shopping_full_curriculum.sh")

        self.assertIn("shopping_full_curriculum_manifest.json", qwen_script)
        self.assertIn("shopping_full_curriculum_manifest.json", liquid_script)
        self.assertNotIn("qwen_shopping_full_curriculum_manifest.json", qwen_script)
        self.assertNotIn("qwen_shopping_full_curriculum_manifest.json", liquid_script)
        self.assertIn("--split-manifest", qwen_script)
        self.assertIn("--split-manifest", liquid_script)

    def test_qwen_full_launcher_keeps_preflight_fallback(self) -> None:
        qwen_script = _read_local_script("run_qwen_shopping_full_curriculum.sh")

        self.assertIn("qwen_shopping_full_curriculum_preflight_v1/preflight.json", qwen_script)
        self.assertIn("elif [[ -f \"$PREFLIGHT_PATH\" ]]", qwen_script)

    def test_bootstrap41_launchers_use_checked_in_manifest(self) -> None:
        qwen_script = _read_local_script("run_qwen_bootstrap41_curriculum.sh")
        liquid_script = _read_local_script("run_liquid_bootstrap41_curriculum.sh")

        self.assertIn("bootstrap41_curriculum_manifest.json", qwen_script)
        self.assertIn("bootstrap41_curriculum_manifest.json", liquid_script)
        self.assertIn("--split-manifest", qwen_script)
        self.assertIn("--split-manifest", liquid_script)

    def test_web_mix88_launchers_use_checked_in_manifest(self) -> None:
        qwen_script = _read_local_script("run_qwen_web_mix88_curriculum.sh")
        liquid_script = _read_local_script("run_liquid_web_mix88_curriculum.sh")

        self.assertIn("web_mix88_curriculum_manifest.json", qwen_script)
        self.assertIn("web_mix88_curriculum_manifest.json", liquid_script)
        self.assertIn("--split-manifest", qwen_script)
        self.assertIn("--split-manifest", liquid_script)

    def test_qwen_tmux_launchers_write_tmux_session_metadata(self) -> None:
        shopping_full_tmux = _read_local_script("run_qwen_shopping_full_curriculum_v3_balanced_tmux.sh")
        bootstrap41_tmux = _read_local_script("run_qwen_bootstrap41_curriculum_tmux.sh")
        bootstrap41_qlora_tmux = _read_local_script("run_qwen_bootstrap41_curriculum_qlora_tmux.sh")
        bootstrap41_qlora_rewardtune_tmux = _read_local_script("run_qwen_bootstrap41_curriculum_qlora_rewardtune_tmux.sh")
        bootstrap41_qlora_stepweight_tune_tmux = _read_local_script("run_qwen_bootstrap41_curriculum_qlora_stepweight_tune_tmux.sh")
        bootstrap41_qlora_gitlabsteps_tmux = _read_local_script("run_qwen_bootstrap41_curriculum_qlora_gitlabsteps_tmux.sh")
        bootstrap41_qlora_gitlabsteps_cleandemos_tmux = _read_local_script("run_qwen_bootstrap41_curriculum_qlora_gitlabsteps_cleandemos_tmux.sh")
        bootstrap41_qlora_gitlabcoverage_tmux = _read_local_script("run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_tmux.sh")
        bootstrap41_qlora_gitlabcoverage_cleandemos_tmux = _read_local_script("run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_cleandemos_tmux.sh")
        bootstrap41_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux = _read_local_script("run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux.sh")
        web_mix88_qlora_tmux = _read_local_script("run_qwen_web_mix88_curriculum_qlora_tmux.sh")
        web_mix88_qlora_gitlabcoverage_tmux = _read_local_script("run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_tmux.sh")
        web_mix88_qlora_gitlabcoverage_warmupoversample_tmux = _read_local_script("run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_warmupoversample_tmux.sh")

        self.assertIn('OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_shopping_full_curriculum_v3_balanced}"', shopping_full_tmux)
        self.assertIn('TMUX_SESSION_PATH="$OUT_DIR/tmux_session.txt"', shopping_full_tmux)
        self.assertIn('RUN_PID_PATH="$OUT_DIR/run.pid"', shopping_full_tmux)
        self.assertIn('RUN_COMMIT_PATH="$OUT_DIR/run_commit.txt"', shopping_full_tmux)
        self.assertIn('RUN_BRANCH_PATH="$OUT_DIR/run_branch.txt"', shopping_full_tmux)
        self.assertIn("tmux new-session -d -s", shopping_full_tmux)
        self.assertIn("printf '%s\\n' \"$SESSION_NAME\" >\"$TMUX_SESSION_PATH\"", shopping_full_tmux)
        self.assertIn("tmux list-panes -t \"$SESSION_NAME\" -F '#{pane_pid}' | head -n 1 >\"$RUN_PID_PATH\"", shopping_full_tmux)
        self.assertIn('git -C "$REPO_ROOT" rev-parse HEAD >"$RUN_COMMIT_PATH"', shopping_full_tmux)
        self.assertIn('git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD >"$RUN_BRANCH_PATH"', shopping_full_tmux)
        self.assertIn("export PYTHONUNBUFFERED=1", shopping_full_tmux)
        self.assertIn("exec python3 -u scripts/run_family_curriculum.py", shopping_full_tmux)

        self.assertIn('OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v1}"', bootstrap41_tmux)
        self.assertIn('TMUX_SESSION_PATH="$OUT_DIR/tmux_session.txt"', bootstrap41_tmux)
        self.assertIn('RUN_PID_PATH="$OUT_DIR/run.pid"', bootstrap41_tmux)
        self.assertIn('RUN_COMMIT_PATH="$OUT_DIR/run_commit.txt"', bootstrap41_tmux)
        self.assertIn('RUN_BRANCH_PATH="$OUT_DIR/run_branch.txt"', bootstrap41_tmux)
        self.assertIn("bootstrap41_curriculum_manifest.json", bootstrap41_tmux)
        self.assertIn('RUN_EXTRA_ARGS="${RUN_EXTRA_ARGS:-}"', bootstrap41_tmux)
        self.assertIn('$RUN_EXTRA_ARGS \\\\', bootstrap41_tmux)
        self.assertIn("tmux new-session -d -s", bootstrap41_tmux)
        self.assertIn("printf '%s\\n' \"$SESSION_NAME\" >\"$TMUX_SESSION_PATH\"", bootstrap41_tmux)
        self.assertIn("tmux list-panes -t \"$SESSION_NAME\" -F '#{pane_pid}' | head -n 1 >\"$RUN_PID_PATH\"", bootstrap41_tmux)
        self.assertIn('git -C "$REPO_ROOT" rev-parse HEAD >"$RUN_COMMIT_PATH"', bootstrap41_tmux)
        self.assertIn('git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD >"$RUN_BRANCH_PATH"', bootstrap41_tmux)
        self.assertIn("export PYTHONUNBUFFERED=1", bootstrap41_tmux)
        self.assertIn("exec python3 -u scripts/run_family_curriculum.py", bootstrap41_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v6_qlora}"', bootstrap41_qlora_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v6_qlora}"', bootstrap41_qlora_tmux)
        self.assertIn('--quantization-mode bnb_4bit', bootstrap41_qlora_tmux)
        self.assertIn('run_qwen_bootstrap41_curriculum_tmux.sh', bootstrap41_qlora_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v7_qlora_rewardtune}"', bootstrap41_qlora_rewardtune_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v7_qlora_rewardtune}"', bootstrap41_qlora_rewardtune_tmux)
        self.assertIn('--quantization-mode bnb_4bit', bootstrap41_qlora_rewardtune_tmux)
        self.assertIn('--success-bonus 1.25', bootstrap41_qlora_rewardtune_tmux)
        self.assertIn('--same-page-repeat-action-penalty 0.06', bootstrap41_qlora_rewardtune_tmux)
        self.assertIn('run_qwen_bootstrap41_curriculum_tmux.sh', bootstrap41_qlora_rewardtune_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v8_qlora_stepweight_tune}"', bootstrap41_qlora_stepweight_tune_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v8_qlora_stepweight_tune}"', bootstrap41_qlora_stepweight_tune_tmux)
        self.assertIn('--quantization-mode bnb_4bit', bootstrap41_qlora_stepweight_tune_tmux)
        self.assertIn('--step-weight-terminal-success-bonus 2.0', bootstrap41_qlora_stepweight_tune_tmux)
        self.assertIn('--step-weight-error-step-multiplier 0.15', bootstrap41_qlora_stepweight_tune_tmux)
        self.assertIn('run_qwen_bootstrap41_curriculum_tmux.sh', bootstrap41_qlora_stepweight_tune_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v14_qlora_gitlabsteps}"', bootstrap41_qlora_gitlabsteps_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v14_qlora_gitlabsteps}"', bootstrap41_qlora_gitlabsteps_tmux)
        self.assertIn('--task-group-max-steps site_gitlab=8', bootstrap41_qlora_gitlabsteps_tmux)
        self.assertIn('--warmup-demo-task-group-max-steps site_gitlab=8', bootstrap41_qlora_gitlabsteps_tmux)
        self.assertIn('run_qwen_bootstrap41_curriculum_tmux.sh', bootstrap41_qlora_gitlabsteps_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v15_qlora_gitlabsteps_cleandemos}"', bootstrap41_qlora_gitlabsteps_cleandemos_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v15_qlora_gitlabsteps_cleandemos}"', bootstrap41_qlora_gitlabsteps_cleandemos_tmux)
        self.assertIn('--task-group-max-steps site_gitlab=8', bootstrap41_qlora_gitlabsteps_cleandemos_tmux)
        self.assertIn('--warmup-demo-task-group-max-steps site_gitlab=8', bootstrap41_qlora_gitlabsteps_cleandemos_tmux)
        self.assertIn('run_qwen_bootstrap41_curriculum_tmux.sh', bootstrap41_qlora_gitlabsteps_cleandemos_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v16_qlora_gitlabcoverage}"', bootstrap41_qlora_gitlabcoverage_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v16_qlora_gitlabcoverage}"', bootstrap41_qlora_gitlabcoverage_tmux)
        self.assertIn('--task-group-max-steps site_gitlab=8', bootstrap41_qlora_gitlabcoverage_tmux)
        self.assertIn('--warmup-demo-task-group-max-steps site_gitlab=8', bootstrap41_qlora_gitlabcoverage_tmux)
        self.assertIn('--task-group-groups-per-task site_gitlab=4', bootstrap41_qlora_gitlabcoverage_tmux)
        self.assertIn('--warmup-demo-task-group-episodes site_gitlab=4', bootstrap41_qlora_gitlabcoverage_tmux)
        self.assertIn('--warmup-demo-task-group-limit-per-task site_gitlab=4', bootstrap41_qlora_gitlabcoverage_tmux)
        self.assertIn('run_qwen_bootstrap41_curriculum_tmux.sh', bootstrap41_qlora_gitlabcoverage_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v17_qlora_gitlabcoverage_cleandemos}"', bootstrap41_qlora_gitlabcoverage_cleandemos_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v17_qlora_gitlabcoverage_cleandemos}"', bootstrap41_qlora_gitlabcoverage_cleandemos_tmux)
        self.assertIn('run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_tmux.sh', bootstrap41_qlora_gitlabcoverage_cleandemos_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v18_qlora_gitlabcoverage_cleandemos_warmupoversample}"', bootstrap41_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v18_qlora_gitlabcoverage_cleandemos_warmupoversample}"', bootstrap41_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)
        self.assertIn('--warmup-task-group-sample-multiplier site_gitlab=2.5', bootstrap41_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)
        self.assertIn('run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_cleandemos_tmux.sh', bootstrap41_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)

        self.assertIn('OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_web_mix88_curriculum_v1_qlora}"', web_mix88_qlora_tmux)
        self.assertIn('SESSION_NAME="${TMUX_SESSION_NAME:-qwen_web_mix88_v1_qlora}"', web_mix88_qlora_tmux)
        self.assertIn('web_mix88_curriculum_manifest.json', web_mix88_qlora_tmux)
        self.assertIn('--quantization-mode bnb_4bit', web_mix88_qlora_tmux)
        self.assertIn('export PYTHONUNBUFFERED=1', web_mix88_qlora_tmux)
        self.assertIn('exec python3 -u scripts/run_family_curriculum.py', web_mix88_qlora_tmux)
        self.assertIn('--family web_mix88', web_mix88_qlora_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_web_mix88_curriculum_v2_qlora_gitlabcoverage}"', web_mix88_qlora_gitlabcoverage_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_web_mix88_v2_qlora_gitlabcoverage}"', web_mix88_qlora_gitlabcoverage_tmux)
        self.assertIn('--task-group-max-steps site_gitlab=8', web_mix88_qlora_gitlabcoverage_tmux)
        self.assertIn('--warmup-demo-task-group-max-steps site_gitlab=8', web_mix88_qlora_gitlabcoverage_tmux)
        self.assertIn('--task-group-groups-per-task site_gitlab=4', web_mix88_qlora_gitlabcoverage_tmux)
        self.assertIn('--warmup-demo-task-group-episodes site_gitlab=4', web_mix88_qlora_gitlabcoverage_tmux)
        self.assertIn('--warmup-demo-task-group-limit-per-task site_gitlab=4', web_mix88_qlora_gitlabcoverage_tmux)
        self.assertIn('run_qwen_web_mix88_curriculum_qlora_tmux.sh', web_mix88_qlora_gitlabcoverage_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_web_mix88_curriculum_v3_qlora_gitlabcoverage_warmupoversample}"', web_mix88_qlora_gitlabcoverage_warmupoversample_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_web_mix88_v3_qlora_gitlabcoverage_warmupoversample}"', web_mix88_qlora_gitlabcoverage_warmupoversample_tmux)
        self.assertIn('--warmup-task-group-sample-multiplier site_gitlab=2.5', web_mix88_qlora_gitlabcoverage_warmupoversample_tmux)
        self.assertIn('run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_tmux.sh', web_mix88_qlora_gitlabcoverage_warmupoversample_tmux)

    def test_refresh_wrappers_exist_for_exact_full_and_bootstrap41_manifests(self) -> None:
        expected_files = (
            "refresh_shopping_exact_curriculum_manifest.sh",
            "refresh_shopping_exact_curriculum_manifest.ps1",
            "refresh_shopping_full_curriculum_manifest.sh",
            "refresh_shopping_full_curriculum_manifest.ps1",
            "refresh_bootstrap41_curriculum_manifest.sh",
            "refresh_bootstrap41_curriculum_manifest.ps1",
            "refresh_web_mix88_curriculum_manifest.sh",
            "refresh_web_mix88_curriculum_manifest.ps1",
        )

        for filename in expected_files:
            self.assertTrue((LOCAL_SCRIPTS / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
