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

    def test_bootstrap44_launchers_use_checked_in_manifest(self) -> None:
        qwen_script = _read_local_script("run_qwen_bootstrap44_curriculum.sh")
        liquid_script = _read_local_script("run_liquid_bootstrap44_curriculum.sh")

        self.assertIn("bootstrap44_curriculum_manifest.json", qwen_script)
        self.assertIn("bootstrap44_curriculum_manifest.json", liquid_script)
        self.assertIn("--split-manifest", qwen_script)
        self.assertIn("--split-manifest", liquid_script)

    def test_shopping_order_launchers_use_checked_in_manifest(self) -> None:
        qwen_script = _read_local_script("run_qwen_shopping_order_curriculum.sh")
        liquid_script = _read_local_script("run_liquid_shopping_order_curriculum.sh")

        self.assertIn("shopping_order_curriculum_manifest.json", qwen_script)
        self.assertIn("shopping_order_curriculum_manifest.json", liquid_script)
        self.assertIn("--split-manifest", qwen_script)
        self.assertIn("--split-manifest", liquid_script)

    def test_shopping_order_full_launchers_use_checked_in_manifest(self) -> None:
        qwen_script = _read_local_script("run_qwen_shopping_order_full_curriculum.sh")
        liquid_script = _read_local_script("run_liquid_shopping_order_full_curriculum.sh")

        self.assertIn("shopping_order_full_curriculum_manifest.json", qwen_script)
        self.assertIn("shopping_order_full_curriculum_manifest.json", liquid_script)
        self.assertIn("--split-manifest", qwen_script)
        self.assertIn("--split-manifest", liquid_script)

    def test_web_mix88_launchers_use_checked_in_manifest(self) -> None:
        qwen_script = _read_local_script("run_qwen_web_mix88_curriculum.sh")
        liquid_script = _read_local_script("run_liquid_web_mix88_curriculum.sh")

        self.assertIn("web_mix88_curriculum_manifest.json", qwen_script)
        self.assertIn("web_mix88_curriculum_manifest.json", liquid_script)
        self.assertIn("--split-manifest", qwen_script)
        self.assertIn("--split-manifest", liquid_script)

    def test_web_mix91_launchers_use_checked_in_manifest(self) -> None:
        qwen_script = _read_local_script("run_qwen_web_mix91_curriculum.sh")
        liquid_script = _read_local_script("run_liquid_web_mix91_curriculum.sh")

        self.assertIn("web_mix91_curriculum_manifest.json", qwen_script)
        self.assertIn("web_mix91_curriculum_manifest.json", liquid_script)
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
        bootstrap44_qlora_tmux = _read_local_script("run_qwen_bootstrap44_curriculum_qlora_tmux.sh")
        bootstrap44_qlora_gitlabcoverage_tmux = _read_local_script("run_qwen_bootstrap44_curriculum_qlora_gitlabcoverage_tmux.sh")
        bootstrap44_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux = _read_local_script("run_qwen_bootstrap44_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux.sh")
        shopping_order_qlora_tmux = _read_local_script("run_qwen_shopping_order_curriculum_qlora_tmux.sh")
        shopping_order_full_qlora_tmux = _read_local_script("run_qwen_shopping_order_full_curriculum_qlora_tmux.sh")
        web_mix88_qlora_tmux = _read_local_script("run_qwen_web_mix88_curriculum_qlora_tmux.sh")
        web_mix88_qlora_gitlabcoverage_tmux = _read_local_script("run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_tmux.sh")
        web_mix88_qlora_gitlabcoverage_warmupoversample_tmux = _read_local_script("run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_warmupoversample_tmux.sh")
        web_mix91_qlora_tmux = _read_local_script("run_qwen_web_mix91_curriculum_qlora_tmux.sh")
        web_mix91_qlora_gitlabcoverage_tmux = _read_local_script("run_qwen_web_mix91_curriculum_qlora_gitlabcoverage_tmux.sh")
        web_mix91_qlora_gitlabcoverage_warmupoversample_tmux = _read_local_script("run_qwen_web_mix91_curriculum_qlora_gitlabcoverage_warmupoversample_tmux.sh")

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

        self.assertIn('OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap44_curriculum_v1_qlora}"', bootstrap44_qlora_tmux)
        self.assertIn('SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap44_v1_qlora}"', bootstrap44_qlora_tmux)
        self.assertIn('bootstrap44_curriculum_manifest.json', bootstrap44_qlora_tmux)
        self.assertIn('--quantization-mode bnb_4bit', bootstrap44_qlora_tmux)
        self.assertIn('export PYTHONUNBUFFERED=1', bootstrap44_qlora_tmux)
        self.assertIn('exec python3 -u scripts/run_family_curriculum.py', bootstrap44_qlora_tmux)
        self.assertIn('--family bootstrap44', bootstrap44_qlora_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap44_curriculum_v2_qlora_gitlabcoverage}"', bootstrap44_qlora_gitlabcoverage_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap44_v2_qlora_gitlabcoverage}"', bootstrap44_qlora_gitlabcoverage_tmux)
        self.assertIn('--task-group-max-steps site_gitlab=8', bootstrap44_qlora_gitlabcoverage_tmux)
        self.assertIn('--warmup-demo-task-group-max-steps site_gitlab=8', bootstrap44_qlora_gitlabcoverage_tmux)
        self.assertIn('--task-group-groups-per-task site_gitlab=4', bootstrap44_qlora_gitlabcoverage_tmux)
        self.assertIn('--warmup-demo-task-group-episodes site_gitlab=4', bootstrap44_qlora_gitlabcoverage_tmux)
        self.assertIn('--warmup-demo-task-group-limit-per-task site_gitlab=4', bootstrap44_qlora_gitlabcoverage_tmux)
        self.assertIn('run_qwen_bootstrap44_curriculum_qlora_tmux.sh', bootstrap44_qlora_gitlabcoverage_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap44_curriculum_v2_qlora_gitlabcoverage_cleandemos_warmupoversample}"', bootstrap44_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap44_v2_qlora_gitlabcoverage_cleandemos_warmupoversample}"', bootstrap44_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)
        self.assertIn('--task-group-max-steps site_gitlab=8', bootstrap44_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)
        self.assertIn('--warmup-demo-task-group-max-steps site_gitlab=8', bootstrap44_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)
        self.assertIn('--task-group-groups-per-task site_gitlab=4', bootstrap44_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)
        self.assertIn('--warmup-demo-task-group-episodes site_gitlab=4', bootstrap44_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)
        self.assertIn('--warmup-demo-task-group-limit-per-task site_gitlab=4', bootstrap44_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)
        self.assertIn('--warmup-task-group-sample-multiplier site_gitlab=2.5', bootstrap44_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)
        self.assertIn('run_qwen_bootstrap44_curriculum_qlora_tmux.sh', bootstrap44_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux)

        self.assertIn('OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_shopping_order_curriculum_v1_qlora}"', shopping_order_qlora_tmux)
        self.assertIn('SESSION_NAME="${TMUX_SESSION_NAME:-qwen_shopping_order_v1_qlora}"', shopping_order_qlora_tmux)
        self.assertIn('shopping_order_curriculum_manifest.json', shopping_order_qlora_tmux)
        self.assertIn('--quantization-mode bnb_4bit', shopping_order_qlora_tmux)
        self.assertIn('export PYTHONUNBUFFERED=1', shopping_order_qlora_tmux)
        self.assertIn('exec python3 -u scripts/run_family_curriculum.py', shopping_order_qlora_tmux)
        self.assertIn('--family shopping_order', shopping_order_qlora_tmux)

        self.assertIn('OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_shopping_order_full_curriculum_v1_qlora}"', shopping_order_full_qlora_tmux)
        self.assertIn('SESSION_NAME="${TMUX_SESSION_NAME:-qwen_shopping_order_full_v1_qlora}"', shopping_order_full_qlora_tmux)
        self.assertIn('shopping_order_full_curriculum_manifest.json', shopping_order_full_qlora_tmux)
        self.assertIn('--quantization-mode bnb_4bit', shopping_order_full_qlora_tmux)
        self.assertIn('export PYTHONUNBUFFERED=1', shopping_order_full_qlora_tmux)
        self.assertIn('exec python3 -u scripts/run_family_curriculum.py', shopping_order_full_qlora_tmux)
        self.assertIn('--family shopping_order_full', shopping_order_full_qlora_tmux)

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

        self.assertIn('OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_web_mix91_curriculum_v1_qlora}"', web_mix91_qlora_tmux)
        self.assertIn('SESSION_NAME="${TMUX_SESSION_NAME:-qwen_web_mix91_v1_qlora}"', web_mix91_qlora_tmux)
        self.assertIn('web_mix91_curriculum_manifest.json', web_mix91_qlora_tmux)
        self.assertIn('--quantization-mode bnb_4bit', web_mix91_qlora_tmux)
        self.assertIn('export PYTHONUNBUFFERED=1', web_mix91_qlora_tmux)
        self.assertIn('exec python3 -u scripts/run_family_curriculum.py', web_mix91_qlora_tmux)
        self.assertIn('--family web_mix91', web_mix91_qlora_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_web_mix91_curriculum_v2_qlora_gitlabcoverage}"', web_mix91_qlora_gitlabcoverage_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_web_mix91_v2_qlora_gitlabcoverage}"', web_mix91_qlora_gitlabcoverage_tmux)
        self.assertIn('--task-group-max-steps site_gitlab=8', web_mix91_qlora_gitlabcoverage_tmux)
        self.assertIn('--warmup-demo-task-group-max-steps site_gitlab=8', web_mix91_qlora_gitlabcoverage_tmux)
        self.assertIn('--task-group-groups-per-task site_gitlab=4', web_mix91_qlora_gitlabcoverage_tmux)
        self.assertIn('--warmup-demo-task-group-episodes site_gitlab=4', web_mix91_qlora_gitlabcoverage_tmux)
        self.assertIn('--warmup-demo-task-group-limit-per-task site_gitlab=4', web_mix91_qlora_gitlabcoverage_tmux)
        self.assertIn('run_qwen_web_mix91_curriculum_qlora_tmux.sh', web_mix91_qlora_gitlabcoverage_tmux)

        self.assertIn('RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_web_mix91_curriculum_v2_qlora_gitlabcoverage_warmupoversample}"', web_mix91_qlora_gitlabcoverage_warmupoversample_tmux)
        self.assertIn('TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_web_mix91_v2_qlora_gitlabcoverage_warmupoversample}"', web_mix91_qlora_gitlabcoverage_warmupoversample_tmux)
        self.assertIn('--task-group-max-steps site_gitlab=8', web_mix91_qlora_gitlabcoverage_warmupoversample_tmux)
        self.assertIn('--warmup-demo-task-group-max-steps site_gitlab=8', web_mix91_qlora_gitlabcoverage_warmupoversample_tmux)
        self.assertIn('--task-group-groups-per-task site_gitlab=4', web_mix91_qlora_gitlabcoverage_warmupoversample_tmux)
        self.assertIn('--warmup-demo-task-group-episodes site_gitlab=4', web_mix91_qlora_gitlabcoverage_warmupoversample_tmux)
        self.assertIn('--warmup-demo-task-group-limit-per-task site_gitlab=4', web_mix91_qlora_gitlabcoverage_warmupoversample_tmux)
        self.assertIn('--warmup-task-group-sample-multiplier site_gitlab=2.5', web_mix91_qlora_gitlabcoverage_warmupoversample_tmux)
        self.assertIn('run_qwen_web_mix91_curriculum_qlora_tmux.sh', web_mix91_qlora_gitlabcoverage_warmupoversample_tmux)

    def test_refresh_wrappers_exist_for_exact_full_and_bootstrap41_manifests(self) -> None:
        expected_files = (
            "refresh_shopping_exact_curriculum_manifest.sh",
            "refresh_shopping_exact_curriculum_manifest.ps1",
            "refresh_shopping_full_curriculum_manifest.sh",
            "refresh_shopping_full_curriculum_manifest.ps1",
            "refresh_qwen_shopping_full_current_stack.sh",
            "refresh_qwen_shopping_full_current_stack.ps1",
            "refresh_bootstrap41_curriculum_manifest.sh",
            "refresh_bootstrap41_curriculum_manifest.ps1",
            "refresh_bootstrap44_curriculum_manifest.sh",
            "refresh_bootstrap44_curriculum_manifest.ps1",
            "refresh_bootstrap44_current_stack_scorecard.sh",
            "refresh_bootstrap44_current_stack_scorecard.ps1",
            "refresh_shopping_order_current_stack_scorecard.sh",
            "refresh_shopping_order_current_stack_scorecard.ps1",
            "refresh_shopping_order_curriculum_manifest.sh",
            "refresh_shopping_order_curriculum_manifest.ps1",
            "refresh_shopping_order_full_current_stack_scorecard.sh",
            "refresh_shopping_order_full_current_stack_scorecard.ps1",
            "refresh_shopping_order_full_curriculum_manifest.sh",
            "refresh_shopping_order_full_curriculum_manifest.ps1",
            "refresh_web_mix88_curriculum_manifest.sh",
            "refresh_web_mix88_curriculum_manifest.ps1",
            "refresh_web_mix91_curriculum_manifest.sh",
            "refresh_web_mix91_curriculum_manifest.ps1",
            "refresh_web_mix91_current_stack_scorecard.sh",
            "refresh_web_mix91_current_stack_scorecard.ps1",
        )

        for filename in expected_files:
            self.assertTrue((LOCAL_SCRIPTS / filename).exists(), filename)

    def test_bootstrap44_current_stack_refresh_wrappers_use_checked_in_inputs(self) -> None:
        bash_script = _read_local_script("refresh_bootstrap44_current_stack_scorecard.sh")
        powershell_script = _read_local_script("refresh_bootstrap44_current_stack_scorecard.ps1")

        for script in (bash_script, powershell_script):
            self.assertIn("build_expanded_family_stage_scorecard.py", script)
            self.assertIn("bootstrap44_curriculum_manifest.json", script)
            self.assertIn("bootstrap41_current_stack_summary.json", script)
            self.assertIn("eval_task_12_bootstrap_reviewcount_v1", script)
            self.assertIn("eval_task_13_bootstrap_reviewcount_v1", script)
            self.assertIn("eval_task_144_spendfix_v1", script)
            self.assertIn("bootstrap44_current_stack_scorecard.json", script)
            self.assertIn("bootstrap44_current_stack_audit.json", script)
            self.assertIn("--audit-out", script)

    def test_order_current_stack_refresh_wrappers_use_checked_in_inputs(self) -> None:
        shopping_order_bash = _read_local_script("refresh_shopping_order_current_stack_scorecard.sh")
        shopping_order_powershell = _read_local_script("refresh_shopping_order_current_stack_scorecard.ps1")
        shopping_order_full_bash = _read_local_script("refresh_shopping_order_full_current_stack_scorecard.sh")
        shopping_order_full_powershell = _read_local_script("refresh_shopping_order_full_current_stack_scorecard.ps1")

        for script in (shopping_order_bash, shopping_order_powershell):
            self.assertIn("build_subset_family_stage_scorecard.py", script)
            self.assertIn("shopping_order_current_stack_scorecard_manifest.json", script)

        for script in (shopping_order_full_bash, shopping_order_full_powershell):
            self.assertIn("build_subset_family_stage_scorecard.py", script)
            self.assertIn("shopping_order_full_current_stack_scorecard_manifest.json", script)

    def test_shopping_full_current_stack_refresh_wrappers_use_checked_in_manifest(self) -> None:
        bash_script = _read_local_script("refresh_qwen_shopping_full_current_stack.sh")
        powershell_script = _read_local_script("refresh_qwen_shopping_full_current_stack.ps1")

        for script in (bash_script, powershell_script):
            self.assertIn("refresh_family_current_stack.py", script)
            self.assertIn("qwen_shopping_full_current_stack_manifest.json", script)

    def test_web_mix91_current_stack_refresh_wrappers_use_checked_in_manifest(self) -> None:
        bash_script = _read_local_script("refresh_web_mix91_current_stack_scorecard.sh")
        powershell_script = _read_local_script("refresh_web_mix91_current_stack_scorecard.ps1")

        for script in (bash_script, powershell_script):
            self.assertIn("build_combined_family_stage_scorecard.py", script)
            self.assertIn("web_mix91_current_stack_scorecard_manifest.json", script)

    def test_order_family_refresh_wrappers_use_checked_in_outputs(self) -> None:
        shopping_order_bash = _read_local_script("refresh_shopping_order_curriculum_manifest.sh")
        shopping_order_powershell = _read_local_script("refresh_shopping_order_curriculum_manifest.ps1")
        shopping_order_full_bash = _read_local_script("refresh_shopping_order_full_curriculum_manifest.sh")
        shopping_order_full_powershell = _read_local_script("refresh_shopping_order_full_curriculum_manifest.ps1")

        for script in (shopping_order_bash, shopping_order_powershell):
            self.assertIn("refresh_family_split_manifest.py", script)
            self.assertIn("shopping_order", script)
            self.assertIn("shopping_order_curriculum_manifest.json", script)

        for script in (shopping_order_full_bash, shopping_order_full_powershell):
            self.assertIn("refresh_family_split_manifest.py", script)
            self.assertIn("shopping_order_full", script)
            self.assertIn("shopping_order_full_curriculum_manifest.json", script)



if __name__ == "__main__":
    unittest.main()
