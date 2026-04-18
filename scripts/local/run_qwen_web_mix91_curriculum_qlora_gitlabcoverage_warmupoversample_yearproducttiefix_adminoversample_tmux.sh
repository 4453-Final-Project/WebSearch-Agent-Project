#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

export RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_web_mix91_curriculum_v5_qlora_gitlabcoverage_warmupoversample_yearproducttiefix_adminoversample}"
export TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_web_mix91_v5_qlora_gitlabcoverage_warmupoversample_yearproducttiefix_adminoversample}"
export RUN_EXTRA_ARGS="${RUN_EXTRA_ARGS:-} --warmup-demo-task-group-episodes site_shopping_admin=4 --warmup-demo-task-group-limit-per-task site_shopping_admin=4 --warmup-task-group-sample-multiplier site_shopping_admin=2.0"

exec bash "$SCRIPT_DIR/run_qwen_web_mix91_curriculum_qlora_gitlabcoverage_warmupoversample_yearproducttiefix_tmux.sh"
