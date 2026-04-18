#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

export RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v21_qlora_gitlabcoverage_cleandemos_warmupoversample_yearproductfix}"
export TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v21_qlora_gitlabcoverage_cleandemos_warmupoversample_yearproductfix}"

exec bash "$SCRIPT_DIR/run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_tmux.sh"
