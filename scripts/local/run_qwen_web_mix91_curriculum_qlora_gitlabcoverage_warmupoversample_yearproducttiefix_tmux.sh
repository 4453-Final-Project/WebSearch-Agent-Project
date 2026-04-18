#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

export RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_web_mix91_curriculum_v4_qlora_gitlabcoverage_warmupoversample_yearproducttiefix}"
export TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_web_mix91_v4_qlora_gitlabcoverage_warmupoversample_yearproducttiefix}"

exec bash "$SCRIPT_DIR/run_qwen_web_mix91_curriculum_qlora_gitlabcoverage_warmupoversample_tmux.sh"
