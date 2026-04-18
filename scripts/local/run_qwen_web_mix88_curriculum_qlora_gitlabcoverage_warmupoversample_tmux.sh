#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

export RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_web_mix88_curriculum_v3_qlora_gitlabcoverage_warmupoversample}"
export TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_web_mix88_v3_qlora_gitlabcoverage_warmupoversample}"
EXTRA_ARGS=(
  --warmup-task-group-sample-multiplier site_gitlab=2.5
)
if [[ -n "${RUN_EXTRA_ARGS:-}" ]]; then
  export RUN_EXTRA_ARGS="${RUN_EXTRA_ARGS} ${EXTRA_ARGS[*]}"
else
  export RUN_EXTRA_ARGS="${EXTRA_ARGS[*]}"
fi

exec bash "$SCRIPT_DIR/run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_tmux.sh"
