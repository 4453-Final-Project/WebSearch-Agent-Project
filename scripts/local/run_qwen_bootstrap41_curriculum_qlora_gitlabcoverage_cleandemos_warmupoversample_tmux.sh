#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

export RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v18_qlora_gitlabcoverage_cleandemos_warmupoversample}"
export TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v18_qlora_gitlabcoverage_cleandemos_warmupoversample}"
if [[ -n "${RUN_EXTRA_ARGS:-}" ]]; then
  export RUN_EXTRA_ARGS="--warmup-task-group-sample-multiplier site_gitlab=2.5 ${RUN_EXTRA_ARGS}"
else
  export RUN_EXTRA_ARGS="--warmup-task-group-sample-multiplier site_gitlab=2.5"
fi

exec bash "$SCRIPT_DIR/run_qwen_bootstrap41_curriculum_qlora_gitlabcoverage_cleandemos_tmux.sh"
