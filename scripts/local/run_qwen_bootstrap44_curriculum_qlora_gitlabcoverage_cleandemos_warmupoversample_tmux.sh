#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

export RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap44_curriculum_v2_qlora_gitlabcoverage_cleandemos_warmupoversample}"
export TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap44_v2_qlora_gitlabcoverage_cleandemos_warmupoversample}"
EXTRA_ARGS=(
  --task-group-max-steps site_gitlab=8
  --warmup-demo-task-group-max-steps site_gitlab=8
  --task-group-groups-per-task site_gitlab=4
  --warmup-demo-task-group-episodes site_gitlab=4
  --warmup-demo-task-group-limit-per-task site_gitlab=4
  --warmup-task-group-sample-multiplier site_gitlab=2.5
)
if [[ -n "${RUN_EXTRA_ARGS:-}" ]]; then
  export RUN_EXTRA_ARGS="${RUN_EXTRA_ARGS} ${EXTRA_ARGS[*]}"
else
  export RUN_EXTRA_ARGS="${EXTRA_ARGS[*]}"
fi

exec bash "$SCRIPT_DIR/run_qwen_bootstrap44_curriculum_qlora_tmux.sh"
