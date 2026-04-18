#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

export RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v15_qlora_gitlabsteps_cleandemos}"
export TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v15_qlora_gitlabsteps_cleandemos}"
EXTRA_ARGS=(
  --quantization-mode bnb_4bit
  --quant-compute-dtype bfloat16
  --quant-type nf4
  --quant-use-double-quant
  --success-bonus 1.25
  --success-step-bonus 0.10
  --repeat-action-penalty 0.04
  --same-page-repeat-action-penalty 0.06
  --per-step-penalty 0.02
  --success-unique-url-bonus 0.03
  --max-unique-url-bonus-urls 5
  --step-weight-later-step-bonus 0.25
  --step-weight-terminal-success-bonus 2.0
  --step-weight-error-step-multiplier 0.15
  --step-weight-min 0.02
  --task-group-max-steps site_gitlab=8
  --warmup-demo-task-group-max-steps site_gitlab=8
)
if [[ -n "${RUN_EXTRA_ARGS:-}" ]]; then
  export RUN_EXTRA_ARGS="${RUN_EXTRA_ARGS} ${EXTRA_ARGS[*]}"
else
  export RUN_EXTRA_ARGS="${EXTRA_ARGS[*]}"
fi

exec bash "$SCRIPT_DIR/run_qwen_bootstrap41_curriculum_tmux.sh"
