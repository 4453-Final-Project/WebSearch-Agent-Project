#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

export RUN_OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v6_qlora}"
export TMUX_SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v6_qlora}"
EXTRA_ARGS=(
  --quantization-mode bnb_4bit
  --quant-compute-dtype bfloat16
  --quant-type nf4
  --quant-use-double-quant
)
if [[ -n "${RUN_EXTRA_ARGS:-}" ]]; then
  export RUN_EXTRA_ARGS="${RUN_EXTRA_ARGS} ${EXTRA_ARGS[*]}"
else
  export RUN_EXTRA_ARGS="${EXTRA_ARGS[*]}"
fi

exec bash "$SCRIPT_DIR/run_qwen_bootstrap41_curriculum_tmux.sh"
