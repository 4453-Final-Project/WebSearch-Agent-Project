#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

cd "$REPO_ROOT"
source "$WORKSPACE_ROOT/.venv/bin/activate"

if [[ -f "$REPO_ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
fi

if [[ -f "$REPO_ROOT/scripts/webarena_env.local.sh" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/scripts/webarena_env.local.sh"
fi

export MODEL_ID='Qwen/Qwen3.5-2B'
export MODEL_DIR_NAME='Qwen3.5-2B'

python scripts/train_grpo_single_task.py \
  --task-id 310 \
  --seed 42 \
  --max-steps 4 \
  --iterations 3 \
  --groups-per-iteration 2 \
  --group-size 2 \
  --ppo-epochs 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --score-batch-size 1 \
  --warmup-demo-dir "$WORKSPACE_ROOT/outputs/warmup_demos/task_0310" \
  --warmup-epochs 6 \
  --warmup-batch-size 1 \
  --warmup-gradient-accumulation-steps 4 \
  --eval-episodes 5 \
  --out-dir "$WORKSPACE_ROOT/outputs/task4-grpo-task310-final"
