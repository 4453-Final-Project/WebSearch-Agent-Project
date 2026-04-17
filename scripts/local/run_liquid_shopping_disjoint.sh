#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
OUT_DIR="$WORKSPACE_ROOT/outputs/liquid_shopping_disjoint_v1"

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

mkdir -p "$OUT_DIR"

for task in 324 325 326 327 328; do
  python3 scripts/eval_single_task.py \
    --policy qwen \
    --model-dir-name LFM2.5-350M \
    --task-id "$task" \
    --episodes 4 \
    --max-steps 4 \
    --temperature 0.0 \
    --out-dir "$OUT_DIR/eval_baseline/task_$task"
done

COMMON_ARGS=(
  --model-dir-name LFM2.5-350M
  --task-id 327
  --task-id 328
  --warmup-task-id 325
  --warmup-task-id 326
  --eval-task-id 324
  --eval-task-id 325
  --eval-task-id 326
  --eval-task-id 327
  --eval-task-id 328
  --holdout-task-id 324
  --warmup-demo-dir "$WORKSPACE_ROOT/outputs/shopping_searchsort_warmup_demos_v1"
  --warmup-demo-limit-per-task 3
  --warmup-epochs 1
  --warmup-batch-size 1
  --warmup-gradient-accumulation-steps 4
  --groups-per-task 5
  --group-size 2
  --iterations 1
  --ppo-epochs 2
  --batch-size 2
  --gradient-accumulation-steps 4
  --score-batch-size 1
  --max-supervised-tokens 512
  --lora-r 4
  --lora-alpha 8
  --lora-dropout 0.05
  --max-new-tokens 64
  --max-steps 4
  --temperature 0.6
  --eval-temperature 0.0
  --eval-episodes 4
  --out-dir "$OUT_DIR"
)

python3 scripts/train_grpo_staged.py --stage warmup "${COMMON_ARGS[@]}"
python3 scripts/train_grpo_staged.py --stage collect "${COMMON_ARGS[@]}"
python3 scripts/train_grpo_staged.py --stage grpo "${COMMON_ARGS[@]}"
python3 scripts/train_grpo_staged.py --stage eval "${COMMON_ARGS[@]}"
python3 scripts/plot_shopping_searchsort_results.py \
  --run-dir "$OUT_DIR" \
  --out-dir "$WORKSPACE_ROOT/outputs/shopping_report_figures" \
  --run-name liquid_shopping_disjoint_v1 \
  --model-name LFM2.5-350M \
  --title-suffix "Liquid Disjoint Run"
