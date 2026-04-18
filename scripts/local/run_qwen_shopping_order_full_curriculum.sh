#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
OUT_DIR="$WORKSPACE_ROOT/outputs/qwen_shopping_order_full_curriculum_v1"
MANIFEST_PATH="$REPO_ROOT/scripts/local/shopping_order_full_curriculum_manifest.json"

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

ARGS=(
  --family shopping_order_full
  --model-dir-name Qwen3.5-2B
  --warmup-demo-episodes 2
  --warmup-demo-max-steps 4
  --warmup-demo-limit-per-task 2
  --warmup-epochs 1
  --groups-per-task 2
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
  --eval-episodes 2
  --out-dir "$OUT_DIR"
)

if [[ -f "$MANIFEST_PATH" ]]; then
  ARGS+=(--split-manifest "$MANIFEST_PATH")
fi

python3 scripts/run_family_curriculum.py "${ARGS[@]}"
