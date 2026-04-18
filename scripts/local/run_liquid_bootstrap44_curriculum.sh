#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
OUT_DIR="$WORKSPACE_ROOT/outputs/liquid_bootstrap44_curriculum_v1"
LOCAL_MANIFEST_PATH="$REPO_ROOT/scripts/local/bootstrap44_curriculum_manifest.json"

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

python3 scripts/run_family_curriculum.py \
  --family bootstrap44 \
  --model-dir-name LFM2.5-350M \
  --warmup-demo-episodes 2 \
  --warmup-demo-max-steps 4 \
  --warmup-demo-limit-per-task 2 \
  --warmup-epochs 1 \
  --groups-per-task 2 \
  --group-size 2 \
  --iterations 1 \
  --ppo-epochs 2 \
  --batch-size 2 \
  --gradient-accumulation-steps 4 \
  --score-batch-size 1 \
  --max-supervised-tokens 512 \
  --lora-r 4 \
  --lora-alpha 8 \
  --lora-dropout 0.05 \
  --max-new-tokens 64 \
  --max-steps 4 \
  --temperature 0.6 \
  --eval-temperature 0.0 \
  --eval-episodes 2 \
  --split-manifest "$LOCAL_MANIFEST_PATH" \
  --out-dir "$OUT_DIR"
