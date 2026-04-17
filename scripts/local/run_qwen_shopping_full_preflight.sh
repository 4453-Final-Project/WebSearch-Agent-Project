#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
OUT_DIR="$WORKSPACE_ROOT/outputs/qwen_shopping_full_curriculum_preflight_v1"

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
  --family shopping_full \
  --model-dir-name Qwen3.5-2B \
  --dry-run \
  --out-dir "$OUT_DIR"
