#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
OUT_PATH="$REPO_ROOT/scripts/local/shopping_order_full_curriculum_manifest.json"

cd "$REPO_ROOT"
source "$WORKSPACE_ROOT/.venv/bin/activate"

python3 scripts/refresh_family_split_manifest.py \
  --family shopping_order_full \
  --out "$OUT_PATH" \
  --split-seed 42
