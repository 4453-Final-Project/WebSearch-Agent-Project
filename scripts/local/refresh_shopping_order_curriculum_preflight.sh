#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

cd "$REPO_ROOT"
if [[ -f "$WORKSPACE_ROOT/.venv/bin/activate" ]]; then
  # Prefer the documented sibling venv when it is available in the current shell.
  source "$WORKSPACE_ROOT/.venv/bin/activate"
fi

python3 scripts/run_family_curriculum.py \
  --family shopping_order \
  --split-manifest "$SCRIPT_DIR/shopping_order_curriculum_manifest.json" \
  --dry-run \
  --out-dir "$WORKSPACE_ROOT/outputs/shopping_order_curriculum_preflight_v1" \
  "$@"
