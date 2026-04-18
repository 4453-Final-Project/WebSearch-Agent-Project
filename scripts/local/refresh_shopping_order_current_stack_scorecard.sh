#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

cd "$REPO_ROOT"
if [[ -f "$WORKSPACE_ROOT/.venv/bin/activate" ]]; then
  source "$WORKSPACE_ROOT/.venv/bin/activate"
fi

python3 "${REPO_ROOT}/scripts/build_subset_family_stage_scorecard.py" \
  --manifest "${SCRIPT_DIR}/shopping_order_current_stack_scorecard_manifest.json" \
  "$@"
