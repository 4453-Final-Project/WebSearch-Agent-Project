#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
ACTIVATE_PATH="$WORKSPACE_ROOT/.venv/bin/activate"

cd "$REPO_ROOT"
if [[ -f "$ACTIVATE_PATH" ]]; then
  # shellcheck disable=SC1090
  source "$ACTIVATE_PATH"
fi

python "$REPO_ROOT/scripts/validate_expanded_family_stage_scorecard.py" \
  --scorecard-path "$WORKSPACE_ROOT/outputs/bootstrap44_current_stack_scorecard.json" \
  --benchmark-blocker 124 \
  --benchmark-blocker 133 \
  --benchmark-blocker 141 \
  "$@"
