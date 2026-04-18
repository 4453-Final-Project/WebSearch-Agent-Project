#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

cd "$REPO_ROOT"
if [[ -f "$WORKSPACE_ROOT/.venv/bin/activate" ]]; then
  source "$WORKSPACE_ROOT/.venv/bin/activate"
fi

python3 "${REPO_ROOT}/scripts/validate_webarena_blocker_audit.py" \
  --audit-path "$WORKSPACE_ROOT/outputs/bootstrap41_blocker_audit_v1.json" \
  --expected-task-id 124 \
  --expected-task-id 133 \
  --expected-task-id 141 \
  "$@"
