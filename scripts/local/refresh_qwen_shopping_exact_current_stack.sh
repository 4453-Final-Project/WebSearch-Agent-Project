#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

python "${REPO_ROOT}/scripts/refresh_family_current_stack.py" \
  --manifest "${SCRIPT_DIR}/qwen_shopping_exact_current_stack_manifest.json" \
  "$@"
