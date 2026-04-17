#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"

cd "$REPO_ROOT"
source "$WORKSPACE_ROOT/.venv/bin/activate"

python3 scripts/refresh_family_split_manifest.py \
  --family bootstrap41 \
  --out "$REPO_ROOT/scripts/local/bootstrap41_curriculum_manifest.json"
