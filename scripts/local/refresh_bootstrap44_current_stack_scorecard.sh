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

python3 scripts/build_expanded_family_stage_scorecard.py \
  --expanded-split-manifest "$SCRIPT_DIR/bootstrap44_curriculum_manifest.json" \
  --base-summary "$WORKSPACE_ROOT/outputs/bootstrap41_current_stack_summary.json" \
  --base-stage current_stack \
  --label current_stack \
  --override "12=$WORKSPACE_ROOT/outputs/eval_task_12_bootstrap_reviewcount_v1/metrics.json" \
  --override "13=$WORKSPACE_ROOT/outputs/eval_task_13_bootstrap_reviewcount_v1/metrics.json" \
  --override "144=$WORKSPACE_ROOT/outputs/eval_task_144_spendfix_v1/metrics.json" \
  --out "$WORKSPACE_ROOT/outputs/bootstrap44_current_stack_scorecard.json" \
  --audit-out "$WORKSPACE_ROOT/outputs/bootstrap44_current_stack_audit.json" \
  "$@"
