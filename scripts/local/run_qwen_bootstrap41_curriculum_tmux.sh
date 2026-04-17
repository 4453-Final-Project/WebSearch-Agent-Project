#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
OUT_DIR="${RUN_OUT_DIR:-$WORKSPACE_ROOT/outputs/qwen_bootstrap41_curriculum_v1}"
SESSION_NAME="${TMUX_SESSION_NAME:-qwen_bootstrap41_v1}"
LOCAL_MANIFEST_PATH="$REPO_ROOT/scripts/local/bootstrap41_curriculum_manifest.json"
RUN_EXTRA_ARGS="${RUN_EXTRA_ARGS:-}"

mkdir -p "$OUT_DIR"

RUNNER_SCRIPT="$OUT_DIR/tmux_runner.sh"
LOG_PATH="$OUT_DIR/run.log"
ERR_PATH="$OUT_DIR/run.stderr.log"
TMUX_SESSION_PATH="$OUT_DIR/tmux_session.txt"
RUN_PID_PATH="$OUT_DIR/run.pid"

cat >"$RUNNER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
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
exec python3 scripts/run_family_curriculum.py \\
  --family bootstrap41 \\
  --model-dir-name Qwen3.5-2B \\
  --warmup-demo-episodes 2 \\
  --warmup-demo-max-steps 4 \\
  --warmup-demo-limit-per-task 2 \\
  --warmup-epochs 1 \\
  --groups-per-task 2 \\
  --group-size 2 \\
  --iterations 1 \\
  --ppo-epochs 2 \\
  --batch-size 2 \\
  --gradient-accumulation-steps 4 \\
  --score-batch-size 1 \\
  --max-supervised-tokens 512 \\
  --lora-r 4 \\
  --lora-alpha 8 \\
  --lora-dropout 0.05 \\
  --max-new-tokens 64 \\
  --max-steps 4 \\
  --temperature 0.6 \\
  --eval-temperature 0.0 \\
  --eval-episodes 2 \\
  --split-manifest "$LOCAL_MANIFEST_PATH" \\
  --out-dir "$OUT_DIR" \\
  $RUN_EXTRA_ARGS \\
  >"$LOG_PATH" 2>"$ERR_PATH"
EOF

chmod +x "$RUNNER_SCRIPT"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" "bash '$RUNNER_SCRIPT'"
printf '%s\n' "$SESSION_NAME" >"$TMUX_SESSION_PATH"
tmux list-panes -t "$SESSION_NAME" -F '#{pane_pid}' | head -n 1 >"$RUN_PID_PATH"
tmux list-sessions
