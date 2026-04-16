#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="${MODEL_ID:-Qwen/Qwen3.5-2B}"
MODEL_DIR_NAME="${MODEL_DIR_NAME:-${MODEL##*/}}"
TARGET_DIR="${TARGET_DIR:-$WORKSPACE_DIR/models/$MODEL_DIR_NAME}"

echo "Downloading $MODEL ..."
mkdir -p "$TARGET_DIR"

hf download "$MODEL" --local-dir "$TARGET_DIR"

echo "Model downloaded to $TARGET_DIR"
