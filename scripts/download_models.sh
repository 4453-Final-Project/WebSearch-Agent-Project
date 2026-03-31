#!/usr/bin/env bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL="Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"
TARGET_DIR="$WORKSPACE_DIR/models/Qwen2.5-3B-Instruct-GPTQ-Int4"

echo "Downloading $MODEL ..."
mkdir -p "$TARGET_DIR"

hf download "$MODEL" --local-dir "$TARGET_DIR"

echo "Model downloaded to $TARGET_DIR"
