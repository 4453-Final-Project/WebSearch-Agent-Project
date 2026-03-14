#!/usr/bin/env bash

set -e

MODEL="Qwen/Qwen2.5-3B-Instruct"
TARGET_DIR="../models/Qwen2.5-3B-Instruct"

echo "Downloading $MODEL ..."
mkdir -p "$TARGET_DIR"

hf download "$MODEL" --local-dir "$TARGET_DIR"

echo "Model downloaded to $TARGET_DIR"

