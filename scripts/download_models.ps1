$MODEL="Qwen/Qwen2.5-3B-Instruct"
$TARGET_DIR="../models/Qwen2.5-3B-Instruct"

Write-Host "Downloading $MODEL ..."

New-Item -ItemType Directory -Force -Path $TARGET_DIR | Out-Null

hf download $MODEL --local-dir $TARGET_DIR

Write-Host "Model downloaded to $TARGET_DIR"
