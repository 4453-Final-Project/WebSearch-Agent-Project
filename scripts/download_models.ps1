$MODEL = if ($env:MODEL_ID) { $env:MODEL_ID } else { "Qwen/Qwen3.5-2B" }
$MODEL_DIR_NAME = if ($env:MODEL_DIR_NAME) { $env:MODEL_DIR_NAME } else { ($MODEL -split "/")[-1] }
$TARGET_DIR = if ($env:TARGET_DIR) {
    [System.IO.Path]::GetFullPath($env:TARGET_DIR)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\..\models\$MODEL_DIR_NAME"))
}

Write-Host "Downloading $MODEL ..."

New-Item -ItemType Directory -Force -Path $TARGET_DIR | Out-Null

hf download $MODEL --local-dir $TARGET_DIR

Write-Host "Model downloaded to $TARGET_DIR"
