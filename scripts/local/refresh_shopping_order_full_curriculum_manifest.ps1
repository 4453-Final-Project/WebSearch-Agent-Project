$ErrorActionPreference = "Stop"

$repoRoot = Split-Path $PSScriptRoot -Parent -Parent
$workspaceRoot = Split-Path $repoRoot -Parent
$outPath = Join-Path $repoRoot "scripts\local\shopping_order_full_curriculum_manifest.json"

Set-Location $repoRoot
. (Join-Path $workspaceRoot ".venv\Scripts\Activate.ps1")

python scripts/refresh_family_split_manifest.py `
    --family shopping_order_full `
    --out $outPath `
    --split-seed 42
