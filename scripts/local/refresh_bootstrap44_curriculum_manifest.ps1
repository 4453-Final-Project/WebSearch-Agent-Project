$repoRoot = Split-Path $PSScriptRoot -Parent -Parent
$workspaceRoot = Split-Path $repoRoot -Parent

Set-Location $repoRoot
. (Join-Path $workspaceRoot ".venv\Scripts\Activate.ps1")

python scripts/refresh_family_split_manifest.py `
    --family bootstrap44 `
    --out (Join-Path $repoRoot "scripts\local\bootstrap44_curriculum_manifest.json")
