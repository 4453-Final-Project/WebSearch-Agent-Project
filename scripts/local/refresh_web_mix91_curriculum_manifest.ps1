$repoRoot = Split-Path $PSScriptRoot -Parent -Parent
$workspaceRoot = Split-Path $repoRoot -Parent

Set-Location $repoRoot
. (Join-Path $workspaceRoot ".venv\Scripts\Activate.ps1")

python scripts/refresh_family_split_manifest.py `
    --family web_mix91 `
    --out (Join-Path $repoRoot "scripts\local\web_mix91_curriculum_manifest.json")
