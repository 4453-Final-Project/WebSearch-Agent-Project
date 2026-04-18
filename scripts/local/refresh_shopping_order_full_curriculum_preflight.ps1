$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$workspaceRoot = Resolve-Path (Join-Path $repoRoot "..")
$activatePath = Join-Path $workspaceRoot ".venv\Scripts\Activate.ps1"

Set-Location $repoRoot
if (Test-Path $activatePath) {
    . $activatePath
}

python scripts/run_family_curriculum.py `
    --family shopping_order_full `
    --split-manifest (Join-Path $PSScriptRoot "shopping_order_full_curriculum_manifest.json") `
    --dry-run `
    --out-dir (Join-Path $workspaceRoot "outputs\shopping_order_full_curriculum_preflight_v1") `
    @args
