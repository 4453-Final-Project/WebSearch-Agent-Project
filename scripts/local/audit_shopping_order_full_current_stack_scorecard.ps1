$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$WorkspaceRoot = Resolve-Path (Join-Path $RepoRoot "..")
$ActivatePath = Join-Path $WorkspaceRoot ".venv\Scripts\Activate.ps1"

Set-Location $RepoRoot
if (Test-Path $ActivatePath) {
    . $ActivatePath
}

python (Join-Path $RepoRoot "scripts\audit_subset_family_stage_scorecard.py") `
    --scorecard-path (Join-Path $WorkspaceRoot "outputs\shopping_order_full_current_stack_scorecard.json") `
    --benchmark-blocker 204 `
    --out (Join-Path $WorkspaceRoot "outputs\shopping_order_full_current_stack_audit.json") `
    @args
