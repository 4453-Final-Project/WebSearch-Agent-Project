$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$workspaceRoot = Resolve-Path (Join-Path $repoRoot "..")
$activatePath = Join-Path $workspaceRoot ".venv\Scripts\Activate.ps1"

Set-Location $repoRoot
if (Test-Path $activatePath) {
    . $activatePath
}

python scripts/build_expanded_family_stage_scorecard.py `
    --expanded-split-manifest (Join-Path $PSScriptRoot "bootstrap44_curriculum_manifest.json") `
    --base-summary (Join-Path $workspaceRoot "outputs\bootstrap41_current_stack_summary.json") `
    --base-stage current_stack `
    --label current_stack `
    --override "12=$(Join-Path $workspaceRoot 'outputs\eval_task_12_bootstrap_reviewcount_v1\metrics.json')" `
    --override "13=$(Join-Path $workspaceRoot 'outputs\eval_task_13_bootstrap_reviewcount_v1\metrics.json')" `
    --override "144=$(Join-Path $workspaceRoot 'outputs\eval_task_144_spendfix_v1\metrics.json')" `
    --out (Join-Path $workspaceRoot "outputs\bootstrap44_current_stack_scorecard.json") `
    --audit-out (Join-Path $workspaceRoot "outputs\bootstrap44_current_stack_audit.json") `
    @args
