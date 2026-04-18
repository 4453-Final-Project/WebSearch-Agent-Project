$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$WorkspaceRoot = Resolve-Path (Join-Path $RepoRoot "..")
$ActivatePath = Join-Path $WorkspaceRoot ".venv\Scripts\Activate.ps1"

Set-Location $RepoRoot
if (Test-Path $ActivatePath) {
    . $ActivatePath
}

python (Join-Path $RepoRoot "scripts\validate_expanded_family_stage_scorecard.py") `
  --scorecard-path (Join-Path $WorkspaceRoot "outputs\bootstrap44_current_stack_scorecard.json") `
  --benchmark-blocker 124 `
  --benchmark-blocker 133 `
  --benchmark-blocker 141 `
  @args
