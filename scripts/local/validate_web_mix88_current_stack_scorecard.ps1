$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$WorkspaceRoot = Resolve-Path (Join-Path $RepoRoot "..")
$ActivatePath = Join-Path $WorkspaceRoot ".venv\Scripts\Activate.ps1"

Set-Location $RepoRoot
if (Test-Path $ActivatePath) {
    . $ActivatePath
}

python (Join-Path $RepoRoot "scripts\validate_combined_family_stage_scorecard.py") `
  --scorecard-path (Join-Path $WorkspaceRoot "outputs\web_mix88_current_stack_scorecard.json") `
  --benchmark-blocker 124 `
  --benchmark-blocker 133 `
  --benchmark-blocker 141 `
  --benchmark-blocker 204 `
  @args
