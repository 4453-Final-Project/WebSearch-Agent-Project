$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\\..")).Path
$workspaceRoot = Split-Path -Parent $repoRoot

if (Test-Path (Join-Path $workspaceRoot ".venv\\Scripts\\Activate.ps1")) {
    . (Join-Path $workspaceRoot ".venv\\Scripts\\Activate.ps1")
}

python `
  "$repoRoot\\scripts\\build_subset_family_stage_scorecard.py" `
  --manifest "$scriptDir\\shopping_order_full_current_stack_scorecard_manifest.json" `
  @args
