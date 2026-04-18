$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$WorkspaceRoot = Resolve-Path (Join-Path $RepoRoot "..")
$ActivatePath = Join-Path $WorkspaceRoot ".venv\Scripts\Activate.ps1"

Set-Location $RepoRoot
if (Test-Path $ActivatePath) {
    . $ActivatePath
}

python (Join-Path $RepoRoot "scripts\build_family_override_summary.py") `
  --manifest (Join-Path $ScriptDir "bootstrap41_current_stack_summary_manifest.json") `
  @args
