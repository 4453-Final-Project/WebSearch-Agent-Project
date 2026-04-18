$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")
$WorkspaceRoot = Resolve-Path (Join-Path $RepoRoot "..")
$ActivatePath = Join-Path $WorkspaceRoot ".venv\Scripts\Activate.ps1"

Set-Location $RepoRoot
if (Test-Path $ActivatePath) {
    . $ActivatePath
}

python (Join-Path $RepoRoot "scripts\validate_webarena_blocker_audit.py") `
  --audit-path (Join-Path $WorkspaceRoot "outputs\bootstrap41_blocker_audit_v1.json") `
  --expected-task-id 124 `
  --expected-task-id 133 `
  --expected-task-id 141 `
  @args
