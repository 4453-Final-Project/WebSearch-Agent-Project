$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..\..")

python (Join-Path $RepoRoot "scripts\refresh_family_current_stack.py") `
  --manifest (Join-Path $ScriptDir "qwen_shopping_exact_current_stack_manifest.json") `
  @args
