param(
    [string]$OutDir = "$(Split-Path $PSScriptRoot -Parent -Parent)\..\outputs\qwen_bootstrap41_curriculum_v1"
)

$repoRoot = Split-Path $PSScriptRoot -Parent -Parent
$workspaceRoot = Split-Path $repoRoot -Parent
$manifestPath = Join-Path $repoRoot "scripts\local\bootstrap41_curriculum_manifest.json"

Set-Location $repoRoot
. (Join-Path $workspaceRoot ".venv\Scripts\Activate.ps1")

$envScript = Join-Path $repoRoot ".env.ps1"
if (Test-Path $envScript) {
    . $envScript
}

$webarenaEnvScript = Join-Path $repoRoot "scripts\webarena_env.local.ps1"
if (Test-Path $webarenaEnvScript) {
    . $webarenaEnvScript
}

python scripts/run_family_curriculum.py `
    --family bootstrap41 `
    --model-dir-name Qwen3.5-2B `
    --warmup-demo-episodes 2 `
    --warmup-demo-max-steps 4 `
    --warmup-demo-limit-per-task 2 `
    --warmup-epochs 1 `
    --groups-per-task 2 `
    --group-size 2 `
    --iterations 1 `
    --ppo-epochs 2 `
    --batch-size 2 `
    --gradient-accumulation-steps 4 `
    --score-batch-size 1 `
    --max-supervised-tokens 512 `
    --lora-r 4 `
    --lora-alpha 8 `
    --lora-dropout 0.05 `
    --max-new-tokens 64 `
    --max-steps 4 `
    --temperature 0.6 `
    --eval-temperature 0.0 `
    --eval-episodes 2 `
    --split-manifest $manifestPath `
    --out-dir $OutDir
