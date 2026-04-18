$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\\..")).Path
$workspaceRoot = Split-Path -Parent $repoRoot

if (-not $env:RUN_OUT_DIR) {
    $env:RUN_OUT_DIR = Join-Path $workspaceRoot "outputs/qwen_shopping_order_full_curriculum_v1_qlora"
}

if (-not $env:TMUX_SESSION_NAME) {
    $env:TMUX_SESSION_NAME = "qwen_shopping_order_full_v1_qlora"
}

$extraArgs = @(
    "--quantization-mode", "bnb_4bit",
    "--quant-compute-dtype", "bfloat16",
    "--quant-type", "nf4",
    "--quant-use-double-quant",
    "--success-bonus", "1.25",
    "--success-step-bonus", "0.10",
    "--repeat-action-penalty", "0.04",
    "--same-page-repeat-action-penalty", "0.06",
    "--per-step-penalty", "0.02",
    "--success-unique-url-bonus", "0.03",
    "--max-unique-url-bonus-urls", "5",
    "--step-weight-later-step-bonus", "0.25",
    "--step-weight-terminal-success-bonus", "2.0",
    "--step-weight-error-step-multiplier", "0.15",
    "--step-weight-min", "0.02"
)

$joinedExtraArgs = [string]::Join(" ", $extraArgs)
if ($env:RUN_EXTRA_ARGS) {
    $env:RUN_EXTRA_ARGS = "$joinedExtraArgs $($env:RUN_EXTRA_ARGS)"
} else {
    $env:RUN_EXTRA_ARGS = $joinedExtraArgs
}

$bashScript = (Resolve-Path (Join-Path $scriptDir "run_qwen_shopping_order_full_curriculum_qlora_tmux.sh")).Path
wsl bash -lc "cd '/mnt/c/Users/tyler/school/RL/WebSearch-Agent-Project' && bash '$([IO.Path]::GetFullPath($bashScript).Replace('\','/').Replace('C:','/mnt/c'))'"
