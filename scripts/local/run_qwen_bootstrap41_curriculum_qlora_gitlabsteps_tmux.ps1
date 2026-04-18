$ErrorActionPreference = "Stop"

function Convert-ToWslPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WindowsPath
    )

    $fullPath = [System.IO.Path]::GetFullPath($WindowsPath)
    $normalized = $fullPath -replace "\\", "/"
    if ($normalized -match "^([A-Za-z]):/(.*)$") {
        $drive = $matches[1].ToLowerInvariant()
        $rest = $matches[2]
        return "/mnt/$drive/$rest"
    }

    throw "Unable to convert Windows path to WSL path: $WindowsPath"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $scriptDir "..\.."))
$workspaceRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot ".."))
$scriptDirWsl = Convert-ToWslPath -WindowsPath $scriptDir
$defaultOutDir = Join-Path $workspaceRoot "outputs\qwen_bootstrap41_curriculum_v14_qlora_gitlabsteps"
$runOutDir = if ($env:RUN_OUT_DIR) { $env:RUN_OUT_DIR } else { $defaultOutDir }
$sessionName = if ($env:TMUX_SESSION_NAME) { $env:TMUX_SESSION_NAME } else { "qwen_bootstrap41_v14_qlora_gitlabsteps" }
$extraArgs = "--quantization-mode bnb_4bit --quant-compute-dtype bfloat16 --quant-type nf4 --quant-use-double-quant --success-bonus 1.25 --success-step-bonus 0.10 --repeat-action-penalty 0.04 --same-page-repeat-action-penalty 0.06 --per-step-penalty 0.02 --success-unique-url-bonus 0.03 --max-unique-url-bonus-urls 5 --step-weight-later-step-bonus 0.25 --step-weight-terminal-success-bonus 2.0 --step-weight-error-step-multiplier 0.15 --step-weight-min 0.02 --task-group-max-steps site_gitlab=8 --warmup-demo-task-group-max-steps site_gitlab=8"
if ($env:RUN_EXTRA_ARGS) {
    $env:RUN_EXTRA_ARGS = "$($env:RUN_EXTRA_ARGS) $extraArgs"
} else {
    $env:RUN_EXTRA_ARGS = $extraArgs
}
$env:RUN_OUT_DIR = $runOutDir
$env:TMUX_SESSION_NAME = $sessionName

$command = @(
    "cd '$scriptDirWsl'"
    "bash ./run_qwen_bootstrap41_curriculum_qlora_gitlabsteps_tmux.sh"
) -join " && "

wsl bash -lc $command
