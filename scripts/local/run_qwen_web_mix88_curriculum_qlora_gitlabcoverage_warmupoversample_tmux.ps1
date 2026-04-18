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
$defaultOutDir = Join-Path $workspaceRoot "outputs\qwen_web_mix88_curriculum_v3_qlora_gitlabcoverage_warmupoversample"
$runOutDir = if ($env:RUN_OUT_DIR) { $env:RUN_OUT_DIR } else { $defaultOutDir }
$sessionName = if ($env:TMUX_SESSION_NAME) { $env:TMUX_SESSION_NAME } else { "qwen_web_mix88_v3_qlora_gitlabcoverage_warmupoversample" }
$extraArgs = "--warmup-task-group-sample-multiplier site_gitlab=2.5"
if ($env:RUN_EXTRA_ARGS) {
    $env:RUN_EXTRA_ARGS = "$($env:RUN_EXTRA_ARGS) $extraArgs"
} else {
    $env:RUN_EXTRA_ARGS = $extraArgs
}
$env:RUN_OUT_DIR = $runOutDir
$env:TMUX_SESSION_NAME = $sessionName

$command = @(
    "cd '$scriptDirWsl'"
    "bash ./run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_warmupoversample_tmux.sh"
) -join " && "

wsl bash -lc $command
