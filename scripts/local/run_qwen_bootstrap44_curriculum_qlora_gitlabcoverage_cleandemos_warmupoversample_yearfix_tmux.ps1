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
$defaultOutDir = Join-Path $workspaceRoot "outputs\qwen_bootstrap44_curriculum_v3_qlora_gitlabcoverage_cleandemos_warmupoversample_yearfix"
$env:RUN_OUT_DIR = if ($env:RUN_OUT_DIR) { $env:RUN_OUT_DIR } else { $defaultOutDir }
$env:TMUX_SESSION_NAME = if ($env:TMUX_SESSION_NAME) { $env:TMUX_SESSION_NAME } else { "qwen_bootstrap44_v3_qlora_gitlabcoverage_cleandemos_warmupoversample_yearfix" }

$command = @(
    "cd '$scriptDirWsl'"
    "bash ./run_qwen_bootstrap44_curriculum_qlora_gitlabcoverage_cleandemos_warmupoversample_yearfix_tmux.sh"
) -join " && "

wsl bash -lc $command
