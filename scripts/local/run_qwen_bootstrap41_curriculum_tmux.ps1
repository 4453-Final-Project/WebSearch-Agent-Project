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
$repoRootWsl = Convert-ToWslPath -WindowsPath $repoRoot

$command = @(
    "cd '$repoRootWsl'"
    "bash scripts/local/run_qwen_bootstrap41_curriculum_tmux.sh"
) -join " && "

wsl bash -lc $command
