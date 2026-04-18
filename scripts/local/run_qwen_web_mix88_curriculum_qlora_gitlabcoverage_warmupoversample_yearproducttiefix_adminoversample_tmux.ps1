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
$defaultOutDir = Join-Path $workspaceRoot "outputs\qwen_web_mix88_curriculum_v6_qlora_gitlabcoverage_warmupoversample_yearproducttiefix_adminoversample"
$env:RUN_OUT_DIR = if ($env:RUN_OUT_DIR) { $env:RUN_OUT_DIR } else { $defaultOutDir }
$env:TMUX_SESSION_NAME = if ($env:TMUX_SESSION_NAME) { $env:TMUX_SESSION_NAME } else { "qwen_web_mix88_v6_qlora_gitlabcoverage_warmupoversample_yearproducttiefix_adminoversample" }
$existingExtraArgs = if ($env:RUN_EXTRA_ARGS) { $env:RUN_EXTRA_ARGS.Trim() } else { "" }
$adminExtraArgs = "--warmup-demo-task-group-episodes site_shopping_admin=4 --warmup-demo-task-group-limit-per-task site_shopping_admin=4 --warmup-task-group-sample-multiplier site_shopping_admin=2.0"
$env:RUN_EXTRA_ARGS = if ($existingExtraArgs) { "$existingExtraArgs $adminExtraArgs" } else { $adminExtraArgs }

$command = @(
    "cd '$scriptDirWsl'"
    "bash ./run_qwen_web_mix88_curriculum_qlora_gitlabcoverage_warmupoversample_yearproducttiefix_adminoversample_tmux.sh"
) -join " && "

wsl bash -lc $command
