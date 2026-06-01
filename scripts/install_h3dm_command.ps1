param(
    [string]$EnvironmentName = "human-3d-motion"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

function Assert-LastCommandSucceeded {
    param([string]$Step)

    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

$prefixOutput = conda run -n $EnvironmentName python -c "import sys; print(sys.prefix)"
Assert-LastCommandSucceeded "Resolving Conda environment prefix"
$environmentPrefix = (
    $prefixOutput |
        Where-Object { $_ -and $_.Trim() } |
        Select-Object -Last 1
).Trim()
if (-not $environmentPrefix) {
    throw "Could not resolve Conda environment prefix for '$EnvironmentName'."
}

$scriptsDir = Join-Path $environmentPrefix "Scripts"
New-Item -ItemType Directory -Force -Path $scriptsDir | Out-Null

$commandPath = Join-Path $scriptsDir "h3dm.cmd"
$command = @"
@echo off
setlocal
set "H3DM_REPO_ROOT=$repoRoot"
cd /d "%H3DM_REPO_ROOT%"
python -m webapp.main %*
"@

Set-Content -Path $commandPath -Value $command -Encoding ASCII
Write-Host "Installed h3dm command: $commandPath"
