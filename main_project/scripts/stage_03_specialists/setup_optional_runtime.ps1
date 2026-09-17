<#
Create the isolated Windows runtime for optional specialist packages.

PaddleOCR is deliberately kept out of the main PyTorch environment. The
script is idempotent: it creates .venv-specialists when needed and then applies
the pinned requirements file.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$mainPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$specialistEnv = Join-Path $repoRoot ".venv-specialists"
$specialistPython = Join-Path $specialistEnv "Scripts\python.exe"
$requirements = Join-Path $repoRoot "main_project\configs\stage_03_specialists\specialists_requirements_windows.txt"

if (-not (Test-Path -LiteralPath $mainPython)) {
    throw "Main environment is missing: $mainPython"
}
if (-not (Test-Path -LiteralPath $specialistPython)) {
    & $mainPython -m venv --system-site-packages $specialistEnv
}
& $specialistPython -m pip install -r $requirements
Write-Host "Optional specialist runtime is ready: $specialistPython"
