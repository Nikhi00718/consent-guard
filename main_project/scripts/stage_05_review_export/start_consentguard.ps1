<#
.SYNOPSIS
    Start ConsentGuard and open it in the browser.

.DESCRIPTION
    One entry point for daily use: checks the environment, builds the interface
    if needed, loads every detector once, waits until the server answers, and
    opens the workspace. Close the window to stop the server.
#>
[CmdletBinding()]
param(
    [int]$Port = 7860,
    [ValidateSet('auto', 'cuda', 'cpu')][string]$Device = 'auto',
    [ValidateSet('personal', 'research')][string]$PolicyMode = 'personal',
    [switch]$NoTiledPass,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location $repo

$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "Python environment missing. Run: powershell -ExecutionPolicy Bypass -File main_project\scripts\stage_02_baseline_model\setup_environment.ps1"
}

if (-not (Test-Path (Join-Path $repo 'main_project\frontend\dist\index.html'))) {
    Write-Host 'Building the interface (first run only)...' -ForegroundColor Cyan
    if (-not (Test-Path (Join-Path $repo 'main_project\frontend\node_modules'))) {
        npm --prefix main_project/frontend ci
    }
    npm --prefix main_project/frontend run build
}

$env:PYTHONPATH = @(
    (Join-Path $repo 'main_project\src'),
    (Join-Path $repo 'main_project\scripts\stage_05_review_export')
) -join ';'

$arguments = @(
    (Join-Path $repo 'main_project\scripts\stage_05_review_export\run_web_app.py'),
    '--port', $Port,
    '--device', $Device,
    '--policy-mode', $PolicyMode
)
if ($NoTiledPass) { $arguments += '--no-tiled-pass' }

Write-Host ''
Write-Host '  ConsentGuard' -ForegroundColor Green
Write-Host "  Loading detectors (about a minute), then opening http://127.0.0.1:$Port"
Write-Host '  Detection is never complete: look at the result before you share it.'
Write-Host ''

$server = Start-Process -FilePath $python -ArgumentList $arguments -PassThru -NoNewWindow
try {
    $ready = $false
    foreach ($attempt in 1..120) {
        Start-Sleep -Seconds 2
        if ($server.HasExited) { throw "The server stopped during startup (exit code $($server.ExitCode))." }
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 3
            if ($health.status -eq 'ok') {
                Write-Host "  Ready with $($health.configured_providers) detectors." -ForegroundColor Green
                $ready = $true
                break
            }
        } catch {
            # Still loading model weights; keep waiting.
        }
    }
    if (-not $ready) { throw 'The server did not become ready in four minutes.' }
    if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$Port" | Out-Null }
    Write-Host '  Press Ctrl+C to stop.' -ForegroundColor DarkGray
    $server.WaitForExit()
} finally {
    if (-not $server.HasExited) {
        Write-Host '  Stopping ConsentGuard...' -ForegroundColor DarkGray
        $server.Kill()
    }
}
