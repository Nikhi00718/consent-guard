[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$logRoot = Join-Path $workspace 'data\download_logs'
$statusPath = Join-Path $logRoot 'visual_redactions_images_status.json'
$finalStatusPath = Join-Path $logRoot 'visual_redactions_finalization_status.json'
$lockPath = Join-Path $logRoot 'visual_redactions_finalizer.lock'
$python = Join-Path $workspace '.venv\Scripts\python.exe'
$validator = Join-Path $PSScriptRoot 'validate_extract_visual_redactions_release.py'

function Write-FinalStatus([string]$State, [string]$Message) {
    [ordered]@{
        state = $State
        message = $Message
        updated_at = (Get-Date).ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath $finalStatusPath -Encoding UTF8
}

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$lock = $null
try {
    $lock = [System.IO.File]::Open(
        $lockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
    Write-FinalStatus 'waiting-for-download' 'Waiting for all official archives to finish.'
    while ($true) {
        $active = @(Get-CimInstance Win32_Process | Where-Object {
            $_.CommandLine -like '*download_visual_redactions_aria2.ps1*' -and
            $_.ProcessId -ne $PID
        })
        if ($active.Count -eq 0) { break }
        Start-Sleep -Seconds 30
    }

    if (-not (Test-Path -LiteralPath $statusPath)) {
        throw 'Download status file is missing.'
    }
    $downloadStatus = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($downloadStatus.state -ne 'downloaded') {
        throw "Download did not complete cleanly. State: $($downloadStatus.state); $($downloadStatus.message)"
    }

    Write-FinalStatus 'validating' 'Running full archive, extraction, identity, geometry, and decode checks.'
    & $python $validator --extract
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset validator failed with exit code $LASTEXITCODE."
    }
    Write-FinalStatus 'valid' 'Official Visual Redactions release passed every configured validation gate.'
}
catch {
    Write-FinalStatus 'failed' $_.Exception.Message
    throw
}
finally {
    if ($lock) { $lock.Dispose() }
}
