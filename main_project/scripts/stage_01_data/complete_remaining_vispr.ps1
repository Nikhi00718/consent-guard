[CmdletBinding()]
param(
    [int]$RetryDelaySeconds = 60
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$resumeScript = Join-Path $PSScriptRoot 'resume_vispr_split_until_complete.ps1'
$finalizer = Join-Path $PSScriptRoot 'finalize_vispr_data.py'
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $venvPython)) {
    throw "Python environment is missing: $venvPython"
}

function Complete-Split {
    param([Parameter(Mandatory = $true)][ValidateSet('val', 'test')][string]$Split)

    while ($true) {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $resumeScript `
            -Split $Split `
            -RetryDelaySeconds $RetryDelaySeconds
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            break
        }
        if ($exitCode -eq 3) {
            Write-Host "[$(Get-Date)] $Split is owned by the active downloader; checking again in $RetryDelaySeconds seconds."
            Start-Sleep -Seconds $RetryDelaySeconds
            continue
        }
        throw "$Split downloader stopped with exit code $exitCode. No extraction was attempted."
    }

    & $venvPython $finalizer --split $Split --extract --rebuild-records
    if ($LASTEXITCODE -ne 0) {
        throw "$Split archive finalization failed with exit code $LASTEXITCODE"
    }
}

Push-Location $workspaceRoot
try {
    Complete-Split -Split val
    Complete-Split -Split test
    Write-Host "[$(Get-Date)] Remaining VISPR archives are downloaded, validated, extracted, and records rebuilt."
}
finally {
    Pop-Location
}
