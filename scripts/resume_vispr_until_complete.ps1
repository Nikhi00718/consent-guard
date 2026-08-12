[CmdletBinding()]
param(
    [int]$RetryDelaySeconds = 60
)

$ErrorActionPreference = 'Continue'
$downloadScript = Join-Path $PSScriptRoot 'download_datasets.ps1'

while ($true) {
    $started = Get-Date
    Write-Host "[$started] Starting/resuming VISPR image downloads."

    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $downloadScript -VisprImages -AcceptLargeDownloads
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[$(Get-Date)] VISPR image downloads completed successfully."
            exit 0
        }
        Write-Warning "Downloader exited with code $LASTEXITCODE."
    }
    catch {
        Write-Warning "Downloader exception: $($_.Exception.Message)"
    }

    Write-Host "[$(Get-Date)] Retrying in $RetryDelaySeconds seconds; partial files will be resumed."
    Start-Sleep -Seconds $RetryDelaySeconds
}
