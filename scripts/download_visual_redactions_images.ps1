[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$archiveRoot = Join-Path $workspace 'data\raw\visual_redactions\image_archives'
$logRoot = Join-Path $workspace 'data\download_logs'
$statusPath = Join-Path $logRoot 'visual_redactions_images_status.json'
$lockPath = Join-Path $logRoot 'visual_redactions_images_supervisor.lock'

New-Item -ItemType Directory -Force -Path $archiveRoot, $logRoot | Out-Null

$archives = @(
    [pscustomobject]@{
        Split = 'train2017'
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/train2017.tar.gz'
        Bytes = [int64]7816171895
    },
    [pscustomobject]@{
        Split = 'val2017'
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/val2017.tar.gz'
        Bytes = [int64]3158549744
    },
    [pscustomobject]@{
        Split = 'test2017'
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/test2017.tar.gz'
        Bytes = [int64]5633037591
    }
)

function Write-Status([string]$State, [string]$CurrentSplit, [string]$Message) {
    $downloadedTotal = [int64]0
    $rows = foreach ($archive in $archives) {
        $path = Join-Path $archiveRoot "$($archive.Split).tar.gz"
        $downloaded = if (Test-Path -LiteralPath $path) { (Get-Item -LiteralPath $path).Length } else { 0 }
        $downloadedTotal += $downloaded
        [ordered]@{
            split = $archive.Split
            expected_bytes = $archive.Bytes
            downloaded_bytes = $downloaded
            percent = [math]::Round(100 * $downloaded / $archive.Bytes, 3)
            complete = ($downloaded -eq $archive.Bytes)
            path = $path
        }
    }
    $payload = [ordered]@{
        state = $State
        current_split = $CurrentSplit
        message = $Message
        updated_at = (Get-Date).ToString('o')
        total_expected_bytes = [int64](($archives | Measure-Object Bytes -Sum).Sum)
        total_downloaded_bytes = $downloadedTotal
        archives = @($rows)
    }
    $temporary = "$statusPath.tmp-$PID"
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

$lockStream = $null
try {
    $lockStream = [System.IO.File]::Open(
        $lockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    throw 'A Visual Redactions image download supervisor is already running.'
}

try {
    $curl = (Get-Command curl.exe -ErrorAction Stop).Source
    Write-Status 'running' '' 'Starting resumable Visual Redactions image downloads.'
    foreach ($archive in $archives) {
        $destination = Join-Path $archiveRoot "$($archive.Split).tar.gz"
        if (Test-Path -LiteralPath $destination) {
            $currentBytes = (Get-Item -LiteralPath $destination).Length
            if ($currentBytes -gt $archive.Bytes) {
                throw "$destination is larger than the verified official size."
            }
            if ($currentBytes -eq $archive.Bytes) {
                Write-Status 'running' $archive.Split 'Archive already has the expected byte length.'
                continue
            }
        }

        Write-Status 'running' $archive.Split 'Downloading with curl byte-range resume.'
        & $curl --location --fail --retry 12 --retry-all-errors --retry-delay 5 `
            --continue-at - --output $destination $archive.Url
        if ($LASTEXITCODE -ne 0) {
            throw "curl failed with exit code $LASTEXITCODE for $($archive.Split)."
        }
        $actualBytes = (Get-Item -LiteralPath $destination).Length
        if ($actualBytes -ne $archive.Bytes) {
            throw "$($archive.Split) size mismatch: expected $($archive.Bytes), got $actualBytes."
        }
        Write-Status 'running' $archive.Split 'Download complete and exact byte length verified.'
    }
    Write-Status 'downloaded' '' 'All official archives have exact expected byte lengths.'
}
catch {
    Write-Status 'failed' '' $_.Exception.Message
    throw
}
finally {
    if ($lockStream) { $lockStream.Dispose() }
}
