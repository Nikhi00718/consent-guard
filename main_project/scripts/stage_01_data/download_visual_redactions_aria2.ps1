[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$archiveRoot = Join-Path $workspace 'data\raw\visual_redactions\image_archives'
$logRoot = Join-Path $workspace 'data\download_logs'
$statusPath = Join-Path $logRoot 'visual_redactions_images_status.json'
$lockPath = Join-Path $logRoot 'visual_redactions_images_aria2.lock'
$expected = [ordered]@{
    train2017 = [int64]7816171895
    val2017 = [int64]3158549744
    test2017 = [int64]5633037591
}
$inputPath = Join-Path $workspace 'main_project\configs\stage_01_data\visual_redactions_aria2_input.txt'
New-Item -ItemType Directory -Force -Path $archiveRoot, $logRoot | Out-Null

function Write-Status([string]$State, [string]$Message) {
    $downloadedTotal = [int64]0
    $rows = foreach ($entry in $expected.GetEnumerator()) {
        $path = Join-Path $archiveRoot "$($entry.Key).tar.gz"
        $bytes = if (Test-Path -LiteralPath $path) { (Get-Item -LiteralPath $path).Length } else { 0 }
        $downloadedTotal += $bytes
        [ordered]@{
            split = $entry.Key
            expected_bytes = $entry.Value
            downloaded_bytes = $bytes
            percent = [math]::Round(100 * $bytes / $entry.Value, 3)
            complete = ($bytes -eq $entry.Value)
            path = $path
        }
    }
    $payload = [ordered]@{
        state = $State
        message = $Message
        updated_at = (Get-Date).ToString('o')
        total_expected_bytes = [int64](($expected.Values | Measure-Object -Sum).Sum)
        total_downloaded_bytes = $downloadedTotal
        archives = @($rows)
    }
    $temporary = "$statusPath.tmp-$PID"
    $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $temporary -Encoding UTF8
    Move-Item -LiteralPath $temporary -Destination $statusPath -Force
}

$lock = $null
try {
    $lock = [System.IO.File]::Open(
        $lockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
    $aria = (Get-Command aria2c.exe -ErrorAction Stop).Source
    Write-Status 'running' 'Downloading all three archives with resumable segmented transfers.'
    # The legacy dataset host currently presents a certificate Windows reports as revoked.
    # Content is therefore accepted only after exact byte-size, full gzip/TAR, and
    # annotation/image identity validation in validate_extract_visual_redactions_release.py.
    & $aria --input-file=$inputPath --check-certificate=false --continue=true --allow-overwrite=false --auto-file-renaming=false `
        --max-concurrent-downloads=3 --max-connection-per-server=8 --split=8 `
        --min-split-size=4M --file-allocation=none --retry-wait=5 --max-tries=0 `
        --summary-interval=5 --console-log-level=notice --enable-rpc=true `
        --rpc-listen-all=false --rpc-listen-port=6800 --dir=$archiveRoot
    if ($LASTEXITCODE -ne 0) { throw "aria2c failed with exit code $LASTEXITCODE" }
    foreach ($entry in $expected.GetEnumerator()) {
        $path = Join-Path $archiveRoot "$($entry.Key).tar.gz"
        $actual = (Get-Item -LiteralPath $path).Length
        if ($actual -ne $entry.Value) {
            throw "$($entry.Key) size mismatch: expected $($entry.Value), got $actual"
        }
    }
    Write-Status 'downloaded' 'All three archives have exact expected byte lengths.'
}
catch {
    Write-Status 'failed' $_.Exception.Message
    throw
}
finally {
    if ($lock) { $lock.Dispose() }
}
