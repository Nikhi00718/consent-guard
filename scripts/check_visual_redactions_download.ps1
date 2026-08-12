$workspace = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$statusPath = Join-Path $workspace 'data\download_logs\visual_redactions_images_status.json'
$finalStatusPath = Join-Path $workspace 'data\download_logs\visual_redactions_finalization_status.json'
$archiveRoot = Join-Path $workspace 'data\raw\visual_redactions\image_archives'
$expected = [int64]16607759230

$files = @(
    @{ Split = 'train2017'; Bytes = [int64]7816171895 },
    @{ Split = 'val2017'; Bytes = [int64]3158549744 },
    @{ Split = 'test2017'; Bytes = [int64]5633037591 }
)
$ariaJobs = @{}
$totalSpeed = [int64]0
try {
    $activeBody = '{"jsonrpc":"2.0","id":"active","method":"aria2.tellActive","params":[["gid","status","totalLength","completedLength","downloadSpeed","files"]]}'
    $waitingBody = '{"jsonrpc":"2.0","id":"waiting","method":"aria2.tellWaiting","params":[0,100,["gid","status","totalLength","completedLength","downloadSpeed","files"]]}'
    $jobs = @()
    $jobs += @(Invoke-RestMethod -Uri 'http://127.0.0.1:6800/jsonrpc' -Method Post -ContentType 'application/json' -Body $activeBody).result
    $jobs += @(Invoke-RestMethod -Uri 'http://127.0.0.1:6800/jsonrpc' -Method Post -ContentType 'application/json' -Body $waitingBody).result
    foreach ($job in $jobs) {
        if (-not $job.files -or -not $job.files[0].path) { continue }
        $name = [System.IO.Path]::GetFileName($job.files[0].path)
        $split = $name -replace '\.tar\.gz$', ''
        $ariaJobs[$split] = $job
        $totalSpeed += [int64]$job.downloadSpeed
    }
}
catch {
    # RPC is available only while the segmented downloader is active.
}
$downloadedTotal = [int64]0
$rows = foreach ($file in $files) {
    $path = Join-Path $archiveRoot "$($file.Split).tar.gz"
    if ($ariaJobs.ContainsKey($file.Split)) {
        $bytes = [int64]$ariaJobs[$file.Split].completedLength
        $speed = [int64]$ariaJobs[$file.Split].downloadSpeed
    }
    else {
        $bytes = if (Test-Path -LiteralPath $path) { (Get-Item -LiteralPath $path).Length } else { 0 }
        $speed = [int64]0
    }
    $downloadedTotal += $bytes
    [pscustomobject]@{
        Split = $file.Split
        DownloadedGiB = [math]::Round($bytes / 1GB, 3)
        ExpectedGiB = [math]::Round($file.Bytes / 1GB, 3)
        Percent = [math]::Round(100 * $bytes / $file.Bytes, 2)
        Complete = ($bytes -eq $file.Bytes)
        SpeedMiBs = [math]::Round($speed / 1MB, 2)
    }
}
$state = if (Test-Path -LiteralPath $statusPath) {
    (Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json).state
} else { 'not-started' }
$finalizationState = if (Test-Path -LiteralPath $finalStatusPath) {
    (Get-Content -LiteralPath $finalStatusPath -Raw | ConvertFrom-Json).state
} else { 'not-started' }
$active = @(Get-CimInstance Win32_Process | Where-Object {
    ($_.CommandLine -like '*download_visual_redactions_images.ps1*' -or
     $_.CommandLine -like '*download_visual_redactions_aria2.ps1*' -or
     ($_.Name -eq 'aria2c.exe' -and $_.CommandLine -like '*orekondy18cvpr*')) -and
    $_.ProcessId -ne $PID
} | Select-Object ProcessId, Name)
[pscustomobject]@{
    State = $state
    FinalizationState = $finalizationState
    Active = ($active.Count -gt 0)
    Processes = $active
    DownloadedGiB = [math]::Round($downloadedTotal / 1GB, 3)
    ExpectedGiB = [math]::Round($expected / 1GB, 3)
    OverallPercent = [math]::Round(100 * $downloadedTotal / $expected, 2)
    TotalSpeedMiBs = [math]::Round($totalSpeed / 1MB, 2)
    EstimatedSecondsRemaining = if ($totalSpeed -gt 0) { [math]::Round(($expected - $downloadedTotal) / $totalSpeed) } else { $null }
    Archives = $rows
    FreeGiB = [math]::Round((Get-PSDrive C).Free / 1GB, 2)
} | ConvertTo-Json -Depth 5
