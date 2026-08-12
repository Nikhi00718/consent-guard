[CmdletBinding()]
param()

$workspaceRoot = Split-Path -Parent $PSScriptRoot
$rawRoot = Join-Path $workspaceRoot 'data\raw'

function Format-ProgressLine {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int64]$ExpectedBytes
    )

    $currentBytes = if (Test-Path -LiteralPath $Path) {
        (Get-Item -LiteralPath $Path).Length
    }
    else {
        0
    }
    $percent = if ($ExpectedBytes -gt 0) {
        [math]::Round(($currentBytes / $ExpectedBytes) * 100, 2)
    }
    else {
        0
    }

    [PSCustomObject]@{
        Dataset = $Name
        DownloadedGB = [math]::Round($currentBytes / 1GB, 3)
        ExpectedGB = [math]::Round($ExpectedBytes / 1GB, 3)
        Percent = $percent
    }
}

$visprRoot = Join-Path $rawRoot 'vispr'
$expectedVispr = @(
    @{ Name = 'VISPR train'; File = 'train2017.tar.gz'; Bytes = 22377737446 },
    @{ Name = 'VISPR val'; File = 'val2017.tar.gz'; Bytes = 9435240408 },
    @{ Name = 'VISPR test'; File = 'test2017.tar.gz'; Bytes = 17367722420 }
)

$progress = foreach ($entry in $expectedVispr) {
    Format-ProgressLine `
        -Name $entry.Name `
        -Path (Join-Path $visprRoot $entry.File) `
        -ExpectedBytes $entry.Bytes
}
$progress | Format-Table -AutoSize

$redactionsRoot = Join-Path $rawRoot 'visual_redactions'
$expectedRedactions = @(
    @{ Name = 'Redactions train masks'; File = 'train2017.json'; Bytes = 71097278 },
    @{ Name = 'Redactions val masks'; File = 'val2017.json'; Bytes = 30457149 },
    @{ Name = 'Redactions test masks'; File = 'test2017.json'; Bytes = 57474452 },
    @{ Name = 'Redactions train weak'; File = 'train2017-extra.zip'; Bytes = 65824250 },
    @{ Name = 'Redactions val weak'; File = 'val2017-extra.zip'; Bytes = 27110759 },
    @{ Name = 'Redactions test weak'; File = 'test2017-extra.zip'; Bytes = 51760164 }
)

$redactionProgress = foreach ($entry in $expectedRedactions) {
    Format-ProgressLine `
        -Name $entry.Name `
        -Path (Join-Path $redactionsRoot $entry.File) `
        -ExpectedBytes $entry.Bytes
}
$redactionProgress | Format-Table -AutoSize

$vpdRoot = Join-Path $rawRoot 'vpd_public'
$vpdFiles = Get-ChildItem -LiteralPath $vpdRoot -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notmatch '\\.cache\\huggingface\\download' }
$vpdPartialFiles = Get-ChildItem -LiteralPath (Join-Path $vpdRoot '.cache\huggingface\download') `
    -File -Recurse -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like '*.incomplete' }
$vpdCommittedBytes = (($vpdFiles | Measure-Object Length -Sum).Sum)
$vpdPartialBytes = (($vpdPartialFiles | Measure-Object Length -Sum).Sum)
$vpdExpectedBytes = 33006447236
[PSCustomObject]@{
    Dataset = 'VPD public repository'
    Files = @($vpdFiles).Count
    CommittedGiB = [math]::Round($vpdCommittedBytes / 1GB, 3)
    PartialGiB = [math]::Round($vpdPartialBytes / 1GB, 3)
    DownloadedGiB = [math]::Round(($vpdCommittedBytes + $vpdPartialBytes) / 1GB, 3)
    ExpectedGiB = [math]::Round($vpdExpectedBytes / 1GB, 3)
    Percent = [math]::Round((($vpdCommittedBytes + $vpdPartialBytes) / $vpdExpectedBytes) * 100, 2)
} | Format-Table -AutoSize

$processInspectionLimited = $false
try {
    $downloadProcesses = Get-CimInstance Win32_Process -ErrorAction Stop |
        Where-Object {
            $_.CommandLine -match 'download_datasets.ps1|hf.exe.+Visual_Privacy_Dataset|curl.exe.+orekondy'
        } |
        Where-Object { $_.CommandLine -notmatch 'check_dataset_status.ps1' } |
        Select-Object ProcessId, ParentProcessId, Name, CommandLine
}
catch {
    $processInspectionLimited = $true
    $downloadProcesses = Get-Process -Name curl,hf,python,powershell -ErrorAction SilentlyContinue |
        Select-Object @{Name='ProcessId';Expression={$_.Id}}, Name, CPU, StartTime
}

Write-Host 'Active dataset processes:'
if ($downloadProcesses) {
    $downloadProcesses | Format-Table -AutoSize
    if ($processInspectionLimited) {
        Write-Host '  Command-line inspection was denied; these are candidate process names only.'
    }
}
else {
    Write-Host '  None'
}

$driveName = ([System.IO.Path]::GetPathRoot($workspaceRoot)).TrimEnd('\').TrimEnd(':')
$freeBytes = $null
try {
    $drive = Get-PSDrive -Name $driveName -ErrorAction Stop
    if ($drive.Free -gt 0) {
        $freeBytes = $drive.Free
    }
}
catch {
    $freeBytes = $null
}
if ($null -eq $freeBytes) {
    try {
        $driveInfo = [System.IO.DriveInfo]::new("$driveName`:\")
        if ($driveInfo.IsReady) {
            $freeBytes = $driveInfo.AvailableFreeSpace
        }
    }
    catch {
        $freeBytes = $null
    }
}
if ($null -ne $freeBytes) {
    Write-Host "Free space: $([math]::Round($freeBytes / 1GB, 1)) GB"
}
else {
    Write-Host 'Free space: unavailable in this shell; run Get-PSDrive C in your terminal.'
}
Write-Host "Logs: $(Join-Path $workspaceRoot 'data\download_logs')"
