[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('val2017', 'test2017')]
    [string]$Split
)

$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$archiveRoot = Join-Path $workspace 'data\raw\visual_redactions\image_archives'
$logRoot = Join-Path $workspace 'data\download_logs'
$definitions = @{
    val2017 = @{
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/val2017.tar.gz'
        Bytes = [int64]3158549744
    }
    test2017 = @{
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/test2017.tar.gz'
        Bytes = [int64]5633037591
    }
}
New-Item -ItemType Directory -Force -Path $archiveRoot, $logRoot | Out-Null
$destination = Join-Path $archiveRoot "$Split.tar.gz"
$lockPath = Join-Path $logRoot "$Split.parallel.lock"
$lock = [System.IO.File]::Open(
    $lockPath,
    [System.IO.FileMode]::OpenOrCreate,
    [System.IO.FileAccess]::ReadWrite,
    [System.IO.FileShare]::None
)
try {
    $definition = $definitions[$Split]
    $curl = (Get-Command curl.exe -ErrorAction Stop).Source
    & $curl --location --fail --retry 12 --retry-all-errors --retry-delay 5 `
        --continue-at - --output $destination $definition.Url
    if ($LASTEXITCODE -ne 0) { throw "curl failed with exit code $LASTEXITCODE" }
    $actual = (Get-Item -LiteralPath $destination).Length
    if ($actual -ne $definition.Bytes) {
        throw "$Split size mismatch: expected $($definition.Bytes), got $actual"
    }
}
finally {
    $lock.Dispose()
}
