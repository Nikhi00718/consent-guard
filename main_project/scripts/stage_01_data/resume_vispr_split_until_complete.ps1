[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('val', 'test')]
    [string]$Split,

    [int]$RetryDelaySeconds = 60
)

$ErrorActionPreference = 'Continue'
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$destinationRoot = Join-Path $workspaceRoot 'data\raw\vispr'

$files = @{
    val = @{
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/val2017.tar.gz'
        Name = 'val2017.tar.gz'
        ExpectedBytes = 9435240408
    }
    test = @{
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/test2017.tar.gz'
        Name = 'test2017.tar.gz'
        ExpectedBytes = 17367722420
    }
}

$file = $files[$Split]
$destination = Join-Path $destinationRoot $file.Name
$curl = (Get-Command curl.exe -ErrorAction Stop).Source
$lockPath = "$destination.download.lock"

function Test-TarGzipIntegrity {
    param([Parameter(Mandatory = $true)][string]$Path)
    $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
    if (-not $tar) {
        Write-Error 'tar.exe is required for completion integrity checks.'
        return $false
    }
    & $tar.Source -tzf $Path *> $null
    return $LASTEXITCODE -eq 0
}

try {
    $lockStream = [System.IO.File]::Open(
        $lockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    Write-Error "Another process already owns the download lock for $destination. Refusing a duplicate writer."
    exit 3
}

try {
  while ($true) {
    $currentBytes = if (Test-Path -LiteralPath $destination) {
        (Get-Item -LiteralPath $destination).Length
    }
    else {
        0
    }

    if ($currentBytes -eq $file.ExpectedBytes) {
        Write-Host "[$(Get-Date)] $Split reached its expected size; validating the complete gzip/tar stream."
        if (Test-TarGzipIntegrity -Path $destination) {
            Write-Host "[$(Get-Date)] $Split archive is complete and readable."
            exit 0
        }
        Write-Error "$Split archive has the expected size but failed integrity validation. Preserve it for audit and restart from a clean file."
        exit 4
    }

    Write-Host "[$(Get-Date)] Resuming $Split archive at $currentBytes bytes."
    & $curl `
        --location `
        --fail `
        --retry 20 `
        --retry-all-errors `
        --retry-delay 10 `
        --continue-at - `
        --output $destination `
        $file.Url

    if ($LASTEXITCODE -eq 0) {
        $downloadedBytes = (Get-Item -LiteralPath $destination).Length
        if ($downloadedBytes -eq $file.ExpectedBytes) {
            Write-Host "[$(Get-Date)] Downloaded expected bytes; validating the complete gzip/tar stream."
            if (Test-TarGzipIntegrity -Path $destination) {
                Write-Host "[$(Get-Date)] Completed and validated $destination."
                exit 0
            }
            Write-Error "$split archive failed full integrity validation. Preserve it for audit and restart from a clean file."
            exit 4
        }
        Write-Warning "$Split transfer returned success but has $downloadedBytes of $($file.ExpectedBytes) bytes."
    }
    else {
        Write-Warning "$Split curl exited with code $LASTEXITCODE."
    }

    Write-Host "[$(Get-Date)] Retrying $Split in $RetryDelaySeconds seconds."
    Start-Sleep -Seconds $RetryDelaySeconds
  }
}
finally {
    $lockStream.Dispose()
}
