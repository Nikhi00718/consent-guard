[CmdletBinding()]
param(
    [switch]$VisprAnnotations,
    [switch]$VisprImages,
    [switch]$VisualRedactionsAnnotations,
    [switch]$VisualRedactionsInfo,
    [switch]$VpdInspect,
    [switch]$VpdPublicVideos,
    [switch]$ExtractVispr,
    [switch]$ComputeHashes,
    [switch]$AcceptLargeDownloads
)

$ErrorActionPreference = 'Stop'

$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$rawDataRoot = Join-Path $workspaceRoot 'data\raw'
$visprRoot = Join-Path $rawDataRoot 'vispr'
$redactionsRoot = Join-Path $rawDataRoot 'visual_redactions'
$vpdRoot = Join-Path $rawDataRoot 'vpd_public'

$selectedAction = $VisprAnnotations -or $VisprImages -or $VisualRedactionsAnnotations -or $VisualRedactionsInfo -or
    $VpdInspect -or $VpdPublicVideos -or $ExtractVispr -or $ComputeHashes

if (-not $selectedAction) {
    Write-Host 'No action selected.'
    Write-Host ''
    Write-Host 'Recommended order:'
    Write-Host '  .\scripts\download_datasets.ps1 -VisprAnnotations'
    Write-Host '  .\scripts\download_datasets.ps1 -VisualRedactionsAnnotations'
    Write-Host '  .\scripts\download_datasets.ps1 -VisprImages -AcceptLargeDownloads'
    Write-Host '  .\scripts\download_datasets.ps1 -VpdInspect'
    Write-Host '  .\scripts\download_datasets.ps1 -VpdPublicVideos -AcceptLargeDownloads'
    Write-Host ''
    Write-Host 'The VPD public download currently contains videos, not a confirmed'
    Write-Host '100,000-image bounding-box training package.'
    exit 0
}

New-Item -ItemType Directory -Force $visprRoot, $redactionsRoot, $vpdRoot | Out-Null

function Get-FreeSpaceGB {
    $driveName = ([System.IO.Path]::GetPathRoot($workspaceRoot)).TrimEnd('\').TrimEnd(':')
    $drive = Get-PSDrive -Name $driveName
    return [math]::Round($drive.Free / 1GB, 1)
}

function Get-HfCommand {
    $hf = Get-Command hf -ErrorAction SilentlyContinue
    if ($hf) {
        return $hf.Source
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        $pythonExecutable = (& python -c "import sys; print(sys.executable)" 2>$null)
        if ($pythonExecutable) {
            $candidate = Join-Path (Split-Path -Parent $pythonExecutable) 'Scripts\hf.exe'
            if (Test-Path -LiteralPath $candidate) {
                return $candidate
            }
        }
    }

    return $null
}

function Invoke-ResumableDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    $lockPath = "$Destination.download.lock"
    try {
        $lockStream = [System.IO.File]::Open(
            $lockPath,
            [System.IO.FileMode]::OpenOrCreate,
            [System.IO.FileAccess]::ReadWrite,
            [System.IO.FileShare]::None
        )
    }
    catch {
        throw "Another process already owns the download lock for $Destination. Refusing a duplicate writer."
    }

    try {
      $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
      if (-not $curl) {
          throw 'curl.exe was not found. Install curl or use a current Windows installation.'
      }

    $remoteLength = $null
    try {
        $head = Invoke-WebRequest -Uri $Url -Method Head -UseBasicParsing
        $lengthHeader = $head.Headers['Content-Length']
        if ($lengthHeader) {
            $remoteLength = [int64]$lengthHeader
        }
    }
    catch {
        Write-Warning "Could not read remote size for $Url. The resumable download will still be attempted."
    }

    if (Test-Path -LiteralPath $Destination) {
        $localLength = (Get-Item -LiteralPath $Destination).Length
        if ($remoteLength -and $localLength -eq $remoteLength) {
            Write-Host "Already complete: $Destination"
            return
        }
        if ($remoteLength -and $localLength -gt $remoteLength) {
            throw "Local file is larger than the remote file: $Destination"
        }
        Write-Host "Resuming: $Destination"
    }
    else {
        Write-Host "Downloading: $Destination"
    }

    & $curl.Source `
        --location `
        --fail `
        --retry 5 `
        --retry-delay 5 `
        --continue-at - `
        --output $Destination `
        $Url

    if ($LASTEXITCODE -ne 0) {
        throw "Download failed with curl exit code $LASTEXITCODE for $Url"
    }

      if ($remoteLength) {
          $downloadedLength = (Get-Item -LiteralPath $Destination).Length
          if ($downloadedLength -ne $remoteLength) {
              throw "Size verification failed for $Destination"
          }
      }
    }
    finally {
        if ($lockStream) {
            $lockStream.Dispose()
        }
    }
}

$visprAnnotationFiles = @(
    @{
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/train2017_anno.tar.gz'
        Name = 'train2017_anno.tar.gz'
    },
    @{
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/val2017_anno.tar.gz'
        Name = 'val2017_anno.tar.gz'
    },
    @{
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/test2017_anno.tar.gz'
        Name = 'test2017_anno.tar.gz'
    }
)

$visprImageFiles = @(
    @{
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/train2017.tar.gz'
        Name = 'train2017.tar.gz'
        Approx = '21 GB'
    },
    @{
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/val2017.tar.gz'
        Name = 'val2017.tar.gz'
        Approx = '8.8 GB'
    },
    @{
        Url = 'https://datasets.d2.mpi-inf.mpg.de/orekondy17iccv/test2017.tar.gz'
        Name = 'test2017.tar.gz'
        Approx = '17 GB'
    }
)

$visualRedactionsAnnotationFiles = @(
    @{
        Url = 'https://www.datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/annotations/train2017.json'
        Name = 'train2017.json'
    },
    @{
        Url = 'https://www.datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/annotations/val2017.json'
        Name = 'val2017.json'
    },
    @{
        Url = 'https://www.datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/annotations/test2017.json'
        Name = 'test2017.json'
    },
    @{
        Url = 'https://www.datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/annotations-extra/train2017.zip'
        Name = 'train2017-extra.zip'
    },
    @{
        Url = 'https://www.datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/annotations-extra/val2017.zip'
        Name = 'val2017-extra.zip'
    },
    @{
        Url = 'https://www.datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/annotations-extra/test2017.zip'
        Name = 'test2017-extra.zip'
    }
)

if ($VisprAnnotations) {
    Write-Host "Downloading VISPR annotations to $visprRoot"
    foreach ($file in $visprAnnotationFiles) {
        Invoke-ResumableDownload `
            -Url $file.Url `
            -Destination (Join-Path $visprRoot $file.Name)
    }
}

if ($VisprImages) {
    if (-not $AcceptLargeDownloads) {
        throw 'VISPR images are about 47 GB compressed. Re-run with -AcceptLargeDownloads.'
    }

    Write-Host "Free space before VISPR image download: $(Get-FreeSpaceGB) GB"
    foreach ($file in $visprImageFiles) {
        Write-Host "Expected archive size: $($file.Approx)"
        Invoke-ResumableDownload `
            -Url $file.Url `
            -Destination (Join-Path $visprRoot $file.Name)
    }
}

if ($VisualRedactionsAnnotations) {
    Write-Host "Downloading Visual Redactions annotations to $redactionsRoot"
    foreach ($file in $visualRedactionsAnnotationFiles) {
        Invoke-ResumableDownload `
            -Url $file.Url `
            -Destination (Join-Path $redactionsRoot $file.Name)
    }

    $extraRoot = Join-Path $redactionsRoot 'annotations-extra'
    New-Item -ItemType Directory -Force -Path $extraRoot | Out-Null
    foreach ($archive in Get-ChildItem -LiteralPath $redactionsRoot -Filter '*-extra.zip' -File) {
        $splitName = $archive.BaseName -replace '-extra$', ''
        $destination = Join-Path $extraRoot $splitName
        New-Item -ItemType Directory -Force -Path $destination | Out-Null
        Write-Host "Extracting $($archive.Name)"
        Expand-Archive -LiteralPath $archive.FullName -DestinationPath $destination -Force
    }
}

if ($VisualRedactionsInfo) {
    Write-Host ''
    Write-Host 'Visual Redactions official project page:'
    Write-Host 'Official page:'
    Write-Host '  https://resources.mpi-inf.mpg.de/d2/orekondy/redactions/'
    Write-Host ''
    Write-Host 'Download Train/Val/Test annotations with:'
    Write-Host '  .\scripts\download_datasets.ps1 -VisualRedactionsAnnotations'
    Write-Host 'They will be saved under:'
    Write-Host "  $redactionsRoot"
    Write-Host ''
    Write-Host 'The project-page host returns HTTP 403, but its dataset host is accessible.'
    Write-Warning 'Do not reuse VISPR pixels for Visual Redactions masks. Matching ID strings do not prove image identity.'
    Write-Host 'Separate official image archives (15.467 GiB compressed total):'
    Write-Host '  https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/train2017.tar.gz'
    Write-Host '  https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/val2017.tar.gz'
    Write-Host '  https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/test2017.tar.gz'
    Write-Warning 'Free substantial extraction headroom and add byte-length verification before downloading.'
}

if ($VpdInspect) {
    $datasetId = 'XiaoyuSunANU/Visual_Privacy_Dataset'
    $encodedId = [uri]::EscapeDataString($datasetId)
    $sizeInfo = Invoke-RestMethod "https://datasets-server.huggingface.co/size?dataset=$encodedId"
    $splitInfo = Invoke-RestMethod "https://datasets-server.huggingface.co/splits?dataset=$encodedId"

    Write-Host 'Current Hugging Face Dataset Viewer status:'
    Write-Host "  Dataset: $datasetId"
    Write-Host "  Rows: $($sizeInfo.size.dataset.num_rows)"
    Write-Host "  Splits: $(($splitInfo.splits | ForEach-Object { $_.split }) -join ', ')"
    Write-Host '  Current rows point to video files.'
    Write-Host ''

    $hf = Get-HfCommand
    if (-not $hf) {
        Write-Warning 'The hf CLI is not installed.'
        Write-Host 'Install it with:'
        Write-Host '  python -m pip install --upgrade huggingface_hub'
        Write-Host 'Then open a new PowerShell window and run this inspection again.'
    }
    else {
        $dryRunPath = Join-Path $vpdRoot 'vpd_download_dry_run.txt'
        Write-Host "Writing the hf dry-run inventory to $dryRunPath"
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $dryRunOutput = & $hf download $datasetId `
            --type dataset `
            --dry-run 2>&1
        $hfExitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorActionPreference
        $dryRunOutput | Set-Content -LiteralPath $dryRunPath -Encoding utf8

        if ($hfExitCode -ne 0) {
            throw "hf dry-run failed with exit code $hfExitCode"
        }
        Write-Host "Inventory lines: $($dryRunOutput.Count)"
    }
}

if ($VpdPublicVideos) {
    if (-not $AcceptLargeDownloads) {
        throw 'The current VPD public repository is about 33 GB. Re-run with -AcceptLargeDownloads.'
    }

    $hf = Get-HfCommand
    if (-not $hf) {
        throw 'Install the hf CLI first: python -m pip install --upgrade huggingface_hub'
    }

    Write-Warning 'This currently downloads public VPD videos, not confirmed VPD-100K image annotations.'
    Write-Host "Free space before VPD download: $(Get-FreeSpaceGB) GB"
    & $hf download XiaoyuSunANU/Visual_Privacy_Dataset `
        --type dataset `
        --local-dir $vpdRoot `
        --max-workers 4

    if ($LASTEXITCODE -ne 0) {
        throw "VPD download failed with exit code $LASTEXITCODE"
    }
}

if ($ExtractVispr) {
    Write-Host "Free space before extraction: $(Get-FreeSpaceGB) GB"
    $finalizer = Join-Path $PSScriptRoot 'finalize_vispr_data.py'
    & py -3.11 $finalizer --split all --extract
    if ($LASTEXITCODE -ne 0) {
        throw 'Safe VISPR archive validation/extraction failed.'
    }
}

if ($ComputeHashes) {
    $manifestPath = Join-Path $rawDataRoot 'sha256_manifest.txt'
    $files = Get-ChildItem -LiteralPath $rawDataRoot -Recurse -File |
        Where-Object { $_.FullName -ne $manifestPath }

    if (-not $files) {
        Write-Warning "No files found under $rawDataRoot"
    }
    else {
        $hashLines = foreach ($file in $files) {
            $hash = Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256
            $relativePath = [System.IO.Path]::GetRelativePath($rawDataRoot, $file.FullName)
            "$($hash.Hash)  $relativePath"
        }
        $hashLines | Set-Content -LiteralPath $manifestPath -Encoding utf8
        Write-Host "Wrote SHA-256 manifest: $manifestPath"
    }
}

Write-Host ''
Write-Host "Done. Free space: $(Get-FreeSpaceGB) GB"
