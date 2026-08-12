[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [switch]$SkipPreflight
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
$setupLockPath = Join-Path $workspaceRoot '.venv.setup.lock'

try {
    $setupLock = [System.IO.File]::Open(
        $setupLockPath,
        [System.IO.FileMode]::OpenOrCreate,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
}
catch {
    throw 'Another environment setup process already owns .venv.setup.lock.'
}

function Invoke-Checked {
    param([Parameter(Mandatory = $true)][scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

Push-Location $workspaceRoot
try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        Invoke-Checked { py -3.11 -m venv .venv }
    }
    Invoke-Checked { & $venvPython -m pip install --upgrade pip setuptools wheel }
    if ($CpuOnly) {
        Invoke-Checked { & $venvPython -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu }
    }
    else {
        $wheelRoot = Join-Path $workspaceRoot 'data\cache\wheels'
        New-Item -ItemType Directory -Path $wheelRoot -Force | Out-Null
        $wheelSpecs = @(
            @{
                Name = 'torch-2.13.0+cu126-cp311-cp311-win_amd64.whl'
                Url = 'https://download-r2.pytorch.org/whl/cu126/torch-2.13.0%2Bcu126-cp311-cp311-win_amd64.whl'
                Bytes = [int64]2594548547
                Sha256 = '8095729db14e7fd5178a39676fdd679208eff4041407ea34e3d898336c90f5c5'
            },
            @{
                Name = 'torchvision-0.28.0+cu126-cp311-cp311-win_amd64.whl'
                Url = 'https://download-r2.pytorch.org/whl/cu126/torchvision-0.28.0%2Bcu126-cp311-cp311-win_amd64.whl'
                Bytes = [int64]8520320
                Sha256 = '8a976240db376f83dda566bc71320071cbf5f0a013c87b3f34e0e81f2ca96da8'
            }
        )
        $curl = (Get-Command curl.exe -ErrorAction Stop).Source
        foreach ($wheelSpec in $wheelSpecs) {
            $wheelPath = Join-Path $wheelRoot $wheelSpec.Name
            $currentBytes = if (Test-Path -LiteralPath $wheelPath) { (Get-Item -LiteralPath $wheelPath).Length } else { 0 }
            if ($currentBytes -gt $wheelSpec.Bytes) {
                throw "Local wheel is larger than expected: $wheelPath"
            }
            if ($currentBytes -lt $wheelSpec.Bytes) {
                Invoke-Checked {
                    & $curl --location --fail --retry 20 --retry-all-errors --retry-delay 10 --continue-at - --output $wheelPath $wheelSpec.Url
                }
            }
            $downloadedBytes = (Get-Item -LiteralPath $wheelPath).Length
            if ($downloadedBytes -ne $wheelSpec.Bytes) {
                throw "Wheel size verification failed for $wheelPath ($downloadedBytes/$($wheelSpec.Bytes) bytes)"
            }
            if ($wheelSpec.ContainsKey('Sha256')) {
                $wheelHash = (Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($wheelHash -ne $wheelSpec.Sha256) {
                    throw "Wheel SHA-256 verification failed for $wheelPath ($wheelHash)"
                }
            }
        }
        Invoke-Checked { & $venvPython -m pip install (Join-Path $wheelRoot $wheelSpecs[0].Name) (Join-Path $wheelRoot $wheelSpecs[1].Name) }

        $torchHome = Join-Path $workspaceRoot 'data\cache\torch'
        $env:TORCH_HOME = $torchHome
        $checkpointRoot = Join-Path $torchHome 'hub\checkpoints'
        New-Item -ItemType Directory -Path $checkpointRoot -Force | Out-Null
        $modelWeightName = 'maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth'
        $modelWeightPath = Join-Path $checkpointRoot $modelWeightName
        $modelWeightBytes = [int64]185828065
        $modelWeightUrl = 'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth'
        $currentModelBytes = if (Test-Path -LiteralPath $modelWeightPath) { (Get-Item -LiteralPath $modelWeightPath).Length } else { 0 }
        if ($currentModelBytes -gt $modelWeightBytes) {
            throw "Local model weight is larger than expected: $modelWeightPath"
        }
        if ($currentModelBytes -lt $modelWeightBytes) {
            Invoke-Checked {
                & $curl --location --fail --retry 20 --retry-all-errors --retry-delay 10 --continue-at - --output $modelWeightPath $modelWeightUrl
            }
        }
        if ((Get-Item -LiteralPath $modelWeightPath).Length -ne $modelWeightBytes) {
            throw "Model weight size verification failed: $modelWeightPath"
        }
        $modelWeightHash = (Get-FileHash -LiteralPath $modelWeightPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($modelWeightHash -ne '73cbd0190fcbe3ba339921fbce2c3a0b6bb9126c9a133c85e43a2a8e060a109e') {
            throw "Model weight SHA-256 verification failed: $modelWeightHash"
        }
    }
    Invoke-Checked { & $venvPython -m pip install -r requirements\base.txt -r requirements\dev.txt }
    Invoke-Checked { & $venvPython -m pip install -e . --no-deps }
    if (-not $SkipPreflight) {
        Invoke-Checked { & $venvPython scripts\preflight_environment.py --config configs\train_smoke.yaml }
    }
}
finally {
    Pop-Location
    $setupLock.Dispose()
}
