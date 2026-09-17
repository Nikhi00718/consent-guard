<#
.SYNOPSIS
    Run every check locally, in the order that fails fastest.

.DESCRIPTION
    There is no hosted CI for this project: the tests need the local GPU, the
    trained checkpoints, and the dataset records, none of which leave this
    machine. This script is the equivalent gate. Run it before committing.

    Stages:
      1. Python unit/integration tests (pytest)
      2. Frozen baseline manifest verification (hashes of checkpoint + records)
      3. Frontend type-check and production build
      4. Frontend unit tests (vitest)
      5. Browser end-to-end tests (Playwright, model-free fixture API)
      6. Optional: a real one-photo run through every detector (-IncludeModels)
#>
[CmdletBinding()]
param(
    [switch]$IncludeModels,
    [switch]$SkipBrowser
)

$ErrorActionPreference = 'Continue'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $repo

$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Python environment missing at $python" }

$env:PYTHONPATH = @(
    (Join-Path $repo 'main_project\src'),
    (Join-Path $repo 'main_project\scripts\stage_05_review_export')
) -join ';'
$env:CONSENTGUARD_PYTHON = $python

$results = [System.Collections.Generic.List[object]]::new()

function Invoke-Stage {
    param([string]$Name, [scriptblock]$Body)
    Write-Host ''
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    $started = Get-Date
    & $Body
    $ok = $LASTEXITCODE -eq 0
    $results.Add([pscustomobject]@{
        Stage   = $Name
        Result  = if ($ok) { 'PASS' } else { "FAIL (exit $LASTEXITCODE)" }
        Seconds = [math]::Round(((Get-Date) - $started).TotalSeconds, 1)
    })
    if (-not $ok) { Write-Host "$Name failed" -ForegroundColor Red }
}

Invoke-Stage 'Python tests' { & $python -m pytest -q }
Invoke-Stage 'Frozen baseline manifest' { & $python main_project\scripts\stage_02_baseline_model\verify_baseline.py | Out-Null }
Invoke-Stage 'Frontend build' { npm --prefix main_project/frontend run build }
Invoke-Stage 'Frontend unit tests' { npm --prefix main_project/frontend test }
if (-not $SkipBrowser) {
    Invoke-Stage 'Browser end-to-end' { npm --prefix main_project/frontend run test:e2e }
}
if ($IncludeModels) {
    Invoke-Stage 'Real-model photo run' {
        & $python main_project\scripts\stage_05_review_export\run_analysis_pipeline.py `
            --config main_project/configs/stage_02_baseline_model/train_maskrcnn_moderate_v2_negatives_10ep.yaml `
            --checkpoint artifacts/checkpoints/maskrcnn_moderate_v2_negatives_10ep/last.pt `
            --input data/raw/visual_redactions/images/val2017/2017_10024630.jpg `
            --threshold-profile main_project/configs/stage_04_fusion_calibration/threshold_profile_personal_max_coverage.yaml `
            --yunet-model artifacts/specialists/opencv_zoo/face_detection_yunet_2023mar.onnx `
            --plate-yunet-model artifacts/specialists/opencv_zoo/license_plate_detection_lpd_yunet_2023mar.onnx `
            --ppocr-model artifacts/specialists/opencv_zoo/text_detection_en_ppocrv3_2023may.onnx `
            --with-barcode --device auto `
            --report reports/ci_local_pipeline_smoke.json | Out-Null
    }
}

Write-Host ''
Write-Host '=== Summary ===' -ForegroundColor Cyan
$results | Format-Table -AutoSize
$failed = @($results | Where-Object { $_.Result -ne 'PASS' })
if ($failed.Count -gt 0) {
    Write-Host "$($failed.Count) stage(s) failed." -ForegroundColor Red
    exit 1
}
Write-Host 'All stages passed.' -ForegroundColor Green
exit 0
