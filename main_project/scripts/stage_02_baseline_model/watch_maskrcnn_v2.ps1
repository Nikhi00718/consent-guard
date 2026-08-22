param(
    [int]$RefreshSeconds = 5,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$artifactRoot = Join-Path (Join-Path $projectRoot 'artifacts') 'checkpoints'
$runDirectory = Join-Path $artifactRoot 'maskrcnn_4gb_v2'
$metricsPath = Join-Path $runDirectory 'metrics.jsonl'
$configPath = Join-Path (Join-Path $projectRoot 'configs') 'train_maskrcnn_4gb_v2.yaml'
$recordsPath = Join-Path (Join-Path (Join-Path (Join-Path $projectRoot 'data') 'processed') 'visual_redactions') 'records_train2017.jsonl'

function Get-YamlInteger([string]$Name, [int]$Fallback) {
    if (-not (Test-Path -LiteralPath $configPath)) { return $Fallback }
    $match = [regex]::Match((Get-Content -LiteralPath $configPath -Raw), "(?m)^\s*$Name\s*:\s*(\d+)\s*$")
    if ($match.Success) { return [int]$match.Groups[1].Value }
    return $Fallback
}

function New-Bar([double]$Fraction, [int]$Width = 34) {
    $bounded = [Math]::Max(0.0, [Math]::Min(1.0, $Fraction))
    $filled = [int][Math]::Floor($bounded * $Width)
    return ('#' * $filled) + ('-' * ($Width - $filled))
}

$epochCount = Get-YamlInteger 'epochs' 30
$accumulation = Get-YamlInteger 'gradient_accumulation_steps' 4
$trainImages = if (Test-Path -LiteralPath $recordsPath) { (Get-Content -LiteralPath $recordsPath | Measure-Object -Line).Lines } else { 3873 }
$stepsPerEpoch = [int][Math]::Ceiling($trainImages / [double]$accumulation)
$totalSteps = $epochCount * $stepsPerEpoch

do {
    Clear-Host
    Write-Host 'ConsentGuard - Live Mask R-CNN v2 Training Monitor' -ForegroundColor Cyan
    Write-Host ("Updated: {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor DarkGray
    Write-Host ''

    $trainer = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match 'train_maskrcnn.py' -and $_.CommandLine -match 'train_maskrcnn_4gb_v2.yaml' } |
        Select-Object -First 1
    if ($null -ne $trainer) {
        Write-Host ("Trainer: RUNNING (PID {0})" -f $trainer.ProcessId) -ForegroundColor Green
    } else {
        Write-Host 'Trainer: not detected (finished or stopped)' -ForegroundColor Yellow
    }

    if (-not (Test-Path -LiteralPath $metricsPath)) {
        Write-Host 'Waiting for the first metrics entry; trainer is initialising.' -ForegroundColor Yellow
    } else {
        $entries = Get-Content -LiteralPath $metricsPath | ForEach-Object {
            if ($_.Trim()) { $_ | ConvertFrom-Json }
        }
        $lastStep = $entries | Where-Object { $_.event -eq 'train_step' } | Select-Object -Last 1
        $lastEpoch = $entries | Where-Object { $_.event -eq 'epoch_complete' } | Select-Object -Last 1
        $lastEvaluation = $entries | Where-Object { $_.event -eq 'evaluation' } | Select-Object -Last 1

        if ($null -eq $lastStep) {
            Write-Host 'Metrics file exists; waiting for the first optimizer step.' -ForegroundColor Yellow
        } else {
            $step = [int]$lastStep.global_step
            $fraction = $step / [double]$totalSteps
            $activeEpoch = [Math]::Min($epochCount, [int][Math]::Floor(($step - 1) / $stepsPerEpoch) + 1)
            $epochStep = (($step - 1) % $stepsPerEpoch) + 1
            Write-Host ("Overall [{0}] {1,6:P1}  ({2}/{3} optimizer steps)" -f (New-Bar $fraction), $fraction, $step, $totalSteps) -ForegroundColor Cyan
            $epochFraction = $epochStep / [double]$stepsPerEpoch
            Write-Host ("Epoch {0}/{1} [{2}] {3,6:P1}  ({4}/{5} steps)" -f $activeEpoch, $epochCount, (New-Bar $epochFraction), $epochFraction, $epochStep, $stepsPerEpoch) -ForegroundColor Cyan
            Write-Host ("Latest loss: {0:N4}   LR: {1:E2}" -f [double]$lastStep.loss, [double]$lastStep.learning_rate)

            $firstStep = $entries | Where-Object { $_.event -eq 'train_step' } | Select-Object -First 1
            if ($null -ne $firstStep -and $step -ge 20) {
                $elapsedSeconds = [Math]::Max(1.0, [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [double]$firstStep.time_unix)
                $remainingSeconds = [Math]::Max(0.0, ($totalSteps - $step) * ($elapsedSeconds / $step))
                $finish = (Get-Date).AddSeconds($remainingSeconds)
                Write-Host ("Estimated remaining: {0}   Finish: {1}" -f ([TimeSpan]::FromSeconds($remainingSeconds).ToString('g')), $finish.ToString('yyyy-MM-dd HH:mm')) -ForegroundColor Yellow
            } else {
                Write-Host 'Estimated remaining: calibrating from the first 20 optimizer steps.' -ForegroundColor Yellow
            }

            if ($null -ne $lastEpoch) {
                Write-Host ("Completed epochs: {0}   Last checkpoint: epoch {1}" -f ([int]$lastEpoch.epoch + 1), ([int]$lastEpoch.epoch + 1))
            }
            if ($null -ne $lastEvaluation) {
                $metrics = $lastEvaluation.metrics
                Write-Host ("Validation mask mAP: {0:N5}   mAP@50: {1:N5}   Recall@100: {2:N5}" -f [double]$metrics.segm_map, [double]$metrics.segm_map_50, [double]$metrics.segm_mar_100) -ForegroundColor Magenta
            } else {
                Write-Host 'Validation: waiting for epoch 1 to complete.' -ForegroundColor DarkGray
            }
        }
    }

    try {
        $gpu = & nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ($gpu) { Write-Host ("GPU (memory MiB, utilisation %, temperature C, power W): {0}" -f $gpu) -ForegroundColor Gray }
    } catch {}
    Write-Host ''
    Write-Host ("Refreshes every {0} seconds. Ctrl+C closes this monitor only." -f $RefreshSeconds) -ForegroundColor DarkGray

    if (-not $Once) { Start-Sleep -Seconds $RefreshSeconds }
} while (-not $Once)
