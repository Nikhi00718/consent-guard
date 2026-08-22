param(
    [string]$Config = 'main_project/configs/stage_02_baseline_model/train_maskrcnn_verified_visual.yaml',
    [int]$RefreshSeconds = 5,
    [switch]$Once
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$configPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }
$configPath = [IO.Path]::GetFullPath($configPath)
$configText = Get-Content -LiteralPath $configPath -Raw

function Get-YamlValue([string]$Name, [string]$Fallback) {
    $match = [regex]::Match($configText, "(?m)^\s*$Name\s*:\s*([^#\r\n]+)")
    if ($match.Success) { return $match.Groups[1].Value.Trim().Trim("'", '"') }
    return $Fallback
}

function Resolve-ProjectPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $projectRoot ($Path -replace '/', '\')))
}

function New-Bar([double]$Fraction, [int]$Width = 34) {
    $bounded = [Math]::Max(0.0, [Math]::Min(1.0, $Fraction))
    $filled = [int][Math]::Floor($bounded * $Width)
    return ('#' * $filled) + ('-' * ($Width - $filled))
}

$runDirectory = Resolve-ProjectPath (Get-YamlValue 'output_dir' 'artifacts/checkpoints/unknown')
$metricsPath = Join-Path $runDirectory 'metrics.jsonl'
$resultPath = Join-Path $runDirectory 'training_result.json'
$recordsPath = Resolve-ProjectPath (Get-YamlValue 'train_records' 'data/processed/unknown.jsonl')
$epochCount = [int](Get-YamlValue 'epochs' '30')
$accumulation = [int](Get-YamlValue 'gradient_accumulation_steps' '1')
$trainImages = if (Test-Path -LiteralPath $recordsPath) {
    (Get-Content -LiteralPath $recordsPath | Measure-Object -Line).Lines
} else { 0 }
$stepsPerEpoch = if ($trainImages -gt 0) {
    [int][Math]::Ceiling($trainImages / [double]$accumulation)
} else { 1 }
$totalSteps = $epochCount * $stepsPerEpoch
$configLeaf = Split-Path -Leaf $configPath

do {
    Clear-Host
    Write-Host 'ConsentGuard - Verified Visual Mask R-CNN Training' -ForegroundColor Cyan
    Write-Host ("Updated: {0}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss')) -ForegroundColor DarkGray
    Write-Host ("Config:  {0}" -f $configLeaf) -ForegroundColor DarkGray
    Write-Host ("Data:    {0} verified images, {1} optimizer steps/epoch" -f $trainImages, $stepsPerEpoch) -ForegroundColor DarkGray
    Write-Host ''

    $trainer = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match 'train_maskrcnn.py' -and $_.CommandLine -like "*$configLeaf*" } |
        Select-Object -First 1
    if ($null -ne $trainer) {
        Write-Host ("Trainer: RUNNING (PID {0})" -f $trainer.ProcessId) -ForegroundColor Green
    } elseif (Test-Path -LiteralPath $resultPath) {
        Write-Host 'Trainer: FINISHED' -ForegroundColor Green
    } else {
        Write-Host 'Trainer: not detected (initialising, stopped, or failed)' -ForegroundColor Yellow
    }

    if (-not (Test-Path -LiteralPath $metricsPath)) {
        Write-Host 'Waiting for the first metrics entry; trainer is initialising.' -ForegroundColor Yellow
    } else {
        $entries = Get-Content -LiteralPath $metricsPath | ForEach-Object {
            if ($_.Trim()) { $_ | ConvertFrom-Json }
        }
        $steps = @($entries | Where-Object { $_.event -eq 'train_step' })
        $lastStep = $steps | Select-Object -Last 1
        $lastEpoch = $entries | Where-Object { $_.event -eq 'epoch_complete' } | Select-Object -Last 1
        $lastEvaluation = $entries | Where-Object { $_.event -eq 'evaluation' } | Select-Object -Last 1
        $resumeEvent = $entries | Where-Object { $_.event -eq 'resumed' } | Select-Object -Last 1

        if ($null -eq $lastStep) {
            Write-Host 'Metrics file exists; waiting for the first optimizer step.' -ForegroundColor Yellow
        } else {
            $step = [int]$lastStep.global_step
            $fraction = $step / [double]$totalSteps
            $activeEpoch = [Math]::Min($epochCount, [int][Math]::Floor(($step - 1) / $stepsPerEpoch) + 1)
            $epochStep = (($step - 1) % $stepsPerEpoch) + 1
            Write-Host ("Overall [{0}] {1,6:P1}  ({2}/{3} steps)" -f (New-Bar $fraction), $fraction, $step, $totalSteps) -ForegroundColor Cyan
            $epochFraction = $epochStep / [double]$stepsPerEpoch
            Write-Host ("Epoch {0}/{1} [{2}] {3,6:P1}  ({4}/{5} steps)" -f $activeEpoch, $epochCount, (New-Bar $epochFraction), $epochFraction, $epochStep, $stepsPerEpoch) -ForegroundColor Cyan
            Write-Host ("Latest loss: {0:N4}   LR: {1:E2}" -f [double]$lastStep.loss, [double]$lastStep.learning_rate)

            $firstStep = $steps | Select-Object -First 1
            if ($null -ne $resumeEvent) {
                $runStartStep = [int]$resumeEvent.global_step
                $runStartUnix = [double]$resumeEvent.time_unix
            } else {
                $runStartStep = [Math]::Max(0, [int]$firstStep.global_step - 1)
                $runStartUnix = [double]$firstStep.time_unix
            }
            $observedSteps = $step - $runStartStep
            if ($null -ne $firstStep -and $observedSteps -ge 10) {
                $elapsedSeconds = [Math]::Max(1.0, [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $runStartUnix)
                $remainingSeconds = [Math]::Max(0.0, ($totalSteps - $step) * ($elapsedSeconds / [Math]::Max(1, $observedSteps)))
                $finish = (Get-Date).AddSeconds($remainingSeconds)
                Write-Host ("Elapsed: {0}   Estimated remaining: {1}" -f ([TimeSpan]::FromSeconds($elapsedSeconds).ToString('g')), ([TimeSpan]::FromSeconds($remainingSeconds).ToString('g'))) -ForegroundColor Yellow
                Write-Host ("Estimated finish: {0}" -f $finish.ToString('yyyy-MM-dd HH:mm')) -ForegroundColor Yellow
            } else {
                Write-Host 'Estimated remaining: calibrating from the first 10 optimizer steps.' -ForegroundColor Yellow
            }

            $completedEpochs = if ($null -ne $lastEpoch) {
                [int]$lastEpoch.epoch + 1
            } elseif ($null -ne $resumeEvent) {
                [int][Math]::Floor([int]$resumeEvent.global_step / [double]$stepsPerEpoch)
            } else { 0 }
            Write-Host ("Completed epochs: {0}" -f $completedEpochs)
            if ($null -ne $lastEvaluation) {
                $metrics = $lastEvaluation.metrics
                Write-Host ("Validation mask mAP: {0:N5}   mAP@50: {1:N5}   Recall@100: {2:N5}" -f [double]$metrics.segm_map, [double]$metrics.segm_map_50, [double]$metrics.segm_mar_100) -ForegroundColor Magenta
            } else {
                Write-Host ("Validation: waiting for epoch {0} to complete." -f $activeEpoch) -ForegroundColor DarkGray
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
