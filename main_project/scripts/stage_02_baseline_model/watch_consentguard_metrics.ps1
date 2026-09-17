param(
    [string]$Config = 'main_project/configs/stage_02_baseline_model/train_maskrcnn_verified_class_agnostic_10ep.yaml',
    [int]$RefreshSeconds = 5,
    [switch]$Once
)

$ErrorActionPreference = 'SilentlyContinue'
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

function New-Bar([double]$Fraction, [int]$Width = 32) {
    $bounded = [Math]::Max(0.0, [Math]::Min(1.0, $Fraction))
    $filled = [int][Math]::Floor($bounded * $Width)
    return ('#' * $filled) + ('-' * ($Width - $filled))
}

function Format-Number([object]$Value, [string]$Format = 'N4') {
    if ($null -eq $Value) { return 'n/a' }
    return ([double]$Value).ToString($Format)
}

function Get-JsonProperty([object]$Object, [string]$Name) {
    if ($null -eq $Object) { return $null }
    $property = $Object.psobject.Properties[$Name]
    if ($null -ne $property) { return $property.Value }
    return $null
}

function Read-JsonLines([string]$Path) {
    $items = @()
    if (-not (Test-Path -LiteralPath $Path)) { return $items }
    foreach ($line in (Get-Content -LiteralPath $Path)) {
        if (-not [string]::IsNullOrWhiteSpace($line)) {
            try { $items += ($line | ConvertFrom-Json) } catch {}
        }
    }
    return $items
}

function Show-PerClassEvaluation([object]$Evaluation) {
    if ($null -eq $Evaluation -or $null -eq $Evaluation.metrics.per_class) {
        Write-Host 'Per-class validation metrics: waiting for the first evaluation.' -ForegroundColor DarkGray
        return
    }
    Write-Host 'Per-class validation AP / AR@100' -ForegroundColor Cyan
    Write-Host ('{0,-27} {1,9} {2,9}' -f 'Class', 'Mask AP', 'Recall') -ForegroundColor DarkGray
    foreach ($property in ($Evaluation.metrics.per_class.psobject.Properties | Sort-Object Name)) {
        $item = $property.Value
        Write-Host ('{0,-27} {1,9} {2,9}' -f $property.Name, (Format-Number $item.map 'P1'), (Format-Number $item.mar_100 'P1'))
    }
}

function Show-Diagnostics([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host ("{0}: not available yet ({1})" -f $Label, (Split-Path -Leaf $Path)) -ForegroundColor DarkGray
        return
    }
    try { $diagnostics = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json } catch { return }
    Write-Host ("{0}" -f $Label) -ForegroundColor Cyan
    if ($null -ne $diagnostics.privacy) {
        Write-Host ('  Privacy union recall: {0}   Leakage: {1}   Over-redaction: {2}' -f (Format-Number $diagnostics.privacy.sensitive_pixel_recall 'P1'), (Format-Number $diagnostics.privacy.leakage_rate 'P1'), (Format-Number $diagnostics.privacy.over_redaction_fraction 'P2')) -ForegroundColor Magenta
    }
    if ($null -ne $diagnostics.rpn) {
        Write-Host ('  RPN recall IoU50: {0}   IoU75: {1}   Mean best IoU: {2}' -f (Format-Number (Get-JsonProperty $diagnostics.rpn.overall 'recall_at_iou_0.50') 'P1'), (Format-Number (Get-JsonProperty $diagnostics.rpn.overall 'recall_at_iou_0.75') 'P1'), (Format-Number $diagnostics.rpn.overall.mean_best_iou 'P3'))
        Write-Host ('  RPN small IoU50: {0}   Medium: {1}   Large: {2}' -f (Format-Number (Get-JsonProperty $diagnostics.rpn.by_size.small 'recall_at_iou_0.50') 'P1'), (Format-Number (Get-JsonProperty $diagnostics.rpn.by_size.medium 'recall_at_iou_0.50') 'P1'), (Format-Number (Get-JsonProperty $diagnostics.rpn.by_size.large 'recall_at_iou_0.50') 'P1'))
    }
    if ($null -ne $diagnostics.privacy.per_class) {
        Write-Host '  Privacy recall by class' -ForegroundColor DarkGray
        foreach ($property in ($diagnostics.privacy.per_class.psobject.Properties | Sort-Object Name)) {
            Write-Host ('    {0,-27} {1,9} leakage {2,9}' -f $property.Name, (Format-Number $property.Value.sensitive_pixel_recall 'P1'), (Format-Number $property.Value.leakage_rate 'P1'))
        }
    }
}

$runName = Get-YamlValue 'name' 'maskrcnn_run'
$runDirectory = Resolve-ProjectPath (Get-YamlValue 'output_dir' 'artifacts/checkpoints/unknown')
$metricsPath = Join-Path $runDirectory 'metrics.jsonl'
$resultPath = Join-Path $runDirectory 'training_result.json'
$recordsPath = Resolve-ProjectPath (Get-YamlValue 'train_records' 'data/processed/unknown.jsonl')
$epochCount = [int](Get-YamlValue 'epochs' '10')
$accumulation = [int](Get-YamlValue 'gradient_accumulation_steps' '1')
$trainImages = if (Test-Path -LiteralPath $recordsPath) { (Get-Content -LiteralPath $recordsPath | Measure-Object -Line).Lines } else { 0 }
$stepsPerEpoch = if ($trainImages -gt 0) { [int][Math]::Ceiling($trainImages / [double]$accumulation) } else { 1 }
$totalSteps = $epochCount * $stepsPerEpoch
$runDiagnostics = Join-Path $projectRoot ("reports\{0}_diagnostics.json" -f $runName)
$moderateDiagnostics = Join-Path $projectRoot 'reports\maskrcnn_verified_moderate_balance_10ep_diagnostics.json'
$aggressiveDiagnostics = Join-Path $projectRoot 'reports\maskrcnn_verified_class_balanced_10ep_diagnostics.json'
$configLeaf = Split-Path -Leaf $configPath

do {
    Clear-Host
    Write-Host 'ConsentGuard - Live Metrics Dashboard' -ForegroundColor Cyan
    Write-Host ("Updated: {0}   Config: {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $configLeaf) -ForegroundColor DarkGray
    Write-Host ''

    $trainer = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match 'train_maskrcnn.py' -and $_.CommandLine -like "*$configLeaf*" } |
        Select-Object -First 1
    if ($null -ne $trainer) {
        Write-Host ("TRAINING: RUNNING  PID {0}" -f $trainer.ProcessId) -ForegroundColor Green
    } elseif (Test-Path -LiteralPath $resultPath) {
        Write-Host 'TRAINING: FINISHED' -ForegroundColor Green
    } else {
        Write-Host 'TRAINING: not detected (initialising, stopped, or failed)' -ForegroundColor Yellow
    }

    $entries = Read-JsonLines $metricsPath
    $steps = @($entries | Where-Object { $_.event -eq 'train_step' })
    $lastStep = $steps | Select-Object -Last 1
    $evaluations = @($entries | Where-Object { $_.event -eq 'evaluation' })
    $lastEvaluation = $evaluations | Select-Object -Last 1
    $lastEpoch = $entries | Where-Object { $_.event -eq 'epoch_complete' } | Select-Object -Last 1

    if ($null -ne $lastStep) {
        $step = [int]$lastStep.global_step
        $fraction = $step / [double]$totalSteps
        $activeEpoch = [Math]::Min($epochCount, [int][Math]::Floor(($step - 1) / $stepsPerEpoch) + 1)
        $epochStep = (($step - 1) % $stepsPerEpoch) + 1
        Write-Host ("Progress [{0}] {1,6:P1}   Step {2}/{3}" -f (New-Bar $fraction), $fraction, $step, $totalSteps) -ForegroundColor Cyan
        Write-Host ("Epoch {0}/{1}   Epoch step {2}/{3}" -f $activeEpoch, $epochCount, $epochStep, $stepsPerEpoch)
        Write-Host ("Loss {0}   cls {1}   mask {2}   box {3}   obj {4}   rpn-box {5}   LR {6}" -f (Format-Number $lastStep.loss), (Format-Number $lastStep.loss_classifier), (Format-Number $lastStep.loss_mask), (Format-Number $lastStep.loss_box_reg), (Format-Number $lastStep.loss_objectness), (Format-Number $lastStep.loss_rpn_box_reg), (Format-Number $lastStep.learning_rate 'E2'))

        $firstStep = $steps | Select-Object -First 1
        if ($null -ne $firstStep -and $step -ge 20) {
            $elapsed = [Math]::Max(1.0, [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - [double]$firstStep.time_unix)
            $remaining = [Math]::Max(0.0, ($totalSteps - $step) * ($elapsed / $step))
            Write-Host ("Elapsed {0}   Estimated remaining {1}   Finish {2}" -f ([TimeSpan]::FromSeconds($elapsed).ToString('g')), ([TimeSpan]::FromSeconds($remaining).ToString('g')), (Get-Date).AddSeconds($remaining).ToString('yyyy-MM-dd HH:mm')) -ForegroundColor Yellow
        }
    } else {
        Write-Host 'Waiting for the first training metric.' -ForegroundColor Yellow
    }

    if ($null -ne $lastEvaluation) {
        $metrics = $lastEvaluation.metrics
        $bestEvaluation = $evaluations | Sort-Object { [double]$_.metrics.segm_map } -Descending | Select-Object -First 1
        Write-Host ("Latest validation: mask mAP {0}   AP50 {1}   AP75 {2}   AP-small {3}   AR100 {4}" -f (Format-Number $metrics.segm_map 'P2'), (Format-Number $metrics.segm_map_50 'P2'), (Format-Number $metrics.segm_map_75 'P2'), (Format-Number $metrics.segm_map_small 'P2'), (Format-Number $metrics.segm_mar_100 'P2')) -ForegroundColor Magenta
        if ($null -ne $bestEvaluation) { Write-Host ("Best validation so far: mAP {0} at optimizer step {1}" -f (Format-Number $bestEvaluation.metrics.segm_map 'P2'), $bestEvaluation.global_step) -ForegroundColor Magenta }
        Show-PerClassEvaluation $lastEvaluation
    } else {
        Write-Host 'Validation metrics: waiting for the first completed epoch.' -ForegroundColor DarkGray
    }

    Show-Diagnostics $runDiagnostics 'Current-run privacy/RPN diagnostics'
    if (-not (Test-Path -LiteralPath $runDiagnostics)) {
        Show-Diagnostics $moderateDiagnostics 'Control diagnostics: moderate sampler'
        Show-Diagnostics $aggressiveDiagnostics 'Control diagnostics: aggressive sampler'
    }

    try {
        $gpu = & nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ($gpu) { Write-Host ("GPU memory MiB / total, utilization %, temperature C, power W: {0}" -f $gpu) -ForegroundColor Gray }
    } catch {}
    Write-Host ''
    Write-Host ("Refresh: {0}s   Ctrl+C closes this dashboard only." -f $RefreshSeconds) -ForegroundColor DarkGray
    if (-not $Once) { Start-Sleep -Seconds $RefreshSeconds }
} while (-not $Once)
