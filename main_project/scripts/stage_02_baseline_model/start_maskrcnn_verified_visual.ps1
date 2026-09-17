param(
    [string]$Config = 'main_project/configs/stage_02_baseline_model/train_maskrcnn_verified_visual.yaml'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..\..')).Path
$configPath = if ([IO.Path]::IsPathRooted($Config)) { $Config } else { Join-Path $projectRoot $Config }
$configPath = [IO.Path]::GetFullPath($configPath)
$configLeaf = Split-Path -Leaf $configPath
$configText = Get-Content -LiteralPath $configPath -Raw
$outputMatch = [regex]::Match($configText, '(?m)^\s*output_dir\s*:\s*([^#\r\n]+)')
if (-not $outputMatch.Success) { throw 'Config is missing experiment.output_dir.' }
$outputRelative = $outputMatch.Groups[1].Value.Trim().Trim("'", '"') -replace '/', '\'
$runDirectory = if ([IO.Path]::IsPathRooted($outputRelative)) {
    $outputRelative
} else {
    Join-Path $projectRoot $outputRelative
}
$metricsPath = Join-Path $runDirectory 'metrics.jsonl'

$existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'train_maskrcnn.py' -and $_.CommandLine -like "*$configLeaf*" } |
    Select-Object -First 1
if ($null -ne $existing) {
    throw "Training is already running for $configLeaf (PID $($existing.ProcessId))."
}
if (Test-Path -LiteralPath $metricsPath) {
    throw "Run directory already contains metrics: $metricsPath. Resume explicitly or select a new output directory."
}

$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
$trainScript = Join-Path $PSScriptRoot 'train_maskrcnn.py'
$monitorScript = Join-Path $PSScriptRoot 'watch_maskrcnn.ps1'

$trainingCommand = @"
Set-Location -LiteralPath '$projectRoot'
& '$pythonPath' -u '$trainScript' --config '$configPath'
`$trainerExitCode = `$LASTEXITCODE
if (`$trainerExitCode -eq 0) {
    Write-Host 'Training finished successfully.' -ForegroundColor Green
} else {
    Write-Host ("Training exited with code {0}. Keep this window open for the traceback." -f `$trainerExitCode) -ForegroundColor Red
}
"@
$trainerWindow = Start-Process powershell.exe -WindowStyle Normal -PassThru -ArgumentList @(
    '-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $trainingCommand
)
Start-Sleep -Seconds 4
$monitorWindow = Start-Process powershell.exe -WindowStyle Normal -PassThru -ArgumentList @(
    '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', $monitorScript, '-Config', $configPath
)

[pscustomobject]@{
    Config = $configPath
    TrainerTerminalPid = $trainerWindow.Id
    MonitorTerminalPid = $monitorWindow.Id
    Metrics = $metricsPath
} | Format-List
