<#
.SYNOPSIS
    Put a "ConsentGuard" shortcut on the desktop.

.DESCRIPTION
    Creates (or refreshes) a shortcut that runs start_consentguard.ps1, so the
    tool opens with one double-click. Run it again after moving the repository.
    Remove the shortcut by deleting it from the desktop.
#>
[CmdletBinding()]
param(
    [string]$Name = 'ConsentGuard',
    [int]$Port = 7860
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$launcher = Join-Path $repo 'main_project\scripts\stage_05_review_export\start_consentguard.ps1'
if (-not (Test-Path $launcher)) { throw "Launcher not found at $launcher" }

$desktop = [Environment]::GetFolderPath('Desktop')
$shortcutPath = Join-Path $desktop "$Name.lnk"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = (Get-Command powershell.exe).Source
$shortcut.Arguments = "-NoLogo -ExecutionPolicy Bypass -File `"$launcher`" -Port $Port"
$shortcut.WorkingDirectory = $repo
$shortcut.Description = 'Erase private content from a photo before sharing it (local only)'
$shortcut.IconLocation = "$env:SystemRoot\System32\imageres.dll,68"
$shortcut.Save()

Write-Host "Shortcut created: $shortcutPath" -ForegroundColor Green
Write-Host 'Double-click it to start ConsentGuard.'
