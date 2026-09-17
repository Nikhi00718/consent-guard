param(
    [string]$OutputPath = "C:\consentGuard\output\pdf\ConsentGuard_End_to_End_Project_Handbook.pdf"
)

$ErrorActionPreference = "Stop"

$handbookRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = Resolve-Path (Join-Path $handbookRoot "..\..\..")
$tectonicPath = Join-Path $repositoryRoot ".tools\tectonic\tectonic.exe"
$temporaryOutput = Join-Path $repositoryRoot "tmp\pdfs\consentguard_handbook_build"

if (-not (Test-Path -LiteralPath $tectonicPath)) {
    throw "Portable Tectonic compiler not found at $tectonicPath"
}

if (-not (Test-Path -LiteralPath $temporaryOutput)) {
    New-Item -ItemType Directory -Path $temporaryOutput | Out-Null
}

Push-Location $handbookRoot
try {
    & $tectonicPath -X compile main.tex `
        --outdir $temporaryOutput `
        --keep-logs `
        --keep-intermediates `
        --print
    if ($LASTEXITCODE -ne 0) {
        throw "Tectonic compilation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$compiledPdf = Join-Path $temporaryOutput "main.pdf"
if (-not (Test-Path -LiteralPath $compiledPdf)) {
    throw "Tectonic did not produce $compiledPdf"
}

$outputDirectory = Split-Path -Parent $OutputPath
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory | Out-Null
}

Copy-Item -LiteralPath $compiledPdf -Destination $OutputPath -Force
Get-Item -LiteralPath $OutputPath
