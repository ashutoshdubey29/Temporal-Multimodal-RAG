# PowerShell Launcher for Temporal-Multimodal-RAG
$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $PSScriptRoot

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "====================================================" -ForegroundColor Red
    Write-Host "[ERROR] Python virtual environment not found!" -ForegroundColor Red
    Write-Host "  Target: $VenvPython" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Please run '.\setup.bat' first to configure the environment."
    Write-Host "====================================================" -ForegroundColor Red
    exit 1
}

$env:PYTHONUNBUFFERED = "1"
& $VenvPython (Join-Path $PSScriptRoot "run.py") @args
