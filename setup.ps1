# PowerShell Setup Script for Temporal-Multimodal-RAG
$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location $PSScriptRoot

& (Join-Path $PSScriptRoot "setup.bat") @args
