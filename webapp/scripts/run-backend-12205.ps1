[CmdletBinding()]
param(
    [string]$HostAddress = "0.0.0.0",
    [int]$Port = 12205
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $repoRoot

Write-Host "Starting taiko-diffusion backend on http://$HostAddress`:$Port"
python -m uvicorn webapp.backend.main:app --host $HostAddress --port $Port
