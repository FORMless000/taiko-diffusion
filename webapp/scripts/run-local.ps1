param(
    [string]$BackendHost = "127.0.0.1",
    [int]$BackendPort = 8000,
    [string]$FrontendHost = "127.0.0.1",
    [int]$FrontendPort = 3000
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$FrontendRoot = Join-Path $RepoRoot "webapp\frontend"
$ConfigPath = Join-Path $FrontendRoot "public\config.js"
$BackendUrl = "http://$BackendHost`:$BackendPort"
$FrontendUrl = "http://$FrontendHost`:$FrontendPort"

Set-Content -Path $ConfigPath -Encoding UTF8 -Value @"
window.__TAIKO_CONFIG__ = {
  apiBaseUrl: "$BackendUrl"
};
"@

Push-Location $FrontendRoot
npm run build
Pop-Location

$backendCommand = "Set-Location '$RepoRoot'; `$env:TAIKO_WEBAPP_ALLOW_ORIGINS='*'; python -m uvicorn webapp.backend.main:app --host $BackendHost --port $BackendPort"
$frontendCommand = "Set-Location '$RepoRoot'; python -m http.server $FrontendPort --bind $FrontendHost --directory 'webapp/frontend/out'"

$backendProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCommand -PassThru
$frontendProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCommand -PassThru

Write-Host "Backend:  $BackendUrl"
Write-Host "Frontend: $FrontendUrl"
Write-Host "Backend PID:  $($backendProc.Id)"
Write-Host "Frontend PID: $($frontendProc.Id)"
Write-Host "Close the spawned PowerShell windows to stop the local services."
