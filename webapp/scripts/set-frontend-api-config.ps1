[CmdletBinding()]
param(
    [string]$ApiBaseUrl = "https://ec2-18-117-249-161.us-east-2.compute.amazonaws.com"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$configPath = Join-Path $repoRoot "webapp\frontend\public\config.js"

Set-Content -Path $configPath -Encoding UTF8 -Value @"
window.__TAIKO_CONFIG__ = {
  apiBaseUrl: "$ApiBaseUrl"
};
"@

Write-Host "Updated frontend runtime API URL:"
Write-Host "  $ApiBaseUrl"
Write-Host "Config file: $configPath"
