# WSL 2 Firewall Configuration Script
param(
    [int]$Port = 3400,
    [string]$Protocol = "TCP"
)

# Check admin
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Error: Run as Administrator!" -ForegroundColor Red
    exit 1
}

Write-Host "WSL 2 Firewall Configuration" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan

# Get Windows IP
$windowsIP = (Get-NetIPConfiguration | Where-Object {$_.IPv4DefaultGateway -ne $null}).IPv4Address.IPAddress | Select-Object -First 1

if (-not $windowsIP) {
    Write-Host "Error: Could not find Windows IP" -ForegroundColor Red
    exit 1
}

Write-Host "Windows IP: $windowsIP" -ForegroundColor Green
Write-Host "Port: $Port" -ForegroundColor Green

# Create rule
$ruleName = "WSL 2 Sensor Fusion (Port $Port)"
Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule -Force -ErrorAction SilentlyContinue

New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Action Allow -Protocol $Protocol -LocalPort $Port -ErrorAction SilentlyContinue | Out-Null

Write-Host ""
Write-Host "Done! Connect your phone to:" -ForegroundColor Green
Write-Host "  $windowsIP`:$Port" -ForegroundColor Yellow
