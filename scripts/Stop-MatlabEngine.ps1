# Stop-MatlabEngine.ps1
#
# Kills all running MATLAB processes. Run after stopping game sessions
# to free memory if the shared engine was not stopped cleanly.
#
# Usage:
#   .\scripts\Stop-MatlabEngine.ps1

$procs = Get-Process | Where-Object { $_.Name -like "*MATLAB*" } -ErrorAction SilentlyContinue

if (-not $procs) {
    Write-Host "No MATLAB processes found."
    exit 0
}

$totalMB = [int](($procs | Measure-Object WorkingSet -Sum).Sum / 1MB)
Write-Host "Found $($procs.Count) MATLAB process(es) using ~$totalMB MB — stopping..."
$procs | Stop-Process -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 2

$remaining = Get-Process | Where-Object { $_.Name -like "*MATLAB*" } -ErrorAction SilentlyContinue
if ($remaining) {
    Write-Host "Warning: $($remaining.Count) process(es) still running."
} else {
    Write-Host "Done. All MATLAB processes stopped."
}
