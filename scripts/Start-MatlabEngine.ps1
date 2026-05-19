# Start-MatlabEngine.ps1
#
# Starts a single shared MATLAB engine that all game sessions will reuse.
# Run this BEFORE launching game sessions to avoid one MATLAB instance per session.
#
# Usage:
#   .\scripts\Start-MatlabEngine.ps1
#
# Keep this terminal open while games are running. Close it (or press Enter)
# to stop MATLAB cleanly when all game sessions are done.

$scriptDir = Split-Path -Parent $PSCommandPath
$projectDir = Split-Path -Parent $scriptDir

Set-Location $projectDir

Write-Host "Starting shared MATLAB engine..."
Write-Host "(all game sessions launched after this will connect to this instance)"
Write-Host ""

poetry run python -c @"
import matlab.engine, sys

print('Launching MATLAB...', flush=True)
eng = matlab.engine.start_matlab('-nodesktop -nosplash')
eng.eval('matlab.engine.shareEngine', nargout=0)
print('MATLAB engine ready and shared.', flush=True)
print('Leave this window open while games run.', flush=True)
print('Press Enter to stop MATLAB cleanly...', flush=True)
try:
    input()
except (EOFError, KeyboardInterrupt):
    pass
print('Stopping MATLAB...', flush=True)
eng.quit()
print('Done.', flush=True)
"@
