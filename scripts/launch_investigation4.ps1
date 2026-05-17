#Requires -Version 7
<#
.SYNOPSIS
    Launch 10 parallel Investigation 4 sessions (terminal state mapping).

.DESCRIPTION
    Pre-creates one shared 50,000-game run in the DB, then starts 10 background
    play_tangled.py processes, each logging in with a separate tangled-game.com
    account (tangled1@linknode.com .. tangled10@linknode.com).

    Credentials: password is read from TANGLED_PASSWORD env var (or .env file via
    dotenv). It is NEVER written to this script or any log file.

.PARAMETER Games
    Total planned games for the shared run (default: 50000).

.PARAMETER Sessions
    Number of parallel sessions (default: 10).

.PARAMETER Opponent
    Opponent to play against (default: melissa).

.PARAMETER MctTime
    MCTS time per move in seconds (default: 5).

.EXAMPLE
    $env:TANGLED_PASSWORD = "mypassword"
    .\scripts\launch_investigation4.ps1

.EXAMPLE
    .\scripts\launch_investigation4.ps1 -Games 10000 -Sessions 3
#>

param(
    [int]$Games    = 50000,
    [int]$Sessions = 10,
    [string]$Opponent = "melissa",
    [float]$MctTime   = 5.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$LogDir      = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

# ------------------------------------------------------------------
# 1. Verify TANGLED_PASSWORD is set (never log or echo it)
# ------------------------------------------------------------------
if (-not $env:TANGLED_PASSWORD) {
    # Try loading from .env
    $envFile = Join-Path $ProjectRoot ".env"
    if (Test-Path $envFile) {
        Get-Content $envFile | ForEach-Object {
            if ($_ -match "^\s*TANGLED_PASSWORD\s*=\s*(.+)\s*$") {
                $env:TANGLED_PASSWORD = $matches[1].Trim('"').Trim("'")
            }
        }
    }
}

if (-not $env:TANGLED_PASSWORD) {
    Write-Error "TANGLED_PASSWORD is not set. Set it in your environment or .env file."
    exit 1
}
Write-Host "[launcher] Password loaded from environment (not logged)."

# ------------------------------------------------------------------
# 2. Pre-create the shared run so all sessions join the same one
# ------------------------------------------------------------------
Write-Host "[launcher] Pre-creating shared run ($Games games, opponent=$Opponent, lut=calib)..."

# Write to a temp file — multiline Python via -c loses stdout in PowerShell
$preCreatePy = Join-Path $ProjectRoot "scripts\_precreate_run.py"
@"
from snowdrop_tangled_agents.stats.collector import StatsCollector
c = StatsCollector()
run_id, game_num = c.get_or_create_run(
    planned_games=$Games,
    strategy='terminal_explorer',
    opponent='$Opponent',
    seat=1,
    lut_variant='calib',
)
print('run_id=' + str(run_id), flush=True)
print('start_game=' + str(game_num), flush=True)
"@ | Out-File -FilePath $preCreatePy -Encoding utf8

$preCreateResult = & poetry -C $ProjectRoot run python $preCreatePy
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to pre-create run. Output:`n$preCreateResult"
    exit 1
}
Remove-Item $preCreatePy -ErrorAction SilentlyContinue

$runId = ($preCreateResult | Where-Object { $_ -match "^run_id=" }) -replace "^run_id=", ""
$runId = $runId.Trim()
Write-Host "[launcher] Shared run: $runId"
Write-Host "[launcher] Starting $Sessions parallel sessions..."

# ------------------------------------------------------------------
# 3. Launch N sessions in background
# ------------------------------------------------------------------
$jobs = @()
for ($i = 1; $i -le $Sessions; $i++) {
    $user    = "tangled${i}@linknode.com"
    $logFile = Join-Path $LogDir "inv4_session_${i}.log"

    $cmd = "poetry -C `"$ProjectRoot`" run python `"$ProjectRoot\play_tangled.py`" " +
           "--opponent $Opponent " +
           "--run $Games " +
           "--strategy terminal_explorer " +
           "--lut-variant calib " +
           "--mcts-time $MctTime " +
           "--headless " +
           "--no-dashboard " +
           "--username `"$user`""

    Write-Host "[launcher] Session $i → $user  (log: logs\inv4_session_${i}.log)"

    # WindowStyle Hidden gives each child its own console so SIGINT from the
    # launcher terminal does not propagate and kill the background sessions.
    $proc = Start-Process -FilePath "powershell" `
        -ArgumentList "-NonInteractive", "-Command", $cmd `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError  "$logFile.err" `
        -WindowStyle Hidden `
        -PassThru `
        -WorkingDirectory $ProjectRoot

    $jobs += [PSCustomObject]@{ Session = $i; User = $user; Pid = $proc.Id; Log = $logFile }
}

# ------------------------------------------------------------------
# 4. Summary
# ------------------------------------------------------------------
Write-Host ""
Write-Host "=========================================="
Write-Host "Investigation 4 launched — $Sessions sessions"
Write-Host "Run ID: $runId"
Write-Host "Total planned games: $Games"
Write-Host "=========================================="
$jobs | Format-Table -AutoSize

Write-Host ""
Write-Host "Monitor progress:"
Write-Host "  Get-Content logs\inv4_session_1.log -Tail 20 -Wait"
Write-Host ""
Write-Host "Check DB:"
Write-Host "  poetry run python play_tangled.py --stats"
Write-Host ""
Write-Host "Kill all sessions:"
Write-Host "  poetry run python play_tangled.py --kill-active"
