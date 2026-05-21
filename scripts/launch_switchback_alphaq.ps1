#Requires -Version 7
<#
.SYNOPSIS
    Launch 10 parallel switchback-vs-AlphaQ sessions (Track 2 validation at scale).

.DESCRIPTION
    Pre-creates one shared run in the DB, then starts 10 background
    play_tangled.py processes (tangled1@linknode.com .. tangled10@linknode.com),
    each playing 50 games with the pure switchback geometric solver.

    Strategy: hybrid_solver with --solver-adversary switchback
    Opening: forced E7G (--oracle-override 15 7 G), matching MATLAB run 147 baseline.
    All subsequent moves: pure structural switchback scoring. No MCTS, no minimax.

    Credentials: password is read from TANGLED_PASSWORD env var (or .env file).
    It is NEVER written to this script or any log file.

.PARAMETER Games
    Total planned games for the shared run (default: 500 = 10 sessions x 50).

.PARAMETER Sessions
    Number of parallel sessions (default: 10).

.EXAMPLE
    $env:TANGLED_PASSWORD = "mypassword"
    .\scripts\launch_switchback_alphaq.ps1

.EXAMPLE
    .\scripts\launch_switchback_alphaq.ps1 -Games 200 -Sessions 4
#>

param(
    [int]$Games    = 500,
    [int]$Sessions = 10
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
Write-Host "[launcher] Pre-creating shared run ($Games games, opponent=alphaq, strategy=hybrid_solver/switchback)..."

$preCreatePy = Join-Path $ProjectRoot "scripts\_precreate_switchback_run.py"
@"
from snowdrop_tangled_agents.stats.collector import StatsCollector
c = StatsCollector()
run_id, game_num = c.get_or_create_run(
    planned_games=$Games,
    strategy='hybrid_solver/switchback',
    opponent='alphaq',
    seat=1,
)
print('run_id=' + str(run_id), flush=True)
print('start_game=' + str(game_num), flush=True)
"@ | Out-File -FilePath $preCreatePy -Encoding utf8

$preCreateResult = & poetry -C $ProjectRoot run python $preCreatePy
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to pre-create run. Output:`n$preCreateResult"
    Remove-Item $preCreatePy -ErrorAction SilentlyContinue
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
    $logFile = Join-Path $LogDir "switchback_alphaq_${i}.log"

    $cmd = "poetry -C `"$ProjectRoot`" run python `"$ProjectRoot\play_tangled.py`" " +
           "--opponent alphaq " +
           "--run $Games " +
           "--strategy hybrid_solver " +
           "--solver-adversary switchback " +
           "--oracle-override 15 7 G " +
           "--headless " +
           "--no-dashboard " +
           "--username `"$user`""

    Write-Host "[launcher] Session $i -> $user  (log: logs\switchback_alphaq_${i}.log)"

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
Write-Host "Switchback vs AlphaQ launched -- $Sessions sessions"
Write-Host "Run ID: $runId"
Write-Host "Total planned games: $Games"
Write-Host "Strategy: hybrid_solver/switchback (pure geometric, no oracle)"
Write-Host "=========================================="
$jobs | Format-Table -AutoSize

Write-Host ""
Write-Host "Monitor progress:"
Write-Host "  Get-Content logs\switchback_alphaq_1.log -Tail 20 -Wait"
Write-Host ""
Write-Host "Check results:"
Write-Host "  poetry run python play_tangled.py --stats"
Write-Host ""
Write-Host "Kill all sessions:"
Write-Host "  poetry run python play_tangled.py --kill-active"
