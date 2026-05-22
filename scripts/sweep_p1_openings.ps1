#Requires -Version 7
<#
.SYNOPSIS
    Exhaustive sweep of all 30 P1 opening moves vs AlphaQ in one browser session.

.DESCRIPTION
    Plays all 30 openings (edges 0-14 x {Green, Purple}) in a single
    play_tangled.py session — one login, Play Again between games.
    Uses --oracle-sequence-file to apply a different first-move override
    per game. Total time ~35 min (no browser restart overhead).

    Results are written to logs/sweep_p1_combined.log.err.
    Run _analyze_p1_sweep.py --combined-log afterwards for the table.

    Credentials: password read from TANGLED_PASSWORD env var or .env.
    Never written to a script, log, or console.

.PARAMETER Account
    Tangled account to use (default: tangled1@linknode.com).

.EXAMPLE
    .\scripts\sweep_p1_openings.ps1 -Account "murray@murraykopit.com"
#>

param(
    [string]$Account = "tangled1@linknode.com"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$LogDir      = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force $LogDir | Out-Null

# ------------------------------------------------------------------
# 1. Verify TANGLED_PASSWORD is available (never echo it)
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
Write-Host "[sweep] Password loaded from environment (not logged)."

# ------------------------------------------------------------------
# 2. Build the sequence file — one "GREY EDGE COLOR" line per game
# ------------------------------------------------------------------
$seqFile     = Join-Path $LogDir "sweep_p1_sequence.txt"
$combinedLog = Join-Path $LogDir "sweep_p1_combined.log.err"
$combinedOut = Join-Path $LogDir "sweep_p1_combined.log"

$lines = @()
for ($edge = 0; $edge -le 14; $edge++) {
    foreach ($color in @('G', 'P')) {
        $lines += "15 $edge $color"
    }
}
$total = $lines.Count   # 30
$lines | Set-Content $seqFile -Encoding UTF8

Write-Host "[sweep] Account  : $Account"
Write-Host "[sweep] Games    : $total (one browser session, Play Again between games)"
Write-Host "[sweep] Seq file : $seqFile"
Write-Host "[sweep] Log      : $combinedLog"
Write-Host ""

# ------------------------------------------------------------------
# 3. Run one play_tangled.py session for all 30 games
# ------------------------------------------------------------------
$cmd = "poetry -C `"$ProjectRoot`" run python `"$ProjectRoot\play_tangled.py`" " +
       "--opponent alphaq " +
       "--games $total " +
       "--strategy hybrid_solver " +
       "--solver-adversary switchback " +
       "--oracle-sequence-file `"$seqFile`" " +
       "--no-dashboard " +
       "--username `"$Account`""

Write-Host "[sweep] Launching single-session sweep..."

$proc = Start-Process -FilePath "pwsh" `
    -ArgumentList "-NonInteractive", "-Command", $cmd `
    -RedirectStandardOutput $combinedOut `
    -RedirectStandardError  $combinedLog `
    -WindowStyle Normal `
    -WorkingDirectory $ProjectRoot `
    -PassThru

# ------------------------------------------------------------------
# 4. Monitor progress: tail combined log for GAME OVER lines
# ------------------------------------------------------------------
$gamesReported = 0
$startTime     = Get-Date
$ansiStrip     = { param($s) $s -replace '\x1b\[[0-9;]*m', '' }

Write-Host "[sweep] Monitoring progress (Ctrl+C to abort)..."
Write-Host ""

while ($true) {
    Start-Sleep -Milliseconds 800

    if (Test-Path $combinedLog) {
        $content    = Get-Content $combinedLog -Raw -ErrorAction SilentlyContinue
        $stripped   = & $ansiStrip $content
        $gameOvers  = [regex]::Matches($stripped, 'GAME OVER: (WIN|DRAW|LOSS)')

        while ($gamesReported -lt $gameOvers.Count) {
            $idx     = $gamesReported
            $result  = $gameOvers[$idx].Groups[1].Value
            $opening = $lines[$idx]   # "15 EDGE COLOR"
            $parts   = $opening -split ' '
            $label   = "E$($parts[1])$($parts[2])"

            $elapsed = (Get-Date) - $startTime
            $eta = if ($gamesReported -gt 0) {
                $secPer = $elapsed.TotalSeconds / ($gamesReported + 1)
                $remain = ($total - $gamesReported - 1) * $secPer
                [TimeSpan]::FromSeconds($remain).ToString("hh\:mm\:ss")
            } else { '?' }

            Write-Host ("[sweep] {0,2}/{1}  {2}  => {3}  ETA {4}" -f `
                ($idx + 1), $total, $label, $result, $eta)
            $gamesReported++
        }
    }

    if ($proc.HasExited -and $gamesReported -ge $total) { break }
    if ($proc.HasExited -and $gamesReported -lt $total) {
        Write-Warning "[sweep] Process exited early after $gamesReported/$total games."
        break
    }
}

$proc.WaitForExit()

# ------------------------------------------------------------------
# 5. Run analysis on the combined log
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[sweep] All $gamesReported games complete. Running analysis..."
Write-Host ""

& poetry -C $ProjectRoot run python "$ProjectRoot\scripts\_analyze_p1_sweep.py" `
    --combined-log $combinedLog --sequence-file $seqFile
