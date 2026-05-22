#Requires -Version 7
<#
.SYNOPSIS
    Exhaustive sweep of all valid P2 responses to AlphaQ's opening.

.DESCRIPTION
    Phase 1: Plays one observation game as Player 2 (no override) to
    determine what AlphaQ plays as P1 — AlphaQ is fully deterministic
    so this is always the same opening.

    Phase 2: For each of the 28 valid P2 first responses (14 remaining
    edges × {G, P}, excluding AlphaQ's already-colored edge), plays one
    game with switchback handling all subsequent moves.

    Our first move as P2 happens when grey_count=14 (AlphaQ has played
    one edge), so we use --oracle-override 14 {edge} {color}.

    Total: ~29 games, ~45 minutes.
    Logs: logs/sweep_p2_obs.log.err and logs/sweep_p2_E{N}{C}.log.err

    Credentials: password read from TANGLED_PASSWORD env var or .env.
    Never written to a script, log, or console.

.PARAMETER Account
    Tangled account to use (default: tangled1@linknode.com).
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
# 1. Verify TANGLED_PASSWORD
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
Write-Host "[p2-sweep] Password loaded from environment (not logged)."

# ------------------------------------------------------------------
# Helper: run one headless game, wait for exit
# ------------------------------------------------------------------
function Run-Game {
    param(
        [string]$LogLabel,
        [string]$ExtraArgs
    )
    $logFile = Join-Path $LogDir "${LogLabel}.log"
    $errFile = Join-Path $LogDir "${LogLabel}.log.err"

    $cmd = "poetry -C `"$ProjectRoot`" run python `"$ProjectRoot\play_tangled.py`" " +
           "--opponent alphaq " +
           "--games 1 " +
           "--strategy hybrid_solver " +
           "--solver-adversary switchback " +
           "--seat 2 " +
           "--headless " +
           "--no-dashboard " +
           "--username `"$Account`" " +
           $ExtraArgs

    Start-Process -FilePath "powershell" `
        -ArgumentList "-NonInteractive", "-Command", $cmd `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError  $errFile `
        -WindowStyle Hidden `
        -WorkingDirectory $ProjectRoot `
        -Wait

    return $errFile
}

# ------------------------------------------------------------------
# 2. Phase 1: observation game (no override) — learn AlphaQ's P1 opening
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[p2-sweep] Phase 1: observation game as P2 (no override)..."

$obsLog = Run-Game -LogLabel "sweep_p2_obs" -ExtraArgs ""

# Parse AlphaQ's opening from the move history: "1. [OPP] E{N} Green/Purple"
$ansiStrip = { param($s) $s -replace '\x1b\[[0-9;]*m', '' }

$alphaQEdge  = $null
$alphaQColor = $null

if (Test-Path $obsLog) {
    $content = Get-Content $obsLog -Raw
    $content = & $ansiStrip $content
    if ($content -match '(?m)\s1\.\s+\[OPP\]\s+E(\d+)\s+(Green|Purple)') {
        $alphaQEdge  = [int]$Matches[1]
        $alphaQColor = if ($Matches[2] -eq 'Green') { 'G' } else { 'P' }
        Write-Host "[p2-sweep] AlphaQ (P1) opened: E${alphaQEdge}${alphaQColor}"
    } else {
        Write-Warning "[p2-sweep] Could not parse AlphaQ's opening from $obsLog — check log manually."
        # Try to show last GAME OVER line
        $gameOver = Select-String -Path $obsLog -Pattern "GAME OVER:" -SimpleMatch |
                    Select-Object -Last 1
        if ($gameOver) {
            Write-Host "  Obs result: $((& $ansiStrip $gameOver.Line).Trim())"
        }
        exit 1
    }
    # Show observation result
    $gameOver = Select-String -Path $obsLog -Pattern "GAME OVER:" -SimpleMatch | Select-Object -Last 1
    if ($gameOver) {
        Write-Host "  Obs result: $((& $ansiStrip $gameOver.Line).Trim())"
    }
} else {
    Write-Error "Observation log not found: $obsLog"
    exit 1
}

# ------------------------------------------------------------------
# 3. Phase 2: sweep all 28 valid P2 responses
# ------------------------------------------------------------------
$openings = @()
for ($edge = 0; $edge -le 14; $edge++) {
    if ($edge -eq $alphaQEdge) { continue }   # that edge is already colored
    foreach ($color in @('G', 'P')) {
        $openings += [PSCustomObject]@{ Edge = $edge; Color = $color }
    }
}

$total = $openings.Count   # should be 28
Write-Host ""
Write-Host "[p2-sweep] Phase 2: sweeping $total P2 responses (AlphaQ opened E${alphaQEdge}${alphaQColor})"
Write-Host "[p2-sweep] Our first P2 move override: --oracle-override 14 {edge} {color}"
Write-Host ""

$startTime = Get-Date
$done = 0

foreach ($o in $openings) {
    $done++
    $label = "sweep_p2_E$($o.Edge)$($o.Color)"

    $elapsed = (Get-Date) - $startTime
    $eta = if ($done -gt 1) {
        $secPer = $elapsed.TotalSeconds / ($done - 1)
        $remain = ($total - $done + 1) * $secPer
        [TimeSpan]::FromSeconds($remain).ToString("hh\:mm\:ss")
    } else { "?" }

    Write-Host ("[p2-sweep] {0,2}/{1}  P2 response E{2}{3}  ETA {4}" -f $done, $total, $o.Edge, $o.Color, $eta)

    $errFile = Run-Game `
        -LogLabel $label `
        -ExtraArgs "--oracle-override 14 $($o.Edge) $($o.Color)"

    if (-not (Test-Path $errFile) -or (Get-Item $errFile).Length -eq 0) {
        Write-Warning "  E$($o.Edge)$($o.Color): empty log — session may have failed"
    } else {
        $gameOver = Select-String -Path $errFile -Pattern "GAME OVER:" -SimpleMatch |
                    Select-Object -Last 1
        if ($gameOver) {
            $line = & $ansiStrip $gameOver.Line
            Write-Host "  => $($line.Trim())"
        } else {
            Write-Host "  => (no GAME OVER line found)"
        }
    }
}

# ------------------------------------------------------------------
# 4. Run analysis
# ------------------------------------------------------------------
Write-Host ""
Write-Host "[p2-sweep] All $total responses complete. AlphaQ opened E${alphaQEdge}${alphaQColor}."
Write-Host "[p2-sweep] Running analysis..."
Write-Host ""

& poetry -C $ProjectRoot run python "$ProjectRoot\scripts\_analyze_p2_sweep.py" `
    --alphaq-opening "E${alphaQEdge}${alphaQColor}"
