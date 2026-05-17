# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-05-17

### Added

- **Investigation 2 implementation** (`scripts/analyse_alphaq_policy.py`)
  - Extracts (state_before, AlphaQ_move) decision pairs from the 1,574-game AlphaQ corpus
  - Computes per-state Shannon response entropy, with Miller-Madow bias correction for global MI
  - Stratifies MI by grey count (game phase)
  - Identifies exploit candidates (n >= 10 obs, H >= 0.5 bits) and decision-boundary candidates (n <= 6, varying moves)
  - Generates three plots: entropy histogram, entropy-vs-grey-count, per-edge color preference
  - Auto-writes `docs/INVESTIGATION_2_RESULTS.md` with full findings table and decision-gate verdict

- **Investigation 2 results doc** (`docs/INVESTIGATION_2_RESULTS.md`)
  - **OPTIMISTIC verdict.** 97.5% of well-observed states are deterministic, but 6 exploit candidates exist with 10+ observations and >= 0.5 bits of response entropy
  - 5 of 6 candidates are at grey=8 — suggests AlphaQ has a mid-game decision-boundary phase
  - All candidates are binary choice points (2 distinct responses, dominant chosen 60-79%)
  - These are concrete search targets for Phases 2-4 of the AlphaQ-targeted plan

### Changed

- Investigation 4 hard-stopped (40 processes terminated) to free compute for AlphaQ-targeted work

## [0.3.0] - 2026-05-17

### Added

- **Methodological review** (`docs/METHODOLOGICAL_REVIEW_AND_REDIRECTION_TOWARD_ALPHAQ.md`)
  - Critical review of the current investigation program at the Investigation 4 inflection point
  - Identifies the goal-misalignment problem (Investigation 4 collects Melissa data; goal is AlphaQ defeat)
  - Documents structural concerns with the calibrated oracle (R²=0.60, anneal_time as possible model-class artefact, opponent-conditional calibration not done with AlphaQ data)
  - Surfaces the minimax-vs-actual-policy mismatch in the hybrid solver
  - Recommends prioritising Investigation 2 (MI/entropy analysis of AlphaQ policy) before further data collection

- **Two-week project plan** (`docs/PROJECT_PLAN_ALPHAQ_TARGETED_INVESTIGATION.md`)
  - Five-phase plan implementing the review's recommendations: MI analysis, predictive opponent model, AlphaQ-conditional calibration, expected-value solver, decision gate
  - Timeline: 10–13 days from kick-off to decision gate (vs 9–17 days for Investigation 4 alone)
  - Includes Investigation 4 disposition recommendation (redirect to AlphaQ or pause)
  - Risk register and per-phase decision gates

## [0.2.2] - 2026-05-17

### Added

- **Leaderboard report** (`scripts/leaderboard_report.py`)
  - Playwright-based script that logs in, scrapes the tangled-game.com leaderboard, and reports stats for Investigation 4 accounts (murr1–murr10)
  - Columns: Rank, Player, ELO, W, L, D, Win%, Games
  - Aggregate view: total games, wins, losses, draws, win %, ELO range, rank range
  - Exact-name matching excludes unrelated accounts (murr2k, murr2k@gmail.com)
  - Usage: `poetry run python scripts/leaderboard_report.py [--no-headless] [--dump-html]`

- **Username logged on login** (`play_tangled.py`)
  - "Logged in as {username}" line confirms per-session account identity in logs

## [0.2.1] - 2026-05-17

### Added

- **`--username` CLI flag** (`play_tangled.py`)
  - Overrides `TANGLED_USERNAME` env var so multiple parallel sessions can each log in with a different account
  - Password always sourced from `TANGLED_PASSWORD` env var only — never a CLI arg

- **Investigation 4 parallel launcher** (`scripts/launch_investigation4.ps1`)
  - PowerShell script that pre-creates one shared 50,000-game run in the DB, then starts 10 background sessions
  - Each session logs in with `tangled{N}@linknode.com`; password read exclusively from `TANGLED_PASSWORD` env var
  - Each session writes to `logs/inv4_session_{N}.log`

## [0.2.0] - 2026-05-17

### Added

- **Investigation 4 Resume Mechanism** (`play_tangled.py`, `collector.py`, `migrations.py`, `terminal_explorer_strategy.py`, `scripts/test_resume.py`)
  - DB migration v12: `lut_variant` column added to `runs` table so oracle variants never accidentally share runs
  - `start_run` / `get_or_create_run` accept `lut_variant`; resume matches on all config fields including oracle variant
  - `update_run_completed` now excludes `abandoned` results — NULL-result games (connection drops) don't inflate completed count
  - `TerminalExplorerStrategy`: `opening_index_start` parameter restores round-robin position from `completed_games` on restart
  - `play_tangled.py`: outer retry loop survives network drops and browser crashes; `explorer_opening_index` recomputed from DB before each session
  - 17-assertion unit test suite (`scripts/test_resume.py`) covering all resume scenarios
  - Live-tested: kill after 2 games, resume joins correct run at game 5/6 with Explorer opening 3/30: E1G

- **`--lut-variant calib` support** (`play_tangled.py`)
  - Selects the website-calibrated oracle (`expanded_lut_calib.mat`, anneal_time=1.85 ns)
  - Intended default for Investigation 4

### Fixed

- **Game loop hang — background publish** (`play_tangled.py`)
  - `_get_dashboard_stats()` + `publisher.publish_state()` moved to a daemon thread so SQLite busy-wait or WebSocket send-buffer stall can never block the game loop

- **Game loop hang — Phase 2 board-state detection** (`play_tangled.py`)
  - `wait_for_opponent_to_play()` previously relied solely on `'your turn'` page text, which can lag or be absent when the opponent plays quickly
  - Now polls both text signal and board-state (grey edge count drop) in parallel; whichever fires first unblocks the loop
  - `our_grey_count` passed as `initial_grey_count` reference point

- **`terminal_explorer` fallback ignored `--mcts-time`** (`play_tangled.py`)
  - Fallback `MCTSStrategy` was hardcoded with `time_limit=float('inf')`; changed to respect `mcts_time` parameter

## [0.1.0] - 2026-05-16

### Added

- **SA Oracle — Full Game Coverage (Priorities 6–10)** (`scripts/generate_sa_oracle.py`, `ExpandedLUT.m`, `HybridTangledSolver.m`)
  - Retrograde minimax DP over all 32,768 terminal boards and grey levels 0–15
  - `expanded_lut_sa.mat` (371 MB, untracked): complete oracle lookup table
  - Game-start oracle value (all grey, optimal play): **+0.0075** — P1 has a forced SA-optimal first move
  - Oracle-optimal openings: E14P, E10G, E11G (near-identical; game is balanced near draw)
  - Oracle fires at every P1 turn via `for e = greyEdges` loop in `HybridTangledSolver.m:solveOracle()`
  - `ExpandedLUT.m`: `readMatOrH5()` helper supports both MATLAB v7 and h5py HDF5 formats
  - Regression tests for all oracle grey levels (7, 9, 11, 13, 15)

- **Win Investigation Closure (2026-05-16)**
  - 1,563+ games against AlphaQ Up across all 30 openings and all strategies: **0 wins**
  - SchrLUT calibration disproved: local Schrödinger adjudicator (epsilon=0.0005, anneal_time=40ns) predicts `winner=red` on ALL boards including confirmed losses; website uses different quantum parameters
  - AlphaQ adaptively steers every game to near-zero Schrödinger energy boards (SchrLUT ≈ 0.006–0.012)
  - SchrLUT outcome distribution: ∈ [7, 10) → 51 games, 0 wins, 42 draws; ∈ [3, 5) → 136 games, 0 wins, 77 draws
  - Petersen graph vertex-transitivity + color-flip symmetry implies V(∅) = 0 under optimal play
  - Best stable strategies: E7G MCTS (98% draw, SA 0.75–0.83) and SA oracle with `--oracle-override 11 11 G` (SA 0.67–0.81)

- **Research Report** (`quantum_tangled_proof.html`)
  - Standalone HTML report with KaTeX formulae and SVG diagrams documenting the full campaign
  - Covers: Petersen graph setup, 10-priority optimization campaign, SA oracle construction, SchrLUT calibration disproof, AlphaQ game tree analysis, Thompson sampling, symmetry proof, and conclusions

- **Opponent Model Rebuild from Oracle-Era Games (Priority 7)** (`scripts/rebuild_opponent_model.py`)
  - Fixed `games.id` TEXT comparison bug: `id >= 1438` (TEXT > INTEGER returns all rows in SQLite); corrected to `rowid >= 3199`
  - Oracle-era corpus: 32 games / 180 moves (19/30 response contexts populated)
  - Dirichlet smoothing (alpha=0.1) fills unseen contexts with uniform prior

- **Thompson Posterior Rebuild (Priority 7c)** (`scripts/rebuild_thompson_state.py`)
  - Fixed MCTS convergence to suboptimal E3G (value −0.0031) at grey=11; oracle-optimal is E11G (+0.1844)
  - Thompson posteriors corrected: E13P → mean=0.200, E5P → mean=0.143, E3P → mean=0.200

### Fixed

- **Cadence and Timeout** (`play_tangled.py`, `matlab_strategy.py`)
  - `--mcts-time` default changed from `inf` → `30s`
  - All strategy `time_limit` defaults raised `10s` → `30s` (Python + MATLAB)
  - Python `timeout_seconds` formula corrected: was `max(3600, iters//2)` = 250,000 s; now `2×time_limit + 120` (cap 7200 for inf)
  - Phase 1 move-acceptance timeout reduced 30s → 3s

- **`solveOracle` greyEdges transposition** (`HybridTangledSolver.m`)
  - `for e = greyEdges'` transposes row→column, iterating once with the entire column vector; fixed to `for e = greyEdges`

- **`ExpandedLUT.loadExpandedLUT` HDF5 compatibility** (`ExpandedLUT.m`)
  - MATLAB's `load()` cannot read h5py-written HDF5 files; `readMatOrH5()` helper tries `load()` then falls back to `h5read()`

- **AlphaQExplorer LUT and thresholds** (Priorities 5, 6b)
  - `AlphaBetaSearch.m` LUT corrected from Schrödinger to SA
  - `EarlyGameThreshold` reset to 0 (oracle supersedes greedy prior at grey=13,11)
  - `LateGameBoostThreshold` reset to 0 (oracle handles all P1 decisions)

### Added

- **Concurrent Game Sessions** (`play_tangled.py`, `snowdrop_tangled_agents/stats/`)
  - Multiple independent `play_tangled.py` processes can now run simultaneously
  - SQLite WAL mode + 30s busy timeout via `connect_db()` helper eliminates "database is locked" errors
  - Per-PID process tracking in `~/.tangled/active_processes/` replaces single-process guard
  - PID-based run isolation: each process gets its own `run_id`, with resume support after reboot
  - Migration v10 adds `pid` column to `runs` table
  - Exclusive transaction guard on schema migrations prevents race conditions
  - PID-suffixed `opponent_model.mat` output avoids concurrent write conflicts
  - `--process-status` shows all active sessions; `--kill-active` stops all

- **Headless Mode** (`play_tangled.py`)
  - `--headless` flag runs browser without visible window, useful for background campaigns

- **AlphaQ Adjudicator Exploration Features** (`terminal_explorer_strategy.py`, `play_tangled.py`)
  - Multi-point random move injection: `--random-turns 0,2,4` injects random moves at configurable turn indices to break AlphaQ's deterministic funneling into ~120 terminal states
  - Seat swap: `--seat 2` plays as Player 2 (Blue) to test adjudicator seat-dependency hypothesis
  - Novel branch forcing: `--novel-branch` queries historical moves from DB and deliberately plays untried (edge, color) pairs at each board state to explore new game tree branches

- **Event-Driven Browser Waits** (`play_tangled.py`)
  - Replaced all Python polling loops with Playwright `wait_for_function()` — JS conditions evaluate inside the browser and only notify Python when triggered
  - Five polling loops converted: main loop idle wait, initial turn wait, move verification, opponent wait, and `wait_for_turn`
  - CPU usage during waits drops from ~12.5% (1 full core) to near zero per process
  - Each wait has a logged timeout: main loop 5min, initial turn 60s, move acceptance 60s, opponent 240s, `wait_for_turn` 180s
  - Game loop wrapped in try/except with full traceback logging to catch silent crashes

- **Move Verification Overhaul** (`play_tangled.py`)
  - Turn-based move acceptance: verify moves by polling the turn indicator (switches to opponent's turn) instead of the slow-to-update board SVG
  - Three acceptance signals in priority order: turn indicator switch, game over popup, board SVG update
  - Same-edge retry: failed moves retry the same edge/color via `pending_move` instead of re-asking the strategy for a different edge, preserving intended terminal states
  - Per-move retry cap (3 cycles) prevents infinite loops on genuinely unplayable edges
  - Full game summary: move history now includes both our and opponent moves with `[US]`/`[OPP]` tags

- **Empirical Closure of the Petersen Graph Under Quantum Adjudication** (`docs/EMPIRICAL_CLOSURE_OF_THE_PETERSEN_GRAPH_UNDER_QUANTUM_ADJUDICATION.md`)
  - Formal writeup of the empirical negative result: 0 wins across 120 diverse terminal states against AlphaQ
  - Documents polarity inversion between SA and quantum adjudication (r = -0.396, p < 0.0001)
  - Statistical confidence analysis: 95% confidence winning terminals < 2.5% of reachable space
  - Sampling bias caveat: classical strategies may be unable to reach non-classical winning states
  - Falsifiability framework and formal proof requirements (complete enumeration, game tree analysis, symmetry reduction)
  - 35 references grounding the work in quantum annealing, order-by-disorder, graph theory, and quantum game theory
  - References Geordie Rose's Tangled blog, $10,000 AlphaQ Up Challenge, and D-Wave adjudication methodology

- **Empirical LUT Construction Campaign** (`docs/EMPIRICAL_LUT_CONSTRUCTION_RESULTS.md`)
  - Results from 65 games (48 oracle routes + 17 terminal explorer) testing website-calibrated LUT
  - Confirms AlphaQ zero-loss equilibrium: max website score +0.861, far below +2 threshold
  - SA polarity inversion quantified: SA [+2,+10) maps to website avg -1.3 to -2.5
  - Tools created: `build_website_lut.py`, `terminal_explorer_strategy.py`, route cycling mode

- **Strategy Analysis Documents** (`docs/POST_SA_LUT_STRATEGY_OPTIONS.md`, `docs/HIGH_COMPUTE_STRATEGY_OPTIONS.md`)
  - 8 strategy alternatives after SA LUT proven anti-correlated with quantum evaluation
  - High-compute options for GPU + MATLAB + multi-core environments
  - Phased approach: exhaustive mapping (9-17 days) then conditional AlphaZero (42 days)
  - Resource allocation for parallel execution of multiple approaches

- **Website-Calibrated LUT Builder** (`snowdrop_tangled_agents/tools/build_website_lut.py`)
  - Mines calibration table for empirical terminal state scores from game database
  - Outputs website_scores.bin (32,768 x f32 LE) with NaN for unobserved states
  - Correlation and compression analysis, score distribution reporting

- **Terminal Explorer Strategy** (`snowdrop_tangled_agents/strategy/terminal_explorer_strategy.py`)
  - Systematic round-robin through all 30 possible openings (15 edges x 2 colors)
  - MCTS fallback for mid-game, designed to maximize terminal state diversity

- **Oracle Route Cycling** (`snowdrop_tangled_agents/strategy/oracle_route_strategy.py`)
  - Added `route_mode` parameter: `fixed` (default) or `cycle` (round-robin through all routes)
  - `--route-mode` and `--routes-file` CLI arguments in `play_tangled.py`

- **Parallel MCTS Rollouts via `parfor`** (`TangledMCTS.m`)
  - The parallel pool of 6 workers was allocated (~900MB each, 5.4GB total) but never used — no
    `parfor` existed in the file and all rollouts ran serially on the main thread
  - Each MCTS iteration now runs `NumWorkers` rollouts in parallel using `parfor`, then averages
    the results.  With 6 workers, each leaf visit produces 6 independent rollouts instead of 1,
    yielding higher-quality value estimates and 6x more samples per iteration
  - **Static method architecture:** MATLAB `parfor` cannot call methods on handle objects, so all
    rollout logic was duplicated as static methods (`parallelSimulate`, `evaluateTerminalStatic`,
    `heuristicActionStatic`, `computeRolloutPriorStatic`) that take a pre-packed context struct
    instead of accessing `this.*` properties
  - **Serial fallback preserved:** When `UseParallel = false` or pool fails to initialize, the
    original `this.simulate()` path is used unchanged
  - **Progress reporting updated:** `sims` counter now reflects actual rollout count
    (e.g. 20K iterations × 6 workers = 120K simulations)
  - Verified: 50 iterations with 4 workers → 200 simulations (exact 4x), all workers active

### Added

- **ANALYSIS_OF_GEORDIE_ROSE_FEEDBACK.md** (New Document)
  - Comprehensive 11,500-word technical analysis of Geordie Rose's feedback on Schrödinger adjudicator optimization work
  - Validates Rose's core claims: 20-vertex limit for exact methods, tensor networks as state-of-the-art, exponential scaling barriers
  - Identifies undervalued contributions: algorithm documentation, SA bias discovery, RL implications
  - Provides strategic roadmap: hybrid approach using exact methods for validation (≤12 vertices), tensor networks for scaling (15-20+ vertices)
  - 18 citations from quantum annealing and tensor network literature

- **MCTS Parameter Documentation** (`play_tangled.py`, `README.md`)
  - Updated module docstring to include usage examples for `--mcts-iterations` and `--mcts-time` parameters
  - Updated README with comprehensive examples showing new defaults (500K iterations, unlimited time)
  - Added fast testing mode examples (100K iterations, 30s time limit) and maximum strength mode (1M iterations)
  - Documented performance characteristics: 500K = ~14 min per move (elite quality), 100K = ~3 min per move (fast testing)
  - Updated all strategy examples throughout Development Progress section with parameter guidance
  - Documented all current strategies with hybrid_solver as default

### Changed

- **Markdown Style Compliance** (`docs/THE_MATHEMATICS_OF_TANGLED_GAME.md`, `docs/ANALYSIS_OF_GEORDIE_ROSE_FEEDBACK.md`)
  - Fixed all ordered lists to use `1.` for every item per MARKDOWN_STYLE.md ruleset (80+ items updated)
  - Removed unintentional indentation from document title
  - Ensures consistent rendering across Markdown Monster and GitHub

### Added

- **MATLAB Readiness Check Before Game Start** (`play_tangled.py`)
  - Added `verify_matlab_readiness()` function that runs before browser opens
  - Checks MATLAB connection, cleans up stale parallel pools, verifies responsiveness
  - Fails fast if MATLAB isn't ready, preventing browser from opening unnecessarily
  - Provides clear diagnostic output showing connection status, pool cleanup, and responsiveness test
  - Prevents the issue where browser opens and game starts but then fails due to MATLAB issues

### Fixed

- **MATLAB MCTS Out-of-Memory Error Handling** (`TangledMCTS.m`, `matlab_strategy.py`)
  - Added try-catch block around main MCTS search loop to catch out-of-memory and other errors
  - Fixed critical bug in error handler where `memInfo` was referenced before being defined
  - MATLAB now reports memory errors with diagnostic message instead of hanging silently
  - Python side now detects and reports MATLAB out-of-memory errors with actionable guidance
  - Error messages include iteration count and memory usage at failure point
  - Suggests reducing --mcts-iterations if memory issues occur
  - Prevents silent hangs when MATLAB runs out of memory during tree expansion or simulation

- **MATLAB Parallel Pool Management** (`TangledMCTS.m`, `matlab_strategy.py`)
  - Fixed parallel pool not being cleaned up between games, causing worker exhaustion
  - Now always deletes and recreates parallel pool to ensure clean worker state
  - Dynamically queries MATLAB for available worker count instead of assuming fixed number
  - Added `cleanupPool()` method to explicitly release workers after each game
  - Python side now calls cleanup in `end_game()` to free resources for next game
  - Added diagnostic logging to show worker allocation and pool lifecycle events
  - Prevents "pool already exists" errors and stale worker states across games

- **MATLAB Execution Timeout** (`matlab_strategy.py`)
  - Added 5-minute timeout for MATLAB solver calls using async execution mode
  - Prevents indefinite hangs when MATLAB enters infinite loop or deadlock
  - Timeout errors are caught and reported with diagnostic information
  - Suggests checking MATLAB console for diagnostic output when timeout occurs

- **MATLAB Diagnostic Logging** (`TangledMCTS.m`)
  - Added progress reporting every 10 seconds during MCTS search
  - Shows iteration count, nodes expanded, simulations run, and tree depth
  - Added logging for root node creation and MCTS loop initialization
  - Helps diagnose where MATLAB hangs when execution stalls without throwing exceptions

- **setup_env.py Installation Robustness**
  - Added missing `websocket-client>=1.6.0` package (was in pyproject.toml but missing from pip install list)
  - Now upgrades pip, setuptools, and wheel before installing packages to ensure proper pyproject.toml handling
  - Enhanced verification to explicitly test all 7 core packages with checkmarks
  - Added helpful error message with install command if imports fail
  - Fixes installation failures in fresh environments where coloredlogs, python-dotenv, playwright, and websocket-client were not being installed properly

### Added

- **Thompson Sampling Opening Selection for AlphaQExplorerStrategy** (`matlab_strategy.py`, `test_matlab_integration.py`)
  - Replaced broken two-phase explore/exploit strategy with principled Bayesian opening selection
  - **Root cause fix:** Previous greedy ranking by `(wins DESC, avg_score DESC)` degenerated to `avg_score DESC` when all openings had 0 wins, selecting E9G (89% loss rate) and E11G (100% loss rate) for exploitation
  - **Thompson Sampling approach:** All 30 openings tracked with win/draw/loss counts, converted to Beta distribution posteriors. Each game samples from `Beta(α, β)` where `α = 1 + wins + 0.5×draws`, `β = 1 + losses + 0.5×draws`. Opening with highest sample is played. Draws count as half-wins to avoid double-penalizing safe openings in 0-win scenarios
  - **Learning-rate gating:** REINFORCE disabled for first 10 games (MIN_GAMES_BEFORE_LEARNING) to gather clean opening data uncontaminated by solver adaptation
  - **State versioning:** v1→v2 migration path for backward compatibility with legacy `alphaq_explorer_state.json` files
  - **Comprehensive test suite:** 8 unit tests covering Thompson Sampling correctness, exploration dynamics, state migration, and learning gating
  - **Full documentation:** `docs/ALPHAQ_STRATEGY.md` provides mathematical foundation, error analysis, reproduction instructions, and expected convergence metrics
  - **Mid-run analysis (Run 55, Game 310/500):** See `docs/ALPHAQ_THOMPSON_SAMPLING_MID_RUN_ANALYSIS.md` for critical findings on algorithm correctness and solver limitations

### Analysis

- **ALPHAQ_THOMPSON_SAMPLING_MID_RUN_ANALYSIS.md** (New Document)
  - Mid-run analysis of Thompson Sampling implementation at game 310/500
  - **Critical Finding:** Thompson Sampling is working perfectly (safe openings 6-9% frequency, risky openings 2-3% frequency, correct posterior distributions), but solver achieves 0% win rate (0 wins in 386 games) against AlphaQ Up
  - **Root Cause Identified:** The original plan's assumption that "safe openings lead to wins" is false. Preventing bad openings prevents catastrophic failure but does not enable wins. AlphaQ Up appears to prevent all wins regardless of opening selection
  - **Weaknesses Documented:**
    1. Zero win rate (0/386 games) - identical to broken greedy strategy
    2. Safe openings degrading (E0G went from 10D/0L to 10D/5L)
    3. REINFORCE learning rate too aggressive (0.03) without validation mechanism
    4. Original plan assumption invalidated by evidence
  - **Recommendations:** Reduce learning rate to 0.001-0.005, add REINFORCE validation metrics, investigate E0G degradation, establish ground truth comparison (AlphaQ's win rate vs other solvers), consider architectural solver changes
  - **Conclusion:** Thompson Sampling successfully fixed the ranking problem but could not overcome the fundamental solver weakness vs AlphaQ Up

- **P(win) Calibration for MCTS Terminal Evaluation** (`TangledMCTS.m`, `generate_calibration.py`)
  - Analysis of 1 511 calibrated games revealed that the displayed score on tangled-game.com does
    not reliably determine the winner: 59 % of losses show positive final scores. P(win) is a step
    function of score — below +2 the adjudicator outcome is dominated by noise (P(win) = 6 % at
    score +0.5, 43 % at +0.8). SA predicted scores are noisier still: SA confidence in [+2, +5]
    only wins 71 % of the time vs 98.5 % for the website score in that range.
  - `generate_calibration.py` fits a monotonic P(win) curve via quantile-binned isotonic regression
    on game history and saves it as `calibration_pwin.mat` (32 knots). The curve maps SA predicted
    scores to empirical win probabilities so the MCTS search optimises for states that reliably win
    rather than states with high expected score.
  - `TangledMCTS.m`: `loadCalibration()` loads the curve at construction; `calibrateScore()` applies
    `interp1` interpolation and returns `2·P(win) − 1 ∈ [−1, +1]`. `evaluateTerminal` now applies
    calibration after the LUT lookup (or heuristic fallback). A `tanh` sigmoid fallback is used when
    the .mat file is unavailable.
  - `docs/SCORE_OUTCOME_DISCREPANCY.md`: full analysis of the score–outcome mismatch, two
    hypotheses (display bug vs intentional quantum measurement stochasticity), a distinguishing
    test for the game backend, and the impact on AI agents.

- **AlphaQ Explorer Strategy with Closed Learning Loop** (`matlab_strategy.py`, `TangledMCTS.m`, `HybridTangledSolver.m`)
  - Two-phase explore/exploit strategy designed for the new "AlphaQ Up" opponent
  - **Exploration phase** (games 0–29): Cycles all 30 possible first moves with learning disabled,
    recording per-opening win/score data to identify weaknesses in AlphaQ's play
  - **Exploitation phase** (games 30+): Re-enables REINFORCE learning (rate 0.03) and rotates
    only the top-N openings ranked by (wins, avg_score). After each game the updated
    `edge_adjustments` are forwarded into MATLAB via `hybridSolver.setEdgeBias()`, closing the
    learning loop so the next game's MCTS rollout priors reflect what was learned
  - **Closed learning loop** — the critical new capability. Previously `HybridSolverStrategy`
    ran REINFORCE and accumulated `edge_adjustments` in Python but never applied them to the
    MATLAB solver. Now `TangledMCTS` carries an `EdgeBias` vector (1×15) that is added to the
    heuristic prior inside `computeRolloutPrior` and clamped to [0.001, 0.999]. `setEdgeBias()`
    on both `TangledMCTS` and `HybridTangledSolver` propagates updates, and `setPlayer()`
    re-applies any existing bias to the freshly constructed MCTS instance
  - Persistent state at `~/.tangled/alphaq_explorer_state.json` (phase, exploration results,
    exploitation openings, index). Safe to resume mid-run: exploitation phase re-enables learning
    and re-pushes accumulated edge bias at `initialize()` time
  - New opponent `"alphaq"` ("AlphaQ Up") added to `play_tangled.py`
  - CLI: `python play_tangled.py --strategy alphaq_explorer --opponent alphaq --run 30`

- **Run Tracking & Resume System** (Migration v8)
  - New `runs` table tracks planned game batches with progress
  - Games store `run_id` and `game_number` for batch tracking
  - `--run N` flag to start/resume a run of N planned games
  - Automatic resume: interrupted runs continue from where they left off
  - Session stats shows run progress: `run = 2`, `games = 5/750`
  - `get_or_create_run()`, `get_active_run()`, `abandon_game()` collector methods

- **Stuck Opponent Detection & Recovery**
  - 3-minute timeout when waiting for opponent to play
  - `StuckOpponentError` exception for timeout handling
  - Automatic browser restart on stuck opponent
  - Games marked as "abandoned" with reason tracking
  - If final game of run is stuck, automatically starts new run

- **Process Tracking for Safe Management**
  - Active process info stored in `~/.tangled/active_process.json`
  - Prevents accidentally starting multiple game sessions
  - `--process-status` flag shows active game process info
  - `--kill-active` flag safely terminates running game process
  - Automatic cleanup on process exit via atexit handler

- **Policy Versioning & Model Metrics Tracking** (Migration v7)
  - `policy_id` field tracks code version via git tag or commit hash
  - Opponent model metrics: `model_entropy`, `model_games_learned`, `model_moves_learned`
  - Prediction accuracy metrics: `model_top3_hit`, `prediction_accuracy`
  - `GameMetricsTracker` class for per-game prediction tracking
  - `get_policy_id()` auto-detects version from git tag/commit
  - `backfill_policy_id.py` tool for historical game backfill
  - Integrated into `play_tangled.py` game loop for automatic metrics capture

- **Live Session Statistics** (`snowdrop_tangled_agents/stats/session_stats.py`)
  - Real-time session statistics with automatic session boundary detection
  - Watch mode: `--watch` refreshes stats periodically (press `q` to exit)
  - Configurable refresh interval: `--interval` (default 60s)
  - Configurable session gap: `--gap` (default 30 minutes)
  - Trend analysis: score trend, win rate trend, recent 5 results
  - Estimated end time based on play rate
  - UTC storage with local timezone display
  - Cleanup tool for stale in-progress games: `--cleanup --force`
  - JSON output: `--json`

- **D-Wave Inspired Hybrid Solver** (`snowdrop_tangled_agents/matlab/rl/HybridTangledSolver.m`)
  - Integrated solver combining Alpha-Beta Minimax, MCTS, and Tabu Search
  - Automatic strategy selection based on game phase:
    - Late game (≤3 grey edges): Pure minimax with guaranteed optimal play
    - Mid game (4-8 grey edges): Hybrid minimax + MCTS
    - Early game (>8 grey edges): MCTS + Tabu refinement
  - D-Wave MST2-inspired Tabu Search (`TabuSearch.m`) with multistart optimization
  - Alpha-Beta Pruning (`AlphaBetaSearch.m`) with transposition tables
  - Time-budgeted search with configurable allocation (35% minimax, 55% MCTS, 10% tabu)
  - Python integration via `HybridSolverStrategy` in `matlab_strategy.py`
  - **REINFORCE-style learning** from game outcomes (added to Python strategy):
    - Temporal credit assignment with discount factor (γ=0.9)
    - Edge value adjustments learned from wins/losses/draws
    - Persistent storage of learned adjustments (`~/.tangled/hybrid_solver_adjustments.json`)
    - Configurable learning rate (default 0.03)
  - CLI: `python play_tangled.py --strategy hybrid_solver`

- **Expanded Lookup Table (19M Entries)** (`snowdrop_tangled_agents/matlab/rl/ExpandedLUT.m`)
  - Pre-computed exact minimax values for 0-3 grey edge states
  - Coverage breakdown:
    - 0 grey (terminal): 32,768 states
    - 1 grey: 491,520 states
    - 2 grey: 3,440,640 states
    - 3 grey: 14,909,440 states
    - **Total: 18,874,368 exact values**
  - Guarantees optimal play for final 4 moves of every game
  - Generation scripts: `generate_expanded_lut.m`, `generate_expanded_lut_parallel.m`, `extend_lut_three_grey_parallel.m`
  - 3-grey extension generated in ~12.6 minutes using 6 parallel workers

- **Hybrid Solver Test Suite** (`snowdrop_tangled_agents/matlab/rl/test_hybrid_solver.m`)
  - 27 unit tests covering ExpandedLUT, TabuSearch, AlphaBetaSearch, HybridTangledSolver
  - Integration tests for full game simulation and solver consistency
  - LUT generation file verification tests

- **MATLAB MCTS Strategy** (`snowdrop_tangled_agents/strategy/matlab_mcts_strategy.py`)
  - High-compute MCTS using MATLAB TangledMCTS engine (5000 iterations, 20s time limit)
  - `MCTSParams` dataclass for tunable parameters with JSON persistence
  - Opening book for first 3 moves (E9→E11→E10 Green sequence)
  - Adaptive exploration based on score momentum (boost when losing, reduce when winning)
  - Fallback chain: MATLAB MCTS → Python MCTS → heuristic
  - Integration with stats collection for edge bonus learning
  - CLI: `python play_tangled.py --strategy matlab_mcts`

- **TangledMCTS MATLAB Engine** (`snowdrop_tangled_agents/matlab/rl/TangledMCTS.m`)
  - Full MCTS with UCB1 selection and progressive bias
  - Domain-specific heuristic rollout policy
  - Calibrated terminal evaluation with priority system (MY_EDGES → OPP_EDGES → HUB_EDGES)
  - Compute effort diagnostics using MATLAB's `cputime` and `memory` functions
  - Session-level statistics tracking (total CPU time, total iterations)
  - `getComputeEffort()` method for detailed performance metrics

- **MCTSNode Class** (`snowdrop_tangled_agents/matlab/rl/MCTSNode.m`)
  - Tree node implementation with containers.Map for children
  - UCB1 selection with configurable exploration and prior weight
  - Action masking for valid moves only
  - Visit counting and value backpropagation

- **MATLAB MCTS Documentation** (`docs/MATLAB_MCTS_STRATEGY.md`)
  - Theory of operation for the MATLAB MCTS strategy
  - Petersen graph edge classification (MY_EDGES, OPP_EDGES, HUB_EDGES)
  - Terminal evaluation scoring values and priority system
  - Critical discoveries: E12 G and E2 G cause score collapse
  - Compute diagnostics and performance profiling guide
  - Turn-based state reading analysis and fix
  - Roadmap for future improvements

- **MATLAB RL System** (`snowdrop_tangled_agents/matlab/rl/`)
  - `TangledEnvironment.m`: RL environment with 50-element observation space, 30 discrete actions
  - `createPPOAgent.m`: PPO agent with actor-critic networks (16K+ parameters each)
  - `trainParallel.m`: Parallel self-play training with worker pools
  - `collectEpisode.m`: Experience collection with action masking
  - `SQLiteExperienceBuffer.m`: Persistent replay buffer with base64 serialization
  - `SimulatedOpponent.m`: Configurable opponent for self-play (random, heuristic, copy)
  - `getActionMask.m`: Valid action filtering (15 edges × 2 colors)

- **Phase 5: Continuous Deployment Pipeline** (`snowdrop_tangled_agents/matlab/rl/`)
  - `ModelRegistry.m`: SQLite-backed model version management
  - `tangled_agent_inference.m`: Compiled inference with hot-reload (60s refresh)
  - `autoDeploy.m`: Automatic deployment when win rate improves
  - `build_rl_package.m`: MATLAB Compiler SDK build script
  - `rl_bridge.py`: Python bridge with fallback chain (compiled → engine → heuristic)

- **CI/CD Integration** (`.github/workflows/test.yml`)
  - GitHub Actions workflow for Python tests (3.11, 3.12, 3.13)
  - Self-hosted runner support for MATLAB tests
  - Manual trigger option for MATLAB test suite
  - pytest markers: `matlab` (requires MATLAB), `slow` (training tests)

- **MATLAB Test Suite** (`snowdrop_tangled_agents/matlab/rl/run_all_tests.m`)
  - 23 regression tests covering Phases 0-4
  - Isolated test directories to protect production database
  - GPU detection with graceful CPU fallback
  - pytest wrapper: `snowdrop_tangled_agents/tests/test_matlab_rl.py`

- **MATLAB Toolbox Integration** (`snowdrop_tangled_agents/matlab/`)
  - Deep Learning Toolbox integration for neural network position evaluation
  - Statistics and ML Toolbox for opponent clustering and style classification
  - Database Toolbox for direct MATLAB-SQLite access in training pipelines
  - MATLAB Compiler SDK support for Python-callable compiled packages
  - `unified_bridge.py`: Automatic backend selection with fallback chain
    - Compiled packages (fastest, MATLAB Runtime only)
    - MATLAB Engine API (full functionality, requires license)
    - Pure Python heuristics (always available)
  - `compiled_bridge.py`: Bridge to pre-compiled MATLAB packages
  - `training.py`: Training orchestration from Python
  - `matlab_strategy.py`: MCTS enhanced with neural network priors
  - Value network architecture: 50-input features → FC(128-64-32) → tanh output
  - Opponent modeling with 20-element feature vectors and K-means clustering
  - CLI: `--strategy matlab --use-nn --adapt-opponent`

- **Database Schema Migrations** (`snowdrop_tangled_agents/stats/migrations.py`)
  - Automatic schema versioning and migration runner
  - v2: `models` table for trained network metadata
  - v3: `opponents` table for opponent profiles and clustering
  - v4: `training_data` table for ML training samples
  - v5: `opponent_history` table for move-level opponent tracking

- **Compiled MATLAB Packages** (for deployment without MATLAB license)
  - `tangled_value_network`: Neural network inference
  - `tangled_opponent_model`: Opponent classification and prior adaptation
  - `tangled_training`: Model training and opponent clustering
  - Requires MATLAB Runtime R2026a (free download)

- **Adjudicator Calibration System** (`snowdrop_tangled_agents/stats/`)
  - Compares our terminal state evaluation to actual tangled-game.com scores
  - New `calibration` table in SQLite database
  - Automatic data collection at game end when all edges are colored
  - `record_calibration()` method in StatsCollector
  - Analysis queries: `get_calibration_summary()`, `get_calibration_details()`
  - `print_calibration_report()` for detailed analysis output
  - CLI access: `python play_tangled.py --calibration`
  - Error distribution tracking (exact, close, moderate, large)
  - Systematic bias detection and interpretation

- **MCTS Strategy Engine** (`snowdrop_tangled_agents/strategy/mcts_strategy.py`)
  - Monte Carlo Tree Search with UCB1 selection for deep lookahead
  - Progressive Bias: heuristic priors guide early exploration, decay with visits
  - Action prioritization: good moves expanded first based on domain knowledge
  - Domain-specific rollout policy using Tangled heuristics
  - Terminal state evaluation using official `SimulatedAnnealingAdjudicator`
  - LRU cache for efficient repeated state evaluations
  - Edge classifications derived from 50+ game empirical analysis:
    - GOOD_PURPLE_EDGES: E0, E1, E3, E5, E12, E13
    - BAD_PURPLE_EDGES: E2, E4, E6, E7, E8, E14

- **Hybrid Strategy** (`snowdrop_tangled_agents/strategy/mcts_strategy.py`)
  - Combines heuristic opening, MCTS midgame, and exhaustive endgame
  - Opening sequence: E9→E10→E11 Green, E5→E12→E13 Purple
  - Adaptive time allocation: 3x more time for critical late-game moves
  - Exhaustive minimax for positions with ≤2 edges remaining
  - REINFORCE-style learning from game outcomes
  - Edge adjustment tracking across games

- **SQLite Statistics Collection** (`snowdrop_tangled_agents/stats/`)
  - `collector.py`: StatsCollector class for game/move recording
  - `queries.py`: Analysis functions for pattern discovery
  - Database schema: games table + moves table with full indexing
  - Automatic integration with play_tangled.py
  - CLI access: `python play_tangled.py --stats`
  - Analysis queries:
    - `get_edge_effectiveness()`: Edge/color performance ranking
    - `get_winning_patterns()`: Move patterns leading to wins
    - `get_score_progression()`: Score trajectory by game result
    - `get_opening_sequences()`: Common openings and outcomes
    - `get_critical_positions()`: Large score swing analysis
    - `get_opponent_patterns()`: Opponent behavior analysis

- **Strategy CLI Options** (`play_tangled.py`)
  - `--strategy {heuristic,mcts,hybrid}`: Select strategy type
  - `--mcts-time SECONDS`: MCTS time limit per move
  - `--mcts-iterations N`: Maximum MCTS iterations
  - `--stats`: Show statistics summary and exit

- **Petersen Strategy Engine** (`snowdrop_tangled_agents/strategy/petersen_strategy.py`)
  - Parameterized strategy calculator for Petersen graph games
  - Edge priority scoring based on vertex ownership (MY_VERTEX=5, OPP_VERTEX=7, HUB_VERTEX=6)
  - Configurable opening sequence override for first N moves
  - Adaptive color selection based on score thresholds and strategy mode
  - Momentum tracking from recent score history
  - Opponent pattern analysis to detect valued edges
  - REINFORCE-style learning from game outcomes with discounted returns
  - Parameter persistence to JSON for learning across sessions
  - Game statistics tracking (wins/losses/draws)

- **Petersen Agent** (`snowdrop_tangled_agents/agents/petersen_agent.py`)
  - SDK-compatible wrapper implementing `GameAgentBase`
  - Translates SDK game state to strategy state string format
  - Supports external score injection for web play
  - Move history tracking for learning updates

- **Web Player** (`play_tangled.py`)
  - Playwright-based automation for tangled-game.com
  - Dynamic vertex discovery from SVG line endpoints
  - Angle-based vertex alignment (outer pentagon, inner pentagram)
  - Robust edge-to-line mapping using nearest-vertex matching
  - Color button detection with multiple text pattern matching
  - Turn detection with explicit state checking
  - Automatic browser cleanup on exit/signal/exception
  - Game outcome recording with full score history
  - Command-line interface with configurable opponent and game count

- **Strategy Module** (`snowdrop_tangled_agents/strategy/__init__.py`)
  - Package exports for PetersenStrategy class

- **Documentation**
  - `CLAUDE.md` - Project guidance for Claude Code
  - `docs/THEORY_OF_OPERATION.md` - Comprehensive system documentation (moved from root)
    - Added Adjudicator Calibration section
    - Added Mermaid diagrams for data flow and gameplay transaction flow
  - `docs/MATLAB_INTEGRATION.md` - Complete MATLAB integration guide
  - `docs/tangled-bot-v28.txt` - Reference JavaScript bot implementation
  - `docs/THE_MATHEMATICS_OF_TANGLED_GAME.md` - Added Section 9: HybridSolverStrategy
    - D-Wave inspired hybrid search architecture
    - Alpha-Beta minimax with transposition tables
    - MCTS with Tabu Search refinement
    - 19M-entry expanded lookup table theory
    - REINFORCE-style adaptive learning with temporal credit assignment
    - Database integration and persistence mechanisms
    - Performance analysis and research extensions for graduate students

### Added

- **Game Analytics & Visualization** (`snowdrop_tangled_agents/tools/plot_progress.py`)
  - Progress tracking: win rate and score trends over time with rolling averages
  - Edge effectiveness analysis: score delta and win rate by edge/color
  - Opening sequence analysis: identify winning opening patterns
  - Timestamped output: `plots/{type}_{YYYYMMDD}_{HHMMSS}.png`
  - CLI: `python -m snowdrop_tangled_agents.tools.plot_progress --all`
  - See `docs/GAME_ANALYTICS.md` for full documentation

### Changed

- **Unified Dashboard Messaging** (`snowdrop_tangled_agents/stats/websocket_publisher.py`)
  - Consolidated `publish_move()`, `publish_stats()`, and `publish_session_stats()` into single `publish_state()` function
  - Dashboard now receives exactly ONE message type (`full_state`) containing complete state
  - Every message includes: move info, board state, vertex colors, session stats, score stats, trends, model metrics, ETA
  - Removes partial message handling complexity from both publisher and dashboard
  - Dashboard always renders full UI regardless of message content

- **Run-Specific Dashboard Stats** (`play_tangled.py`, `snowdrop_tangled_agents/stats/session_stats.py`)
  - Added `get_run_stats(run_id)` function for run-specific statistics (not time-based session)
  - Dashboard now publishes immediately when game ends (after edge 15), not waiting for next game loop
  - All dashboard messages now use run-specific stats when a run_id is active
  - Removed redundant post-game publish block in main loop (single publisher at game end)
  - Fixes issue where dashboard showed stats from previous runs mixed with current run

- **Complete Dashboard State Every Publish** (`play_tangled.py`)
  - Added `_get_dashboard_stats()` helper that computes all dashboard fields
  - Every publish now sends complete state: session, results, scores, trends, and model metrics
  - Dashboard displays retain values during gameplay (no more blank stats mid-game)
  - Simplifies dashboard code: just display what's received, no caching needed

- **Default Strategy Changed** (`play_tangled.py`)
  - Changed default strategy from `heuristic` to `hybrid_solver`
  - HybridSolverStrategy with REINFORCE learning is now the recommended approach
  - Combines MATLAB's powerful search (minimax + MCTS + Tabu) with adaptive learning
  - Requires MATLAB Engine connection; falls back to Python MCTS if unavailable

- **SimulatedOpponent Real MCTS Mode** (`snowdrop_tangled_agents/matlab/rl/SimulatedOpponent.m`)
  - Added real MCTS opponent using `TangledMCTS` engine for realistic training
  - Lazy-initialized MCTS engine to avoid overhead when not needed
  - Renamed previous MCTS approximation to `fast_mcts` style
  - Increased default iterations from 100 to 500 for stronger play
  - Added `MCTSTimeLimit` parameter (default 1.0s)
  - Available styles: `random`, `heuristic`, `mcts` (real), `fast_mcts`, `petersen`, `defensive`, `aggressive`

- **TangledEnvironment Auto-Correction** (`snowdrop_tangled_agents/matlab/rl/TangledEnvironment.m`)
  - Enabled `AutoCorrectInvalidActions` by default to improve RL learning
  - Reduced invalid action penalty from -0.5 to -0.1 when action is remapped
  - Kept larger -0.5 penalty when auto-correct is disabled
  - Added `info.OriginalAction` tracking for debugging remapped moves

- **Curriculum Training Worker Detection** (`snowdrop_tangled_agents/matlab/rl/train_curriculum_ensemble.m`)
  - Added automatic detection of maximum available parallel workers
  - Graceful degradation when requested workers exceed cluster capacity
  - Changed `AutoCorrect` from false to true in `trainLevel()`, `trainSelfPlayEnsemble()`, and `evaluateAgent()`
  - Improved logging to show when worker count is reduced

- **Opponent Model Data** (`snowdrop_tangled_agents/matlab/rl/data/opponent_model.mat`)
  - Updated with latest learned patterns from online gameplay
  - Contains response-conditional and phase-conditional probability matrices

- Moved `THEORY_OF_OPERATION.md` from project root to `docs/` directory
- Updated `README.md` with Development Progress section documenting all implementation steps
- Extended `StatsCollector` with model and opponent management methods
- Added migration support to stats collector initialization

- Updated `snowdrop_tangled_agents/__init__.py` to export PetersenAgent
- Updated `snowdrop_tangled_agents/agents/__init__.py` to include PetersenAgent
- Updated `pyproject.toml` with new dependencies (playwright, python-dotenv, coloredlogs)
- Updated `pyproject.toml` with pytest markers (`matlab`, `slow`)
- Updated `docs/TEST_SUITE.md` with MATLAB RL test documentation and CI/CD workflow

### Fixed

- Terminal state evaluation accuracy (`mcts_strategy.py:evaluate_terminal_state`)
  - Replaced incorrect brute-force spin enumeration with official `SimulatedAnnealingAdjudicator`
  - Calibration improved from ~3-4 point errors to <0.02 point errors
  - Now matches tangled-game.com scores exactly (within stochastic variance)
  - Added LRU caching for efficient repeated evaluations

- Edge mapping between strategy edge indices and website SVG lines
  - Implemented consistent dynamic vertex discovery algorithm
  - Fixed angle wrap-around handling for vertex rotation
  - Aligned inner pentagram and outer pentagon vertex numbering

- Color button detection reliability
  - Increased dialog appearance wait time
  - Added multiple button text patterns (Green/FM/Ferromagnetic)
  - Extended retry logic with longer delays

- Turn detection accuracy
  - Made detection more conservative with explicit checks only
  - Added negative indicators for opponent's turn
  - Removed aggressive fallback assumptions

- Browser session cleanup
  - Added signal handlers for SIGTERM/SIGINT
  - Implemented atexit cleanup handler
  - Added context manager support for automatic cleanup

- MATLAB RL training bugs (`trainParallel.m`)
  - Fixed brace indexing error: SQLiteExperienceBuffer returns matrices, not cell arrays
  - Fixed integer/double type mismatch with explicit `double()` conversions
  - Fixed System ID Toolbox conflict: use `forward(getModel(getCritic(agent)), ...)` instead of `predict()`

- MATLAB test reliability
  - Tests check for "PASSED" message in stdout instead of return code
  - Handles MATLAB prerelease shutdown crashes gracefully

- **MATLAB MCTS Terminal Evaluation** (`TangledMCTS.m:evaluateTerminal`)
  - Fixed E12 G scoring: Was +0.2 (OPP + HUB double-count), now -0.8 penalty
    - E12 connects hub (V6) to opponent vertex (V7)
    - Green/ferromagnetic coupling helps opponent in quantum adjudication
  - Fixed E2 G scoring: Was +0.3 (hub control), now -0.5 penalty
    - Game data showed E2 G at move 4 caused -2.6 point collapses
  - Added priority system to avoid double-counting overlapping edge categories
  - Updated rollout policy: Hub edges 25% green (was 70%)

- **Turn-Based State Reading** (`play_tangled.py`)
  - Fixed stale DOM state issue causing invalid move attempts
  - Root cause: `is_our_turn()` returned True before DOM fully updated with opponent's move
  - Added 0.5s delay after turn change detection before reading board state
  - Clarified that game is strictly turn-based (no true race conditions)

- **Score-Weighted Draw Reward + Opponent-Conditional Calibration** (`matlab_strategy.py`, `TangledMCTS.m`, `HybridTangledSolver.m`)
  - Motivated by Run 47 (0W/27L/33D vs AlphaQ Up): wins are rare enough that the flat ±0.1 draw
    reward starves the REINFORCE loop of gradient.  The draw branch now returns `score × 0.65`,
    giving near-miss draws (e.g. +0.78 → reward +0.507) meaningful positive signal while staying
    below the minimum win reward (1.0).  Win and loss branches are structurally unchanged.
  - `loadCalibration()` is now opponent-conditional.  When an opponent name is provided, the solver
    looks for `calibration_<sanitized_name>.mat` first.  If no per-opponent file exists it falls back
    to the `tanh` sigmoid rather than loading the generic (Melissa-fitted) curve — this prevents
    Melissa's noise profile from biasing the solver against other opponents.  The generic
    `calibration_pwin.mat` path is only taken when no opponent name is supplied (legacy behaviour).
  - `calibration_melissa.mat` added (copy of `calibration_pwin.mat`) so Melissa as a named opponent
    resolves to the existing fitted curve.
  - `OpponentName` threaded from Python through `HybridTangledSolver` into `TangledMCTS` via a new
    `'Opponent'` constructor argument on both classes.
  - Verified: Run 50 vs AlphaQ Up produced non-uniform edge adjustments (+0.035 on good-draw edges,
    −0.081 on loss-associated edges) and draw rewards of 0.34–0.50 (vs flat 0.1 previously).

### Fixed

- **MATLAB string-literal concatenation in `loadCalibration`** (`TangledMCTS.m`)
  - `fprintf` format string was split across two lines with `...` continuation. MATLAB does not
    auto-concatenate adjacent single-quoted strings (unlike C); the second literal became a
    separate argument, causing a parse error on every solver construction. Consolidated onto one
    line so the calibration curve loads correctly.

## [0.0.5] - 2026-01-20

### Changed

- Preparation for version 0.0.5 release

## [0.0.4] - 2026-01-20

### Changed

- Preparation for version 0.0.4 release

## [0.0.3] - 2026-01-20

### Changed

- Updated dependencies
- Preparation for version 0.0.3 release

## [0.0.2] - 2026-01-20

### Changed

- Preparation for version 0.0.2 release

## [0.0.1] - 2026-01-20

### Added

- Initial commit with base agent framework
- Random Randy agent implementation
- Local tournament runner with parallel execution
- Support for multiple X-Prize graphs (2, 11, 12, 18, 19, 20)
- Simulated annealing and Schrodinger equation adjudicators

[Unreleased]: https://github.com/murr2k/snowdrop-tangled-agents/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/murr2k/snowdrop-tangled-agents/compare/v0.0.5...v0.1.0
[0.0.5]: https://github.com/murr2k/snowdrop-tangled-agents/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/murr2k/snowdrop-tangled-agents/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/murr2k/snowdrop-tangled-agents/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/murr2k/snowdrop-tangled-agents/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/murr2k/snowdrop-tangled-agents/releases/tag/v0.0.1
