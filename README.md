# Snowdrop Tangled Agents

**AI agents that compete in the quantum game of Tangled**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MATLAB R2024a+](https://img.shields.io/badge/MATLAB-R2024a+-orange.svg)](https://www.mathworks.com/products/matlab.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Playwright](https://img.shields.io/badge/Playwright-automated-brightgreen.svg)](https://playwright.dev/)

---

A comprehensive framework for building intelligent agents that play [Tangled](https://tangled-game.com), a quantum game where players color edges on graph structures and outcomes are determined by quantum annealing. This project combines **Monte Carlo Tree Search**, **MATLAB-powered compute**, and **machine learning** to compete against the game's built-in AI opponents.

### Tech Stack

| Component | Technology |
|-----------|------------|
| **Core Language** | Python 3.11+ |
| **High-Performance Compute** | MATLAB R2024a+ (Deep Learning, Statistics, Parallel Computing Toolboxes) |
| **Web Automation** | Playwright (Chromium) |
| **Database** | SQLite with automatic migrations |
| **Game Engine** | [snowdrop-tangled-game-engine](https://github.com/snowdropquantum/snowdrop-tangled-game-engine) |
| **Quantum Adjudication** | Simulated Annealing, Schrödinger Equation solvers |

### Highlights

- **Complete SA Oracle** — Retrograde DP over all 16 grey levels (0–15), guaranteeing game-theoretically optimal play under SA adjudication; 371 MB lookup table covers every P1 decision
- **Win Investigation Complete** — 1,563+ games against AlphaQ prove draws are the achievable ceiling; AlphaQ steers to zero-energy boards under any classical strategy
- **Thompson Sampling Opening Selection** — Bayesian exploration-exploitation for opening selection in `AlphaQExplorerStrategy`
- **MCTS Strategy Engine** — Monte Carlo Tree Search with parallel `parfor` rollouts (6 workers × 100 rollouts per iteration)
- **MATLAB Integration** — High-compute hybrid solver: minimax + MCTS + oracle lookup
- **Live Web Play** — Automated gameplay on [tangled-game.com](https://tangled-game.com) via Playwright
- **Statistical Analysis** — SQLite-backed game history; 1,563+ AlphaQ games recorded and analyzed
- **Research Report** — [`quantum_tangled_proof.html`](quantum_tangled_proof.html) — full mathematical narrative with SVG diagrams and KaTeX formulae

---

## Quick Start

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.11 – 3.13 | |
| **Poetry** | 2.x | Package manager. Install: `pipx install poetry` |
| **Git** | any | |
| **MATLAB** | R2024a+ | Required for competitive play (see [MATLAB Setup](#matlab-setup)) |

### Installation

```bash
# Clone and install
git clone https://github.com/snowdropquantum/snowdrop-tangled-agents.git
cd snowdrop-tangled-agents
poetry install
poetry run playwright install chromium
```

Or use the automated setup script (installs Poetry if missing):

```bash
# Windows:
setup.bat

# macOS / Linux:
chmod +x setup.sh && ./setup.sh
```

### Configuration

```bash
cp .env.example .env
# Edit .env: add your tangled-game.com credentials (sign up at https://tangled-game.com)
```

See `.env.example` for all available settings including the optional live
stats dashboard.

### Play a Game

```bash
# Play 1 game against MCTS Melissa (default opponent, hybrid_solver strategy)
poetry run python play_tangled.py --games 1

# Play against AlphaQ Up using the AlphaQ Explorer strategy (5K MCTS iterations)
poetry run python play_tangled.py --strategy alphaq_explorer --opponent alphaq \
  --games 3 --mcts-iterations 5000

# View game statistics from the local database
poetry run python play_tangled.py --stats

# Best stable strategy: SA oracle (consistent draws at SA 0.67–0.81)
poetry run python play_tangled.py --strategy hybrid_solver \
  --lut-variant sa --oracle-override 11 11 G --games 5 --headless
```

### MATLAB Setup

MATLAB powers the high-performance MCTS engine. Without it, the system falls
back to a pure-Python heuristic that is not competitive against strong opponents.

**Required toolboxes:**

| Toolbox | Used for |
|---------|----------|
| **Parallel Computing** | `parfor` rollouts — 6 workers × 100 rollouts/iter |
| **Statistics and Machine Learning** | `interp1` P(win) calibration curve |

**Optional toolboxes (for advanced features):**

| Toolbox | Used for |
|---------|----------|
| Deep Learning | Neural network position evaluation priors |
| Database | Direct SQLite access from MATLAB training pipelines |
| MATLAB Compiler SDK | Building standalone Python-callable packages |
| Reinforcement Learning | RL training experiments |

**Install the MATLAB Engine API for Python:**

```bash
# From the MATLAB installation directory:
cd "C:\Program Files\MATLAB\R2026a\extern\engines\python"   # Windows
# cd /Applications/MATLAB/R2026a/extern/engines/python       # macOS
# cd /usr/local/MATLAB/R2026a/extern/engines/python           # Linux

python -m pip install .
```

The engine starts automatically in headless mode when you run a game — no need
to open the MATLAB GUI.  All `.m` source files live in
`snowdrop_tangled_agents/matlab/rl/` within this repository.

**Verify MATLAB is detected:**

```bash
poetry run python test_matlab_detection.py
```

### Strategies

| Strategy | Engine | Best for |
|----------|--------|----------|
| `hybrid_solver` | MATLAB Minimax + MCTS + Tabu | Default. Strong general play. |
| `alphaq_explorer` | MATLAB MCTS + Thompson Sampling | Competing against AlphaQ Up. |
| `matlab_mcts` | MATLAB MCTS only | Pure MCTS experiments. |
| `mcts` | Pure Python MCTS | No MATLAB required. Weaker. |
| `heuristic` | Python heuristics | Fast, always available. Not competitive. |

### Running Experiments

After installation and MATLAB setup, you can run the current flagship
experiment — systematic re-exploration of all 30 Petersen graph openings
against AlphaQ Up:

```bash
# Phase 1: Screen all 30 openings, 3 games each (est. ~8 hours)
poetry run python play_tangled.py \
  --strategy alphaq_explorer \
  --opponent alphaq \
  --games 90 \
  --mcts-iterations 5000 \
  --opening-mode round_robin
```

See `docs/EXPERIMENT_OPENING_RE_EXPLORATION.md` for the full experiment
design and rationale.

### Analyzing Results

All game data is recorded in a local SQLite database at
`~/.tangled/game_stats.db`.  Every game stores:

- Result, final score, opening move, strategy, MCTS iteration setting
- Move-by-move: edge, color, score, board state, thinking time
- MCTS diagnostics: iterations, total rollout simulations, tree depth

```bash
# Quick summary
poetry run python play_tangled.py --stats

# Direct SQL queries
sqlite3 ~/.tangled/game_stats.db "SELECT opening_edge, opening_color, result, final_score FROM games WHERE opening_mode='round_robin' ORDER BY opening_edge, opening_color"
```

### Live Dashboard (Optional)

For real-time monitoring during long experiments, deploy the
[tangled-stats-dashboard](https://github.com/murr2k/tangled-workspace/tree/main/tangled-stats-dashboard)
to [fly.io](https://fly.io) or any host that supports WebSockets:

1. Deploy the dashboard app (see its `.env.example` for configuration)
2. Set `TANGLED_DASHBOARD_URL` and `TANGLED_DASHBOARD_API_KEY` in your `.env`
3. Game telemetry streams automatically during play

If no dashboard is configured, experiments run normally and all data is
available for post-analysis from the SQLite database.

---

## Results

After 1,563+ games against AlphaQ Up across all 30 possible openings:

| Strategy | Games | Wins | Draws | Losses | Avg SA Score |
|----------|-------|------|-------|--------|--------------|
| E7G MCTS (`alphaq_explorer`) | 257 | 0 | 252 | 5 | 0.823 |
| SA oracle (`hybrid_solver`) | 58+ | 0 | 28+ | 0 | 0.749 |
| E10G oracle | 132 | 0 | 131 | 1 | 0.252 |

**Conclusion:** AlphaQ steers every game to a near-zero Schrödinger energy board.
The local SA oracle value at game start is +0.0075 — confirming the game is drawn
under optimal play. Winning requires knowledge of the website's exact quantum
parameters (unknown) or a full Schrödinger minimax solver (computationally infeasible).

See [`quantum_tangled_proof.html`](quantum_tangled_proof.html) for the full mathematical analysis.

---

## How to build your own agent
This repo includes a random agent (see `RandomRandyAgent` in `random_agent.py`). To build your
own agent, follow the instructions and pattern for that agent. The process is straightforward - your
agent gets a Game instance as input/observable, which is a snapshot of the current game being played 
which includes the current state of the game, and you output an action/move. The move is just which 
edge to pick, and what color to apply to it. That's all! Obviously the underlying process you use to 
select this move can be arbitrarily complex but the actual agent interface is simple.

## Testing agents
A test script is included in `run_local_parallel_tournament.py`. Out of the box it runs 100,000 games
for each of the X-Prize graphs (not including the mutant C60 graph) where the adjudicator used is
simulated annealing, and the two agents in the competition are both random agents.

## Suggested next steps
If you want to build your own agent, I would start by building your own adjudicator (look at both
the adjudicators in the `snowdrop-adjudicators` repo). The better you
can spoof the D-Wave hardware, the better quality reward your agent will have to learn from. To do this,
I would recommend you build a look-up table adjudicator, by first enumerating all the terminal states
for a small game graph, and then adjudicating all of them using the schrodinger equation adjudicator (this will
be limited to small game graphs, but it's still a great warm-up for doing something more ambitious). Once
you have your lookup table adjudicator, you can then build an agent that uses it to give you the correct
answer every time. You can use `run_local_parallel_tournament.py` - add your adjudicator and new agents to
and see how well they do against the random agent and any other agents you might have built!

---

## Development Progress

This section documents the actual development steps taken to build a competitive Tangled agent.

### 1. MCTS Strategy Engine

Implemented a Monte Carlo Tree Search (MCTS) strategy with the following features:

- **UCB1 exploration** with configurable exploration constant
- **Simulated Annealing adjudicator** for terminal state evaluation
- **Opening book** with pre-computed strong opening moves for the Petersen graph
- **Time and iteration limits** for move calculation
- **Statistics collection** for game analysis

```bash
# Play with MCTS strategy (default: 500K iterations, unlimited time)
python play_tangled.py --strategy mcts --games 3

# Fast mode for testing (100K iterations, 30s per move)
python play_tangled.py --strategy mcts --mcts-iterations 100000 --mcts-time 30 --games 3
```

### 2. Web Automation Bot

Built a Playwright-based bot to play on tangled-game.com:

- **Automatic login** with credentials from `.env`
- **Game state parsing** from the web interface
- **Move execution** via browser automation
- **Score tracking** and game result detection
- **Keep-alive mode** for continuous play sessions

```bash
# Play games on tangled-game.com
python play_tangled.py --strategy hybrid --games 10 --keep-open
```

### 3. Statistics Collection System

Created a SQLite-based statistics system:

- **Game history** tracking (opponent, result, scores, timestamps)
- **Move-by-move analysis** with state snapshots
- **Adjudicator calibration** data collection
- **Win rate analysis** by opponent and strategy

```bash
# View statistics
python play_tangled.py --stats
python play_tangled.py --calibration
```

### 4. MATLAB Toolbox Integration

Extended the system with MATLAB integration for advanced ML capabilities:

| Toolbox | Purpose |
|---------|---------|
| **Deep Learning Toolbox** | Neural network training for position evaluation |
| **Statistics and ML Toolbox** | Opponent clustering and style classification |
| **Database Toolbox** | Direct MATLAB-SQLite access for training pipelines |
| **MATLAB Compiler SDK** | Package MATLAB code as Python-callable modules |

#### Neural Network Architecture

**Value Network** (position evaluation):
- Input: 50-element feature vector (board state, edge categories, game phase)
- Hidden: FC(128) → ReLU → Dropout → FC(64) → ReLU → FC(32) → ReLU
- Output: Tanh → value ∈ [-1, +1]

**Opponent Model** (style classification):
- 20-element opponent feature vector (edge preferences, aggression metrics)
- K-means clustering into play styles (aggressive, defensive, hub-focused)
- Adaptive prior adjustment based on opponent tendencies

### 5. Compiled Package Deployment

Built standalone Python packages using MATLAB Compiler SDK:

| Package | Functions | Purpose |
|---------|-----------|---------|
| `tangled_value_network` | `evaluate_position_nn` | Neural network inference |
| `tangled_opponent_model` | `classify_opponent`, `adapt_to_opponent` | Opponent modeling |
| `tangled_training` | `train_value_network`, `cluster_opponents` | Model training |

**For users without MATLAB:**
1. Download MATLAB Runtime R2026a (free) from [MathWorks](https://www.mathworks.com/products/compiler/matlab-runtime.html)
2. Install compiled packages: `pip install tangled_value_network tangled_opponent_model`

**Important:** Runtime version must exactly match the compile version (R2026a).

### 6. Backend Fallback Chain

The system automatically selects the best available backend:

```
1. Compiled Packages (fastest, MATLAB Runtime only)
       ↓ if unavailable
2. MATLAB Engine API (full functionality, requires license)
       ↓ if unavailable
3. Pure Python Heuristics (always available)
```

```bash
# Check what's available
python play_tangled.py --training-status

# Play with MATLAB enhancement
python play_tangled.py --strategy matlab --use-nn --adapt-opponent --games 5
```

### 7. D-Wave Inspired Hybrid Solver (January 2026)

A hybrid Minimax-MCTS solver inspired by D-Wave's classical optimization techniques:

**Architecture:**
- **Alpha-Beta Minimax** - Exact search at shallow depths with transposition tables
- **Tabu Search** - D-Wave MST2-inspired multistart optimization for rollout refinement
- **MCTS** - Monte Carlo Tree Search with progressive widening for deep exploration
- **Expanded LUT** - 19 million pre-computed exact minimax values

**Lookup Table Coverage:**
| Grey Edges | States | Method |
|------------|--------|--------|
| 0 (terminal) | 32,768 | Direct lookup |
| 1 | 491,520 | Minimax depth-1 |
| 2 | 3,440,640 | Minimax depth-2 |
| 3 | 14,909,440 | Minimax depth-3 |
| **Total** | **18,874,368** | **Exact values** |

This gives guaranteed optimal play for the last 4 moves of every game.

**References:**
- D-Wave qbsolv decomposition strategy
- D-Wave dwave-tabu (MST2 algorithm)
- Palubeckis (2004) "Multistart Tabu Search Strategies"

```bash
# Play with Hybrid Solver (default: 500K iterations, unlimited time)
python play_tangled.py --strategy hybrid_solver --games 3

# Competitive mode with maximum strength (1M iterations)
python play_tangled.py --strategy hybrid_solver --mcts-iterations 1000000 --games 1
```

See `docs/HYBRID_MINIMAX_MCTS_PLAN.md` for complete implementation plan.

### 8. MATLAB MCTS Strategy (January 2026)

High-compute MCTS implementation in MATLAB for live play against MCTS Melissa:

**Key Features:**
- 500K iterations per move with unlimited time (default, elite quality - ~14 min per move)
- Configurable 100K-1M iterations via `--mcts-iterations` for speed/quality tradeoff
- **Parallel `parfor` rollouts** — each MCTS iteration runs `NumWorkers` (default 6) independent rollouts in parallel and averages results, producing 6x more samples per leaf visit while using the already-allocated worker pool that previously sat idle
- Domain-specific terminal evaluation calibrated from 140+ games
- Opening book securing our vertex edges (E9, E10, E11 Green)
- Adaptive exploration based on score momentum
- Compute diagnostics using MATLAB's `cputime` and `memory` functions

**Critical Discoveries:**
- **E12 Green is catastrophic**: Connecting hub (V6) to opponent vertex (V7) with ferromagnetic coupling helps the opponent. Added -0.8 penalty.
- **E2 Green causes score collapse**: Observed -2.6 point drops at move 4. Added -0.5 penalty.
- **Turn-based timing**: Game is strictly turn-based; apparent "race conditions" were actually stale DOM state from reading board before browser fully updated.

**Results:**
- Before fixes: Score peaked at +2.95, collapsed to -2.6 at move 4
- After fixes: Stable score progression, competitive games (draws and close losses)

```bash
# Play with MATLAB MCTS (default: 500K iterations, unlimited time - elite quality)
python play_tangled.py --strategy matlab_mcts --games 3

# Fast testing mode (100K iterations)
python play_tangled.py --strategy matlab_mcts --mcts-iterations 100000 --games 5

# View game statistics
python play_tangled.py --stats
```

See `docs/MATLAB_MCTS_STRATEGY.md` for complete theory of operation.

### 9. Game Analytics & Visualization (January 2026)

An instrumented research system for tracking performance, discovering patterns, and measuring improvement:

**Visualization Tools:**
- **Progress plots** — Rolling win rate and score trends over time
- **Edge effectiveness** — Score delta and win rate by edge/color combination
- **Opening analysis** — Identify winning opening sequences with sample sizes

**Design Philosophy:**
- Observability over raw metrics — answer *why*, not just *what*
- Every plot maps to a subsystem lever you can actually adjust
- Query-first architecture — plots are views, not logic
- Timestamped outputs for historical comparison

```bash
# Generate all analysis plots
python -m snowdrop_tangled_agents.tools.plot_progress --all

# Individual plot types
python -m snowdrop_tangled_agents.tools.plot_progress -t progress
python -m snowdrop_tangled_agents.tools.plot_progress -t edge
python -m snowdrop_tangled_agents.tools.plot_progress -t opening
```

Plots are saved to `plots/` with naming: `{type}_{YYYYMMDD}_{HHMMSS}.png`

See `docs/GAME_ANALYTICS.md` for complete documentation and research applications.

### 10. AlphaQ Explorer — Closed-Loop Learning Against a New Opponent (January 2026)

A two-phase explore/exploit strategy built to contend with the new **AlphaQ Up** opponent.
The key advance over previous opponent-specific strategies is a **closed learning loop**: learned
edge-value adjustments now actually modify the MATLAB MCTS rollout priors, so the solver gets
stronger with every game it plays.

**How it works:**

| Phase | Games | Behaviour |
|-------|-------|-----------|
| Exploration | 0–29 | Cycles all 30 possible first moves (15 edges × 2 colors). Learning is disabled so each opening is tested under identical solver conditions. Win/score is recorded per opening. |
| Exploitation | 30+ | Re-enables REINFORCE learning. Rotates only the top-N openings (default 5) ranked by wins then avg score. After every game, the updated `edge_adjustments` vector is pushed into MATLAB via `hybridSolver.setEdgeBias()`. |

**The closed loop in detail:**

Previously `HybridSolverStrategy` ran REINFORCE and accumulated per-edge adjustments in Python,
but those values were never fed back into the MATLAB solver — the MCTS rollouts kept using the
same static heuristic priors every game. This change adds:

1. An `EdgeBias` property (1×15 double) to both `TangledMCTS` and `HybridTangledSolver`
2. A `setEdgeBias()` method on both classes, with the solver forwarding to its MCTS instance
3. Two lines at the end of `computeRolloutPrior`: `prior += EdgeBias(edge)`, clamped to [0.001, 0.999]
4. Bias preservation across `setPlayer()` so a new MCTS instance inherits learned adjustments
5. `AlphaQExplorerStrategy._push_edge_bias()` which calls `hybridSolver.setEdgeBias([...])` after
   each exploitation game, closing the loop

**Persistence and resumability:** Full strategy state (phase, exploration results, exploitation
openings, index) is saved to `~/.tangled/alphaq_explorer_state.json`. If a run is interrupted
during exploitation, the next run re-enables learning and re-pushes the accumulated bias before
playing.

```bash
# Run exploration games followed by exploitation (default: 500K iterations per move)
python play_tangled.py --strategy alphaq_explorer --opponent alphaq --run 60

# Fast exploration mode for development (100K iterations)
python play_tangled.py --strategy alphaq_explorer --opponent alphaq --mcts-iterations 100000 --run 60

# Check what the explorer learned
cat ~/.tangled/alphaq_explorer_state.json
cat ~/.tangled/hybrid_solver_adjustments.json
```

### 11. Score-Weighted Draw Reward + Opponent-Conditional Calibration (January 2026)

Two fixes targeting the learning stall observed in Run 47 (0W/27L/33D vs AlphaQ Up).

**Fix 1 — Score-weighted draw reward**

Against strong opponents wins are rare; most games end in draws.  The old REINFORCE reward for a
draw was a flat ±0.1 regardless of the final score, which gave the policy gradient almost no signal.
The draw reward is now `score × 0.65`, so a near-miss draw at +0.78 produces reward +0.507 instead
of +0.1.  The multiplier 0.65 is chosen so that the maximum realistic draw reward (+0.65 at score
+1.0) stays below the minimum win reward (1.0), preserving the win > draw > loss ordering.  Win and
loss branches are unchanged.

| Scenario | Old reward | New reward |
|----------|-----------|-----------|
| Win at +2.0 | 2.0 | 2.0 |
| Draw at +0.78 | +0.1 | **+0.507** |
| Draw at 0.0 | +0.1 | 0.0 |
| Draw at −0.5 | −0.1 | −0.325 |
| Loss at −1.89 | −1.945 | −1.945 |

**Fix 2 — Opponent-conditional calibration**

The P(win) calibration curve was fitted on Melissa game data.  Loading it for every opponent made the
solver complacent at moderate scores against opponents with different noise profiles.  `loadCalibration()`
now follows a two-phase lookup:

```
Named opponent with fitted curve  →  load calibration_<name>.mat
Named opponent, no fitted curve   →  tanh sigmoid fallback
No opponent name (legacy path)    →  load calibration_pwin.mat
```

`calibration_melissa.mat` is shipped alongside `calibration_pwin.mat` so Melissa resolves correctly.
New per-opponent curves can be added by dropping a `calibration_<name>.mat` into `data/` — the
sanitisation rule is: lowercase, replace non-`[a-z0-9]` with `_`, collapse runs, strip trailing `_`.

**Verification (Run 50, 10 games vs AlphaQ Up):**
- Tanh fallback fires correctly (no `calibration_alphaq_up.mat` exists)
- Draw rewards: 0.34–0.50 per game (vs flat 0.1 previously)
- Final edge adjustments are non-uniform: +0.035 on edges played in good draws, −0.081 on edges
  associated with losses

```bash
# Play with the updated solver
python play_tangled.py --strategy alphaq_explorer --opponent alphaq --run 10

# Inspect learned bias
cat ~/.tangled/hybrid_solver_adjustments.json
```

### 12. Thompson Sampling Opening Selection for AlphaQ Explorer (February 2026)

The two-phase explore/exploit strategy (Section 10) had a critical failure: the greedy opening ranking by `(wins DESC, avg_score DESC)` **degenerated to `avg_score DESC`** when all 74 games had 0 wins. This caused the strategy to select E9G (89% loss rate) and E11G (100% loss rate) for exploitation, producing 0% win rate during games 29–56 and collapse of the entire strategy.

**Root Cause:** With zero wins across all openings, the sort key `(wins, avg_score)` defaults to average score. E9G's single draw at high score (0.728) and E11G's high-scoring losses (~0.58) ranked higher than E0G, E6G, E7G—the only openings that never lost.

**Solution: Thompson Sampling** — a principled Bayesian approach to exploration-exploitation that naturally avoids catastrophic losers:

**Key Innovation: Half-Draw Credits**

Each opening's posterior: `Beta(α, β)` where:
```
α = 1 + wins + 0.5 × draws    (successes, half-weighted)
β = 1 + losses + 0.5 × draws  (failures, half-weighted)
```

Why draws are half-weighted: In a 0-win regime, draws represent partial success (game didn't lose). Without this credit, all openings collapse to `Beta(1, 1+losses)`, losing all signal about safety. With it:
- E0G (0W/10D/0L) → Beta(6, 6), mean=0.5, strong selection
- E11G (0W/0D/10L) → Beta(1, 11), mean=0.08, almost never selected
- Untried openings → Beta(1, 1), mean=0.5, explored freely

**How it works:**

Each game:
1. For all 30 openings: sample from `Beta(α, β)`
2. Play the opening with the highest sample
3. Record result and update counts
4. Learning-rate gating: REINFORCE disabled for first 10 games (MIN_GAMES_BEFORE_LEARNING) to collect clean opening data, then enabled

**Test Results (8 comprehensive tests, all passing):**

| Test | Expected | Result |
|------|----------|--------|
| Thompson favors safe openings | E0G (5 wins) selected >150/1000 | **186/1000 (18.6%)** ✓ |
| Thompson explores untried | E0G (untried among 10L openings) >50/1000 | **54–80/1000 (5–8%)** ✓ |
| v1→v2 migration | 74 games migrated losslessly | **0 data loss** ✓ |
| Learning gating | Learning disabled games 1–9, enabled game 10+ | **Correct transition** ✓ |
| State persistence | Save/load across sessions | **All counts match** ✓ |

**Expected Improvements (vs. old two-phase strategy):**

| Metric | Old Strategy | Thompson Sampling |
|--------|--------------|-------------------|
| Safe openings (0L) selected | ~30% | 80%+ |
| Catastrophic losers (100L) selected | ~20% | <5% |
| Win rate trajectory | Cliff collapse at game 30 | Monotonic improvement |
| Opening convergence | Broken (selects worst) | Principled (Bayesian) |

**Migration & Backward Compatibility:**

- Old v1 state files (`alphaq_explorer_state.json`) automatically migrate to v2 on first load
- All 74 existing games successfully migrated with zero data loss
- State versioning ensures future compatibility

**Code & Documentation:**

- Implementation: `snowdrop_tangled_agents/matlab/matlab_strategy.py` (complete rewrite)
- Tests: 8 unit tests in `snowdrop_tangled_agents/tests/test_matlab_integration.py`
- Full technical docs: `docs/ALPHAQ_STRATEGY.md` (mathematical foundation, error analysis, reproduction instructions)

**Usage:**

```bash
# Run trial with Thompson Sampling (default: 500K iterations, ~14 min per move)
python play_tangled.py --strategy alphaq_explorer --opponent alphaq --games 10

# Fast mode for testing algorithm behavior (100K iterations, ~3 min per move)
python play_tangled.py --strategy alphaq_explorer --opponent alphaq --mcts-iterations 100000 --games 10

# Inspect opening posteriors
cat ~/.tangled/alphaq_explorer_state.json
```

The Thompson Sampling approach generalizes to any opening-selection problem in Tangled and other stochastic games where a strategy needs to learn which first moves lead to success against a specific opponent.

### 13. Opening Re-Exploration Experiment (February 2026)

All prior conclusions about optimal openings against AlphaQ Up — including the
"Nash Equilibrium" claim from Runs 60–66 — were made with **broken MCTS
parallelization**. The `parfor` worker pool allocated 6 workers but never dispatched
work to them: every iteration performed a single serial rollout while workers sat
idle at 0% CPU. This means historical tests used **5,000 rollouts per move** when
they should have used **3,000,000** (a 600x deficit).

Additionally, the MCTS terminal evaluation used a `tanh(score × 0.4)` calibration
fallback that is catastrophically miscalibrated for AlphaQ: it predicts 66% P(win)
at score +0.8, where the actual win rate is **0%** across 1,297 games. This caused
the search to satisfice on the E7G opening line instead of exploring alternatives.

**What changed:**

| Fix | Before | After |
|-----|--------|-------|
| Parallel rollouts | 1 per iteration (broken) | 600 per iteration (6 workers × 100) |
| Calibration fallback | `tanh(score × 0.4)` — 66% at +0.8 | Generic `calibration_pwin.mat` — 9.5% at +0.8 |
| Total rollouts/move (5K iter) | 5,000 | 3,000,000 |

**The experiment:** Systematically re-test all 30 possible opening moves
(15 edges × 2 colors) against AlphaQ on the Petersen graph with functional
parallel MCTS and correct calibration.

| Phase | Openings | Games | Iterations | Rollouts/move |
|-------|----------|-------|------------|---------------|
| Phase 1: Screening | All 30 | 90 (3 each) | 5,000 | 3,000,000 |
| Phase 2: Deep test | Promising | 10 each | 10,000 | 6,000,000 |

```bash
# Run Phase 1: systematic round-robin exploration of all 30 openings (est. ~8 hours)
poetry run python play_tangled.py \
  --strategy alphaq_explorer \
  --opponent alphaq \
  --games 90 \
  --mcts-iterations 5000 \
  --opening-mode round_robin
```

See `docs/EXPERIMENT_OPENING_RE_EXPLORATION.md` for the full experiment design,
calibration analysis, and expected results.

### 14. Empirical Closure of the Petersen Graph Under Quantum Adjudication (February 2026)

A formal investigation into whether AlphaQ Up can be defeated on the Petersen graph.

**Key findings:**

- **0 wins across 120 diverse terminal states** against AlphaQ, from 2,436+ game observations
- **SA polarity inversion**: Simulated annealing scores are anti-correlated (r = -0.396) with the website's quantum adjudicator against AlphaQ -- states the SA LUT predicts as wins are actually losses
- **AlphaQ zero-loss equilibrium**: Max observed website score +0.861, far below the +2 win threshold. AlphaQ's policy restricts the reachable terminal state basin to [-8.8, +0.9]
- **Statistical confidence**: 95% confidence that winning terminals comprise < 2.5% of the reachable state space (binomial model)
- **Sampling bias caveat**: Classical strategies may be biased toward classically accessible regions; winning terminals may exist only in non-classical portions of the state space

The analysis is grounded in 35 references spanning quantum annealing theory (Kadowaki & Nishimori 1998), order-by-disorder phenomena (Villain et al. 1980), Petersen graph symmetry (Schmidt 2018), and Geordie Rose's Tangled game design and $10,000 AlphaQ Up Challenge.

**Tools created during this investigation:**
- `tools/build_website_lut.py` -- empirical website-calibrated LUT builder
- `strategy/terminal_explorer_strategy.py` -- diversity-maximizing terminal state exploration
- Oracle route cycling mode (`--route-mode cycle`)

See `docs/EMPIRICAL_CLOSURE_OF_THE_PETERSEN_GRAPH_UNDER_QUANTUM_ADJUDICATION.md` for the full paper.

### Documentation

- `docs/EMPIRICAL_CLOSURE_OF_THE_PETERSEN_GRAPH_UNDER_QUANTUM_ADJUDICATION.md` - Formal paper on the empirical negative result against AlphaQ
- `docs/EMPIRICAL_LUT_CONSTRUCTION_RESULTS.md` - Campaign results: 65 games, SA polarity inversion data
- `docs/HIGH_COMPUTE_STRATEGY_OPTIONS.md` - GPU + MATLAB strategy options for unlimited compute
- `docs/POST_SA_LUT_STRATEGY_OPTIONS.md` - Strategy alternatives after SA LUT proven anti-correlated
- `docs/DEPENDENCY_REPOSITORIES.md` - Complete documentation of the three dependency repos (game engine, adjudicators, tangled-adjudicate)
- `docs/ALPHAQ_STRATEGY.md` - Thompson Sampling mathematical foundation, test results, reproduction guide
- `docs/THEORY_OF_OPERATION.md` - Comprehensive system documentation
- `docs/MATLAB_INTEGRATION.md` - Complete MATLAB integration guide
- `docs/MATLAB_MCTS_STRATEGY.md` - MATLAB MCTS theory of operation and tuning guide
- `docs/HYBRID_MINIMAX_MCTS_PLAN.md` - D-Wave inspired hybrid solver implementation plan
- `docs/MCTS_DEPTH_ENHANCEMENT_RESEARCH.md` - MCTS depth enhancement literature review
- `docs/GAME_ANALYTICS.md` - Game analytics and visualization system
- `docs/TEST_SUITE.md` - Regression test documentation
- `docs/THE_MATHEMATICS_OF_TANGLED_GAME.md` - Game theory analysis
- `docs/EXPERIMENT_OPENING_RE_EXPLORATION.md` - Opening re-exploration experiment design and calibration analysis

### 15. SA Oracle and Win Investigation (May 2026)

A complete retrograde minimax oracle covering the entire Petersen graph game tree, followed by a
definitive investigation into whether AlphaQ Up can be defeated.

**Oracle construction (Priorities 6–10):**

- `scripts/generate_sa_oracle.py` — retrograde DP from all 32,768 terminal boards upward through
  all 16 grey levels. Generates `expanded_lut_sa.mat` (371 MB, levels 0–15, untracked).
- `ExpandedLUT.m` — loads all 16 levels via `readMatOrH5()` helper (supports both MATLAB v7 and
  h5py HDF5 formats). Oracle fires on every P1 turn via `solveOracle()` in `HybridTangledSolver.m`.
- Game-start oracle value (all grey, optimal play): **+0.0075** — P1 has a forced SA-optimal first
  move (E11G, E10G, or E14P; all near-identical in value).

```bash
# Generate oracle (requires ~30s total, ~1 GB RAM peak)
poetry run python scripts/generate_sa_oracle.py --end-level 15 --validate

# Play with full oracle (best stable strategy)
poetry run python play_tangled.py --strategy hybrid_solver \
  --lut-variant sa --oracle-override 11 11 G --games 5 --headless
```

**Win investigation findings:**

- **0 wins in 1,563+ games** across all 30 openings and all strategies.
- Local Schrödinger adjudicator (epsilon=0.0005, anneal_time=40ns) predicts `winner=red` on every
  board — including confirmed losses. The website uses different quantum parameters; our LUT is
  irrelevant to predicting website outcomes.
- AlphaQ adaptively steers every game to a near-zero Schrödinger energy board (SchrLUT ≈ 0.006).
  Even reaching SchrLUT=7.97 produces draws and losses, not wins.
- Petersen graph vertex-transitivity + color-flip symmetry of the SA Hamiltonian implies V(∅) = 0
  under optimal play — the game is fundamentally drawn.

See [`quantum_tangled_proof.html`](quantum_tangled_proof.html) for the complete mathematical treatment.