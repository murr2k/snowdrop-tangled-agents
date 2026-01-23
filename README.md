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

- **MCTS Strategy Engine** - Monte Carlo Tree Search with 5000+ iterations per move
- **MATLAB Integration** - High-compute search with parallel processing and neural network evaluation
- **Live Web Play** - Automated gameplay on tangled-game.com via Playwright
- **Statistical Analysis** - SQLite-backed game history with pattern discovery
- **Calibrated Heuristics** - Terminal evaluation tuned from 140+ real games

---

## Quick Start

### Prerequisites

- Python 3.11 - 3.13
- Git
- MATLAB R2024a+ (optional, for high-performance strategies)

### Installation

**Automated Setup (Recommended)**

```bash
# Clone the repository
git clone https://github.com/snowdropquantum/snowdrop-tangled-agents.git
cd snowdrop-tangled-agents

# Run the setup script
# On Windows:
setup.bat

# On macOS/Linux:
chmod +x setup.sh
./setup.sh
```

**Manual Setup**

```bash
# Using Poetry (recommended)
poetry install
poetry run playwright install chromium

# Using pip
pip install -e .
pip install snowdrop-tangled-game-engine snowdrop-adjudicators
python -m playwright install chromium
```

### Configuration

Copy `.env.example` to `.env` and add your tangled-game.com credentials:

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Play a Game

```bash
# Play against MCTS Melissa on tangled-game.com
python play_tangled.py --games 1

# Use different strategies
python play_tangled.py --strategy mcts --mcts-time 5 --games 3
python play_tangled.py --strategy hybrid --games 3

# View game statistics
python play_tangled.py --stats

# View adjudicator calibration
python play_tangled.py --calibration
```

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
# Play with MCTS strategy
python play_tangled.py --strategy mcts --mcts-time 5 --games 3
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
# Play with Hybrid Solver
python play_tangled.py --strategy hybrid_solver --games 3
```

See `docs/HYBRID_MINIMAX_MCTS_PLAN.md` for complete implementation plan.

### 8. MATLAB MCTS Strategy (January 2026)

High-compute MCTS implementation in MATLAB for live play against MCTS Melissa:

**Key Features:**
- 5000 iterations per move with 20s time limit
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
# Play with MATLAB MCTS
python play_tangled.py --strategy matlab_mcts --games 3

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

### Documentation

- `docs/THEORY_OF_OPERATION.md` - Comprehensive system documentation
- `docs/MATLAB_INTEGRATION.md` - Complete MATLAB integration guide
- `docs/MATLAB_MCTS_STRATEGY.md` - MATLAB MCTS theory of operation and tuning guide
- `docs/HYBRID_MINIMAX_MCTS_PLAN.md` - D-Wave inspired hybrid solver implementation plan
- `docs/MCTS_DEPTH_ENHANCEMENT_RESEARCH.md` - MCTS depth enhancement literature review
- `docs/GAME_ANALYTICS.md` - Game analytics and visualization system
- `docs/TEST_SUITE.md` - Regression test documentation
- `docs/THE_MATHEMATICS_OF_TANGLED_GAME.md` - Game theory analysis