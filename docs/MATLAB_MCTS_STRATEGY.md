# MATLAB MCTS Strategy for Tangled Game

This document describes the theory of operation for the MATLAB-based Monte Carlo Tree Search (MCTS) strategy used to play against MCTS Melissa on tangled-game.com.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Petersen Graph Edge Classification](#petersen-graph-edge-classification)
- [Terminal Evaluation Function](#terminal-evaluation-function)
  - [Priority System](#priority-system-avoids-double-counting)
  - [Scoring Values](#scoring-values)
  - [Critical Discoveries](#critical-discoveries)
- [Rollout Policy](#rollout-policy)
- [Opening Book](#opening-book)
- [Adaptive Exploration](#adaptive-exploration)
- [Compute Diagnostics and Performance Profiling](#compute-diagnostics-and-performance-profiling)
  - [Metrics Tracked](#metrics-tracked)
  - [How Profiling Helped](#how-profiling-helped)
  - [Interpreting the Metrics](#interpreting-the-metrics)
- [Improvements Made (January 2026)](#improvements-made-january-2026)
  - [1. Fixed E12 G Evaluation Bug](#1-fixed-e12-g-evaluation-bug)
  - [2. Fixed E2 G Evaluation Bug](#2-fixed-e2-g-evaluation-bug)
  - [3. Added Compute Effort Diagnostics](#3-added-compute-effort-diagnostics)
  - [4. Fixed Turn-Based State Reading](#4-fixed-turn-based-state-reading)
- [Results](#results)
- [File Locations](#file-locations)
- [Roadmap for Future Work](#roadmap-for-future-work)
- [Configuration](#configuration)
- [References](#references)

---

## Overview

The strategy uses a high-compute MCTS implementation in MATLAB with domain-specific heuristics calibrated from game statistics. The key insight is that the quantum adjudicator's behavior doesn't match simple heuristics - certain moves that seem strategically sound cause catastrophic score collapses.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Python Orchestration                      │
│                      (play_tangled.py)                       │
│  - Browser automation via Playwright                         │
│  - Turn detection and state reading                          │
│  - Move execution and score tracking                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  MatlabMCTSStrategy                          │
│              (matlab_mcts_strategy.py)                       │
│  - Opening book (first 3 moves)                              │
│  - Adaptive exploration based on momentum                    │
│  - Parameter persistence and learning                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     TangledMCTS.m                            │
│  - UCB1 selection with progressive bias                      │
│  - Heuristic rollout policy                                  │
│  - Calibrated terminal evaluation                            │
│  - Compute effort diagnostics                                │
└─────────────────────────────────────────────────────────────┘
```

## Petersen Graph Edge Classification

The Petersen graph has 15 edges connecting 10 vertices. For Player 1 (Red), edges are classified as:

| Category | Edges (0-indexed) | Description |
|----------|-------------------|-------------|
| MY_EDGES | E9, E10, E11 | Touch our special vertex (V5) |
| OPP_EDGES | E5, E12, E13 | Touch opponent's vertex (V7) |
| HUB_EDGES | E2, E10, E12 | Touch the hub vertex (V6) |

Note: E10 is in both MY_EDGES and HUB_EDGES. E12 is in both OPP_EDGES and HUB_EDGES.

## Terminal Evaluation Function

The evaluation function scores terminal states from our perspective. Key principles:

### Priority System (Avoids Double-Counting)

```matlab
1. MY_EDGES (highest priority) - defense is critical
2. OPP_EDGES (skip if in MY_EDGES) - attacks have modest value
3. HUB_EDGES (skip if already scored) - hub control is situational
```

### Scoring Values

| Edge Type | Green | Purple |
|-----------|-------|--------|
| MY_EDGES | +1.2 | -1.0 |
| OPP_EDGES (E5) | -0.15 | +0.25 |
| OPP_EDGES (E13) | -0.15 | +0.35 |
| OPP_EDGES (E12) | **-0.8** | +0.15 |
| HUB_EDGES (E2) | **-0.5** | +0.1 |

### Critical Discoveries

**E12 Green is catastrophic (-0.8 penalty)**
E12 connects hub (V6) to opponent vertex (V7). Making it green/ferromagnetic creates strong coupling between the hub and opponent's special vertex, which dramatically helps the opponent in quantum adjudication.

**E2 Green causes score collapse (-0.5 penalty)**
E2 connects inner vertex (V0) to hub (V6). Game data showed E2 G at move 4 consistently caused 2.6-point score drops. The quantum physics interaction is unclear, but the empirical evidence is strong.

## Rollout Policy

The heuristic rollout policy uses weighted random selection with these priors:

| Edge Type | Our Turn (Green%) | Opponent Turn (Green%) |
|-----------|-------------------|------------------------|
| MY_EDGES | 95% | 15% |
| OPP_EDGES | 5% | 95% |
| HUB_EDGES | 25% | 55% |
| Other | 55% | 55% |

The low green percentage for hub edges (25%) reflects the discovery that E2 G is harmful.

## Opening Book

The first 3 moves use a predefined sequence rather than MCTS:

```
Move 1: E9 Green  (secure our edge)
Move 2: E11 Green (secure our edge)
Move 3: E10 Green or E5 Purple (depending on availability)
```

This opening consistently achieves +1.0 score by securing our vertex edges.

## Adaptive Exploration

Exploration is adjusted based on score momentum:

- **Losing momentum** (score dropped >0.5): Boost exploration by 1.3x
- **Winning momentum** (score gained >0.5): Reduce exploration by 0.8x

This encourages more aggressive search when behind and consolidation when ahead.

## Compute Diagnostics and Performance Profiling

Understanding whether MATLAB is actually doing meaningful work is critical for debugging. Early in development, there was uncertainty about whether the MATLAB engine was truly computing or just returning quickly with poor results.

### Metrics Tracked

The MATLAB engine tracks detailed compute metrics using built-in profiling functions:

```matlab
% In TangledMCTS.search()
cpuStart = cputime;                    % MATLAB built-in CPU time
startTime = tic;                       % Wall clock time

% ... search loop ...

cpuElapsed = cputime - cpuStart;       % Total CPU seconds
wallTime = toc(startTime);             % Total wall seconds
memInfo = memory;                      % MATLAB memory struct
memUsedMB = memInfo.MemUsedMATLAB / (1024 * 1024);
```

The info struct returned by `search()` includes:

| Metric | Description | Typical Value |
|--------|-------------|---------------|
| `cpuTime` | CPU seconds consumed | 5-12s |
| `wallTime` | Wall clock seconds | 6-10s |
| `cpuEfficiency` | CPU/wall ratio (>1 = parallel) | 1.2-1.3 |
| `nodesExpanded` | Tree nodes created | 5000 |
| `simulations` | Rollout simulations run | 2500-5000 |
| `treeDepth` | Maximum search depth | 4-6 |
| `memoryUsedMB` | MATLAB memory usage | ~500MB |

### How Profiling Helped

**1. Confirmed MATLAB Was Actually Computing**

Early debugging showed concerning behavior - MCTS would return moves that seemed random or poorly considered. The first question was: "Is MATLAB actually running 5000 iterations?"

The diagnostics confirmed yes:
```
MCTS: E12 P (5000 iter in 8.1s, CPU=9.69s [119%], nodes=5000, sims=5000, depth=5)
```

This ruled out issues like:
- MATLAB engine not starting properly
- Early termination due to errors
- Time limit cutting off search prematurely

**2. Revealed Parallel Computing Was Working**

CPU efficiency >100% indicates multiple cores are being used. Seeing `[119%]` or `[125%]` confirmed that MATLAB's parallel computing was engaged, providing meaningful speedup.

**3. Identified Search Depth Limitations**

The `treeDepth` metric showed searches were reaching depth 4-6, which is reasonable for a 15-move game. If depth had been stuck at 1-2, it would indicate expansion problems.

**4. Helped Diagnose Move Quality Issues**

When MCTS returned bad moves despite high iteration counts, the diagnostics proved the problem wasn't computational effort - it was the **evaluation function**. This redirected debugging from "is MATLAB working?" to "what's wrong with our heuristics?"

Example from logs:
```
MCTS: E2 G (5000 iter in 6.1s, CPU=5.47s [97%], nodes=5000, sims=5000, depth=5)
Move 4: E2 G -> Score: -0.05  (score collapsed from +2.98)
```

MCTS did 5000 iterations with proper depth and simulation count - it genuinely believed E2 G was good. The problem was the terminal evaluation, not the search.

**5. Tracked Session-Level Statistics**

Cumulative metrics help understand total computational investment:
```matlab
effort.sessionTotalCPU        % Total CPU time across all searches
effort.sessionTotalIterations % Total iterations this session
```

### Interpreting the Metrics

| Observation | Indicates |
|-------------|-----------|
| CPU efficiency < 50% | Parallel pool not initialized or single-threaded |
| CPU efficiency > 100% | Parallel computing working correctly |
| Low node count vs iterations | Heavy tree reuse (good) |
| Simulations << iterations | Many terminal states reached early |
| Depth stuck at 1-2 | Expansion problem or very constrained state |
| High memory (>1GB) | Large tree, may need pruning |

### Python-Side Logging

The Python wrapper logs diagnostics for each move:

```python
logger.info(
    f"MCTS: E{edge} {color} ({iterations} iter in {search_time:.1f}s, "
    f"CPU={cpu_time:.2f}s [{cpu_eff:.0%}], nodes={nodes}, sims={sims}, depth={depth})"
)
```

This creates a clear audit trail in game logs, making it easy to correlate move quality with computational effort.

## Improvements Made (January 2026)

### 1. Fixed E12 G Evaluation Bug

**Problem:** MCTS was choosing E12 G, causing -1.5 point score collapses.

**Root Cause:** E12 is in both OPP_EDGES and HUB_EDGES. The evaluation was double-counting: -0.2 (OPP) + 0.4 (HUB) = +0.2 net, when it should be strongly negative.

**Fix:**
- Added priority system to avoid double-counting
- Added -0.8 penalty specifically for E12 G
- E12 is now processed only as OPP_EDGE

### 2. Fixed E2 G Evaluation Bug

**Problem:** Move 4 (first MCTS move) consistently chose E2 G, causing -2.6 point collapses.

**Root Cause:** E2 was scored as +0.3 (hub control), but quantum adjudication shows it's harmful.

**Fix:**
- Added -0.5 penalty for E2 G
- Changed hub edge rollout prior from 70% green to 25% green

### 3. Added Compute Effort Diagnostics

**Problem:** No visibility into whether MATLAB was actually doing significant computation.

**Fix:** Added tracking for CPU time, nodes expanded, simulations, tree depth, and memory usage using MATLAB's built-in `cputime` and `memory` functions.

### 4. Fixed Turn-Based State Reading

**Problem:** Occasional errors where MCTS would calculate a move for an edge, but the edge was already taken when we tried to click it.

**Initial Misdiagnosis:** I initially thought this was a race condition - that the opponent was "stealing" edges while our MCTS was calculating. This led to complex retry logic and recalculation attempts.

**Key Realization:** The game is **turn-based**. When it's our turn, the opponent waits. They cannot move until we complete our move. This means true race conditions are impossible during our turn.

**Actual Root Cause:** The issue was **stale DOM state**, not race conditions. The sequence was:

```
1. Opponent finishes their move
2. Game signals "your turn" (detected via text on page)
3. We immediately read board state
4. But DOM hasn't fully updated with opponent's last move yet
5. We calculate move based on stale state
6. Move attempt fails because edge is already taken
```

The `is_our_turn()` check was returning True slightly before the DOM fully reflected the opponent's move. This is a timing issue with browser rendering, not game logic.

**Fix:** Added a 0.5s delay after turn change detection, before reading board state:

```python
while not self.is_game_over():
    if not self.is_our_turn():
        time.sleep(0.3)
        continue

    # Brief delay after turn change to ensure DOM is fully updated
    # This prevents reading stale state from before opponent's move
    time.sleep(0.5)

    state = self.read_board()  # Now guaranteed to be current
```

**Implications for Future Work:**

Since the game is turn-based, we have opportunities during the opponent's turn:
- Start MCTS search speculatively while opponent thinks
- Build search tree for likely opponent responses
- Refine search when opponent's actual move is revealed

This "ponder" mode could significantly improve move quality by using the opponent's ~20s thinking time productively.

## Results

### Before Fixes
- Score peaked at +2.95, collapsed to -2.6 at move 4
- MCTS frequently chose E2 G and E12 G
- Typical game: Loss by 0.5-1.5 points

### After Fixes
- Score drops are modest (~0.4-0.6 points)
- MCTS correctly avoids E2 G and E12 G
- Typical game: Draw or close loss (within 0.1-0.3 points)

## File Locations

| File | Purpose |
|------|---------|
| `snowdrop_tangled_agents/matlab/rl/TangledMCTS.m` | MCTS engine |
| `snowdrop_tangled_agents/matlab/rl/MCTSNode.m` | Tree node class |
| `snowdrop_tangled_agents/strategy/matlab_mcts_strategy.py` | Python wrapper |
| `~/.tangled/matlab_mcts_params.json` | Tunable parameters |
| `~/.tangled/game_stats.db` | Game statistics database |

## Roadmap for Future Work

### Short-Term Improvements

1. **Integrate Real Adjudicator in MCTS**
   - Bridge to Python's SimulatedAnnealingAdjudicator for terminal evaluation
   - Would give accurate scores instead of heuristic approximations
   - Challenge: Performance (adjudicator is slow)

2. **Opponent Move Prediction During Their Turn**
   - Start MCTS search while opponent is thinking
   - Build search tree for likely opponent responses
   - Refine when actual move is revealed

3. **Move-Specific Learning**
   - Current learning penalizes ALL moves in lost games
   - Should identify which specific moves caused problems
   - Use score delta per move, not just final outcome

### Medium-Term Improvements

4. **Neural Network Position Evaluation**
   - Train value network on game history
   - Replace heuristic evaluation with learned function
   - See plan file for Deep Learning Toolbox integration

5. **Opponent Modeling**
   - Cluster opponents by play style
   - Adapt strategy based on opponent tendencies
   - See plan file for Statistics Toolbox integration

6. **Opening Book Expansion**
   - Learn optimal responses to different opponent openings
   - Currently vulnerable when opponent takes E11 first

### Long-Term Goals

7. **Quantum-Aware Evaluation**
   - Understand WHY E2 G and E12 G are bad
   - Model quantum superposition effects
   - Predict adjudicator behavior from first principles

8. **Compiled MATLAB Packages**
   - Use MATLAB Compiler SDK to package as Python module
   - Eliminate need for MATLAB license at runtime
   - Faster startup, easier deployment

9. **Win Rate Target**
   - Current: ~10% win, ~50% draw, ~40% loss
   - Goal: >30% win rate against MCTS Melissa

## Configuration

### MCTSParams (matlab_mcts_params.json)

```json
{
  "iterations": 5000,
  "time_limit": 20.0,
  "exploration": 1.414,
  "prior_weight": 2.0,
  "opening_sequence": [9, 11, 10, 5, 13, 12],
  "opening_moves": 3,
  "losing_exploration_boost": 1.3,
  "winning_exploration_reduction": 0.8
}
```

### Running Games

```bash
# Single game with MATLAB MCTS
python play_tangled.py --strategy matlab_mcts --games 1

# With debug logging
python play_tangled.py --strategy matlab_mcts --games 1 --debug

# View statistics
python play_tangled.py --stats
```

## References

- [Tangled Game Rules](https://tangled-game.com/rules)
- [MCTS Algorithm](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)
- [Petersen Graph](https://en.wikipedia.org/wiki/Petersen_graph)
- [Quantum Annealing](https://en.wikipedia.org/wiki/Quantum_annealing)
