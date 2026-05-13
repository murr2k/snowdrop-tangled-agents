# Algorithm Analysis: Draw Ceiling vs AlphaQ Up

**Date:** 2026-05-12  
**Context:** 1,419+ games vs AlphaQ Up, 0 wins, consistent draws at +0.77–+0.82.  
**Question:** Are there fundamental errors or misconceptions in our algorithm keeping us from doing better than a draw?

---

## Architecture summary

`AlphaQExplorerStrategy` wraps MATLAB's `HybridTangledSolver`:

- **Move 1:** Forced opening (E7G by default) or Thompson/round-robin sampling across all 30 openings
- **Moves 2–8:** MATLAB `HybridTangledSolver` — alpha-beta minimax (depth 4) + MCTS with tabu rollouts + expanded LUT (~19M states for 0–3 grey edges) + REINFORCE edge-bias learning
- **Python MCTS** (`mcts_strategy.py`): fallback when MATLAB unavailable

---

## Finding 1: Wrong objective function (highest priority)

The MCTS terminal evaluator loads `terminal_scores.mat` — the **Schrödinger ground truth LUT**. But §6 of WAYPOINT_2026-05-07 establishes (95.8% confidence from 48 games / 18 states) that the server adjudicates with **simulated annealing**, not Schrödinger.

These adjudicators disagree on sign in meaningful cases:

| Terminal | Schrödinger | SA | Server (vs AlphaQ) |
|---|---:|---:|---|
| `PGPGGGPPGGPPGPG` | **+7.94** | **−8.81** | 6/6 LOSSES |
| `PGGGGPPGGPGGGPP` | ~0 | +0.779 | 162/162 DRAWS |

MCTS is backpropagating values from the wrong evaluator. The tree steers toward states that Schrödinger likes but SA penalizes — which is exactly what AlphaQ exploits.

`terminal_scores_sa.mat` already exists (generated 2026-05-07/08, 100K SA reads, 51.9 min runtime). It uses the **same field name** (`terminal_scores`) and same shape (32768×1 float32) as the Schrödinger file — drop-in compatible with `TangledMCTS.m`'s `loadLUT()`.

**The fix is parameterizing the LUT filename across 3 files:**

1. `TangledMCTS.m` — add `LUTFile` property + parameter to `loadLUT()`
2. `HybridTangledSolver.m` — accept `LUTFile` name/value arg, forward to `TangledMCTS`
3. `matlab_strategy.py` — add `lut_file` param to `HybridSolverStrategy.__init__`, embed in MATLAB eval string

**Important caveat:** `ExpandedLUT.m` (covers 0–3 grey edge states via minimax) was generated from the Schrödinger LUT. Switching TangledMCTS to SA while leaving ExpandedLUT on Schrödinger creates a mixed objective. Full consistency requires also regenerating `expanded_lut.mat` from SA scores. This is Phase 1b (see §Implementation Plan).

---

## Finding 2: Early-game MCTS is effectively random

At move 2 (13 grey edges), branching factor ~26, 6 plies remaining: game tree has ~26⁶ ≈ 300 million nodes. At 1,000 MCTS iterations we sample 1-in-300,000 paths — noise, not signal.

Heuristic priors and Progressive Bias help, but the search cannot see the consequence of a move 6 plies deep. **Moves 2–4 are functionally determined by the priors, not the search.** We spend 90 seconds computing a move that the prior alone would produce in milliseconds.

**Speed improvement:** Skip MCTS for moves 1–4 (≥11 grey edges). Use the priors directly (or a small fixed lookup) and concentrate compute on moves 5–8 where the tree depth is tractable. This yields 3–4× game speed improvement with no quality loss in the early game.

---

## Finding 3: Rollouts don't model AlphaQ

`_heuristic_action()` uses `compute_action_prior()` — a fixed empirical prior from 40+ game analysis. It has no knowledge of AlphaQ's actual policy. Every simulated game during MCTS is a fantasy game between "heuristic-us" and "heuristic-opponent," not between us and AlphaQ.

We have 1,419+ recorded AlphaQ games with full move histories in the stats DB. AlphaQ's responses are effectively deterministic from a given board state (per WAYPOINT). Using that empirical distribution to drive the opponent's moves in rollouts would make every simulation predictive of the real game.

**This is the largest addressable improvement.** Opponent-aware rollouts would also naturally explore the 9,339 oracle gaps — MCTS routed through AlphaQ's actual policy will reach previously-unobserved states.

---

## Finding 4: REINFORCE cannot escape Nash

The REINFORCE loop adjusts `edge_adjustments` after each game. Against a Nash-optimal opponent, this converges to drawing more efficiently — but cannot find a win that does not exist in the reachable game tree. REINFORCE is a hill-climber; if the hill's peak is a draw, that is where it goes.

---

## Finding 5: The ceiling is confirmed, but one path remains

WAYPOINT §8.4: Among the 110 oracle-reachable terminals, H2 (Nash within explored tree) is **mathematically confirmed**. The 16 marginal SA-win candidates (SA ≥ +1.09) in the reachable set have been tested 24+ times and produced 0 wins.

The one remaining path: **9,339 oracle gaps** — states where AlphaQ's response is empirically unknown. If AlphaQ has a non-Nash deviation in unexplored territory, we have not seen it because we have not reached those states. Opponent-aware rollouts (Finding 3) are the natural way to explore this territory.

---

## Prioritized implementation plan

| Priority | Change | Files | Expected impact |
|---|---|---|---|
| **1a** | Switch TangledMCTS terminal eval to `terminal_scores_sa.mat` | `TangledMCTS.m`, `HybridTangledSolver.m`, `matlab_strategy.py` | Aligns MCTS objective with server scoring |
| **1b** | Regenerate `expanded_lut.mat` from SA terminal scores | `generate_expanded_lut.m` (config), `ExpandedLUT.m` (optional param) | Removes mixed-objective inconsistency for near-terminal states |
| **2** | Opponent-aware rollouts using empirical AlphaQ move DB | `TangledMCTS.m` rollout logic, `matlab_strategy.py` | Makes tree values predictive rather than decorative; explores oracle gaps |
| **3** | Skip MCTS for moves 1–4; use prior-weighted selection | `matlab_strategy.py` or `HybridTangledSolver.m` | 3–4× faster per game; no quality loss early game |
| **4** | Concentrate boosted iterations on moves ≥10 colored (not just score >0.75) | `matlab_strategy.py:1834` | Better compute allocation late game |

---

## Honest assessment

The draw ceiling is AlphaQ's play quality, not ours. Items 1a/1b are the only changes that could theoretically break it — and only if there exists a reachable game line we have not yet discovered. Items 2–4 improve efficiency and coverage without strategic downside.

Even a perfect implementation of all four priorities may not produce a win. The value of the effort is (a) maximizing P(discovering a reachable win) via correct objective function and realistic rollouts, and (b) gathering that evidence efficiently.
