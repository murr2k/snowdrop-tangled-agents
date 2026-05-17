# Investigation Avenues — Post-Closure Assessment

**Date:** 2026-05-16  
**Context:** Win investigation formally closed after 1,563+ games vs AlphaQ (0 wins).  
This document catalogues all identified avenues for potential future investigation,
including those tried and ruled out, those partially explored, and genuinely novel angles.

See `docs/INVESTIGATION_ROADMAP.md` for execution status.

---

## Recommended Order of Investigation

| Priority | Investigation | Feasibility | Potential | Notes |
|----------|--------------|-------------|-----------|-------|
| 1 | Player 2 seat swap | ✅ Done | — | 0W/0D/10L; exposed oracle inconsistency |
| 2 | Spectral/MI analysis of AlphaQ policy | ~1 day | Diagnostic | Proven on Amara |
| 3 | Adjudicator parameter recovery | ✅ Done | — | anneal_time=1.85 ns, 83% accuracy |
| 4 | Exhaustive terminal state mapping | 9–17 days (parallel) | Definitive | Needs calibrated oracle (now available) |
| 5 | Tensor network simulation | 3–6 months | Highest | Rose recommendation |

---

## A. Tried and Explicitly Ruled Out

### A1. SA as Terminal Evaluator
SA exhibits systematic winner flips near the draw boundary (anti-correlation r = −0.396 with
website scores). SA LUT maps to website avg −1.3 to −2.5 in regions that should be positive.
**Verdict:** Ruled out.  
**Files:** `docs/EMPIRICAL_LUT_CONSTRUCTION_RESULTS.md`, `docs/SCHRODINGER_SOLVER_ANALYSIS.md`

### A2. Bayesian Optimization (BoTorch)
Assessed 2026-05-16. Game space is discrete, evaluation points are opponent-controlled, and
the outcome distribution is constant (all draws/losses → no GP signal).  
**Verdict:** Ruled out.  
**File:** `snowdrop_tangled_agents/botorch/ASSESSMENT.md`

### A3. RuVector Quantum Platform
QAOA/VQE solve the wrong problem (combinatorial optimization vs adversarial routing).
Petersen graph (32,768 terminal states) is trivially enumerable — no quantum speedup at this
scale.  
**Verdict:** Ruled out.  
**File:** `docs/RUVECTOR_APPLICABILITY_ASSESSMENT.md`

### A4. Pure MCTS Without LUT
Achieves 0% win rate against AlphaQ with no terminal guidance.
**Verdict:** Ruled out.

### A5. SchrLUT Threshold Theory
Reaching SchrLUT > 7 was hypothesised to predict wins. Empirically disproved: SchrLUT ∈
[7, 10) → 51 games, 0 wins, 42 draws, 9 losses. Local adjudicator (ε=0.0005,
anneal_time=40ns) predicts `winner=red` on ALL boards including confirmed losses.  
**Verdict:** Ruled out.  
**File:** `docs/next_priorities.md` (memory)

---

## B. Mentioned but Never Tried

### B1. Tensor Networks (MPS/DMRG) ★★★★★
Identified by Geordie Rose as the correct tool for quantum simulation beyond 12 vertices.
Scales O(χ³) vs O(2^n). Literature demonstrates simulation of 1,000-qubit QA (Luchnikov
et al.) and 127-qubit systems (Tindall et al.).  
**Why not done:** High implementation barrier; project previously focussed on tractable
split-operator method.  
**File:** `docs/ANALYSIS_OF_GEORDIE_ROSE_FEEDBACK.md` §8.2–8.4

### B2. Website Adjudicator Reverse-Engineering ★★★★☆
Fit a parameterised Schrödinger model to observed (terminal_board → website_score) pairs
from Melissa/Amara games (diverse terminal boards, unlike AlphaQ). Recover ε, anneal_time,
transverse field strength. Once known, oracle becomes genuinely predictive.  
**Data needed:** ~50,000 games vs Melissa/Amara for 30% terminal coverage.  
**File:** `docs/RUVECTOR_APPLICABILITY_ASSESSMENT.md` §3.2

### B3. AlphaZero Self-Play RL ★★★☆☆
Neural network (policy + value heads) + MCTS guided by NN + self-play training.  
**Estimated cost:** 42–92 days with GPU + 16-core.  
**File:** `docs/HIGH_COMPUTE_STRATEGY_OPTIONS.md` Option B

### B4. Evolutionary Strategy Optimisation ★★☆☆☆
Genetic algorithm over strategy parameters (MCTS constants, edge weights).  
**Estimated cost:** 60–80 days.  
**File:** `docs/HIGH_COMPUTE_STRATEGY_OPTIONS.md` Option C

### B5. GPU-Accelerated MCTS ★★★☆☆
20–50× speedup for 1M iterations/move. Requires CUDA kernel development.  
**File:** `docs/HIGH_COMPUTE_STRATEGY_OPTIONS.md` Option D

### B6. Opponent Policy Probing for Instabilities ★★★☆☆
Systematically navigate low-observation AlphaQ states to detect policy shifts. AlphaQ
was observed changing moves on states with ≤6 observations.  
**File:** `docs/RUVECTOR_APPLICABILITY_ASSESSMENT.md` §3.3

### B7. Margin Amplification Search ★★★☆☆
Start from near-miss draws (0 < E(s_T) ≤ ε); amplify margin by delaying symmetry breaking.  
**File:** `docs/PERSISTENT_EXPLOITABILITY_UNDER_EVALUATOR-POLICY_MISALIGNMENT.md` Phase 3

---

## C. Partially Tried, Not Fully Explored

### C1. Player 2 / Seat Swap ★★★★★ (Investigation 1) ✅ Complete
0W / 0D / 10L. All 10 games identical (deterministic oracle + deterministic AlphaQ → same
terminal board every game). Three P2 transposition bugs were found and fixed before running.
Primary value: exposed the oracle's internal inconsistency (−0.954 → +0.003 in one round),
motivating the Oracle Revision Project. **Status:** See `INVESTIGATION_ROADMAP.md`.

### C2. Spectral / MI Analysis of AlphaQ Policy ★★★★☆ (Investigation 2)
Applied once to Amara (PSD + autocorrelation analysis) — correctly predicted that opening
diversity would break equilibrium, leading to 83.3% win rate. Never applied to AlphaQ.
Computing mutual information between board state and AlphaQ's move would distinguish "true
Nash equilibrium" from "locally optimal in explored regions".  
**Cost:** ~1 day of analysis script writing.  
**File:** `docs/AMARA_STRATEGY_ANALYSIS.md`

### C3. Adjudicator Parameter Recovery ★★★★☆ (Investigation 3) ✅ Complete
Used 1061 (terminal_state, website_score) pairs from existing DB (no new games needed).
MATLAB split-operator grid search over anneal_time (5–20,000 ns) + fminbnd refinement.  
**Result:** anneal_time=1.85 ns, epsilon=0.0, R²=0.60, 83.1% win/draw/loss classification
accuracy. Calibrated terminal LUT rebuilt. Expanded LUT rebuild in progress.  
**Surprising finding:** Website uses an extremely short anneal_time (1.85 ns vs 40 ns
default) — system barely evolves before measurement.

### C4. Exhaustive Terminal State Mapping ★★★★☆ (Investigation 4)
~2,436 terminals observed (7.43% coverage). Target: 30% coverage (~50,000 games vs
Melissa/Amara). Rust `oracle-solver/` already built for route enumeration; underutilised.  
**Cost:** 9–17 days at 16-core parallelism.  
**File:** `docs/HIGH_COMPUTE_STRATEGY_OPTIONS.md` Option A

### C5. Adversarial Opening Library for AlphaQ ★★★☆☆
Exact methodology that achieved 83.3% wins vs Amara never applied to AlphaQ. E14P: 84.6%
vs Amara, E1G: 100% (single trial). AlphaQ is much stronger but approach is proven.  
**File:** `docs/AMARA_STRATEGY_ANALYSIS.md`

### C6. RL/MC Ensemble ★★★☆☆
Complete MATLAB implementation exists (`MCRollout.m`, `EnsemblePolicy.m`, etc.) but
never systematically trained against AlphaQ with curriculum learning.  
**File:** `docs/RL_MC_ENSEMBLE.md`

### C7. RL from Historical Database ★★☆☆☆
Full plan documented in `docs/TRAIN_FROM_DATABASE_PLAN.md`. Blocked: PPO is on-policy;
MATLAB's `trainFromData` only supports SAC/DQN/TD3. Fix: switch to SAC agent.

---

## D. Novel Angles from Codebase Analysis

### D1. Information-Theoretic Analysis ★★★★☆
Compute mutual information I(AlphaQ policy; board state). High MI = strong state-dependent
strategy. Low MI = degenerate/equilibrium. Definitively distinguishes "unbeatable" (Nash)
from "untested" (unexplored states). No existing analysis in codebase.

### D2. Active Learning / Uncertainty Sampling ★★★☆☆
Preferentially play states where AlphaQ's response entropy is highest. High-entropy states
suggest AlphaQ is near a decision boundary — potentially exploitable. Requires real-time
opponent model query.

### D3. Persistent Exploitability Under Evaluator Mismatch ★★★☆☆
700-line theoretical framework exists. If sign(E_SA) ≠ sign(E_quantum) for some states,
and AlphaQ was trained under E_quantum, those states may be exploitable via E_SA-optimal play.  
**File:** `docs/PERSISTENT_EXPLOITABILITY_UNDER_EVALUATOR-POLICY_MISALIGNMENT.md`

### D4. Reverse Annealing Schedule Exploration ★★☆☆☆
D-Wave supports reverse annealing (restart from known state). If adjudicator uses this,
different schedules might produce different outcomes for the same terminal board. Requires
D-Wave access.

---

## Notes on Feasibility

- Investigations 1–4 are executable with existing infrastructure.
- Investigation 5 (tensor networks) requires new implementation and is only worthwhile
  after Investigation 3 recovers the website's quantum parameters.
- Wins against AlphaQ require either: a full Schrödinger minimax oracle at correct params,
  knowledge of AlphaQ's specific policy weaknesses, or AlphaQ changing its strategy.
