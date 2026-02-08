# Oracle Solver: Deterministic Route Enumeration Against a Quantum-Trained Agent

**Author:** Murray Kopit  
**Date:** February 2026  
**Status:** Live trial complete; LUT mismatch confirmed
**Related:**
[ALPHAQ_ORACLE_STRATEGY.md](ALPHAQ_ORACLE_STRATEGY.md),
[PERSISTENT_EXPLOITABILITY_UNDER_EVALUATOR-POLICY_MISALIGNMENT.md](PERSISTENT_EXPLOITABILITY_UNDER_EVALUATOR-POLICY_MISALIGNMENT.md),
[SCORE_OUTCOME_DISCREPANCY.md](SCORE_OUTCOME_DISCREPANCY.md),
[ALPHAQ_STRATEGY.md](ALPHAQ_STRATEGY.md)

---

## 1  Purpose

This document describes the design, implementation, and results of a computational experiment that searches for deterministic winning strategies against AlphaQ, a reinforcement learning agent trained with access to quantum computation, in the Tangled game.

The core question is:

> **Can a classical agent, armed only with historical game data and the ground-truth terminal evaluation, find a reproducible move sequence that forces a win against a quantum-trained but stationary opponent?**

This question has implications beyond the game itself. If the answer is yes, it demonstrates that a policy optimised under one evaluation function (even one derived from quantum hardware) can harbour exploitable blind spots when the adjudication function differs from the training signal—a failure mode we formalise as *persistent exploitability under evaluator–policy misalignment*.

---

## 2  Background and Motivation

### 2.1  The Tangled Game

Tangled is a two-player, perfect-information, turn-based game played on the Petersen graph (10 vertices, 15 edges). Players alternate turns colouring edges either Green (ferromagnetic coupling, J = −1) or Purple (antiferromagnetic coupling, J = +1). The game terminates when all 15 edges are coloured. The resulting edge-colouring defines an Ising model Hamiltonian on the graph. The terminal state is adjudicated by computing the ground-state energy of this Hamiltonian; the player whose colouring strategy produces a lower-energy configuration wins.

Key parameters:

| Parameter | Value |
|-----------|-------|
| Graph | Petersen (10 vertices, 15 edges) |
| Terminal states | 2^15 = 32,768 |
| Draw threshold (ε) | 0.0005 |
| Adjudicator | D-Wave quantum hardware (enumerated, stored as lookup table) |
| Moves per game | 15 (≈8 per player) |

### 2.2  AlphaQ

AlphaQ is the primary competitive agent on the Tangled platform. It was trained using reinforcement learning with access to quantum computation during the training process. At inference time, AlphaQ's policy is **fixed**: it does not learn online, does not adapt to opponents, and does not condition on game history across matches. These properties were confirmed directly by Geordie Rose (platform creator):

> *"The agent does not learn on the job, its performance is fixed."*

### 2.3  Prior Results

Over 1,376 games against AlphaQ across multiple strategy iterations (MCTS with 5K–500K iterations, hybrid minimax+MCTS solvers, Thompson sampling over openings, opponent modelling), the classical agent achieved **zero wins**. The best outcomes were consistent draws. Analysis of these games revealed three critical observations:

1. **AlphaQ is highly predictable.** Of 202 distinct board states observed two or more times in the opponent history, 198 (98.0%) produce a fully deterministic response. The average top-response probability across all observed states is 99.2%.

2. **The explored state space is small.** Only 86 distinct terminal states were reached across 1,376 games, out of 32,768 possible Petersen graph colourings (0.26% coverage). AlphaQ's determinism funnels play into a narrow set of endpoints.

3. **The evaluation function used during search is misaligned with the adjudicator.** The simulated annealing (SA) evaluator used by the classical agent's MCTS assigns positive scores to terminal states that the D-Wave ground-truth adjudicator classifies as losses. Of 204 observed losses, 48 had positive SA scores. This means the search is optimising toward states the server declares as losses—a fundamental mismatch.

### 2.4  Theoretical Basis

The theoretical framework for this exploit is formalised in *Proposition 1: Persistent Exploitability Under Evaluator–Policy Misalignment* (see companion document). The key result is:

> In a finite, deterministic, turn-based game with a fixed opponent policy and fixed terminal evaluator, the existence of a terminal state with value exceeding the draw threshold implies the existence of a deterministic winning route computable by path enumeration over the induced state graph.

Against a deterministic opponent, the game tree collapses: opponent turns have no branching. What remains is a directed acyclic graph where only the classical player's choices create branches. Finding a win reduces from a game-theoretic problem to a **routing problem**—enumerating paths in a graph and checking terminal values.

---

## 3  Experimental Design

### 3.1  Conditions

The experiment operates under the following confirmed conditions:

1. **Ground truth adjudication.** All 32,768 terminal states of the Petersen graph were evaluated on D-Wave quantum hardware and stored in a lookup table (LUT). This LUT is the sole arbiter of win/draw/loss outcomes. A terminal state with LUT score > ε = 0.0005 is a win for Player 1; score < −ε is a loss; |score| ≤ ε is a draw.

2. **Stationary opponent.** AlphaQ's policy does not change between games or during games. Any winning route discovered is replayable indefinitely.

3. **Perfect information.** Both players observe the full board state at all times.

4. **Deterministic transitions.** Colouring an edge is irreversible and produces a unique successor state.

### 3.2  Assumptions

1. **Oracle fidelity.** The opponent model (oracle) is constructed from historical game data. We assume that AlphaQ's responses to previously-observed board states will not change. For states observed only once, we treat the observed response as deterministic but assign lower confidence. For states never observed, the oracle has no prediction (a "gap").

2. **LUT accuracy.** The exported LUT binary matches the server's adjudication. We verified this by checking known terminal states against their observed server outcomes.

3. **Turn order.** The classical player moves first (Player 1). AlphaQ is Player 2.

### 3.3  Architecture

The solver is implemented as a standalone Rust binary (`oracle-solver`) that operates in four phases:

```
┌──────────────────────────────────────────────────────────┐
│                    ORACLE SOLVER                         │
│                                                          │
│  Phase 1: BUILD ORACLE                                   │
│    Input: game_stats.db (SQLite, 1421 AlphaQ games)      │
│    Output: HashMap<BoardState, OracleEntry>              │
│    Method: Query opponent_history table for all          │
│            (board_state, response) pairs; compute        │
│            primary response and confidence per state     │
│                                                          │
│  Phase 2: LOAD TERMINAL LUT                              │
│    Input: terminal_scores.bin (32768 × f32)              │
│    Source: D-Wave hardware enumeration via MATLAB .mat   │
│    Output: Score lookup: TerminalIndex → f32             │
│                                                          │
│  Phase 3: ENUMERATE GAME TREE                            │
│    Method: Parallel DFS across 30 openings (rayon)       │
│    Our turns: branch over all legal moves                │
│    Opponent turns: follow oracle prediction              │
│      - confidence ≥ 0.9 → single deterministic branch    │
│      - confidence < 0.9 → branch top 3 responses         │
│      - oracle gap → branch top 3 global fallback moves   │
│    Pruning: drop paths with cumulative confidence < 0.5  │
│    Output: set of reachable terminal states + routes     │
│                                                          │
│  Phase 4: SCORE AND RANK                                 │
│    For each terminal: LUT score → classify win/draw/loss │
│    Sort winning routes by score, confidence, gap count   │
│    Output: JSON with winning routes and near-misses      │
└──────────────────────────────────────────────────────────┘
```

#### State Representation

Board states are encoded as a 32-bit integer using 2 bits per edge (00 = grey/uncoloured, 01 = Green, 10 = Purple), allowing compact hashing and O(1) edge queries. Terminal states are further compressed to a 16-bit index (1 bit per edge, G=1, P=0) for LUT lookup.

#### Oracle Construction

The oracle is built from the `opponent_history` table in the game statistics database. For each distinct board state observed before an AlphaQ move, the table records which edge AlphaQ coloured and in what colour. Grouping by board state and counting response frequencies yields:

- **Primary response**: the most frequent (edge, colour) pair
- **Confidence**: primary count / total observations
- **Alternatives**: remaining responses with their probabilities

A global fallback distribution (most common moves across all states) handles oracle gaps.

#### Game Tree Enumeration

The DFS proceeds from each of 30 possible openings (15 edges × 2 colours). At each node:

- **Our turn**: generate all legal moves (grey edges × {G, P}) and recurse into each.
- **Opponent turn**: consult the oracle.
  - If the oracle has high confidence (≥ 0.9), follow the single predicted response—no branching.
  - If confidence is lower, branch over the top 3 predicted responses.
  - If the state is unseen (oracle gap), branch over the top 3 moves from the global fallback distribution and halve the path confidence score.

Paths whose cumulative confidence drops below 0.5 are pruned. The 30 openings are processed in parallel using the Rayon work-stealing thread pool.

#### Why Rust

The game tree enumeration is compute-intensive: with a branching factor of ~10 on our turns (5 grey edges × 2 colours at midgame) over ~8 decision points, the search space is on the order of 10^7 nodes per opening. Rust provides:

- Zero-cost abstractions for the compact state representation
- Efficient hash maps for the oracle and transposition tables
- Data-parallel enumeration via Rayon with no GIL or runtime overhead
- Sub-second total execution time for the full enumeration

---

## 4  Results

### 4.1  Oracle Statistics

| Metric | Value |
|--------|-------|
| Board states observed (≥1 occurrence) | 202 |
| Fully deterministic states (confidence ≥ 0.9) | 198 (98.0%) |
| Average confidence across all states | 99.2% |
| Source games | 1,421 |
| Source opponent moves | 4,903 |

AlphaQ's policy is even more deterministic than initially estimated from the earlier 457-state analysis (which reported 72.2% fully deterministic). With the opponent_history table providing exact board-state-before context, the refined oracle shows 98% determinism.

### 4.2  Enumeration Statistics

| Metric | Value |
|--------|-------|
| Total nodes visited | 212,554 |
| Unique terminal states reached | 76 |
| Oracle hits (known states) | 1,477 |
| Oracle misses (unseen states) | 195,955 |
| Execution time | < 1 second |

The 76 terminal states reached represent a modest expansion beyond the 86 historically observed. The high oracle miss rate (99.2%) reflects that most intermediate states in the tree were not previously encountered—the oracle provides coverage primarily along well-trodden game paths.

### 4.3  Terminal State Classification

| Outcome | Count | Percentage |
|---------|-------|-----------|
| **Win** (score > ε) | **48** | 63.2% |
| Near-miss (0 < score ≤ ε) | 1 | 1.3% |
| Draw (|score| ≤ ε) | 0 | 0.0% |
| Loss (score < −ε) | 27 | 35.5% |

### 4.4  Top Winning Routes

The following routes have **zero oracle gaps** and **100% path confidence**, meaning every opponent response along the path has been observed in prior games and is fully deterministic. These routes are reproducible.

| Rank | Terminal State | LUT Score | Margin / ε | Opening | Our 8 Moves |
|------|---------------|-----------|-----------|---------|-------------|
| 1 | `PGPGGGPPGGPPGPG` | +7.943 | 15,885× | E1G | E1G→E2P→E3G→E6P→E9G→E10P→E12G→E14G |
| 2 | `PPPGPPGGPPGPGPP` | +3.977 | 7,953× | E0P | E0P→E4P→E3G→E6G→E13P→E10G→E8P→E14P |
| 3 | `PGPGGGPPPGPGGPP` | +3.974 | 7,947× | E1G | E1G→E2P→E3G→E7P→E9G→E13P→E11G→E14P |
| 5 | `PGPGGPPPPPPGGPG` | +3.969 | 7,938× | E1G | E1G→E3G→E14G→E4G→E11G→E13P→E7P→E12G |
| 6 | `PGPGGGPPPGPGGPG` | +3.968 | 7,934× | E1G | E1G→E2P→E3G→E7P→E9G→E13P→E11G→E14G |

Route 1 is exceptional: its LUT score of +7.943 exceeds the draw threshold by a factor of 15,885. This is not a marginal win—it represents a decisive quantum-mechanical advantage in the Ising ground state.

### 4.5  The Near-Miss

One terminal state (`PGPGGGPGPGPPPGP`, score = +0.000107) falls just below the win threshold. Its margin is −0.000393, placing it firmly in draw territory. This state exemplifies the "thin margin regime" that characterises the Petersen graph: small perturbations in edge colouring can shift a terminal state across the ε boundary.

---

## 5  Interpretation

### 5.1  Why These Routes Were Never Found by MCTS

The classical MCTS used simulated annealing (SA) as its terminal evaluation function. SA is a stochastic approximation of the true quantum ground state. The score-outcome discrepancy analysis showed that SA assigns positive scores to many states the D-Wave adjudicator classifies as losses. Consequently, MCTS was actively steering play *away* from the winning terminal states and *toward* states that appeared favourable under SA but were adjudicated as draws or losses.

The oracle solver bypasses SA entirely. It uses the D-Wave ground-truth LUT directly, evaluating terminal states by the same function the server uses for adjudication. This eliminates the evaluator mismatch that blinded MCTS.

### 5.2  Why AlphaQ Cannot Adapt

AlphaQ was trained to optimality under its own internal evaluation function. At inference time, it does not observe the adjudicator's output, does not update its weights, and does not condition on opponent history. Even after repeated losses along the same route, no corrective gradient exists within AlphaQ's objective. The exploit is therefore **persistent**: it will produce the same result in every replay, indefinitely.

This is the empirical manifestation of Proposition 1 (Persistent Exploitability). AlphaQ occupies a local optimum in its own evaluation landscape that is a global sub-optimum under the true adjudicator. Without access to the adjudicator's signal during inference, it cannot escape.

### 5.3  Relationship to Quantum Advantage

Geordie Rose's research demonstrates that access to quantum computation during training can improve agent performance. The oracle solver's results do not contradict this. Rather, they show that **quantum advantage in training does not guarantee robustness to evaluator mismatch at deployment**. AlphaQ was trained with quantum resources, but its policy was crystallised into a classical deterministic function. A classical adversary with exact knowledge of that function and access to the ground-truth evaluator can exploit the resulting blind spots.

This suggests that true quantum advantage in adversarial settings requires quantum-consistent evaluation not only during training but also during inference—or alternatively, a policy that retains sufficient stochasticity to avoid deterministic exploitation.

---

## 6  Expected Pre-Trial Outcome

Based on the solver's output, we predict:

1. **Route 1 will produce a win when played live.** The terminal state `PGPGGGPPGGPPGPG` has a LUT score of +7.943, far exceeding ε = 0.0005. Every opponent move along the 15-move path has been observed with 100% consistency in prior games. There are zero oracle gaps. If AlphaQ's policy has not changed since the data was collected, the game will follow the predicted path exactly, and the server will adjudicate a win.

2. **Multiple independent routes will succeed.** The solver identified 48 winning routes. Even if some routes fail due to undetected policy changes or oracle inaccuracies, the redundancy provides robustness. Routes through different openings (E0P, E1G, E3G, E11G, E12P) diversify the attack surface.

3. **The near-miss will remain a draw.** The single near-miss state (score = +0.000107) falls 0.000393 below the threshold. We do not expect this to cross the boundary under any reasonable perturbation.

4. **Routes with oracle gaps (confidence < 1.0) are less reliable.** Routes 4 and 8, which contain one oracle gap each, may diverge from prediction at the unseen state. These should be treated as secondary candidates, played only after gap-free routes are verified.

### Falsification Conditions

The prediction is falsified if:

- AlphaQ plays a different move than the oracle predicts at any point along Route 1, causing the game to diverge from the predicted path. This would indicate either a policy update or insufficient oracle coverage.
- The server adjudicates the predicted terminal state as a draw or loss despite the LUT score exceeding ε. This would indicate a discrepancy between the exported LUT and the server's actual adjudication table.
- The game does not reach the predicted terminal state due to an error in the turn-order model (e.g., AlphaQ moves first rather than second).

---

## 7  Experimental Protocol

### 7.1  Pre-Trial Verification

Before live play, verify:

1. The exported LUT binary matches the server's adjudication for at least 5 known terminal states (states with prior observed server outcomes).
2. The oracle's predictions match AlphaQ's actual responses for at least 10 previously-observed board states.
3. The turn order assumption (we move first) is correct for the target game configuration.

### 7.2  Trial Execution

1. Select Route 1 (highest score, zero gaps, 100% confidence).
2. Play the prescribed opening move (E1G: edge 1, Green).
3. After AlphaQ responds, verify the response matches the oracle's prediction (E0P: edge 0, Purple). If it does not, abort this route and fall back to Route 2.
4. Continue playing prescribed moves, verifying each AlphaQ response.
5. Record the terminal state and server adjudication.

### 7.3  Post-Trial Analysis

- If win: record the exact game transcript; verify terminal state matches prediction; attempt to replay the same route to confirm reproducibility.
- If draw/loss: compare actual game path to predicted path; identify the divergence point; update the oracle with the new observation; re-run the solver.

---

## 8  Implementation Reference

### 8.1  Software

| Component | Language | Location |
|-----------|----------|----------|
| Oracle solver | Rust | `oracle-solver/` |
| Game database | SQLite | `~/.tangled/game_stats.db` |
| Terminal LUT | Binary (f32) | `oracle-solver/data/terminal_scores.bin` |
| LUT source | MATLAB v7.3 .mat | `snowdrop_tangled_agents/matlab/rl/data/terminal_scores.mat` |
| Solver output | JSON | `oracle-solver/output/oracle_routes.json` |

### 8.2  Reproduction

```bash
# 1. Export LUT (one-time, requires h5py)
python -c "import h5py, numpy as np; \
  f = h5py.File('snowdrop_tangled_agents/matlab/rl/data/terminal_scores.mat', 'r'); \
  np.array(f['terminal_scores']).flatten().astype(np.float32).tofile('oracle-solver/data/terminal_scores.bin')"

# 2. Build solver
cd oracle-solver && cargo build --release

# 3. Run
cargo run --release -- \
  --db-path ~/.tangled/game_stats.db \
  --lut-path data/terminal_scores.bin \
  --opponent alphaq

# 4. Examine output
cat output/oracle_routes.json | python -m json.tool
```

### 8.3  Solver Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--confidence` | 0.9 | Oracle confidence threshold for deterministic branching |
| `--max-branch` | 3 | Max branches on uncertain oracle states |
| `--max-gap-branch` | 3 | Max branches on oracle gaps (unseen states) |
| `--min-path-confidence` | 0.5 | Minimum cumulative path confidence to continue exploring |

---

## 9  Limitations

1. **Oracle coverage.** The oracle covers 202 of the many thousands of possible board states. Most intermediate states in the enumeration tree are unseen. The solver mitigates this with confidence pruning, but routes passing through unseen states are inherently less reliable. **Confirmed in trial:** Route 1 failed at a state with only 6 observations.

2. **SA-derived LUT.** The terminal_scores.mat file used in this experiment was generated by the simulated annealing adjudicator, not the D-Wave hardware directly. **Confirmed in trial:** the SA-derived LUT predicted +1.985 (win) for Route 7's terminal state; the website returned +0.03 (draw). This is the same evaluator mismatch that undermined MCTS—now applied to the solver itself. This limitation proved to be the experiment's primary failure mode.

3. **Turn order.** The solver assumes the classical player moves first. This was confirmed correct during trials.

4. **Policy staleness.** AlphaQ may exhibit non-stationarity at sparsely-observed states. **Confirmed in trial:** the state `PGPGGGPPGGPPG--` showed 6/6 E13P historically but 0/4 E13P live. Well-observed states (87+ games) showed perfect consistency.

---

## 10  Live Trial Results

### 10.1  Trial Protocol

Ten live games were played against AlphaQ on the Tangled platform using the `OracleRouteStrategy`, a Python strategy class that plays oracle route moves verbatim and falls back to MCTS upon opponent deviation. Two routes were tested:

- **Route 1** (index 0): target `PGPGGGPPGGPPGPG`, LUT score +7.943
- **Route 7** (index 6): target `PGGPGGPPPGPPPPP`, LUT score +1.985

### 10.2  Route 1 Trials (Games 1–4)

All four attempts with Route 1 failed at the same point: **turn 14 (opponent's 7th move)**. The oracle predicted E13P with 100% confidence, but AlphaQ played E13G in every trial.

| Game | Moves Completed | Deviation Point | Website Score | Result |
|------|----------------|-----------------|---------------|--------|
| 1 | 7/8 | Turn 14: E13G (expected P) | −1.078 | Loss |
| 2 | 7/8 | Turn 14: E13G (expected P) | −1.053 | Loss |
| 3 | 7/8 | Turn 14: E13G (expected P) | −1.114 | Loss |
| 4 | 7/8 | Turn 14: E13G (expected P) | −1.060 | Loss |

**Cause**: The state `PGPGGGPPGGPPG--` (13 edges coloured, before opponent's turn 14) had been observed only 6 times in the historical data, all showing E13P. AlphaQ's true policy at this state is apparently E13G, or at minimum non-deterministic. The 100% confidence was an artefact of small sample size.

This deviation is catastrophic for the route. The original target scored +7.943 (win) in the LUT. The deviated terminal state with E13G scores −5.975 (decisive loss). AlphaQ's choice at this state flips the outcome from a large win to a large loss—suggesting this is exactly the kind of state AlphaQ was trained to navigate.

### 10.3  Route 7 Trials (Games 5–10)

Route 7 was selected after analysing all 48 winning routes against the current database. Its distinguishing property is **depth of observation**: every opponent move along the route passes through states with 87–130 observations, all at 100% confidence.

| Turn | Opponent Move | Observations | Confidence |
|------|--------------|-------------|------------|
| 2 | E0P | 87 | 100% |
| 4 | E3P | 119 | 100% |
| 6 | E5G | 130 | 100% |
| 8 | E7P | 103 | 100% |
| 10 | E10P | 110 | 100% |
| 12 | E8P | 107 | 100% |
| 14 | E11P | 114 | 100% |

All six trials completed the full route with zero deviations:

| Game | Moves Completed | Website Score | Result |
|------|----------------|---------------|--------|
| 5 | 8/8 | +0.031 | Draw |
| 6 | 8/8 | +0.031 | Draw |
| 7 | 8/8 | +0.037 | Draw |
| 8 | 8/8 | +0.015 | Draw |
| 9 | 8/8 | +0.019 | Draw |
| 10 | 8/8 | +0.049 | Draw |

Route 7's target terminal state `PGGPGGPPPGPPPPP` was reached in all six games. Historical database analysis confirms this is the **second most commonly reached terminal state** against AlphaQ (149 occurrences across 1,431 games), always resulting in a draw.

### 10.4  Score Comparison

| Metric | LUT Score | SA Prediction | Website Score |
|--------|-----------|---------------|---------------|
| Route 1 target | +7.943 | — | (not reached) |
| Route 7 target | +1.985 | +1.578 | +0.03 ± 0.01 |

The LUT predicted a decisive win (+1.985); the website adjudicator returned a near-zero draw (+0.03). The low variance in website scores across six trials (σ ≈ 0.01) indicates the website's adjudicator is approximately deterministic—consistent with a lookup table rather than a stochastic quantum computation. This 66× overestimate is consistent with the SA positive bias documented in the Score–Outcome Discrepancy report (mean SA overestimate: +0.237, with qualitative outcome misclassifications at SA scores below +2).

---

## 11  Error Discussion

### 11.1  Falsification of Pre-Trial Predictions

Both falsification conditions identified in Section 6 were triggered:

1. **Route 1: opponent deviation.** AlphaQ deviated from the oracle at a sparsely-observed state (6 observations). The 100% historical confidence was an overfitting artefact—6 consistent observations do not constitute a reliable policy estimate. Route 7, with 87+ observations per state, proved robust.

2. **Route 7: LUT-adjudicator mismatch.** The terminal state was reached exactly as predicted, but the server adjudicated it as a draw (+0.03) despite the LUT scoring it as a win (+1.985). The LUT does not reflect the server's actual evaluation function.

### 11.2  The LUT Is SA-Derived, Not D-Wave Ground Truth

The most consequential error is that the `terminal_scores.mat` file—described in the codebase as a terminal evaluation lookup table and assumed to represent D-Wave ground truth—was **generated by the simulated annealing adjudicator in MATLAB**, not by D-Wave quantum hardware.

This was identified as a risk in Section 9.2 ("Limitations") but treated as unlikely. The live trial has confirmed it as the primary failure mode. The SA evaluator systematically overestimates the score of certain terminal states, producing false-positive "wins" that the actual game adjudicator classifies as draws or losses.

The magnitude of the error is striking: an SA score of +1.985 maps to a website score of +0.03—a 66× overestimate. This is not noise or approximation error; it is a qualitative misclassification of the outcome category.

### 11.3  The Evaluator Mismatch Is Recursive

This result reveals an ironic symmetry. The original hypothesis was that AlphaQ's policy harbours exploitable blind spots because it was trained under an approximate evaluator that diverges from the ground-truth adjudicator. The oracle solver was designed to exploit these blind spots by using the "ground-truth" LUT directly.

But the LUT itself is an approximation (SA-derived). The solver therefore suffers from the **same class of evaluator mismatch** it was designed to exploit. The exploit cannot succeed when the exploit tool and the target system share the same fundamental limitation.

Formally: let $\hat{V}$ be the SA evaluation function, $V^*$ be the true adjudicator, and $\pi_Q$ be AlphaQ's policy.
- AlphaQ was trained to maximise $\hat{V}$, creating blind spots where $\hat{V}(s) \neq V^*(s)$.
- The oracle solver searched for states where the LUT (also $\hat{V}$) predicts a win.
- Because the LUT and AlphaQ's training signal are derived from the same function, the solver's "winning" states are likely states that AlphaQ was already trained to handle well—they are blind spots of $\hat{V}$ shared by both systems.

### 11.4  Oracle Confidence and Sample Size

Route 1's failure demonstrates that oracle confidence without sample size context is dangerously misleading. A state observed once with a single response has confidence = 1.0 but conveys almost no information about AlphaQ's true policy distribution.

The corrected analysis after the trial:

| Route 1 state | Historical obs | Confidence | Live behaviour |
|---------------|---------------|-----------|---------------|
| `PGPGGGPPGGPPG--` | 6 (all E13P) | 100% | 4/4 E13G |

AlphaQ apparently changed its response at this state, or the 6 historical observations were insufficient to capture a stochastic policy. Either way, confidence metrics must be weighted by observation count. Route 7's robustness (87–130 observations) versus Route 1's fragility (6 observations at the failure point) confirms that **minimum observation depth** is a better route selection criterion than LUT score.

### 11.5  AlphaQ's Win Immunity

A broader database analysis reveals that across 1,431 games and 91 unique terminal states reached against AlphaQ, **zero games have been won**. The most favourable terminal state (reached 162 times) scores +0.877 on the website—still classified as a draw.

This raises the question of whether it is possible to win against AlphaQ at all under the current adjudicator, or whether AlphaQ's training has converged to a policy that avoids all losing terminal states regardless of opponent play. Against Melissa (a weaker agent), 329 wins have been recorded with scores up to +15.365, confirming that the adjudicator does produce wins in general.

### 11.6  Correlation with Score–Outcome Discrepancy Report

An earlier analysis ([SCORE_OUTCOME_DISCREPANCY.md](SCORE_OUTCOME_DISCREPANCY.md)), conducted before the oracle solver was designed, identified every element of the failure we observed. The correlation is exact and worth documenting explicitly, because the oracle solver was built despite this prior work being available.

**The score is not the outcome.** The discrepancy report demonstrated that 59% of losses show a positive displayed score, and that the declared winner and the displayed score are produced by independent processes. The oracle solver assumed `LUT score > ε → win`, a threshold model that the discrepancy report had already shown to be false. Wins require a displayed score above approximately +2.0, not +0.0005.

**SA overestimates systematically.** The report quantified SA's positive bias: it overestimates the website score 64.9% of the time, with a mean bias of +0.237. At SA scores of +2 to +5, the actual win rate is only ~71%, not the near-certainty implied by the oracle solver's classification. Route 7's SA LUT score of +1.985 falls just below the +2 threshold where wins become reliable against Melissa—and that threshold does not apply to AlphaQ at all.

**The calibration is opponent-specific.** The discrepancy report's Run 47 tested the Melissa-calibrated solver against AlphaQ and recorded 0 wins in 60 games. It concluded explicitly: *"To calibrate against AlphaQ Up, we need wins. The current calibration curve should not be trusted for this opponent."* The oracle solver ignored this finding by using the same SA-derived LUT regardless of opponent.

**Hypothesis B (quantum measurement) explains the live trial.** The report proposed that the displayed score is an expectation value ⟨H⟩ while the winner is determined by a single quantum measurement. Route 7's consistent website score of +0.03 with low variance (σ ≈ 0.01) fits this model: the expectation value is stable and slightly positive, but too small to produce a winning measurement outcome. The six trials effectively performed the distinguishing test proposed in the report—the score is repeatable and near-deterministic, yet the outcome is always a draw, consistent with the measurement model where P(win) at score +0.03 is effectively zero.

**The P(win) step function quantifies the gap.** The discrepancy report's calibration curve shows:

| Website Score | P(win) vs Melissa |
|---------------|-------------------|
| [0, +0.5) | 5.9% |
| [+0.5, +1) | 43.2% |
| [+1, +2) | 77.4% |
| [+2, +5) | 98.5% |

Route 7's terminal state scores +0.03 on the website. Even against Melissa (the weaker opponent), this corresponds to P(win) ≈ 5.9%. Against AlphaQ, where no wins have ever been recorded at any score, the probability is effectively zero. The oracle solver would need to reach terminal states scoring above +2 on the website to have a realistic chance of winning—a constraint that the SA-derived LUT cannot enforce because it does not predict website scores.

**The recursive mismatch was foreseeable.** Section 11.3 described the ironic symmetry of building an exploit tool that carries the same evaluator mismatch it targets. The discrepancy report made this explicit in its Finding 3: SA is a noisier signal than the website score itself, and the two can qualitatively disagree. The oracle solver amplified this error by treating SA scores as ground truth rather than as one more approximation.

The lesson is not that the discrepancy report was ignored—it was written before the oracle solver was conceived—but that its findings should have been incorporated as hard constraints on the solver's design. Specifically, the solver should have been gated by the requirement that no route can be classified as "winning" unless its terminal state has been independently observed to produce a website score above +2. This would have reduced the 48 "winning" routes to zero and prevented the trial from proceeding under false expectations.

---

## 12  What Would Be Needed

To make the oracle solver approach viable, the following would be required:

1. **A website-calibrated scoring function.** The SA-derived LUT must be replaced with a function that predicts P(win) against AlphaQ, not raw SA scores. The discrepancy report's calibration curve provides a template for Melissa, but AlphaQ requires its own—and building it requires achieving at least some wins to fit the upper tail. Possible approaches:
   - Requesting the actual D-Wave ground-truth adjudicator scores from the platform operators.
   - Building an empirical model from the ~91 terminal states reached against AlphaQ, mapping terminal state → website score → observed outcome. The current data (all draws/losses) constrains the model from above but cannot identify the win threshold.
   - Playing against AlphaQ with diverse strategies to expand the set of reached terminal states, specifically targeting states with website scores above +1 to explore the upper tail of the P(win) curve.

2. **Terminal state targeting above the win threshold.** The discrepancy report established that wins require website scores above approximately +2. The solver must target terminal states that achieve this on the website, not states with high SA scores. Since SA and website scores are weakly correlated in the [0, +2) range, the solver cannot use SA as a proxy for this purpose.

3. **Observation-weighted route selection.** The solver should rank routes not only by predicted score but by the minimum number of observations at any opponent decision point along the route. Routes with min_obs < 20 should be treated as speculative.

4. **Multi-route adaptive play.** Rather than committing to a single route before the game, the strategy should maintain a portfolio of compatible routes and dynamically select based on the opponent's actual responses—switching to an alternative route when a deviation occurs rather than falling back to MCTS.

---

## 13  Conclusion

The oracle solver experiment validated two of its three core hypotheses and falsified the third:

**Validated:**

1. **Opponent determinism enables route-based exploitation.** Route 7 demonstrated that AlphaQ's policy is perfectly deterministic along well-observed paths (87–130 observations per state, 100% consistency across 6 consecutive live trials). The game tree collapse from exponential to polynomial is real and exploitable for reliable path selection.

2. **Historical data produces a high-fidelity oracle.** On paths with sufficient observation depth (>50 games per state), the oracle achieves 100% predictive accuracy. The oracle mechanism itself works.

**Falsified:**

3. **The LUT does not provide ground-truth evaluation.** The `terminal_scores.mat` file is SA-derived, not D-Wave ground truth. The SA score of +1.985 for Route 7's terminal state maps to a website score of +0.03 (draw), a qualitative misclassification. The 48 "winning" routes identified by the solver are not winning routes under the actual adjudicator. Without a correctly calibrated evaluation function, the solver cannot distinguish terminal states that produce wins from those that produce draws.

### 13.1  What Was Accomplished

Despite failing to achieve a win, the experiment produced several durable results:

- A Rust-based game tree enumerator that runs the full 30-opening search in under 1 second.
- An `OracleRouteStrategy` for live play that reliably follows prescribed routes and detects deviations.
- Empirical confirmation that AlphaQ is deterministic on deeply-observed paths.
- Quantification of the SA-to-website score divergence (66× overestimate on the tested terminal state).
- A principled framework for route selection based on observation depth rather than estimated score.

### 13.2  The Fundamental Lesson

The experiment's failure mode is itself a demonstration of evaluator–policy misalignment, applied reflexively. The solver was designed to exploit AlphaQ's blind spots arising from training under an approximate evaluator—but the solver's own evaluation function (the SA-derived LUT) suffered from the same approximation error. A tool built to exploit mismatch cannot succeed when it carries the same mismatch within itself.

This was not an unforeseeable failure. The Score–Outcome Discrepancy report had already established that (1) SA scores do not determine game outcomes, (2) the win threshold is at website score ≈ +2, not ε = 0.0005, (3) SA-to-website calibration does not transfer across opponents, and (4) zero wins have been achieved against AlphaQ at any score. Each of these findings, had it been treated as a design constraint, would have prevented the oracle solver from classifying any routes as "winning."

The path forward requires breaking the symmetry in one of two ways:
- **Obtain a better evaluation function.** A LUT derived from the actual game adjudicator—whether obtained from the platform operators or reverse-engineered from observed outcomes—would allow the solver to target genuinely winning terminal states. The discrepancy report's P(win) curve provides the template: terminal states must score above +2 on the website to produce reliable wins, and the SA score is not a useful proxy for this.
- **Expand the observed terminal state space against AlphaQ.** With only 91 unique terminal states reached across 1,431 games, and all of them draws or losses, the empirical P(win) curve against AlphaQ has no positive support. Reaching new terminal states—particularly those with website scores in the [+1, +3) range—would reveal whether the P(win) step function has a viable threshold against this opponent, or whether AlphaQ's policy has genuinely converged to a zero-loss equilibrium.
