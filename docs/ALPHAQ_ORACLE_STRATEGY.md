# AlphaQ Oracle Strategy

**Status:** Proposed
**Date:** February 2026
**Prerequisite:** Run 86 data (Phase 1 opening re-exploration)
**Related docs:**
[EXPERIMENT_OPENING_RE_EXPLORATION.md](EXPERIMENT_OPENING_RE_EXPLORATION.md),
[ALPHAQ_STRATEGY.md](ALPHAQ_STRATEGY.md),
[OPPONENT_MODELING.md](OPPONENT_MODELING.md),
[LUT_TERMINAL_EVALUATION.md](LUT_TERMINAL_EVALUATION.md),
[SCORE_OUTCOME_DISCREPANCY.md](SCORE_OUTCOME_DISCREPANCY.md)

---

## 1  Executive Summary

After 1,376 games and 0 wins against AlphaQ Up, a new strategy is needed.
The Phase 1 re-exploration (Run 86) with fixed 600x parallel MCTS rollouts
confirmed that brute-force search quality alone cannot break the draw/loss
barrier.  However, the data reveals three exploitable facts that together
form the basis of a winning strategy:

1. **AlphaQ is 94% predictable.**  Of 457 distinct board states observed
   two or more times, 72.2% produce a fully deterministic AlphaQ response.
   The average top-response probability across all states is 94%.

2. **The explored state space is tiny.**  Only 86 distinct terminal states
   have been reached across 1,376 games, out of 2^15 = 32,768 possible
   Petersen-graph terminal colorings (0.26% coverage).

3. **Our evaluation function is wrong.**  Terminal state `PPPPGGGPPPGGPPG`
   scores +0.86 under our SA evaluator but the server classifies it as a
   loss every time.  Terminal state `PGGGGPPGGPGGGPP` scores +0.78 and is
   always a draw.  SA score does not predict server adjudication.

The Oracle strategy exploits AlphaQ's predictability to turn a stochastic
search problem into a near-deterministic game tree that can be solved
offline, then targets unexplored terminal states that the server adjudicator
may evaluate as wins.

---

## 2  Evidence Base

### 2.1  AlphaQ Predictability

Source: 12,000+ opponent moves in `game_stats.db`.

```
States observed 2+ times:       457
Fully deterministic responses:   330 (72.2%)
Average top-response probability: 94%
```

Implication: for any given board position, we can predict AlphaQ's reply
with ~94% accuracy.  This effectively converts MCTS random rollouts into
near-exact minimax lookahead.

### 2.2  Terminal State Coverage

```
Possible terminal states (Petersen):  32,768
Observed terminal states:                 86 (0.26%)
Terminal states producing a draw:         ~60
Terminal states producing a loss:         ~26
Terminal states producing a win:            0
```

86 observed states from 1,376 games means AlphaQ's deterministic responses
funnel the vast majority of games into a handful of endpoints.  From the
data, the best-scoring opening (E7G) leads to the **same terminal state**
(`PGGGGPPGGPGGGPP`) in every game.  We are replaying the same game over and
over.

### 2.3  Score-Outcome Discrepancy

Our SA evaluator and the tangled-game.com server adjudicator disagree
systematically:

| Terminal State | SA Score | Server Result | Games |
|----------------|----------|---------------|-------|
| `PPPPGGGPPPGGPPG` | +0.86 | **Always loss** | 5 |
| `PGGGGPPGGPGGGPP` | +0.78 | Always draw | 162 |
| `GPPGPPPPPGGGGPG` | +0.74 | Always draw | 10 |
| `PPGGPPGPGGGGGGP` | +0.67 | **Always loss** | 9 |
| `PPGPPGGGGGGGGPP` | +0.65 | **Always loss** | 9 |

Losses with positive SA scores (> 0) occur frequently: 48 out of 204 total
losses have SA score > 0.  The draw distribution clusters around
[-0.16, +0.10], while the loss distribution spans [-8.82, +0.89].

The MCTS currently optimizes SA score.  This directs it toward terminal
states that *our* evaluator likes but the *server* may classify as losses.
This is the fundamental reason brute-force search cannot find wins.

### 2.4  Best Historical Results

| Opening | Avg Score | Result | Terminal State |
|---------|-----------|--------|---------------|
| E7G | +0.789 | Draw (3/3) | `PGGGGPPGGPGGGPP` |
| E8G | +0.310 | Draw (3/3) | varies |
| E5G | +0.179 | Draw (3/3) | varies |
| E3P | +0.176 | Draw (3/3) | varies |

No opening has ever produced a win.  E7G is the highest-scoring but
converges to a fixed draw every time.

### 2.5  Game Path Analysis

```
Total unique game paths played:  1,357
Distinct terminal states reached:    86
Average paths per terminal state:   ~16
```

With ~8 of our moves per game and ~5 legal choices per move, the
theoretical game tree has ~5^8 = 390,625 paths per opening.  We have
explored 1,357 of them.  AlphaQ's determinism collapses many branches, but
even accounting for that, we estimate 10,000-50,000 reachable terminal
states -- 100-600x more than the 86 observed.

---

## 3  The Oracle Strategy

### Phase A: Build the AlphaQ Oracle (offline)

**Goal:** Construct a lookup table that predicts AlphaQ's response for any
board state, then enumerate all reachable terminal states.

**A1 -- Response Table Construction**

Extract from the database every (board_state, opponent_edge, opponent_color)
tuple observed across all AlphaQ games.  For each board state, store:

- Primary response: (edge, color) with highest frequency
- Confidence: frequency / total observations
- Alternative responses: for the 28% of states with non-deterministic
  behavior, store the full distribution

Data source: `moves` table, `player = 'opponent'`, joined with previous
move's `state_after` to reconstruct the board state before AlphaQ's turn.

Format: MATLAB struct or Python dict, keyed by 15-character board state
string.

**A2 -- Game Tree Enumeration**

For each of the 30 possible openings (15 edges x 2 colors):

```
function enumerate(state, is_our_turn, depth):
    if no grey edges remain:
        record terminal_state
        return

    if is_our_turn:
        for each grey edge e, for each color c in {G, P}:
            next_state = state with edge e set to c
            enumerate(next_state, false, depth + 1)
    else:
        response = oracle.predict(state)
        if response.confidence >= 0.9:
            # Deterministic: follow single branch
            next_state = state with response applied
            enumerate(next_state, true, depth + 1)
        else:
            # Non-deterministic: branch over top responses
            for each response in oracle.top_responses(state):
                next_state = state with response applied
                enumerate(next_state, true, depth + 1)
```

With AlphaQ's 72% full determinism and 94% average confidence, most
opponent branches collapse to a single path.  The branching factor is
dominated by our own moves (~5 choices at ~8 decision points).

Estimated reachable terminal states: 10,000 - 50,000.

**A3 -- Terminal State Scoring**

For each reachable terminal state, look up the pre-computed LUT score
(32,768 entries covering all possible terminal colorings).  Also flag
whether the state matches any of the 86 known server-adjudicated outcomes.

Output: ranked list of reachable terminal states by LUT score, annotated
with known server outcomes where available.

### Phase B: Calibrate Against the Real Adjudicator

**Goal:** Replace SA-based evaluation with a model trained on actual
server outcomes.

**B1 -- Outcome Model from Known Terminal States**

From the 86 observed terminal states:

```
terminal_state -> {P(win), P(draw), P(loss), n_observations}
```

This is ground truth from the server.  No modeling assumptions needed.

**B2 -- Generalization to Unseen States**

For the ~50,000 reachable terminal states not yet observed, we cannot
directly know the server outcome.  Options:

- **Conservative:** Only target terminal states where the LUT score
  exceeds the highest-scoring known loss (+0.89).  This is extremely
  restrictive but avoids false positives.

- **Feature-based:** Build a classifier mapping terminal state features
  (frustration pattern, edge coloring symmetry, vertex connectivity) to
  server outcome.  Train on the 86 known outcomes.  Even a crude model
  beats raw SA score.

- **Empirical:** Play targeted verification games (Phase C) to observe the
  server outcome for high-priority candidate states.  This is the most
  reliable approach.

**B3 -- Candidate Identification**

Filter the reachable terminal states to find win candidates:

1. LUT score significantly above the known-draw range (> +1.0)
2. Board structure dissimilar to the known "always-loss" states
3. Reachable via a game path where AlphaQ's responses are high-confidence

Estimate: a few hundred to a few thousand candidate states worth
investigating.

### Phase C: Targeted Play (online)

**Goal:** Verify candidates and play winning lines.

**C1 -- Oracle-Guided MCTS**

Replace the heuristic rollout policy with the AlphaQ oracle:

- During MCTS simulation, when it's the opponent's turn, use the oracle's
  predicted response instead of a random or heuristic move.
- When the oracle has high confidence (>0.9), use the deterministic
  prediction.  Otherwise, sample from the oracle's response distribution.
- For terminal evaluation, use the outcome model from Phase B instead of
  raw SA score.

This transforms 3,000,000 random rollouts per move into 3,000,000
near-deterministic simulations.  The effective search quality increase is
enormous -- comparable to increasing search depth by 3-5 plies in chess.

**C2 -- Path Targeting**

From Phase A, we know which sequences of our moves lead to each candidate
terminal state.  Bias the MCTS:

- **Opening selection:** Choose the opening that leads to the most
  promising reachable terminal states.
- **Move prior:** In the UCB1 selection, add a prior bonus for moves that
  are on a path toward a high-value candidate terminal state.
- **Pruning:** Skip branches that lead exclusively to known-loss or
  known-draw terminal states.

**C3 -- Verification Games**

Play a focused set of games targeting specific candidate terminal states:

1. Select the top 20 candidate terminal states from Phase B
2. For each, compute the game path (our moves) that reaches it
3. Play the game, making the prescribed moves
4. Record the server outcome
5. Update the outcome model with the new observation

Each verification game reveals one new ground-truth outcome.  At ~8 minutes
per game with parallel MCTS, 20 verification games take ~3 hours.

---

## 4  Implementation Roadmap

### Step 1: Oracle Response Table (Python script, ~2 hours)

Create `build_oracle.py`:
- Query all opponent moves from `game_stats.db`
- Reconstruct board-state-before from prior move's `state_after`
- Build response table as a dict: `{state: {(edge, color): count, ...}}`
- Export as `.mat` file for MATLAB and `.json` for Python
- Report: coverage statistics, confidence distribution, gaps

### Step 2: Game Tree Enumerator (MATLAB or Python, ~4 hours)

Create `enumerate_game_tree.m` or `enumerate_game_tree.py`:
- Load oracle response table
- Enumerate from each of the 30 openings
- Record all reachable terminal states
- Score each with the LUT
- Report: terminal state count per opening, score distribution,
  overlap with known outcomes

### Step 3: Outcome Model (Python, ~2 hours)

Create `build_outcome_model.py`:
- Extract known terminal state -> server outcome mappings from DB
- Compute features: frustration count, green/purple ratio, vertex
  coloring pattern, edge connectivity
- Train a simple classifier (logistic regression or random forest)
- Cross-validate on the 86 known outcomes
- Score all enumerated terminal states from Step 2
- Rank and output candidate win states

### Step 4: Oracle-Guided MCTS (MATLAB, ~4 hours)

Modify `TangledMCTS.m`:
- Load oracle response table at initialization
- In `parallelSimulate()`, replace opponent heuristic moves with oracle
  predictions when confidence > threshold
- Replace terminal evaluation with outcome model score when available,
  falling back to LUT score
- Add move prior biasing toward candidate terminal states

### Step 5: Verification Campaign (online, ~3 hours)

Run targeted games:
- Select top candidate terminal states
- Use oracle-guided MCTS with path targeting
- Record outcomes
- Iterate: update model, re-rank candidates, play more games

---

## 5  Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| AlphaQ updates its policy | Low | Monitor response divergence; rebuild oracle periodically |
| Oracle coverage gaps at novel states | Medium | Fall back to heuristic for unseen states; expand DB |
| All reachable terminal states are draws/losses | Medium | This would confirm true Nash equilibrium; still valuable knowledge |
| Server adjudicator differs from SA in unpredictable ways | Medium | Empirical verification (Phase C) bypasses modeling assumptions |
| Game tree enumeration too large | Low | Prune early with oracle confidence threshold; limit branching |

---

## 6  Success Criteria

- **Minimum:** Identify at least one reachable terminal state where the
  server adjudicator produces a win.
- **Target:** Develop a repeatable opening + mid-game sequence that
  produces wins against AlphaQ at >10% rate.
- **Stretch:** Achieve >50% win rate against AlphaQ by exploiting its
  deterministic policy with perfect lookahead.

If no winning terminal state exists among the reachable set, this
constitutes strong evidence for a true Nash equilibrium at the
classical-vs-AlphaQ boundary, and the research shifts to understanding
whether quantum resources (which are explicitly out of scope here) are
necessary to break through.
