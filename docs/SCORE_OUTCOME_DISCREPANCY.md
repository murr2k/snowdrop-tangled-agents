# Score–Outcome Discrepancy in tangled-game.com

Murray Kopit

## Summary

Analysis of 1,600 games and 1,511 calibrated terminal states reveals that the
score displayed on tangled-game.com **does not reliably determine the declared
winner**. Sixty percent of our losses show a positive final score. Draws cluster
around zero but also appear at scores as high as +2. The declared winner (via
"Winner: Player X" text) and the displayed score appear to be produced by
independent processes.

This is either a **display bug** — the score element is not updated after final
adjudication — or an **intentional representation of quantum measurement
stochasticity**, where the score is the expectation value and the winner is a
single measurement outcome. The evidence below distinguishes between the two.
Regardless of which interpretation is correct, the discrepancy has measurable
consequences for AI agents that optimize for score.

---

## Data Collection

All data comes from the project's SQLite game database (`game_stats.db`),
populated over 1,600 live games on tangled-game.com. At the end of each game:

1. A 2-second sleep waits for the page to settle after game-end detection.
2. `read_score()` extracts the value after "Score:" from `<body>` inner text.
3. `get_outcome()` checks for "Winner: Player 1", "Winner: Player 2", or "Draw"
   text first. If none match, it falls back to score thresholds (>0.5 = win,
   <-0.5 = loss). In practice the explicit text is present in the vast majority
   of games, so the fallback rarely fires.
4. For games where all 15 edges are colored at game end, the terminal state is
   also evaluated by our SimulatedAnnealingAdjudicator (`predicted_score`) and
   stored alongside the website values in the `calibration` table.

Both `final_score` (from step 2) and `result` (from step 3) are read from the
same page load after the same 2-second wait. They should reflect the same game
state.

---

## Finding 1: Score sign does not determine winner

Across all 1,600 games with a recorded final score:

| Result | Positive score | Negative score | Zero |
|--------|---------------|----------------|------|
| **Win** | 369 | 2 | 0 |
| **Loss** | 374 | 256 | 3 |
| **Draw** | 495 | 63 | 9 |

Key observations:

- **374 losses have positive final scores.** That is 59% of all losses. The
  website simultaneously displays a positive score and declares Player 2 the
  winner.
- **Only 2 wins have negative scores.** The relationship is strongly asymmetric:
  negative score reliably predicts loss, but positive score predicts nothing.
- **184 draws have |score| ≥ 0.5**, including draws with scores above +1.0. The
  draw determination is not based on the score being near zero.

Per opponent, the rate of losses with positive scores is:

| Opponent | Total Losses | Positive-Score Losses | Rate |
|----------|-------------|----------------------|------|
| MCTS Melissa | 623 | 370 | 59.4% |
| AlphaZero Amara | 10 | 4 | 40.0% |

---

## Finding 2: P(win) is a step function of score, not linear

From the 1,511 calibrated games (where both the website score and the game
result are recorded alongside the terminal state), P(win) as a function of the
website's displayed score:

| Website Score | Games | Wins | P(win) |
|---------------|-------|------|--------|
| < −2 | 15 | 0 | 0.0% |
| [−2, −1) | 23 | 2 | 8.7% |
| [−1, −0.5) | 32 | 0 | 0.0% |
| [−0.5, 0) | 169 | 0 | 0.0% |
| **[0, +0.5)** | **269** | **16** | **5.9%** |
| **[+0.5, +1)** | **125** | **54** | **43.2%** |
| [+1, +2) | 186 | 144 | 77.4% |
| **[+2, +5)** | **136** | **134** | **98.5%** |
| > +5 | 17 | 17 | 100% |

The score is only a reliable win predictor above **+2**. In the range [0, +1) —
where the majority of our games land — the outcome is dominated by adjudication
noise. A displayed score of +0.8 wins less than half the time.

---

## Finding 3: SimulatedAnnealing makes the discrepancy worse

Our LUT (32,768 terminal states) and MCTS rollout evaluation both use
SimulatedAnnealingAdjudicator scores. Comparing P(win) binned by SA prediction
versus by website score reveals that SA is a noisier signal:

| Score Range | P(win) if Website score | P(win) if SA predicted |
|-------------|------------------------|----------------------|
| [+0, +0.5) | 5.9% | 2.9% |
| [+0.5, +1) | 43.2% | 36.0% |
| [+1, +2) | 77.4% | 81.9% |
| **[+2, +5)** | **98.5%** | **70.8%** |
| > +5 | 100% | 88.2% |

At SA scores of +2 to +5 — where our solver is most confident — we actually
lose ~30% of the time. SA is systematically overconfident in this range. Across
all calibrated games, SA overestimates the website score 64.9% of the time, with
a mean positive bias of +0.237.

This means our MCTS is searching toward terminal states it believes are strong
wins, but which the actual adjudicator resolves as losses roughly a third of the
time.

---

## Two Hypotheses

### Hypothesis A: Display Bug — Score Not Updated After Adjudication

The score element on the page is updated after each move during active play as a
running estimate. When the final move is played, the game runs adjudication to
determine the winner and displays "Winner: Player X". However, the score element
is **not re-rendered** to reflect the final adjudication result. `read_score()`
therefore returns the last mid-game score, which can disagree with the declared
winner.

**Evidence for:**
- The discrepancy is present on the same page load, after a 2-second wait.
  Both elements should have settled.
- The asymmetry (negative score → almost always loss, positive score → anything)
  is consistent with a running score that drifts positive during play (due to
  our strategy targeting positive-score states) but gets overruled by a
  separate final adjudication.
- The specific cases of website score +2.7 with "Winner: Player 2" are hard to
  explain as intentional behavior — a score that positive normally wins 98.5%
  of the time.

**Evidence against:**
- The 2-second sleep should be sufficient for a DOM update.
- If this were a simple rendering bug, it would likely have been caught during
  basic testing.

### Hypothesis B: Quantum Measurement Mechanics — Score Is Expectation Value

The displayed score is the **expectation value** of the quantum Hamiltonian
(⟨H⟩), computed once. The winner is determined by a **measurement** — a single
sample from the quantum state, implemented via simulated annealing. A positive
expectation value does not guarantee a positive measurement outcome. This is
standard quantum mechanical behavior and would be an intentional design choice
for a game about quantum annealing.

Under this model:
- Score = ⟨H⟩ (continuous, stable)
- Winner = sign(single SA sample) (stochastic, can disagree with ⟨H⟩)
- P(win) increases with ⟨H⟩ but saturates below 100% for finite systems

**Evidence for:**
- The P(win) curve has the right qualitative shape for a quantum measurement:
  near-zero expectation values produce near-random outcomes; large expectation
  values produce reliable outcomes.
- The game is explicitly themed around quantum annealing. Stochastic outcomes
  ARE the point.
- Draws appearing at non-zero scores is natural if "draw" corresponds to an
  indeterminate measurement (e.g., both players' energies within some
  threshold).
- The asymmetry (negative score → reliable loss) is consistent with the
  quantum state having a strong negative bias that measurement almost always
  confirms.

**Evidence against:**
- If this is intentional, it is not documented anywhere visible to players or
  developers building on the platform.
- The UX implication is significant: players see a positive score and lose.
  This is confusing regardless of the underlying physics.

---

## Distinguishing Test

The two hypotheses produce different predictions about the **relationship
between the score element and the adjudication process**:

| Test | Bug prediction | Quantum prediction |
|------|---------------|-------------------|
| Score element updates after "Winner:" appears | No (stale) | Yes (already correct — it's ⟨H⟩, not the outcome) |
| Running multiple adjudications on the same terminal state produces different winners | N/A | Yes |
| The score shown matches the score from mid-game (before final move) | Yes | No — it should be the full-game ⟨H⟩ |
| Adding `num_reads` to the final adjudication changes win rate but not displayed score | N/A | Yes |

**Recommended test:** On the game's backend, compare the score displayed after
game end with (a) the score after the penultimate move, and (b) the score
computed from the final terminal state using the same adjudicator. If (a) matches
and (b) does not, the score element is stale. If (b) matches and the winner was
determined by a separate single-sample adjudication, it's the quantum model.

---

## Impact on AI Agents

Regardless of which hypothesis is correct, the consequence for AI agents is the
same: **optimizing for score is the wrong objective**.

Our MCTS solver maximizes expected score across rollouts. But P(win) is not a
linear function of score. In the [0, +2) range — where most games are decided —
score and P(win) are nearly uncorrelated. A solver that reaches a terminal state
with score +0.9 has only a 43% chance of winning. A solver that reaches +2.1 has
a 98.5% chance.

The correct objective for a win-maximizing agent is:

```
maximize P(win | terminal_state)
```

not

```
maximize E[score | terminal_state]
```

These are different optimization targets, and MCTS will converge to different
moves depending on which one is used. Concretely: if two rollout paths lead to
terminal states with SA scores +1.8 and +2.2 respectively, the current solver
treats them as nearly equivalent. Under P(win) evaluation, the +2.2 state is
worth roughly 2.5× more (98% vs 40% win probability). The solver would
correctly learn to prefer the path to +2.2.

A calibration curve fitted to the 1,511 existing data points can be used to
convert SA predicted scores to P(win) at evaluation time, with no changes to the
MCTS tree search or rollout structure. Only the terminal value returned changes.

---

## Calibration Results: Run 46 (60 games vs MCTS Melissa)

The calibration curve was implemented in `TangledMCTS.m` as described above and
tested over 60 games against MCTS Melissa. The baseline is the immediately
preceding 60-game run (Run 44) using the same strategy, opponent, and learning
loop but with raw SA scores as the terminal evaluation.

### Head-to-head comparison

| Metric | Run 44 (raw SA) | Run 46 (calibrated) |
|--------|-----------------|---------------------|
| Wins | 13 | **22** |
| Losses | 26 | **16** |
| Draws | 21 | 22 |
| Win rate | 21.7 % | **36.7 %** |
| Loss rate | 43.3 % | **26.7 %** |
| Overall avg score | 0.670 | **0.817** |
| Avg score on wins | 2.423 | 1.858 |
| Wins at score ≥ +2 | 4 | **9** |
| Wins at score [+1, +2) | 6 | 7 |
| Wins at score [0, +1) | 3 | 5 |

Win rate rose by **+15 percentage points** and losses dropped by nearly half.
The change was a single substitution in terminal evaluation — no structural
changes to the tree search, rollout policy, or opening strategy.

### Why it works

The calibration reweights the MCTS value landscape so that the difference
between SA score +1.8 and +2.2 — previously treated as noise — becomes the
difference between a ~40 % and ~98 % win. The solver converges on paths to
terminal states above the +2 threshold more aggressively.

Two effects are visible in the data:

1. **More wins in the reliable zone.** Wins at score ≥ +2 more than doubled
   (4 → 9). The solver is successfully steering toward terminal states where
   the adjudicator outcome is nearly deterministic.

2. **Fewer losses overall, not fewer high-confidence losses.** The 10 losses
   eliminated are concentrated in the [0, +1) score range — the coin-flip zone
   where raw SA evaluation treated +0.6 and +0.9 as meaningful wins. Under
   calibration those states correctly evaluate as near-50/50, so the solver
   avoids them in favour of paths that reach higher scores.

### Score divergence and the mid-game bottleneck

Across the 60 calibrated games, winning and losing trajectories diverge at
**moves 3–5**. Winning games maintain a positive score through the mid-game;
losing games drop negative by move 4. By move 8 the gap is stark: winning
games average +2.17, losing games average +0.05. The outcome is essentially
decided before the late-game phase where the solver has the most search depth.

This suggests the next lever is not terminal evaluation — which is now
well-calibrated — but **mid-game search quality in moves 3–5**, where the
solver must commit to a positional trajectory with limited lookahead. The
edge bias learned via REINFORCE (strongest signals: E11 +0.24, E4 +0.23,
E1 +0.20) is beginning to encode this: it biases the rollout prior toward
edges that empirically maintain the positive mid-game trajectory.

---

## AlphaQ Up: Calibration Does Not Transfer (Run 47, 60 games)

Run 47 tested the same calibrated solver against AlphaQ Up using the
AlphaQExplorerStrategy (30-game exploration → 30-game exploitation). The result
was **0 wins in 60 games** (27 losses, 33 draws). AlphaQ Up is a fundamentally
different opponent from Melissa, and the data reveals exactly why.

### The solver is winning positionally; the adjudicator is not

| Outcome | Avg score | Min score | Max score | n |
|---------|-----------|-----------|-----------|---|
| Draw | +0.437 | +0.022 | +0.780 | 33 |
| Loss | +0.186 | −1.888 | +0.728 | 27 |

Draws averaged +0.437 and losses averaged +0.186. We are consistently reaching
positive terminal scores — the MCTS search and calibration are functioning as
designed. The problem is not positional: it is at the adjudication stage.
AlphaQ Up is winning (or drawing) the adjudication from states where our
calibration curve — fitted on Melissa's adjudication — predicted a win.

The most extreme case: a loss recorded at score +0.728. Against Melissa, a
score in that range wins ~43 % of the time. Against AlphaQ Up it lost outright.
Losses at +0.136 and draws at +0.780 are similarly inconsistent with the
Melissa-derived calibration.

### The calibration curve is opponent-specific

The calibration curve in `calibration_pwin.mat` was fitted on 1,511 games
against Melissa. It encodes Melissa's adjudication noise characteristics: the
threshold above which scores reliably convert to wins, the width of the
coin-flip zone, and the saturation behaviour at high scores.

AlphaQ Up has a different adjudication profile. The +2 threshold that yields
98.5 % win rate against Melissa does not hold here. Without wins to measure
against, the exact shape of AlphaQ's P(win) curve is unknown, but the data
shows it is shifted substantially toward higher scores — or that AlphaQ's
adjudicator is more stochastic overall.

**Implication:** To calibrate against AlphaQ Up, we need wins. The current
calibration curve should not be trusted for this opponent.

### REINFORCE degenerates without wins

The closed learning loop (REINFORCE → `setEdgeBias`) requires positive reward
signal to function. With zero wins in 60 games, every edge that appeared in a
losing game was penalised and no edge was ever reinforced. The final bias drifted
uniformly negative:

```
E13: −0.083   E8: −0.081   E6: −0.076   E4: −0.055
E10: −0.053   E7: −0.049   E14: −0.040   E3: −0.038
```

This is not a learned policy — it is the expected output of a one-sided reward
signal applied uniformly. Draws contribute zero gradient under the current
REINFORCE formulation (only `result == 'win'` yields positive reward).

To extract signal from this opponent, the reward function needs to be modified.
Draws are not equivalent to losses here: a draw at score +0.78 is a near-miss
that the solver almost converted, while a loss at −1.89 is a positional
collapse. Treating them identically discards the only useful signal available.

### Exploration found no exploitable openings

All 30 first-move openings produced either draws or losses. The top-5 selected
for exploitation (E9G, E0G, E11G, E7G, E6G) were chosen by average score since
wins were zero across the board. Exploitation converged on E6G and E7G, which
had the best draw scores (+0.506 and +0.518 respectively), but this produced
no wins in 30 exploitation games.

The absence of exploitable openings suggests AlphaQ Up's weakness — if one
exists — is not in its response to a specific first move. It may require
deeper positional exploitation that only emerges at moves 3–5, consistent with
the mid-game bottleneck identified in the Melissa analysis.

---

## Appendix: Raw Numbers

- Total games played: 1,600
- Games with calibration data: 1,511
- Unique terminal states observed: 934
- Terminal states where SA and website disagree on sign: 147 unique states,
  166 occurrences (11.0% of calibrated games)
- SA mean bias vs website: +0.237
- SA overestimates website score: 64.9% of calibrated games
- Losses with positive website score: 374 (59% of all losses)
- Draws with |score| ≥ 0.5: 184 (32% of all draws)
