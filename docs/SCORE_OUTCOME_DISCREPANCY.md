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
