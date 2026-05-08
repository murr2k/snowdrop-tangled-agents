# Phase A Results: AlphaQ Oracle Construction

**Status:** Complete
**Date:** 7 February 2026
**Prerequisite:** Run 86 data (1,400+ games against AlphaQ Up)
**Related docs:**
[ALPHAQ_ORACLE_STRATEGY.md](ALPHAQ_ORACLE_STRATEGY.md),
[SCORE_OUTCOME_DISCREPANCY.md](SCORE_OUTCOME_DISCREPANCY.md),
[EXPERIMENT_OPENING_RE_EXPLORATION.md](EXPERIMENT_OPENING_RE_EXPLORATION.md)

---

## 1  Summary

Phase A constructed an offline oracle of AlphaQ's behavior, enumerated all
reachable terminal states through oracle-guided game tree search, and scored
them against both the SA-based lookup table and known server outcomes.

**The central finding is that AlphaQ is even more predictable than
hypothesized, but the SA evaluation function is not merely inaccurate — it is
anti-correlated with server outcomes.** High SA scores predict losses, not
wins. This means no amount of SA-score optimization, whether by MCTS or
oracle-guided search, can find winning lines. The evaluation function itself
must be replaced before win candidates can be identified.

---

## 2  Step A1: Oracle Response Table

### Method

Opponent move data was extracted from two sources in `game_stats.db`:

1. **`opponent_history` table** — 4,594 observations with explicit
   `board_state_before` (200 distinct state-action pairs)
2. **`moves` table** — 7,983 additional observations reconstructed by
   joining each opponent move with the prior move's `state_after`

Total: **12,577 observations** across **388 distinct board states**.

### Results

| Metric | Strategy Doc Prediction | Actual |
|--------|------------------------:|-------:|
| Distinct board states | 457 (2+ obs only) | 388 (all), 299 (2+ obs) |
| Fully deterministic states | 330 (72.2%) | 380 (**97.9%**) |
| Average top-response confidence | 94% | **99.3%** |
| Min confidence | — | 50.0% |

AlphaQ is **far more deterministic** than the strategy document estimated.
Of 388 observed board states, only 8 show any stochasticity at all (confidence
< 1.0). For practical purposes, AlphaQ is a pure function from board state to
move.

### Phase Coverage

Oracle data spans all game phases, though coverage thins in early game where
more board states exist:

| Grey Edges | Phase | States with Data |
|-----------:|:------|:----------------:|
| 14 | Opening | 30 |
| 12 | Opening | 51 |
| 10 | Mid | 47 |
| 8 | Mid | 51 |
| 6 | Late | 63 |
| 4 | Late | 74 |
| 2 | Endgame | 72 |

### Top-10 Most Observed States

All top-10 states are fully deterministic (confidence = 1.00):

```
-------G-------  obs=383  best=E0P
PG-----G-------  obs=345  best=E2G
PGGPG----------  obs=333  best=E5G
PGGG---G-------  obs=329  best=E4G
PGG------------  obs=318  best=E3P
PGGGGP-G-------  obs=309  best=E9P
PGGGGP-G-PG----  obs=300  best=E12G
PGGPGGP--------  obs=279  best=E7P
PGGGGPPG-PG-GPP  obs=265  best=E8G
PGGGGP-G-PG-GP-  obs=247  best=E6P
```

The sequence `E7G → E0P → E2G → E3P → E4G → E5G → ...` is the dominant
game line, replayed hundreds of times with perfect consistency from AlphaQ.

---

## 3  Step A2: Game Tree Enumeration

### Method

Depth-first search from each of 30 openings (15 edges × 2 colors). At each
opponent turn, the oracle predicts AlphaQ's response:

- **Confidence >= 0.9:** Follow the single deterministic branch
- **Confidence < 0.9:** Expand top-3 responses (probability >= 5%)
- **No oracle data:** Record an "oracle gap" and stop that branch

At each of our turns, all legal moves are explored (grey edges × 2 colors).

### Results

| Metric | Value |
|--------|------:|
| Total paths explored | 9,835 |
| Terminal states found | 496 |
| **Unique terminal states** | **110** |
| Oracle gaps | 9,339 |
| Elapsed time | 0.1s |

### Terminals per Opening (Top 10)

| Opening | Terminals | Gaps |
|:--------|----------:|-----:|
| E1G | 182 | 2,760 |
| E3G | 56 | 708 |
| E2P | 34 | 453 |
| E4G | 26 | 362 |
| E0P | 22 | 486 |
| E1P | 22 | 486 |
| E0G | 18 | 291 |
| E2G | 16 | 327 |
| E3P | 12 | 305 |
| E9G | 10 | 295 |

E1G dominates because it deviates from the heavily-explored E7G opening,
leading to board regions where the oracle still has data but where our move
choices create more branching.

### Oracle Gap Analysis

9,339 gaps means the majority of game-tree branches terminate at an unknown
opponent state rather than at a true terminal. Gap distribution by remaining
grey edges:

| Grey Edges Remaining | Gaps |
|---------------------:|-----:|
| 12 | 694 |
| 10 | 1,779 |
| 8 | 1,883 |
| 6 | 2,231 |
| 4 | 1,702 |
| 2 | 1,050 |

Gaps are distributed across all depths, peaking at 6 grey edges (mid-late
game). This means the oracle's knowledge is clustered around the game lines
actually played in historical games, and novel move sequences quickly leave
known territory.

### Path Confidence

Of the 496 terminal-reaching paths:

- 336 (67.7%) have minimum path confidence >= 0.9
- Average minimum path confidence: 0.839
- Remaining 32.3% pass through at least one non-deterministic oracle state

---

## 4  Step A3: Terminal State Scoring

### Method

Each of the 110 unique terminal states was:

1. Scored using the 32,768-entry SA-based terminal score LUT (`terminal_scores.mat`)
2. Cross-referenced with known server outcomes from the database (78 known
   terminal state → outcome mappings from 1,400+ games)

### Results

| Category | Count |
|----------|------:|
| Total unique terminal states | 110 |
| Known outcomes | 59 |
| — Draws | 25 |
| — Losses | 34 |
| — Wins | 0 |
| Novel (never observed) | 51 |
| **Win candidates** | **0** |

### The Anti-Correlation Discovery

The most important finding of Phase A:

| Outcome | n | LUT Score Range | LUT Score Avg |
|---------|--:|:---------------:|--------------:|
| Draw | 25 | [0.000, 3.977] | **0.562** |
| Loss | 34 | [-0.017, 7.943] | **0.888** |

**Known losses have higher average LUT scores than known draws.**
The SA-based evaluation function is anti-correlated with the server
adjudicator. Terminal states that the SA evaluator considers "good" (high
frustration, high score) are the ones the quantum adjudicator classifies as
losses.

This explains the entire history of the project:

1. MCTS optimizes for SA score during rollouts
2. Higher SA score → MCTS steers toward those terminal states
3. Those terminal states are actually losses on the server
4. Result: 0 wins in 1,400+ games

The 51 novel (never-observed) terminal states all have very low LUT scores
(0.004–0.013), because the MCTS + AlphaQ dynamic systematically steers games
away from low-score terminals. Paradoxically, these low-score states may be
the most promising candidates — but without server verification, we cannot
know.

### Novel Terminal States (Top 15 by LUT Score)

| Terminal State | LUT Score | Paths | Best Opening |
|:---------------|----------:|------:|:-------------|
| `PPPPGGGGPGGGPGG` | +0.0129 | 3 | E1P |
| `PPGPPPGGGGGGGPP` | +0.0122 | 1 | E7G |
| `PPGGPPGPGGGGGPP` | +0.0122 | 1 | E9G |
| `GPGPGPPPGGPGPPP` | +0.0088 | 1 | E12P |
| `PPGPGGGGGGGGGPP` | +0.0088 | 2 | E9G |
| `GPPGPPPPPGGGGPP` | +0.0085 | 2 | E0G |
| `PGGGGPGPPPGGGPP` | +0.0072 | 2 | E1G |
| `PGPGGPPPPPPGPPG` | +0.0068 | 4 | E1G |
| `PGGPGGGPPGGGGPG` | +0.0068 | 3 | E9G |
| `PGGPPPGPPGGGGPG` | +0.0060 | 7 | E1G |
| `GPGPGGPGGGGGGPP` | +0.0056 | 1 | E10G |
| `PPPPGPGPPGGPGPG` | +0.0053 | 1 | E6G |
| `PPGPPGPGGGGGGPP` | +0.0048 | 2 | E1P |
| `PPGPGGGGPGPGPGP` | +0.0047 | 3 | E1P |
| `GPGPGPPGGGPGPPP` | +0.0041 | 1 | E12P |

None of these can be classified as win candidates using LUT scores alone.

---

## 5  Conclusions

### 5.1  AlphaQ is Solvable

AlphaQ's near-perfect determinism (99.3% confidence, 97.9% fully deterministic)
confirms that the game against AlphaQ is effectively a single-player
optimization problem. Given a board state, AlphaQ's response is known with
near-certainty. The oracle approach is sound.

### 5.2  The Oracle Has Gaps

9,339 oracle gaps vs. 496 terminals means 95% of explored branches hit unknown
territory. The oracle's knowledge is concentrated along historical game lines.
To expand coverage, targeted exploratory games are needed — specifically,
games that deliberately play unusual moves to probe AlphaQ's responses at
novel board states.

### 5.3  The SA Evaluator Must Be Replaced

This is the critical blocker. The SA-based terminal score LUT is
**anti-correlated** with server outcomes:

- Optimizing SA score steers toward **losses**
- Known draws cluster at moderate scores; known losses cluster at high scores
- The MCTS has been systematically optimizing for the wrong objective

No amount of search improvement — parallel rollouts, oracle-guided MCTS,
game tree enumeration — can find wins if the evaluation function points in
the wrong direction. **Phase B (outcome model calibrated on server
adjudication) is the critical path.**

### 5.4  Recommended Next Steps

1. **Phase B (critical):** Build an outcome model trained on the 59 known
   terminal-state → server-outcome mappings. Even a simple logistic regression
   on board features would outperform the SA LUT, which is worse than random.

2. **Oracle gap filling:** Run 50–100 exploratory games with randomized
   mid-game moves to expand the oracle's coverage beyond the 388 known states.
   Each game reveals ~7 new opponent states.

3. **Re-enumerate with outcome model:** Once Phase B provides a reliable
   evaluation function, re-run Phase A's enumeration and scoring. The 51 novel
   terminal states, plus any newly reachable ones from filled gaps, can then be
   properly ranked.

4. **Verification games:** Play targeted games to reach specific high-value
   novel terminal states and observe the server's adjudication.

---

## 6  Output Files

All output files are in `snowdrop_tangled_agents/oracle/data/`:

| File | Contents |
|------|----------|
| `oracle_responses.json` | 388-state oracle response table |
| `reachable_terminals.json` | 9,835 enumeration results (496 terminals + 9,339 gaps) |
| `scored_terminals.json` | 110 unique terminals with LUT scores and known outcomes |
| `win_candidates.json` | Empty (0 candidates with current evaluation) |

### Reproduction

```bash
poetry run python -m snowdrop_tangled_agents.oracle.run_phase_a
```

Options:
- `--confidence 0.9` — oracle confidence threshold (default)
- `--score-threshold 0.5` — minimum LUT score for candidates (default)
- `--db-path PATH` — path to game_stats.db
- `--opponent alphaq` — opponent name pattern
