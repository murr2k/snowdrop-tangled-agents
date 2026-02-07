# Experiment: Opening Re-Exploration with Fixed MCTS

## Hypothesis

The "Nash Equilibrium" conclusion drawn in Runs 60-66 (documented in
[ALPHAQ_UP_BOUNTY_ANALYSIS.md](ALPHAQ_UP_BOUNTY_ANALYSIS.md) and
[THE_MATHEMATICS_OF_TANGLED_GAME.md](THE_MATHEMATICS_OF_TANGLED_GAME.md) §10.4)
may be an artifact of broken tooling rather than a true property of the game.
Re-exploring all 30 openings with functional parallel rollouts could reveal
viable winning lines that were incorrectly dismissed.

## Background: The MCTS Parallelization Bug

All prior experiments against AlphaQ ran with a `parfor` worker pool that
allocated 6 workers (~5.4 GB) but **never dispatched work to them**.  Every
MCTS iteration performed a single serial rollout on the main thread while
workers sat idle at 0% CPU.

| Metric | Broken (Runs 60-66) | Fixed (Run 84+) | Ratio |
|--------|--------------------:|----------------:|------:|
| Iterations | 5,000 | 5,000 | 1× |
| Rollouts per iteration | 1 | 600 | 600× |
| **Total rollouts per move** | **5,000** | **3,000,000** | **600×** |
| Avg time per move (15 grey) | 6.4 s | ~250 s | ~40× |

At 10,000 iterations the ratio rises to **1,200×** more rollouts than the
historical 5,000-iteration tests.

**Impact on prior conclusions:**

- **Run 60** (Thompson Sampling, 500 games): Tested all 30 openings with
  5,000 serial rollouts per move.  Openings like E9G (89% loss rate, 9 games)
  and E11G (100% loss rate, 10 games) were dismissed as "catastrophic"
  ([ALPHAQ_STRATEGY.md](ALPHAQ_STRATEGY.md) §2.1).  With 600× more rollouts,
  the MCTS may find defensive lines that change these outcomes.

- **Run 64** (Forced E7G, 100 games): Achieved 0% loss rate and +0.779
  average score.  Declared as Nash Equilibrium.  Run 84 with 1,200× more
  rollouts scored +0.800 — a negligible +0.02 improvement, suggesting E7G
  **is** genuinely at equilibrium.  But this only invalidates "more compute
  on E7G", not "different openings with more compute".

- **Run 65** (Thompson + Winning-Push, 100 games): Re-enabled Thompson
  Sampling with E7G bias.  6% loss rate.  The winning-push heuristic
  (boost iterations 50% when score > 0.75) activated 25 times, produced
  0 wins.  But the heuristic only boosted from 5,000 to 7,500 serial
  rollouts — still far below the fixed system's 3M+.

## The Calibration Problem

A second, independent issue compounds the broken-MCTS problem.  The MCTS
terminal evaluation uses the SA-derived LUT (32,768 entries), then maps
SA scores to value estimates via a calibration function.

No opponent-specific calibration file exists for AlphaQ.  The system falls
back to `tanh(score × 0.4)`, which is catastrophically miscalibrated:

| SA Score | tanh P(win) | Actual P(win) vs AlphaQ | Actual P(win) all opponents | Gap vs AlphaQ |
|----------|-------------|------------------------|-----------------------------|---------------|
| +0.0 | 50.0% | 0.0% (0/544) | 0.1% | −50.0 pp |
| +0.5 | 59.5% | 0.0% (0/111) | 1.1% | −59.5 pp |
| +0.8 | 66.4% | 0.0% (0/42) | 9.5% | −66.4 pp |
| +0.9 | 69.2% | **never reached** | 36.1% | — |
| +1.0 | 71.6% | **never reached** | 45.8% | — |

(P(win) all-opponents data from `games` table, 2,967 games.
AlphaQ data from 1,297 games; see [SCORE_OUTCOME_DISCREPANCY.md](SCORE_OUTCOME_DISCREPANCY.md) §3.)

**Consequence for MCTS search:** The inflated value estimates cause the MCTS
to "satisfice" — it finds the E7G→E0P→E1G line that reaches +0.8, believes
this is a 66% win probability, and stops exploring.  In reality +0.8 has
never produced a win against AlphaQ.  A correctly calibrated value function
would assign value ≈ 0 (coin flip at best), forcing the MCTS to explore
more aggressively.

### LUT Recommendations

The LUT itself (raw SA scores for all 32,768 terminal states) does not need
to change.  It is a fixed property of the Petersen graph and SA adjudicator.

What needs to change is **how LUT scores are interpreted**:

1. **Generate `calibration_alphaq.mat`** from historical game data.
   Problem: with 0 wins in 1,297 games, the curve would be flat at P(win)=0,
   which makes all terminal states equally valued — unhelpful for search.
   **Defer until after Phase 1** produces games at new score levels.

2. **Use the generic cross-opponent calibration** (`calibration_pwin.mat`)
   instead of the tanh fallback.  This curve (fitted from 2,967 games across
   all opponents) correctly maps +0.8 → P(win)=9.5% rather than 66%.
   **Action: load generic calibration when opponent-specific file is missing,
   instead of falling back to tanh.**

3. **After Phase 1**, if any openings produce scores > 0.9 against AlphaQ,
   fit an AlphaQ-specific calibration curve and re-run Phase 2 with it.

## Experiment Design

### Phase 1: Broad Exploration (30 openings × 3 games)

**Goal:** Screen all 30 openings for viability with the fixed MCTS.  Identify
any that reach score > 0.9 or produce different game trees than E7G.

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Iterations | 5,000 | 3M rollouts per move; 600× the historical baseline.  Sufficient for screening: old 5K serial was enough to identify E7G dominance.  See timing estimate below. |
| Games per opening | 3 | Enough to detect catastrophic openings (>67% loss) while keeping total manageable. |
| Total games | 90 | 30 openings × 3 games |
| Opponent | AlphaQ Up | Same opponent as all prior experiments |
| Graph | Petersen | Same graph as all prior experiments |

**Why 5K iterations is sufficient for Phase 1:**

The question is whether 5K iterations with fixed parallel (3M rollouts) gives
enough signal to distinguish viable openings from non-viable ones.

Evidence:
- Old 5K serial (5K rollouts) was sufficient to show E7G scoring +0.779
  while E9G lost 89% of games.  The signal was clear even with broken MCTS.
- Run 84 at 10K (6M rollouts) scored +0.800 vs Run 64's +0.779 at 5K serial.
  The marginal improvement from 5K→10K fixed is ~+0.02, well within noise.
- Phase 1 is a screening step.  We need to identify which openings are worth
  deeper investigation, not produce precise score estimates.

**Time estimate:**

| Move position | Grey edges | Est. time at 5K iters | Moves per game |
|--------------|-----------|----------------------|---------------|
| Move 2 (first MCTS) | 13 | ~120 s | 1 |
| Move 3 | 11 | ~20 s | 1 |
| Moves 4-8 | ≤9 | ~0.2 s (minimax/LUT) | ~5 |
| **Total per game** | | **~140 s + overhead** | |

90 games × ~3.5 min = **~5.3 hours** plus browser overhead ≈ **~8 hours**.

### Phase 2: Deep Exploitation (promising openings × 10 games)

**Trigger:** Any opening from Phase 1 that meets ANY of:
- Average score > 0.5 AND loss rate < 30%
- Maximum score > 0.9 (potential win territory)
- Different game tree structure than E7G line (novel play)

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Iterations | 10,000 | 6M rollouts per move; maximum practical for multi-game runs |
| Games per opening | 10 | Statistical significance for win-rate estimation |
| Calibration | Updated (see §LUT Recommendations) | |

### Controls

- **E7G baseline:** Include 3 games with E7G opening at 5K iterations
  (Phase 1) to confirm it matches historical scores (+0.779 ± 0.05).
  This validates that the fixed MCTS produces comparable results.

- **Serial fallback test:** Run 1 game with `UseParallel = false` to
  confirm serial mode still works and matches historical behavior.

## Expected Results

### Scenario A: E7G confirmed as dominant (most likely, ~60%)

All openings score ≤ 0.9 against AlphaQ.  E7G remains the best opening.
The Nash Equilibrium conclusion holds — but now validated with correct tooling.

**Next steps if Scenario A:** The draw barrier is genuine.  Shift focus to:
- Calibration-driven search (LUT interpretation, not more compute)
- Different graphs where AlphaQ may not be at equilibrium
- Adjudicator exploitation (terminal states that SA misevaluates in our favor)

### Scenario B: Alternative opening discovered (~30%)

One or more openings produce scores > 0.9 or significantly different game
trees.  The "Nash Equilibrium" was an artifact of insufficient search depth
on non-E7G openings.

**Next steps if Scenario B:** Run Phase 2 on the promising opening(s).
Generate AlphaQ-specific calibration.  Test at scale (100+ games).

### Scenario C: Win achieved (~10%)

An opening produces a score > 1.0 and an actual win.  This would definitively
refute the Nash Equilibrium conclusion.

**Next steps if Scenario C:** Analyze the winning game tree.  Determine if
it is reproducible or a one-off adjudication fluke.  Build a targeted
strategy around the winning line.

## Metrics to Record

For each game, record:
- Opening move (edge + color)
- All move scores (full progression)
- Final score
- Result (win/draw/loss)
- MCTS iterations and thinking time per move
- Tree depth reached
- Total simulations (should be ~3M for 5K iters)

For aggregate analysis:
- Score distribution by opening (mean, max, std)
- Loss rate by opening
- Game tree diversity (how many unique move-3 responses does AlphaQ play?)
- Any game with score > 0.9 (potential breakthrough)

## Implementation

The `alphaq_explorer` strategy already has Thompson Sampling for opening
selection.  To force systematic exploration of all 30 openings:

1. Override Thompson Sampling with round-robin opening selection
2. Each opening = (edge 0-14) × (color G or P) = 30 possibilities
3. Play each opening exactly 3 times
4. Record full diagnostics

Alternatively, heavily bias Thompson Sampling priors to force exploration
of under-tested openings (set virtual wins to 0, virtual draws to 0 for
each opening, ensuring each gets selected).

## References

- [ALPHAQ_UP_BOUNTY_ANALYSIS.md](ALPHAQ_UP_BOUNTY_ANALYSIS.md): Original
  Nash Equilibrium claim (§1-3), score analysis (§4), risk assessment (§5)
- [THE_MATHEMATICS_OF_TANGLED_GAME.md](THE_MATHEMATICS_OF_TANGLED_GAME.md):
  Nash Equilibrium "proof" (§10.4), Thompson Sampling theory (§10.1-10.3)
- [ALPHAQ_STRATEGY.md](ALPHAQ_STRATEGY.md): Thompson Sampling implementation
  (§1-3), opening exploration results (§2.1), winning-push heuristic (§5)
- [SCORE_OUTCOME_DISCREPANCY.md](SCORE_OUTCOME_DISCREPANCY.md): SA score vs
  win probability analysis (§3), calibration methodology (§6-7)
- [ALPHAQ_THOMPSON_SAMPLING_MID_RUN_ANALYSIS.md](ALPHAQ_THOMPSON_SAMPLING_MID_RUN_ANALYSIS.md):
  Mid-run analysis of Run 60 opening exploration
- [LUT_TERMINAL_EVALUATION.md](LUT_TERMINAL_EVALUATION.md): LUT design and
  accuracy analysis
