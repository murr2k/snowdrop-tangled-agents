# Phase 4 Results — Expected-Value Solver vs AlphaQ

**Generated:** 2026-05-17
**Run:** 141 (50 games, 2026-05-17 16:48–18:17 UTC, P1 seat, calib LUT, hybrid_solver, `--solver-adversary expected`)
**Opponent model:** `alphaq_policy_mlp.mat` (MLP 64,64; Phase 2 top-1 = 0.866)

---

## Verdict — REVISED 2026-05-17

**PIVOT VERDICT RETRACTED.** **Phase 4 test was confounded by opening
selection.** The expected-value solver was never tested in the relevant
configuration.

The Phase 4 run picked `E0G` as its opening in all 50 games (this is what
MCTS converges to at grey=15 when the opening book is gated off and there's
no explicit forced opening). E0G is a *known-bad* opening: in the broader
hybrid_solver history, 17 games on E0G produced **0 draws and 16 losses**.

The historical record shows that **E7G as P1 opening produces a 96.2% draw
rate across 262 games against AlphaQ** (252 D / 0 W / 5 L, mean score +0.55).
When hybrid_solver direct happens to pick E7G, its draw rate is 5/5 = 100%.
This is a strong, reproducible classical baseline that the Phase 4 test
never exercised.

**Correct next step:** re-run Phase 4 expected-value mode with a forced E7G
opening, comparing 50 games to the E7G+other-solvers baseline (96.2% draws).
Decision rule:
  - Any wins → exploit found, scale.
  - Draws ~ 96.2% → expected-value mode is neutral on top of E7G; no harm done.
  - Significantly fewer draws → expected-value mode actively breaks the E7G
    advantage; THEN pivot.

**Blocker:** tangled-game.com modified its play field (2026-05-17). Our
Playwright automation in `play_tangled.py` will need updates before any
further live games can be played. The re-run is queued behind that work.

The expected-value reformulation **eliminated every draw**: 49.4% of historical
`hybrid_solver` P1 games against AlphaQ ended in draws; Phase 4's 50 games
ended in zero draws and 50 losses, all in a strictly worse score basin than
the historical loss distribution. This is the failure mode the Phase 2 model
card explicitly warned about (state coverage; calibration degrades on OOD
states reached by the new solver) compounding with the Phase 3 finding
(oracle is unreliable on AlphaQ's basin, R² = −0.94 raw SA, R² = −0.56
Melissa-fit Schr).

---

## Results

| Mode | Strategy | n | W/D/L | Draw rate | Mean score | Std | Range |
|------|----------|---:|-------|-----------|-----------:|-----|-------|
| **expected** (Phase 4, run 141) | hybrid_solver | 50 | **0 / 0 / 50** | **0%** | **−0.669** | 0.048 | [−0.79, −0.51] |
| minimax (all-time hybrid_solver P1 history excl. run 141) | hybrid_solver | 83 | 0 / 41 / 42 | 49.4% | +0.202 | 0.515 | [−0.95, +0.81] |
| minimax (run 130 only, most recent prior baseline) | hybrid_solver | 10 | 0 / 0 / 10 | 0% | +0.687 | 0.022 | [+0.66, +0.73] |

### Opening-conditioned breakdown (the real story)

Phase 4 picked E0G on all 50 games. The historical opening-vs-outcome
table (all P1 games vs AlphaQ, all strategies, query in
`scripts/_phase4_actual_openings.py`):

| Opening | n | W / D / L | Draw rate | Mean score |
|---------|--:|-----------|-----------|-----------:|
| **E7G** | **262** | **0 / 252 / 5** | **96.2%** | **+0.55** |
| E0G (what Phase 4 used) | 67 | 0 / 0 / 66 | 0% | varies |
| E9G | 49 | 0 / 28 / 21 | 57.1% | +0.30 |
| E5P | 8 | 0 / 8 / 0 | 100% | +0.02 |

The "correct" minimax baseline for the Phase 4 question is the E7G row.
Phase 4 needed to be tested against E7G's 96.2% draw rate, not against
the unconditional baseline that mixes E7G with much worse openings.

**The 49.4% draw rate is the critical baseline.** Run 130 was an outlier
"vertex-tiebreak loss" cluster in the +0.7 score basin; the broader historical
record shows the minimax-mode hybrid_solver alternates between (~50%) draws
(score near zero) and (~50%) losses in either the negative basin or the
+0.7 vertex-tiebreak basin. Phase 4 collapsed both populations into a single
loss basin at −0.67.

Welch's t-test on score (Phase 4 vs run 130 specifically):
`t = −141.16, p ≈ 0` — but this comparison is misleading because run 130
was an outlier. The fair statement is the W/D/L collapse: Phase 4 destroyed
the ~41 draws (49% of 83) that the baseline produces in this configuration.

### Score histograms

Phase 4 (n = 50, all losses):
```
[-1.0, -0.7):  11 ###########
[-0.7, -0.5):  39 #######################################
[-0.5, -0.2):   0
[-0.2, +0.2):   0
[+0.2, +0.5):   0
[+0.5, +0.7):   0
[+0.7, +1.0):   0
```

Minimax hybrid_solver baseline (n = 83, 41 draws + 42 losses):
```
[-2.0, -0.7):    9 #########
[-0.7, -0.5):    0
[-0.5, -0.2):    5 #####
[-0.2, +0.2):   28 ############################   <- the draw plateau
[+0.2, +0.5):   10 ##########
[+0.5, +0.7):   13 #############
[+0.7, +1.0):   18 ##################              <- vertex-tiebreak loss basin
```

The baseline is bimodal: a draw plateau near zero (28 games) and a positive
vertex-tiebreak loss cluster (18+13 = 31 games at +0.5..+1.0). Phase 4 has
neither — every game collapses into one tight basin around −0.67.

---

## Interpretation

### What went wrong

The Phase 2 MLP achieves 86.6% top-1 prediction accuracy on AlphaQ's response,
but only **38.9%** on the 6 exploit candidate states (the binary 60/40 choice
points the solver was designed to attack). At those states, the
expected-value calculation `max_e E_pi[LUT(grandchild_after_alphaq_response)]`
weights two distinct grandchild positions by their predicted probability and
picks the move with the highest weighted average.

When the model's prediction is wrong (61% of the time at the exploit
candidates), the solver lands in a grandchild state with a *worse* LUT value
than the minimax-worst child it would have produced under
`max_e min_resp LUT(grandchild)`. Over 8 turns of game, this compounds: the
solver is pulled out of the (0, 2) draw zone (where minimax was operating
~50% of the time) and into the negative-score basin AlphaQ owns.

This is consistent with the Phase 3 finding that the calibrated oracle has
**R² = −0.94 raw SA, R² = −0.56 Schr** on the AlphaQ basin — the LUT values
the solver is maximising expectations over are themselves unreliable in this
region. Phase 4 amplified that unreliability into a systematic loss pattern,
collapsing the bimodal minimax outcome distribution (draw plateau + vertex-
tiebreak loss cluster) into a unimodal loss basin.

### Why this is not a tuning issue

The two failure modes (predicted policy inaccurate on exploit candidates;
oracle unreliable past AlphaQ's response) **interact multiplicatively**.
Tuning either knob within the current architecture cannot fix this:

- A more accurate policy (deeper MLP, more features) would reduce the
  prediction error at the exploit candidates, but the 61% residual at those
  states reflects AlphaQ's intrinsic indeterminism, not a feature-set limit
  (the Phase 1 entropy analysis established this).
- A more accurate oracle on the AlphaQ basin requires either (a) much more
  AlphaQ-conditional calibration data (Phase 3 had 102 boards and found no
  signal) or (b) a fundamentally different value function — which is what
  tensor networks would provide.

The minimax baseline was already losing every game in this seat/opponent
configuration (run 130 is 0W/0D/10L). Phase 4 didn't change the W/D/L axis;
it merely repositioned where the losses happen on the score axis, in the
direction that confirms the value-function unreliability hypothesis.

### Why this is still a valid scientific result

The plan was designed to make the classical-defeat question binary within two
weeks. It has done so. The expected-value reformulation is the strongest
classical-strategy improvement available given the existing oracle and the
existing policy model. It failed. The conclusion follows: classical exploit of
AlphaQ on the Petersen graph is unlikely, and progress requires a better value
function — which is the Phase 5 / Investigation 5 tensor-network agenda.

---

## Decision — REVISED

The initial verdict was PIVOT. **That verdict is retracted.** The Phase 4
test did not exercise the relevant configuration (forced E7G opening), so
it does not bear on the question of whether the expected-value mode helps
or hurts vs. the established E7G classical baseline.

Re-test queued, blocked on site automation updates:

```bash
# Once play_tangled.py is updated for the new tangled-game.com play field:
poetry run python play_tangled.py \
    --opponent alphaq --strategy hybrid_solver --lut-variant calib \
    --solver-adversary expected --oracle-override 15 7 G \
    --games 50 --headless
```

Decision rule for the re-test (vs n=262, 96.2% draw E7G baseline):

| Re-test outcome | Action |
|-----------------|--------|
| Any wins | Exploit found — scale to 500 games, characterise winning lines |
| ~96% draws (no significant change) | Expected-value mode is neutral on top of E7G; keep it but no breakthrough |
| Significantly fewer draws than 96% | Expected-value mode breaks E7G; revert to minimax+E7G, then pivot |

If pivot is eventually warranted, Investigation 5 (tensor-network value
function) remains the right path — but the case for pivoting is much
weaker now that we have empirical evidence of a 96% draw classical
baseline that the existing oracle + E7G opening achieves.

---

## Run reproducibility

```bash
poetry run python play_tangled.py \
    --opponent alphaq \
    --strategy hybrid_solver \
    --lut-variant calib \
    --solver-adversary expected \
    --games 50 --headless
```

Analysis: `poetry run python scripts/_phase4_analyse_results.py`

Raw data in `~/.tangled/game_stats.db`, run id 141.

---

## What stays valid

The Phase 2 predictive model (`alphaq_policy_mlp.mat`) remains a usable
characterisation of AlphaQ's policy and may be useful as input to the tensor-
network value function (e.g., as a prior for variational MPS optimisation).
The MATLAB-side `AlphaQPolicy.m` loader/featuriser and the
`HybridTangledSolver.AdversaryMode='expected'` code path remain in the
codebase for re-use; they are correct (8/8 unit tests pass, sklearn-vs-mat
parity verified < 2e-7) but inappropriate for live play given this result.
