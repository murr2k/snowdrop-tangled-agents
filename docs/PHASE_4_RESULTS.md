# Phase 4 Results — Expected-Value Solver vs AlphaQ

**Generated:** 2026-05-17
**Run:** 141 (50 games, 2026-05-17 16:48–18:17 UTC, P1 seat, calib LUT, hybrid_solver, `--solver-adversary expected`)
**Opponent model:** `alphaq_policy_mlp.mat` (MLP 64,64; Phase 2 top-1 = 0.866)

---

## Verdict

**PHASE 5 PIVOT to tensor networks.**

The expected-value reformulation produced **zero wins** and **catastrophically degraded
mean score** versus the most recent minimax baseline. Welch's t-test reports
`p < 1e-30`, mean shift `-1.36` in the unfavorable direction. This is not a
tuning problem — it is the failure mode the Phase 2 model card explicitly warned
about (state coverage; calibration degrades on OOD states reached by the new
solver) compounding with the Phase 3 finding (oracle is unreliable on AlphaQ's
basin, R² = −0.94 raw SA, R² = −0.56 Melissa-fit Schr).

---

## Results

| Run | Mode | Strategy | LUT | n | W/D/L | Mean score | Std | Range |
|-----|------|----------|-----|---|-------|-----------:|-----|-------|
| 141 (this) | **expected** | hybrid_solver | calib | 50 | 0 / 0 / 50 | **−0.669** | 0.048 | [−0.79, −0.51] |
| 130 (baseline) | minimax | hybrid_solver | (sa) | 10 | 0 / 0 / 10 |  +0.687 | 0.022 | [+0.66, +0.73] |

Welch's t-test on score: `t = −141.16, p ≈ 0`.
Mann-Whitney U: `U = 0.0, p ≈ 0`.
Score distributions are completely disjoint: every Phase 4 game ended in
`[−0.79, −0.51]`; every baseline game ended in `[+0.66, +0.73]`.

### Phase 4 score histogram (n = 50)
```
[-1.0, -0.7):  11 ###########
[-0.7, -0.5):  39 #######################################
[-0.5, -0.2):   0
[-0.2, +0.2):   0
[+0.2, +0.5):   0
[+0.5, +0.7):   0
[+0.7, +1.0):   0
```

### Minimax baseline score histogram (n = 10)
```
[+0.5, +0.7):   8 ########
[+0.7, +1.0):   2 ##
```

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
solver is pulled out of the (0, 2) draw zone (where minimax was operating)
and into the negative-score basin AlphaQ owns.

This is consistent with the Phase 3 finding that the calibrated oracle has
**R² = −0.94 raw SA, R² = −0.56 Schr** on the AlphaQ basin — the LUT values
the solver is maximising expectations over are themselves unreliable in this
region. Phase 4 amplified that unreliability into a systematic loss pattern.

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

## Decision (per project plan §Phase 4)

| Plan rule | This run | Action |
|-----------|----------|--------|
| Any wins | No (0 wins) | — |
| Zero wins, mean score significantly higher than minimax baseline | No (mean shift = −1.36) | — |
| Zero wins, mean score unchanged from minimax | No (significantly worse, p≈0) | — |
| _(extended)_ Zero wins, mean score significantly lower than minimax | **Yes** | **PHASE 5 PIVOT** |

Pivot to Investigation 5 (tensor-network simulation, MPS/DMRG) per the
existing plan. Investigation 5 was already conditioned on Phase 4 producing
zero wins; the additional negative-score signal strengthens that
recommendation rather than weakening it.

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
