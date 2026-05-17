# Phase 4 Results — Expected-Value Solver vs AlphaQ

**Generated:** 2026-05-17
**Run:** 141 (50 games, 2026-05-17 16:48–18:17 UTC, P1 seat, calib LUT, hybrid_solver, `--solver-adversary expected`)
**Opponent model:** `alphaq_policy_mlp.mat` (MLP 64,64; Phase 2 top-1 = 0.866)

---

## Verdict

**PHASE 5 PIVOT to tensor networks.**

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

## Decision (per project plan §Phase 4)

| Plan rule | This run | Action |
|-----------|----------|--------|
| Any wins | No (0 wins) | — |
| Zero wins, but draw rate ≥ minimax baseline | No (0% vs baseline 49.4%) | — |
| Zero wins, mean score significantly higher than minimax baseline | No (−0.67 vs +0.20) | — |
| _(extended)_ Zero wins, draw rate collapsed and score basin worse | **Yes** | **PHASE 5 PIVOT** |

The core empirical fact: the minimax `hybrid_solver` baseline produces draws
in 49.4% of P1 games against AlphaQ. Phase 4 produced zero. The expected-
value reformulation eliminated the entire draw plateau the minimax solver
was reliably hitting, replacing it with losses in a worse score basin.

Pivot to Investigation 5 (tensor-network simulation, MPS/DMRG) per the
existing plan. The negative result here is informative: it confirms that
classical exploit of AlphaQ requires either a better value function (which
tensor networks would provide) or a better predictive policy at the exploit
candidate states (which is bounded by AlphaQ's intrinsic 38.9% indeterminism
and cannot improve beyond that).

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
