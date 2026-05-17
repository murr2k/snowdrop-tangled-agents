# Investigation 3 Results — AlphaQ-Conditional Calibration

**Generated:** 2026-05-17
**Source:** `~/.tangled/game_stats.db` (calibration table, joined to games for opponent)

---

## Corpus

| Opponent | Distinct boards | Total observations |
|----------|-----------------|--------------------|
| Melissa | 1338 | (varies) |
| AlphaQ | 102 | (varies) |
| Overlap (both) | 17 | — |

AlphaQ's basin is roughly 13x narrower than Melissa's, consistent with the empirical closure paper's finding that AlphaQ's adversarial policy constrains the reachable terminal-state space.

---

## SA proxy: raw and linear-refit R²

| Basin | N boards | R² (raw SA) | R² (linear refit a·SA + b) | Slope a | Intercept b |
|-------|----------|-------------|----------------------------|----------|-------------|
| Melissa | 1338 | 0.6333 | 0.7009 | 0.8392 | -0.1500 |
| AlphaQ  | 102 | -0.9384 | 0.0024 | 0.0749 | -0.3383 |

**Key finding:** raw SA R² on the AlphaQ basin is **-0.9384** — strongly negative, meaning SA predictions are *worse than predicting the mean*. The best linear refit recovers only R² = 0.0024; the fitted slope (0.0749) is near zero, confirming that the SA signal carries essentially no information about website outcomes on AlphaQ-reachable boards. This is the closure paper's polarity inversion finding (r = −0.396 in that work) made concrete and quantitative on the calibration corpus.

---

## Melissa-fitted MATLAB Schrödinger oracle on each basin

The existing `matlab_calib_results.mat` was produced by Investigation 3 with anneal_time = 1.85 ns and global R² = 0.6047 across 1061 boards (Melissa-dominated). Subsetting its predictions to the boards actually observed in each basin:

| Basin | N matched | R² (Melissa-fitted Schr oracle) | Win/draw/loss classification accuracy |
|-------|-----------|--------------------------------|---------------------------------------|
| Melissa | 957 | 0.7375 | 0.8401 |
| AlphaQ | 101 | -0.5579 | 0.6832 |

**On the AlphaQ basin, R² = -0.5579 — negative.** The Melissa-fitted Schrödinger oracle is no better than (and potentially worse than) predicting the AlphaQ-basin mean. The 1.85 ns anneal-time fit, while explaining 60% of variance globally, does not generalise to the boards where we actually need predictions for AlphaQ-game decisions.

---

## Verdict and next step

**Decision-gate result: NONE — neither SA nor the Melissa-fitted Schrödinger oracle has meaningful predictive power on the AlphaQ basin.** An AlphaQ-fitted Schrödinger calibration would be at best a marginal improvement: the 102-board sample is small, the score variance is dominated by adjudicator noise within the narrow AlphaQ basin (≈81% of boards land in the (0, 2) draw zone), and the residual structure is unlikely to be linearly captured by a one-parameter (anneal_time) Schrödinger model.

**Recommendation:** Phase 4 proceeds with the existing calibrated oracle (`expanded_lut_calib.mat`) as the value function. Do not block on an AlphaQ-conditional MATLAB calibration. The expected-value reformulation against the predictive policy (Phase 2) is the dominant source of expected improvement; the value-function residual error is second-order.

---

## Optional: producing an AlphaQ-fitted MATLAB calibration

`scripts/investigation_3_alphaq_calibration.py` exported the AlphaQ calibration corpus to:

  `snowdrop_tangled_agents/matlab/rl/data/calibration_boards_alphaq.mat`

To produce a true AlphaQ-fitted Schrödinger model, run in MATLAB:

```matlab
cd snowdrop_tangled_agents/matlab/rl
calibrate_schrodinger('../../data/calibration_boards_alphaq.mat')
% writes data/matlab_calib_results_alphaq.mat
```

Then in Python:

```bash
poetry run python scripts/calibrate_adjudicator.py \
  --load-matlab-results snowdrop_tangled_agents/matlab/rl/data/matlab_calib_results_alphaq.mat \
  --opponent alphaq
```

Given the analysis above this is exploratory rather than blocking; Phase 4 can proceed with the existing oracle.
