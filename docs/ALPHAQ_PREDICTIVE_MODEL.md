# AlphaQ Predictive Policy Model — Phase 2 Model Card

**Generated:** 2026-05-17
**Source corpus:** local `~/.tangled/game_stats.db`
**Method:** supervised classification on AlphaQ's empirical move distribution conditional on board state.

---

## Corpus and split

| Metric | Value |
|--------|-------|
| AlphaQ games | 1574 |
| Decision observations | 9341 |
| Train games | 1260 |
| Test games | 314 |
| Train moves | 7447 |
| Test moves | 1894 |
| Feature dimension | 92 |
| Action space | 30 (15 edges x 2 colors) |

Train/test split is by game_id with 20% holdout, so test-set moves are from games not seen in training.

---

## Feature set

| Block | Indices | Count | Description |
|-------|---------|-------|-------------|
| Per-edge state | 0..44 | 45 | 3-way one-hot (grey, green, purple) per edge |
| Per-vertex degree | 45..74 | 30 | green / purple / total coloured degree per vertex |
| 5-cycle frustration | 75..86 | 12 | per-cycle parity of purple count (1=frustrated, 0=satisfied, 0.5=incomplete) |
| Aggregate counts | 87..89 | 3 | grey / green / purple fractions of 15 |
| Parity | 90..91 | 2 | own-turn flag and grey-count parity |

---

## Model accuracy

| Model | Top-1 | Top-3 | Top-5 | Training time (s) |
|-------|-------|-------|-------|-------------------|
| LogReg | 0.8511 | 0.8680 | 0.8728 | 1.04 |
| MLP (64,64) | 0.8659 | 0.8754 | 0.8759 | 1.11 |

Top-k is measured on held-out test games. Predictions are masked to legal actions (only grey edges) and renormalised before ranking.

### Per-grey-count top-1 accuracy

| Grey | LogReg | MLP (64,64) | N test |
|------|---|---|--------|
| 1 | 0.000 | 0.000 | 30 |
| 2 | 0.965 | 0.969 | 227 |
| 3 | 0.000 | 0.000 | 30 |
| 4 | 0.937 | 0.955 | 222 |
| 5 | 0.000 | 0.000 | 30 |
| 6 | 0.961 | 0.974 | 229 |
| 7 | 0.000 | 0.000 | 30 |
| 8 | 0.917 | 0.946 | 242 |
| 9 | 0.000 | 0.000 | 28 |
| 10 | 0.980 | 0.984 | 247 |
| 11 | 0.032 | 0.032 | 31 |
| 12 | 0.951 | 0.976 | 246 |
| 13 | 0.000 | 0.000 | 30 |
| 14 | 0.978 | 1.000 | 272 |

---

## Performance on exploit candidate states (Phase 1 output)

Top-1 accuracy on the 6 exploit candidate states identified in Investigation 2. These are the states with n >= 10 observations and response entropy >= 0.5 bits — the primary search targets for Phase 4.

| Model | Top-1 on exploit candidates | N test samples on these states |
|-------|-----------------------------|--------------------------------|
| LogReg | 0.389 | 18 |
| MLP (64,64) | 0.389 | 18 |

If the model's top-1 accuracy on these states is meaningfully below its overall top-1, that confirms the entropy at these states is real (the model can't reduce it because AlphaQ genuinely picks differently). This is the favourable signal for Phase 4: the expected-value solver can exploit the response variance the model itself cannot collapse.

---

## Known failure modes

1. **Class imbalance.** Some (edge, color) actions are rarely played by AlphaQ. Logistic regression's per-class weight is uniform; rare classes are under-predicted. MLP captures more but still penalises tail classes.
1. **Within-game correlation.** Consecutive AlphaQ moves in the same game share state ancestry. By-game splitting controls for this in test evaluation, but training samples are not strictly i.i.d.
1. **State coverage.** AlphaQ's reachable basin is narrow (see closure paper). The model's predictions outside this basin (e.g. for states reached via the expected-value solver in Phase 4) are extrapolation, not interpolation. Calibration may degrade.
1. **Quantum adjudicator unknown.** This model predicts AlphaQ's behaviour. It says nothing about whether a position is winning under the website's quantum scorer. Pairing with the calibrated oracle (Phase 3) is required for full Phase 4 utility.

---

## Persisted artefacts

- `snowdrop_tangled_agents/matlab/rl/data/alphaq_policy_logreg.pkl` (sklearn pickle, Python use)
- `snowdrop_tangled_agents/matlab/rl/data/alphaq_policy_logreg.mat` (weights for MATLAB Phase 4 consumption)
- `snowdrop_tangled_agents/matlab/rl/data/alphaq_policy_mlp.pkl` (sklearn pickle, Python use)
- `snowdrop_tangled_agents/matlab/rl/data/alphaq_policy_mlp.mat` (weights for MATLAB Phase 4 consumption)
- `docs/ALPHAQ_PREDICTIVE_MODEL.md` (this file)

---

## Decision-gate interpretation

**Strong predictor.** Best top-1 is 0.866 (MLP (64,64)). The model can be used directly as a hard policy approximation for AlphaQ in the Phase 4 expected-value solver.
