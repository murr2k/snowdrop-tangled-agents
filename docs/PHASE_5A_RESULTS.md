# Phase 5A Results — Diagnosing the R²=0.56 Ceiling

**Date:** 2026-05-18
**Phase 5A of:** `docs/INVESTIGATION_5_TENSOR_NETWORKS.md`
**Verdict:** R²=0.56 is the **inherent ceiling** of the Schrödinger TFIM
model at the Advantage2.1.3 schedule. Parameter tuning cannot improve it.
**Phase 5B (alternative model) is required.**

---

## Summary

Phase 5A ran three diagnostics on the current `terminal_scores.mat` calib
oracle versus 1452 website-scored boards from the `calibration` table.
The combined finding is unambiguous: the Schrödinger TFIM model at the
Advantage2.1.3 schedule has a hard ceiling around R²=0.56 against the
website, and the model itself — not its parameters — is the limit.

| Sub-task | Conclusion |
|----------|-----------|
| 5A.1 residual analysis | Structured residual; R² varies from 0.24 (G=4–6) to 0.84 (G=12–14); R² collapses from 0.82 (4 frustrated 5-cycles) to −0.02 (12 frustrated); bias −1.02 on website-negative boards |
| 5A.3 eigsh adiabatic-limit test | Website is NOT using ground-state evaluation; best ground-state R² across s∈[0.5, 0.999] = 0.16 vs calib 0.35 on same 50-board sample |
| 5A.2 joint parameter sweep | 200 combos (10 anneal_times × 5 s_max × 4 sched_reds) on 1452 boards; best R² = 0.5618 at current calib parameters; no improvement found |

---

## 5A.1 — Residual structure (script: `scripts/analyse_calib_residuals.py`)

Residual stratification reveals strong patterns:

### By Green count
| G in | n | R² | bias |
|------|--:|---:|-----:|
| [2, 4) | 46 | 0.38 | −0.08 |
| [4, 6) | 213 | 0.24 | −0.27 |
| [6, 8) | 527 | 0.42 | −0.27 |
| [8, 10) | 465 | 0.54 | −0.21 |
| [10, 12) | 178 | 0.80 | −0.07 |
| [12, 14) | 21 | 0.84 | +0.08 |

→ The oracle is dramatically worse on AFM/Purple-heavy boards.

### By 5-cycle frustration
| frust | n | R² |
|------:|--:|---:|
| 4 | 246 | +0.82 |
| 6 | 722 | +0.57 |
| 8 | 437 | +0.31 |
| 12 | 34 | −0.02 |

→ Frustration is the dominant error axis.

### By website-score sign
| Bucket | n | mean residual |
|--------|--:|-------------:|
| website > +0.1 | 731 | −0.05 |
| website < −0.1 | 222 | **−1.02** |
| |website| ≤ 0.1 | 499 | −0.11 |

→ The oracle systematically under-predicts the magnitude of P2 wins by ~1 unit.

### Linear rescale
Linear fit `website = 0.94 × calib − 0.18` improves R² by only +0.03 (0.56 → 0.59). Per-board values are individually wrong, not just scaled wrong.

**Plot:** `plots/phase5a_calib_residuals.png` (regenerate with the script).

---

## 5A.3 — eigsh independent reference (script: `scripts/calib_eigsh_reference.py`)

Computed exact ground states of H(s) for 50 random boards at six s values
in [0.5, 0.999], then compared the resulting scores to both website and calib:

| s value | R² vs website (n=50) |
|---------|--------------------:|
| 0.7 | +0.121 |
| 0.8 | **+0.156** (best ground state) |
| 0.9 | +0.146 |
| 0.95 | −0.052 |
| 0.99 | −0.173 |
| 0.999 | −0.056 |
| **calib (1.85 ns evolution)** | **+0.352** (same 50-board sample) |

The classical-Ising ground state at s_max=0.999 collapses to only 10
distinct scores across 50 boards (massive degeneracy in the Petersen
Ising). Even at the best s in [0.5, 0.9] where the transverse field
lifts the degeneracy, the ground state R² is half of calib's short-time
dynamics R² on the same sample.

→ The website does NOT compute a ground state. Calib's short-time
evolution captures real structure that no ground state does.

---

## 5A.2 — Joint parameter sweep (script: `scripts/calib_parameter_sweep.py`)

200 combos (10 × 5 × 4) on the full 1452-board set, MATLAB split-operator
solver in parfor (6 workers, 8.2 hr total).

### Sweep grid

| Parameter | Values |
|-----------|--------|
| anneal_time (ns) | 0.1, 0.3, 0.7, 1.0, 1.85, 3.0, 5.0, 10.0, 30.0, 100.0 |
| s_max | 0.5, 0.7, 0.9, 0.99, 0.999 |
| sched_red | 0.25, 0.5, 1.0, 2.0 |

### Top 5 combos by R²

| Rank | tf (ns) | s_max | sched_red | R² | RMSE | bias |
|------|--------:|------:|----------:|---:|-----:|-----:|
| 1 | 1.85 | 0.99 | **0.5** | **+0.5618** | 0.8943 | −0.2174 |
| 2 | 1.85 | 0.999 | 0.5 | +0.5618 | 0.8943 | −0.2174 |
| 3 | 1.85 | 0.9 | 0.5 | +0.5618 | 0.8943 | −0.2174 |
| 4 | 1.85 | 0.7 | 0.5 | +0.5618 | 0.8943 | −0.2174 |
| 5 | 1.85 | 0.5 | 0.5 | +0.5609 | 0.8953 | −0.2155 |

The top 5 (and the next ~10) are all at sched_red=0.5, tf=1.85 ns
(the existing calib parameters), with s_max irrelevant once ≥ 0.5.
**No combo in the entire 200-combo grid improves on the baseline.**

### Best tf per sched_red

| sched_red | best tf | best R² | tf × sched_red |
|----------:|--------:|--------:|---------------:|
| 0.25 | 3.00 | +0.5350 | 0.75 |
| **0.5** | **1.85** | **+0.5618** | **0.93** |
| 1.0 | 1.00 | +0.5485 | 1.00 |
| 2.0 | 0.30 | +0.5323 | 0.60 |

The optimum lies on a curve where `anneal_time × sched_red ≈ constant`.
This is the expected physical degeneracy: the dimensionless evolution
scale is `tf × schedule_amplitude`, so varying both axes is redundant.
Even traversing this degenerate axis finds no improvement above 0.56.

### s_max irrelevance

| s_max | best R² (across tf, sched_red) |
|------:|------------------------------:|
| 0.5 | +0.5609 |
| 0.7 | +0.5618 |
| 0.9 | +0.5618 |
| 0.99 | +0.5618 |
| 0.999 | +0.5618 |

→ The wavefunction equilibrates before s=0.5; nothing to gain from
stopping the anneal earlier.

---

## What Phase 5A rules out

- **Longer anneal times.** (5A.3 + 5A.2 both confirm — adiabatic limit
  is worse than calib, and tf ≥ 10 ns collapses R² to < 0.2.)
- **Different anneal_time alone at the current schedule.** The optimum
  is already at 1.85 ns.
- **Different sched_red at any anneal_time.** Degenerate axis; no
  off-curve improvement exists.
- **Stopping the anneal early (s_max).** Irrelevant once s_max ≥ 0.5.

## What Phase 5A points toward (for Phase 5B)

The structured residual on frustrated/AFM-heavy boards and the
website-not-adiabatic finding together point at **model error**, not
parameter error. Candidate alternative models to test in Phase 5B:

1. **Different annealing schedule file.** Try Advantage System6 or D-Wave
   Pegasus schedules instead of Advantage2.1.3. The Δ/A envelope shapes
   differ across hardware generations and could substantially change the
   short-time dynamics.

2. **Finite-temperature TFIM.** Replace the pure ground state expectation
   with thermal average ⟨σᶻᵢσᶻⱼ⟩_β. Tune β alongside anneal parameters.
   Finite temperature breaks ground-state degeneracy and can mimic
   "soft" measurement.

3. **Open-system dynamics (Lindblad).** Add phenomenological dephasing /
   decoherence terms; the website's hardware necessarily has finite
   coherence time.

4. **QMC adjudicator.** The snowdrop_adjudicators package may include a
   QMC-based adjudicator; the website may be using that instead of
   pure Schrödinger evolution.

5. **Different score formula.** Re-derive the score from the website
   data by sweeping different functional forms (raw correlations, sum
   of magnetizations, etc.). Possible the website uses something other
   than `influence[p1] − influence[p2]`.

The Phase 5A.1 residual structure should be the test bed: any candidate
model that significantly reduces the **frustration-stratified R²
collapse** (0.82 → −0.02) is a contender for the new oracle.

---

## Phase 5A artefacts

| File | Purpose |
|------|---------|
| `scripts/analyse_calib_residuals.py` | 5A.1 residual analysis |
| `scripts/calib_eigsh_reference.py` | 5A.3 eigsh independent reference |
| `scripts/calib_parameter_sweep.py` | 5A.2 launcher (calls MATLAB; `--analyse-only` mode for replays) |
| `snowdrop_tangled_agents/matlab/rl/parameter_sweep_schrodinger.m` | 5A.2 MATLAB sweep with parfor |
| `snowdrop_tangled_agents/matlab/rl/data/phase5a2_sweep.mat` | 5A.2 raw result (200 combo × 1452 boards) |
| `plots/phase5a_calib_residuals.png` | 5A.1 plot (gitignored) |

The current calib oracle (`terminal_scores.mat` at sr=0.5, sm=0.999, tf=1.85)
remains the best Schrödinger-TFIM-based oracle. No parameter change is
warranted. Phase 5B effort should focus on alternative physical models or
schedules.
