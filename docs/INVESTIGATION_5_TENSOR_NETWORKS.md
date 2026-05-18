# Investigation 5 — Tensor Network / High-Fidelity Oracle

**Status:** 🎯 Active (triggered 2026-05-18 by Phase 4 decision gate)
**Goal:** Build a high-fidelity oracle that addresses the underlying value-function unreliability that defeated Phases 3 and 4, then re-test for wins against AlphaQ Up.

---

## Why this investigation, why now

Phases 1–4 of the AlphaQ-targeted plan ruled out the classical exploit
approach on top of the existing oracle:

| Phase | Result |
|-------|--------|
| 1 — MI/entropy analysis | 6 exploit-candidate states identified at grey=2/5/8 |
| 2 — Predictive policy | MLP top-1 = 0.866 overall, 38.9% on exploit candidates (residual entropy is intrinsic) |
| 3 — AlphaQ-conditional calibration | Oracle R² on AlphaQ basin = −0.56 (calib) / −0.94 (SA raw); verdict NONE |
| 4 — Expected-value solver | Run 142 (E7G + expected): 6.3% draws vs 70% minimax baseline; expected-value mode strictly worse |

The fundamental blocker is **value-function unreliability on the boards
AlphaQ actually reaches.** The current LUTs are built from:
- SA: stochastic terminal evaluations, errors compound through 13 DP levels
- Schr (40 ns): deterministic but wrong parameters (matched to MATLAB local solver, not website)
- Calib (1.85 ns): correct parameters, but only achieves R²=0.60 vs website scores

A solver that picks moves to maximise an unreliable value function cannot
exploit a sophisticated opponent. Improving the oracle is the root-cause
intervention.

---

## Hypothesis

The website's Schrödinger adjudicator can be matched to **R² > 0.95** by
either:

- (A) using the existing exact-diagonalization solver at *additional* recovered
  parameters that Investigation 3 didn't search (e.g., different anneal
  schedule, finite-temperature effects, decoherence terms), or
- (B) using a fundamentally different value computation (MPS/DMRG ground
  state of a slightly different Hamiltonian, or a non-quantum reference
  the website actually uses behind the "Schrödinger adjudicator" label).

Either path produces a terminal LUT that, after retrograde DP, gives an
internally-consistent oracle on the AlphaQ basin. The Phase 4 minimax+E7G
solver re-applied with this oracle should produce either wins or a
significantly elevated draw rate above the current 50–70% baseline.

---

## Method

### Phase 5A — Diagnose the R²=0.60 ceiling (3–5 days)

The single most leveraged question: *why does the current exact Schrödinger
solver, at calibrated parameters, only achieve R²=0.60 against website
scores?*

**Tasks:**

1. **Residual analysis.** Plot website_score vs calib_score for all 1,061
   calibration boards. Bin by:
   - terminal_state structure (purple-heavy, green-heavy, balanced)
   - score magnitude
   - graph metric (number of frustrated triangles, etc.)

   Look for systematic residual patterns (sign bias, scale mismatch, specific
   board families where the model fails).

2. **Parameter sensitivity.** Hold `anneal_time=1.85 ns`, vary one parameter
   at a time:
   - `epsilon` (draw boundary): currently 0; try 0.0001–0.01
   - `s_min`, `s_max`: currently (0.001, 0.999); try (0.01, 0.99) and (0.1, 0.9)
   - schedule envelope (`load_schedule_data`): the current schedule is "typical
     D-Wave" — verify the website isn't using a different anneal schedule

   Report R² vs each axis. Sharp peaks indicate parameters Investigation 3
   missed.

3. **Cross-check against an independent reference.** Compute exact ground
   states for 50 random terminal boards using scipy's `eigsh` on the sparse
   Hamiltonian directly (no time evolution, just T→∞ ground state at the
   final point of the schedule). Compare against both website scores AND the
   current calib values. This distinguishes "wrong parameters" from "wrong
   model".

**Decision gate:**

| Finding | Action |
|---------|--------|
| Residual is structured + parameter sweep finds a sharp peak | Re-fit; rebuild calib LUT at new parameters; skip to Phase 5C |
| R² stays ≤ 0.7 across all parameter choices | The Schrödinger model is wrong; proceed to Phase 5B |
| eigsh ground states match calib values but disagree with website | The website is doing something else entirely; consider what (annealing dynamics? finite-T sampling? not quantum at all?) |

### Phase 5B — Alternative model (1–2 weeks, conditional)

Only run this if Phase 5A says the Schrödinger TFIM model is inadequate.

Candidates:

1. **Finite-temperature TFIM.** Replace ground-state expectation with
   thermal average ⟨σᶻᵢ σᶻⱼ⟩_β at some effective temperature β. Tune β
   along with anneal parameters.

2. **Quantum Monte Carlo reference.** Snowdrop publishes a QMC adjudicator
   in the same repo family — check whether the website's "Schrödinger"
   adjudicator might actually be QMC under the hood. Fit QMC parameters
   to the 1,061 calibration boards.

3. **MPS/DMRG ground state at exact parameters.** Use `quimb` to build the
   MPS for the TFIM ground state at the schedule endpoint. For 10 qubits
   (Petersen) this is overkill but provides a different numerical path; for
   larger graphs (n=20+ X-Prize graphs) this becomes the only feasible path.

   - Dependency: `quimb` (~10 MB) — add to `pyproject.toml` if pursued
   - Test on Petersen first; ground state ⟨σᶻᵢσᶻⱼ⟩ must match eigsh to
     machine precision before scaling

**Decision gate:** any candidate reaching R² > 0.9 on a held-out 20% of the
1,061 boards becomes the new LUT source.

### Phase 5C — Rebuild and validate (1 week)

1. **Regenerate terminal LUT** with the new (parameters or model):
   ```bash
   poetry run python snowdrop_tangled_agents/tools/generate_terminal_lut.py \
       --graph 5 --scorer <new_scorer> \
       --output terminal_scores_v5.mat
   ```

2. **Rebuild expanded LUT** via existing retrograde DP — no pipeline changes:
   ```bash
   poetry run python scripts/generate_sa_oracle.py \
       --terminal-lut terminal_scores_v5.mat \
       --output-prefix oracle_v5 \
       --end-level 9
   ```

3. **Add `--lut-variant v5`** to `play_tangled.py` (same pattern as
   `calib`, `schr`, `sa`).

4. **Live validation: 50 games minimax+E7G** with the new oracle:
   ```bash
   poetry run python play_tangled.py \
       --opponent alphaq --strategy hybrid_solver --lut-variant v5 \
       --solver-adversary minimax --oracle-override 15 7 G \
       --games 50 --headless
   ```

**Decision gate (vs the 70% minimax+E7G+calib baseline from run 143):**

| Outcome | Action |
|---------|--------|
| Any wins | Exploit found — characterise winning lines, scale to 500 games, expand to other openings |
| Draw rate ≥ 90% | Significant improvement — investigate whether other openings (E2G, E8G, E10G) also unlock wins with the new oracle |
| 70–90% draws | Modest improvement — diagnose what's still wrong; consider running expected-value adversary again on top of the better oracle |
| ≤ 70% draws | New oracle isn't better in practice — close investigation; the game is effectively drawn under any reachable classical strategy |

### Phase 5D — MPS scaling (optional, 1–3 months)

Only if Phase 5C succeeds AND we want to apply the technique to larger
X-Prize graphs (12, 18, 19, 20 — up to ~20 vertices, 2²⁰ = 1M states beyond
ED feasibility).

- Implement MPS DMRG for variable-graph TFIM in `quimb`
- Bond-dimension sweep; convergence at fixed bond dim
- Benchmark vs ED on Petersen (must agree to machine precision)
- Generate terminal LUTs for graphs 12, 18, 19, 20
- Live games against AlphaQ Up on those graphs (if AlphaQ supports them)

This phase is deferred until the Petersen-graph result is established.

---

## Files this investigation will touch

| File | Action |
|------|--------|
| `scripts/analyse_calib_residuals.py` | CREATE — Phase 5A.1 residual analysis |
| `scripts/calib_parameter_sweep.py` | CREATE — Phase 5A.2 parameter sensitivity |
| `scripts/calib_eigsh_reference.py` | CREATE — Phase 5A.3 independent reference |
| `scripts/build_oracle_v5.py` | CREATE — Phase 5B/C oracle builder |
| `snowdrop_tangled_agents/tools/generate_terminal_lut.py` | MODIFY — add new scorer option |
| `play_tangled.py` | MODIFY — add `--lut-variant v5` |
| `pyproject.toml` | MAYBE MODIFY — add `quimb` if Phase 5B path B is taken |

## Critical reference data (no changes)

| Asset | Path | Role |
|-------|------|------|
| Calibration board scores | `calibration` table in `~/.tangled/game_stats.db` | 1,061 (terminal_state, website_score) pairs |
| Calibrated terminal LUT | `snowdrop_tangled_agents/matlab/rl/data/terminal_scores.mat` | Current best (R²=0.60) |
| Current website parameters | ε=0, anneal_time=1.85 ns | From Investigation 3 |
| Retrograde DP pipeline | `scripts/generate_sa_oracle.py` | Unchanged — accepts any terminal LUT |
| E7G baseline (current) | run 143 (minimax+E7G): 70% draws, mean −0.51 | The bar to beat |

---

## Risk and exit criteria

**Risk:** Phase 5A may find that no parameter choice or model swap closes the
R² gap. If the website uses an opaque/black-box adjudicator (not pure
TFIM Schrödinger), the gap may be irreducible without reverse-engineering
the website's actual implementation.

**Exit criteria (close investigation):**

- Phase 5A: R² stays ≤ 0.7 across every parameter and reasonable model
  variant AND eigsh independent reference matches calib values closely (i.e.,
  our model is internally consistent; the website is doing something else).
- Phase 5C: 50-game validation gives ≤ 70% draws (no improvement over run 143).

If exited, the program-level conclusion becomes: **classical defeat of
AlphaQ Up on the Petersen graph is unreachable with any oracle derivable
from the published Schrödinger adjudicator definition.** At that point the
program either pivots to a non-Petersen graph or closes.

---

## Reproducibility notes

The Phase 4 final state (the bar Phase 5 must clear):

```bash
# Best classical solver currently:
poetry run python play_tangled.py \
    --opponent alphaq --strategy hybrid_solver --lut-variant calib \
    --solver-adversary minimax --oracle-override 15 7 G \
    --games 50 --headless

# Result: 70% draws, 0 wins, mean score −0.51
```

The expected-value adversary mode (`--solver-adversary expected`) remains
in the codebase but should not be used in live play — it is strictly worse
than minimax on this game (run 142: 6.3% draws).

Run 143 raw data: `~/.tangled/game_stats.db`, run id 143.
