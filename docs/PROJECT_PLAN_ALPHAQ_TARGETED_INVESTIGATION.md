# Project Plan: AlphaQ-Targeted Investigation

**Murray Kopit**
2026-05-17

---

## Objective

Determine, within two weeks of focused work, whether AlphaQ Up admits any
exploit reachable by classical strategies on the Petersen graph. If yes,
demonstrate the win and scale exploitation. If no, document the rejection
and commit the program to the tensor-network path (Investigation 5) as the
only remaining route.

This plan implements the recommendations in
[`METHODOLOGICAL_REVIEW_AND_REDIRECTION_TOWARD_ALPHAQ.md`](METHODOLOGICAL_REVIEW_AND_REDIRECTION_TOWARD_ALPHAQ.md).

---

## Success criteria

The plan succeeds if, by the end of Phase 5:

1. The mutual-information and response-entropy structure of AlphaQ's
   policy is documented from the 1,500-game corpus.
1. A predictive opponent model $\pi_{\mathrm{AlphaQ}}(\cdot \mid s)$
   exists, with held-out top-1 prediction accuracy reported.
1. The calibration fit has been re-run against AlphaQ terminal pairs and
   a comparison to the Melissa-fitted parameters is recorded.
1. `HybridTangledSolver.m` supports an expected-value-vs-predicted-policy
   solver mode and 50 games have been played against AlphaQ in this mode.
1. A decision-gate document records whether to scale exploitation or
   pivot to tensor networks.

The plan does not require a win as a success criterion. It requires
either a win or a defensible rejection.

---

## Phase 0: Disposition of Investigation 4

**Duration:** Immediate, before Phase 1 begins.

The current 10 parallel sessions against Melissa are paused for the
duration of this plan. Two acceptable mechanisms:

1. **Hard stop.** Run `poetry run python play_tangled.py --kill-active`
   and let the run resume later if Phase 5 fails and a tensor-network
   pivot does not subsume the Melissa data need.
1. **Redirect.** Modify `scripts/launch_investigation4.ps1` to use
   `--opponent alphaq` and relaunch. This grows the AlphaQ corpus needed
   by Phases 1–3 in parallel with the analysis work.

**Recommendation:** Redirect. The marginal cost is zero (sessions are
already running) and the AlphaQ corpus growth is directly useful.

**Deliverable:** A note in `CHANGELOG.md` documenting the Investigation 4
disposition decision.

---

## Phase 1: AlphaQ policy analysis (Investigation 2)

**Duration:** 1 day. **Owner:** Analyst. **Cost:** Negligible (no new games needed).

### Scope

Compute statistical properties of AlphaQ's move distribution from the
existing 1,500-game corpus. The output is a binary-decision input for
whether classical exploit is plausible.

### Tasks

1. New analysis script: `scripts/analyse_alphaq_policy.py`.
1. Extract all (board_state, alphaq_move) pairs from the `moves` table
   where the game's opponent is `alphaq` and the move's `player` is
   `opponent`.
1. Compute the following statistics:
   - Per-state response entropy $H(\pi_{\mathrm{AlphaQ}}(\cdot \mid s))$
     for every state with observation count $\geq 3$.
   - Aggregate response entropy distribution: histogram, mean, median,
     percentiles.
   - Mutual information $I(\pi_{\mathrm{AlphaQ}}; S)$ where $S$ is the
     board state. Estimate using the plug-in estimator with bias
     correction. Compare to a randomised-policy baseline.
   - List of high-entropy states ($H > 0.5$ bits) sorted by observation
     count. These are exploit candidates.
   - List of low-observation states ($n \leq 6$) where moves were
     observed to differ across observations. These are decision-boundary
     candidates.
1. Generate plots: response entropy histogram, MI vs state visit count,
   per-edge response distribution.
1. Write findings to `docs/INVESTIGATION_2_RESULTS.md`.

### Decision gate

| Finding                                            | Implication                                              |
|----------------------------------------------------|----------------------------------------------------------|
| MI high, entropy low across observed states        | AlphaQ is near-Nash within observed basin. Pessimistic.  |
| Pockets of high entropy on $n \geq 10$ states      | Exploit candidates exist. Proceed to Phase 2 optimistic. |
| Moves vary on low-observation states               | Off-distribution attack is possible. Proceed to Phase 2. |

Even a pessimistic finding does not abort the plan; Phases 2–4 remain
valuable as the definitive test before pivoting to tensor networks. The
gate only modulates expectations.

### Risks

- Sample size per state is small. Many states have $n = 1$ observation,
  which gives no entropy information. Aggregate statistics may be
  dominated by the long tail of singleton observations.
- The MI estimator is biased for small samples; bias correction
  (Miller-Madow or similar) is mandatory.

---

## Phase 2: AlphaQ predictive policy model

**Duration:** 2–3 days. **Owner:** ML engineer. **Cost:** Negligible.

### Scope

Train a model that predicts AlphaQ's response given the board state. The
model is used as $\pi_{\mathrm{AlphaQ}}(\cdot \mid s)$ in the
expected-value solver (Phase 4).

### Tasks

1. New script: `scripts/train_alphaq_policy.py`.
1. Extract training data from the moves table: board state (15-char
   string) + grey count + game phase $\rightarrow$ AlphaQ's chosen
   (edge, color) pair.
1. Featurise the board state. Recommended feature set:
   - Per-edge state (3-way one-hot for grey/green/purple): 45 features.
   - Per-vertex effective degree in green / purple / total subgraph: 30
     features.
   - Frustration indicators on each face of the Petersen graph.
   - Grey count and parity (whose move).
1. Train two model variants and report both:
   - Multinomial logistic regression over (edge, color) pairs. Cheap,
     interpretable.
   - Small feed-forward neural network (e.g., 2 hidden layers, 64 units).
1. Hold out 20% of games as a test set. Report top-1, top-3, and top-5
   move-prediction accuracy on the test set.
1. Persist trained models to
   `snowdrop_tangled_agents/matlab/rl/data/alphaq_policy_logreg.mat`
   and `_nn.mat`.
1. Write a model card to `docs/ALPHAQ_PREDICTIVE_MODEL.md` with
   accuracy figures, calibration curves, and known failure modes.

### Decision gate

| Top-1 accuracy on held-out test            | Implication                                       |
|--------------------------------------------|---------------------------------------------------|
| $\geq 0.70$                                | Strong predictor. Use directly in Phase 4 solver. |
| $0.40$ – $0.70$                            | Useful prior. Use as soft prior, not hard policy. |
| $< 0.40$                                   | Weak predictor; AlphaQ policy is high-entropy or  |
|                                            | the feature set is inadequate.                    |

Top-1 accuracy below 0.40 does not abort the plan; high-entropy AlphaQ
behaviour is itself an exploit opportunity (it means AlphaQ is not
consistent and can be probed).

### Risks

- Feature engineering matters. The 75-feature set above is a starting
  point; if logreg accuracy is poor, expand to pairwise edge features.
- Class imbalance: some (edge, color) pairs are rare in the training
  corpus. Use weighted loss.

---

## Phase 3: AlphaQ-conditional calibration

**Duration:** 1 day. **Owner:** Analyst. **Cost:** Negligible.

### Scope

Re-run the Schrödinger adjudicator calibration using AlphaQ terminal
state and website score pairs. Compare to the existing Melissa-fitted
calibration.

### Tasks

1. New script flag: extend `scripts/calibrate_adjudicator.py` with a
   `--opponent alphaq` filter that selects only AlphaQ games from the
   calibration table.
1. Export AlphaQ board pairs via `--export-mat --opponent alphaq` to
   `data/calibration_boards_alphaq.mat`.
1. In MATLAB, run `calibrate_schrodinger` on the AlphaQ pairs.
1. Load results: `--load-matlab-results data/matlab_calib_results_alphaq.mat`.
1. Report side-by-side: Melissa-fitted vs AlphaQ-fitted (anneal_time,
   epsilon, R², classification accuracy).
1. If AlphaQ-fitted R² is materially better in AlphaQ's basin
   ($\Delta R^{2} > 0.1$ or significant in held-out cross-validation),
   regenerate the terminal LUT using AlphaQ-fitted parameters and
   rebuild the expanded LUT as `expanded_lut_calib_alphaq.mat`.
1. Add `--lut-variant calib_alphaq` to `play_tangled.py`.
1. Write findings to `docs/INVESTIGATION_3_ALPHAQ_CALIBRATION.md`.

### Decision gate

| AlphaQ-fitted R²                    | Action                                                        |
|-------------------------------------|---------------------------------------------------------------|
| $\geq 0.85$                         | Strong AlphaQ-specific oracle. Rebuild LUT, use in Phase 4.   |
| $0.60$ – $0.85$, $\Delta > 0.1$     | Moderate improvement. Rebuild LUT, use in Phase 4.            |
| Comparable to Melissa fit           | No improvement available from opponent conditioning.          |

### Risks

- AlphaQ-reachable terminals are far less diverse than Melissa-reachable
  terminals (this is the closure paper's central finding). The fit may
  be high-R² over a narrow score range and uninformative outside it.
  Report R² conditional on score-range bins to detect this.

---

## Phase 4: Expected-value solver

**Duration:** 5–7 days. **Owner:** Engineer. **Cost:** 50 game-equivalents of compute.

### Scope

Modify the hybrid solver to maximise expected value under the predicted
AlphaQ policy instead of minimax over best-response.

### Tasks

1. Add an opponent-policy interface to `HybridTangledSolver.m`. Method
   signature: `setOpponentPolicy(policyHandle)` where `policyHandle` is
   a function `s -> distribution over (edge, color) pairs`.
1. Implement two adversary modes selectable at construction:
   - `'minimax'` — existing behaviour.
   - `'expected'` — new behaviour: at each opponent node in the search
     tree, replace the min operation with expectation under
     $\pi_{\mathrm{AlphaQ}}$.
1. Implement the policy handle as a callable that loads
   `alphaq_policy_nn.mat` (or logreg variant) and evaluates per-state.
1. Verify correctness on a small synthetic position where minimax and
   expected-value give different answers. Unit test in
   `snowdrop_tangled_agents/tests/test_expected_value_solver.m`.
1. Add `--solver-adversary {minimax,expected}` CLI flag to
   `play_tangled.py`.
1. Run 50 games against AlphaQ with `--solver-adversary expected
   --lut-variant calib_alphaq` (or `calib` if Phase 3 did not produce a
   better variant). Use the existing `terminal_explorer` strategy
   modified to drive the new solver.

### Output

A new section in `INVESTIGATION_ROADMAP.md`: Investigation 6 (Expected
Value Solver). Results table: wins, draws, losses, terminal score
distribution.

### Decision gate

| 50-game result vs AlphaQ                         | Action                                                       |
|--------------------------------------------------|--------------------------------------------------------------|
| Any wins                                         | Scale: 500-game campaign, characterise winning openings.     |
| Zero wins, but mean score significantly higher   | Continue tuning. Add exploration noise, vary openings.       |
| Zero wins, mean score unchanged from minimax     | Classical exploit unlikely. Trigger Phase 5 pivot.           |

### Risks

- Compute cost is the dominant risk. 50 games at current per-game
  duration is 1–2 days of wall-clock. Parallelise across the 10
  Investigation 4 sessions if redirected.
- The expected-value solver may be exploitable if AlphaQ's policy is
  predictable in ways the model captures but the website-evaluator
  is not. Sanity-check: AlphaQ should still respond as expected,
  not deviate because we've changed our solver.
- A null result does not prove classical defeat is impossible; it only
  shows that this specific predictive-model + expected-value
  formulation does not produce wins.

---

## Phase 5: Decision and pivot

**Duration:** 1 day. **Owner:** Project lead.

### Scope

Synthesise findings from Phases 1–4 and commit the program to one of
two paths.

### Tasks

1. Write `docs/PHASE_5_DECISION.md` summarising:
   - Phase 1 MI and entropy findings.
   - Phase 2 predictive model accuracy.
   - Phase 3 calibration comparison.
   - Phase 4 50-game results.
1. Apply the decision rule:
   - If Phase 4 produced wins: continue, scale, and characterise.
   - If Phase 4 produced zero wins: commit to Investigation 5 (tensor
     network simulation) and update the roadmap.
1. If pivoting: write `docs/INVESTIGATION_5_PLAN.md` (or adapt the
   existing brief) covering MPS/DMRG implementation choice (quimb vs
   TeNPy), benchmark plan, and timeline.
1. Update `INVESTIGATION_ROADMAP.md` and `CHANGELOG.md`.

### Output

A single decision document that records the rationale and commits the
program to a path.

---

## Timeline summary

| Phase | Activity                                    | Duration  | Cumulative |
|-------|---------------------------------------------|-----------|------------|
| 0     | Investigation 4 disposition                 | < 1 hour  | 0 d        |
| 1     | AlphaQ policy analysis                      | 1 day     | 1 d        |
| 2     | AlphaQ predictive policy model              | 2–3 days  | 3–4 d      |
| 3     | AlphaQ-conditional calibration              | 1 day     | 4–5 d      |
| 4     | Expected-value solver + 50-game run         | 5–7 days  | 9–12 d     |
| 5     | Decision and pivot                          | 1 day     | 10–13 d    |

Total: 10–13 days from kick-off to decision gate. This is well inside
the 9–17 day window currently planned for Investigation 4 alone.

---

## Dependencies and ordering

- Phase 1 has no dependencies. It can start immediately.
- Phase 2 depends on Phase 1 (the MI findings inform feature engineering).
- Phase 3 has no dependencies and can run in parallel with Phase 2.
- Phase 4 depends on Phases 2 and 3.
- Phase 5 depends on Phase 4.

If staffing permits parallelism, Phases 1 and 3 can be run concurrently,
shortening the critical path by one day.

---

## Risk register

| Risk                                                          | Likelihood | Severity | Mitigation                                                |
|---------------------------------------------------------------|------------|----------|-----------------------------------------------------------|
| Predictive model accuracy too low to drive solver             | Medium     | High     | Use as soft prior; fall back to mixed minimax/expected.   |
| AlphaQ-fitted calibration not materially better than Melissa  | Medium     | Low      | Use Melissa-fitted in Phase 4; no rebuild cost.           |
| Expected-value solver still produces zero wins                | High       | High     | This is the planned-for outcome of the pessimistic path;  |
|                                                               |            |          | Phase 5 pivot is designed for it.                         |
| Investigation 4 sessions cannot be cleanly paused             | Low        | Low      | Use `--kill-active`; data already in DB is preserved.     |
| Phase 4 reformulation introduces bugs in `HybridTangledSolver`| Medium     | Medium   | Unit test on synthetic positions before live games.       |

---

## What this plan does not include

This plan is scoped to the two-week decision-gate window. It does not
include:

- Implementation of tensor-network simulation. That is Phase 5's pivot
  output and would be its own multi-month project plan.
- Exploration of non-classical adversary models (quantum strategies in
  the Eisert-Wilkens-Lewenstein sense). Out of scope for classical
  computation.
- Re-engineering of the underlying retrograde-DP pipeline. The
  calibrated LUT is taken as given; only its inputs (Phase 3 fit) and
  its use (Phase 4 solver) are modified.

These are deliberately deferred. The intent of this plan is to make the
classical-defeat question binary within two weeks, not to expand the
project's surface area.
