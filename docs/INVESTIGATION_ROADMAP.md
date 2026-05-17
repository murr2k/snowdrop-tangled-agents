# Investigation Roadmap

**Goal:** Find wins against AlphaQ at tangled-game.com, or definitively characterise why
winning is impossible.  
**Status as of 2026-05-17:** Oracle Revision complete; Investigation 4 running; methodological
review completed, recommending program redirection to AlphaQ-targeted analysis (Phases 1–5
in `docs/PROJECT_PLAN_ALPHAQ_TARGETED_INVESTIGATION.md`). Investigation 4 disposition pending.  
See `docs/INVESTIGATION_AVENUES.md` for the original candidate assessment and
`docs/METHODOLOGICAL_REVIEW_AND_REDIRECTION_TOWARD_ALPHAQ.md` for the current critique.

---

## Completed Investigations

### Investigation 1 — Player 2 Seat Swap

**Status:** ✅ Complete — 0W / 0D / 10L  
**Hypothesis:** AlphaQ's policy is optimised for the second-mover role (Player 2 / Blue).
Playing as P2 forces AlphaQ into the first-mover position, potentially exposing weaknesses
in its opening policy.  
**Method:** `poetry run python play_tangled.py --opponent alphaq --seat 2 --strategy hybrid_solver --lut-variant sa --games 10 --headless`  
**Success signal:** Any wins, OR terminal boards with SchrLUT significantly above P1-game
baseline (0.006–0.012).  

**Results:** 0W / 0D / 10L. All 10 games identical (deterministic oracle + deterministic AlphaQ).

Three bugs were found and fixed before running (P2 transposition was broken):
1. `play_tangled.py` hardcoded `player=1` — MATLAB always got P1 perspective.
2. Oracle trigger `mod(numGrey,2)==1` never fired on P2's even-grey turns.
3. `solveOracle` always maximised LUT value — P2 should minimise (values are P1-perspective).

After fixes, oracle fired correctly on every P2 move (`strategy=oracle` confirmed in DB).

**Terminal board (every game):** `PPGGPPGGPPPGGPG`  
**AlphaQ (P1) line:** E1P → E2P → E5P → E14P → E3G → E6P (heavily purple)  
**Our (P2) oracle line:** E10P → E12G → E13G → E11P → E9P → E7G → E8G  

**Score trajectory (SA LUT, P1-perspective):**

| Round | AlphaQ | Us   | Score after |
|-------|--------|------|-------------|
| 1     | E1=P   | E10=P | −0.954 (P2 advantage) |
| 2     | E2=P   | E12=G | +0.003 (drawn) |
| 3     | E5=P   | E13=G | +0.022 |
| 4     | E14=P  | E11=P | +0.400 |
| 5     | E3=G   | E9=P  | +0.273 |
| 6     | E6=P   | E7=G  | +1.377 |
| 7     | —      | E8=G  | **+0.928 (loss)** |

**Key observation — LUT internal inconsistency:** After round 1 the SA oracle valued the
position at −0.954 (P2 strongly ahead). After round 2 — one AlphaQ move and one of our
oracle-chosen responses — it read +0.003 (near draw). That is a ~0.957 swing in a single
round. Under a correct minimax oracle this is impossible: the −0.954 at grey=13 is supposed
to already account for AlphaQ's best response. If it did, our minimising reply at grey=12
followed by AlphaQ's move can only change the value by the marginal difference between the
chosen move and the second-best — not by nearly 1.0.

The contradiction exposes a structural flaw in the expanded LUT: SA is run independently at
each grey level, so the values at grey=13 and grey=11 come from separate stochastic runs
with independent noise. The retrograde minimax DP then propagates and compounds these
errors across levels. Nodes far from the terminal (high grey) accumulate the most error.
The −0.954 at grey=13 was noise amplified by six DP steps, not signal.

**Conclusion:** Playing as P2 does not help, but the deeper finding is that the oracle's
in-game evaluations (at intermediate grey levels) are unreliable. The LUT at grey=0
(terminal) is the most trustworthy level; values deteriorate with distance from the
terminal. We cannot trust oracle-guided play at grey=13 or above. This invalidates the
core assumption behind the hybrid-solver's oracle mode for early-game decisions, and
motivates a revised oracle design (see Oracle Revision Project in docs/).

---

## Completed Investigations (continued)

### Investigation 3 — Adjudicator Parameter Recovery

**Status:** ✅ Complete — anneal_time=1.85 ns recovered, R²=0.60, 83% classification accuracy  
**Hypothesis:** The website Schrödinger adjudicator uses specific ε and anneal_time values.
Fitting a parameterised model to observed (terminal_board → website_score) pairs can recover
these parameters and make the oracle predictive.  
**Method:** Used 1061 distinct (terminal_state, website_score) pairs already in DB (no new
games needed). Grid-searched anneal_time over 15 log-spaced values (5–20,000 ns) using the
MATLAB split-operator Schrödinger solver (~1.4 s/board), then refined with fminbnd in
[1.5, 15] ns. Fit epsilon separately by maximising win/draw/loss classification accuracy.  

**Results:**

| Parameter | Value |
|-----------|-------|
| anneal_time | 1.85 ns |
| epsilon | 0.0 (no draw zone at this time scale) |
| R² vs website scores | 0.6047 |
| Classification accuracy | 83.1% |
| Grid search time | ~5.2 hrs (serial for loop) |
| Refinement evaluations | 11 (fminbnd converged) |

R² did not reach the 0.9 target, but 83% classification accuracy is the operationally relevant
metric — it means the calibrated oracle correctly predicts the website winner on 5 in 6
terminal boards. This places it in the "PARTIAL" band (R² 0.5–0.9) per decision rules.

**Key finding:** The website uses an extremely short anneal_time (1.85 ns vs the 40 ns default).
At this timescale the quantum system barely evolves; scores are dominated by graph structure
near the initial state rather than long-time annealing dynamics.

**Follow-on (Oracle Revision Project — ✅ Complete 2026-05-17):**
1. ✅ Calibrated terminal LUT regenerated (`terminal_scores.mat`, 32,768 states, 1.69 hrs)
2. ✅ Expanded LUT rebuilt (`expanded_lut_calib.mat`, retrograde DP levels 0–13, 361 MB)
3. ✅ `--lut-variant calib` added to `play_tangled.py`
4. ✅ Consistency analysis: calib oracle R²=0.7092 at terminal (best of all three oracles)
5. ✅ P1 validation: 10 games vs AlphaQ → 0W / 0D / 10L (same E9=G exploit, oracle cannot recover)
6. ✅ P2 validation: 10 games vs AlphaQ → 0W / 10D / 0L (oracle reliably guides to ~−0.04 terminal, all draws)

**P2 game pattern (all 10 identical):** AlphaQ opens E0=P → we respond E12=G (score −1.04,
P2 strongly ahead) → oracle navigates to near-zero draw by terminal. Final scores range −0.028
to −0.050. The calib oracle successfully avoids the mid-game value collapse that plagued the SA
LUT (0.957 swing → now <0.05 throughout).

**Conclusion:** Calibrated oracle is internally consistent and operationally superior to SA. As
P1 we still lose to AlphaQ's E9=G counter-opening — this is not an oracle problem, it is the
first-move-advantage problem (oracle lacks level-14 coverage, MCTS chooses E0=G which AlphaQ
exploits). As P2 we draw perfectly. Next: Investigation 4 to find winning terminal boards.

---

## Completed Investigations (continued)

### Investigation 2 — Spectral / MI Analysis of AlphaQ Policy

**Status:** ✅ Complete (2026-05-17) — **OPTIMISTIC** verdict; 6 exploit candidates found  
**Hypothesis:** AlphaQ may be at a locally optimal equilibrium rather than true Nash. Mutual
information I(AlphaQ move; board state) and per-state response entropy can distinguish these.  
**Method:** `scripts/analyse_alphaq_policy.py` — extracts 9,341 AlphaQ decisions from 1,574
games in the local DB, reconstructs (state_before, AlphaQ_move) pairs, computes per-state
Shannon entropy and global MI with Miller-Madow bias correction, stratified by grey count.  

**Results:**

| Metric | Value |
|--------|-------|
| AlphaQ decisions observed | 9,341 |
| Distinct states observed | 588 |
| States with n >= 3 observations | 439 |
| Deterministic states (H = 0) | 428 of 439 (97.5%) |
| High-entropy states (H >= 0.5 bits) | 9 of 439 (2.1%) |
| Global MI (Miller-Madow corrected) | 4.31 bits |
| Exploit candidates (n >= 10, H >= 0.5) | 6 |
| Decision-boundary candidates (n <= 6, varying moves) | 2 |

**Key finding:** AlphaQ is near-deterministic on 97.5% of well-observed states but has 6
identifiable positions where its response is genuinely variable across 10+ observations.
Five of the six are at grey=8 (mid-game), suggesting AlphaQ's policy has a decision-boundary
phase in mid-game. All 6 have exactly 2 distinct responses with the dominant response chosen
60-79% of the time — concrete binary choice points.

**Conclusion:** Classical exploit is plausible but narrow. The 6 candidate states are the
primary search frontier for Phases 2-4 of the AlphaQ-targeted plan. See
`docs/INVESTIGATION_2_RESULTS.md` for the full report and `plots/investigation2_*.png` for
the entropy histogram, per-grey-count entropy curve, and per-edge preference plots.

---

## Queued Investigations

---

### Investigation 4 — Exhaustive Terminal State Mapping

**Status:** 🟢 Running — 10 parallel sessions launched 2026-05-17  
**Hypothesis:** Some winning terminal boards may be reachable against Melissa/Amara that have
not been observed in 1,574 games. Exhaustive mapping builds a complete reachable score table.  
**Method:** `terminal_explorer` strategy cycles all 30 openings (15 edges × {G, P}) in
round-robin, uses MCTS for remaining moves. 10 parallel sessions (tangled1–10@linknode.com)
each join shared run 140 (50,000 games planned):
```
.\scripts\launch_investigation4.ps1   # re-launch all 10 sessions
```
Target: 30% coverage of all 32,768 terminal boards (~9,830 distinct boards).  
**Prerequisite:** ✅ Oracle Revision Project complete (2026-05-17)  
**Resume:** Interrupt-safe — DB tracks `completed_games` and `lut_variant`; restart launcher
to rejoin run 140, restoring opening_index from `completed_games % 30`.  
**Estimated cost:** ~4–7 days (parallelism cuts wall-clock ~10×; browser round-trip dominates)  
**Success signal:** Winning terminal boards exist and are reachable under calibrated oracle.

---

### Investigation 5 — Tensor Network Simulation

**Status:** ⏳ Queued (long-term)  
**Hypothesis:** A matrix product state (MPS/DMRG) simulation of the transverse-field Ising
Hamiltonian at the recovered website parameters (from Investigation 3) can compute exact
Schrödinger ground state energies for all 32,768 terminal boards, enabling a complete oracle.  
**Method:** Implement MPS simulation (Python: quimb / TeNPy); benchmark on Petersen graph;
generate full Schrödinger oracle at website parameters.  
**Prerequisite:** Investigation 3 (need website quantum parameters)  
**Estimated cost:** 3–6 months  
**Success signal:** MPS oracle values match website scores; full minimax oracle computable.

---

## Results Log

| Date | Investigation | Games | W | D | L | Key Finding |
|------|--------------|-------|---|---|---|-------------|
| 2026-05-16 | 1 — P2 seat swap | 10 | 0 | 0 | 10 | Oracle LUT internally inconsistent across grey levels; intermediate evaluations unreliable |
| 2026-05-16 | 3 — Adjudicator parameter recovery | 0 | — | — | — | anneal_time=1.85 ns recovered; R²=0.60, 83% classification accuracy; calibrated terminal LUT rebuilt |
| 2026-05-17 | Oracle Revision — P1 validation (calib) | 10 | 0 | 0 | 10 | Same E9=G AlphaQ exploit; oracle consistent but no level-14 coverage for P1 opening |
| 2026-05-17 | Oracle Revision — P2 validation (calib) | 10 | 0 | 10 | 0 | Perfect draw record; calib oracle guides to −0.028..−0.050 terminal; no mid-game swings |
| 2026-05-17 | 2 — AlphaQ policy MI/entropy analysis | 9341 decisions | — | — | — | 97.5% deterministic states; 6 exploit candidates; MI=4.31 bits; OPTIMISTIC verdict — exploit candidates exist as targets for Phases 2-4 |
| 2026-05-17 | Phase 2 — Predictive opponent policy model | 1894 test moves | — | — | — | MLP top-1=0.866, LogReg top-1=0.851; 38.9% on exploit candidates confirms entropy is intrinsic; strong-predictor verdict |
| 2026-05-17 | Phase 3 — AlphaQ-conditional calibration | 102 boards | — | — | — | Melissa-fit Schr R² on AlphaQ basin = −0.56 (vs +0.74 on Melissa); SA raw R² on AlphaQ = −0.94; verdict NONE; existing oracle stays as soft prior for Phase 4 |
| 2026-05-17 | Phase 4 — Expected-value solver (built) | 0 | — | — | — | `HybridTangledSolver` AdversaryMode='expected' uses MLP policy (`alphaq_policy_mlp.mat`) to compute `E_pi[LUT(grandchild)]` at our turns. Wired through `--solver-adversary expected`. Unit test + parity check pass. 50-game decision-gate run vs AlphaQ launched. |
| 2026-05-17 | Phase 4 — Expected-value solver vs AlphaQ (run 141) | 50 | 0 | 0 | 50 | **PHASE 5 PIVOT.** Eliminated the **49.4% draw rate** baseline minimax hybrid_solver hits (n=83 prior P1 games: 0W / 41D / 42L). Phase 4 mean −0.669, collapsed bimodal baseline (draw plateau + vertex-tiebreak loss cluster) into single loss basin. Confirms Phase 3 oracle-unreliability finding amplifies through Phase 2 policy errors. See `docs/PHASE_4_RESULTS.md`. |

---

## Investigation 6 — Expected-Value Solver vs AlphaQ (Phase 4)

**Hypothesis:** AlphaQ's policy has intrinsic uncertainty at ≥ 6 known choice points
(38.9% top-1 prediction accuracy). A solver that maximises expected value under the
predicted AlphaQ policy — rather than minimaxing the worst-case response — will
exploit the unbalanced (60/40 etc.) response distributions at those states.

**Mechanism:** at each of our candidate moves at our turn (oracle path, sub-50 ms),
compute `E_pi[LUT(grandchild after AlphaQ's response)]` instead of `LUT(child after our
move)`. The LUT past the one-step expectation still encodes minimax; deeper expectimax
recursion is deliberately rejected because the policy degrades on out-of-distribution
states reached by the new solver (see model card known-failure-modes).

**Run command:** `poetry run python play_tangled.py --opponent alphaq --strategy hybrid_solver --lut-variant calib --solver-adversary expected --games 50`

**Decision gate (per project plan §Phase 4):**

| 50-game result | Action |
|----------------|--------|
| Any wins | Scale to 500-game campaign; characterise winning openings |
| Zero wins, mean score significantly higher than minimax baseline | Continue tuning (exploration noise, opening variation) |
| Zero wins, mean score unchanged | Trigger Phase 5 pivot to tensor networks (Investigation 5) |

**Result (run 141, 2026-05-17):** 0 W / 0 D / 50 L. Mean score −0.669 ± 0.048.
The proper minimax-mode baseline is the full hybrid_solver P1 vs AlphaQ
history (n=83 games excluding this run): 0 W / 41 D / 42 L (**49.4% draw
rate**), mean +0.202, bimodal distribution (draw plateau near zero +
vertex-tiebreak loss cluster near +0.7). Phase 4 eliminated the entire 49%
draw plateau and collapsed both modes into a single loss basin at −0.67.
**Phase 5 pivot triggered.** See `docs/PHASE_4_RESULTS.md` for the full
analysis.

---

## Decision Rules

- **If Investigation 1 finds wins:** Characterise which openings and terminal boards; expand
  to 100 games; document in this roadmap.
- **If Investigation 1 finds only draws/losses:** Mark complete, proceed to Investigation 2.
- **Investigation 3 result** (R² = 0.60, partial): Parameters recovered. Oracle revision
  underway. Proceed with calibrated oracle for Investigations 2 and 4.
- **If Investigation 4 finds no winning terminals:** Game is definitively drawn under any
  classical strategy. Close programme.
