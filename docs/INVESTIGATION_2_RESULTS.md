# Investigation 2 Results: AlphaQ Policy Analysis

**Source corpus:** local `~/.tangled/game_stats.db` (read-only)
**Method:** information-theoretic analysis of the empirical AlphaQ move distribution, conditional on observed board state.

---

## Corpus summary

| Metric | Value |
|--------|-------|
| AlphaQ games (completed) | 1574 |
| AlphaQ decisions observed | 9341 |
| Distinct states observed | 588 |
| States with n >= 3 observations | 439 |

---

## Per-state response entropy

| Statistic | Value (bits) |
|-----------|--------------|
| Mean | 0.0201 |
| Median | 0.0000 |
| Std dev | 0.1325 |
| 5th percentile | 0.0000 |
| 95th percentile | 0.0000 |
| Max | 1.0000 |

**Deterministic states (H = 0):** 428 of 439 (97.5%)

**High-entropy states (H >= 0.5 bits):** 9 of 439 (2.1%)

![Entropy histogram](../plots/investigation2_entropy_histogram.png)

![Entropy vs grey count](../plots/investigation2_entropy_vs_grey.png)

---

## Mutual information I(pi_AlphaQ; S)

| Quantity | Value |
|----------|-------|
| H(states) [bits] | 7.6775 |
| H(actions) [bits] | 4.3214 |
| MI plug-in [bits] | 4.3067 |
| MI Miller-Madow corrected [bits] | 4.3053 |
| Distinct (state, action) cells | 599 |

**Interpretation:** higher MI means AlphaQ's move is more informative about the board state (more state-dependent). MI is biased upward by sparsity; the Miller-Madow correction subtracts the leading bias term.

### MI stratified by grey count

| Grey edges | N observations | N states | MI plug-in (bits) | MI MM-corrected (bits) | Mean entropy (bits) |
|------------|----------------|----------|-------------------|------------------------|---------------------|
| 1 | 144 | 30 | 3.5619 | 3.5018 | 0.0000 |
| 2 | 1129 | 75 | 3.3429 | 3.3314 | 0.0183 |
| 3 | 145 | 30 | 3.3495 | 3.2898 | 0.0000 |
| 4 | 1102 | 79 | 3.3119 | 3.2982 | 0.0372 |
| 5 | 146 | 28 | 3.4590 | 3.3948 | 0.0326 |
| 6 | 1136 | 68 | 2.7698 | 2.7577 | 0.0089 |
| 7 | 148 | 28 | 2.9427 | 2.8891 | 0.0000 |
| 8 | 1143 | 55 | 2.9351 | 2.9250 | 0.0986 |
| 9 | 141 | 24 | 3.1693 | 3.1181 | 0.0000 |
| 10 | 1235 | 51 | 2.3482 | 2.3424 | 0.0000 |
| 11 | 144 | 22 | 2.8450 | 2.7999 | 0.0000 |
| 12 | 1236 | 53 | 2.4097 | 2.4038 | 0.0235 |
| 13 | 147 | 15 | 1.5687 | 1.5491 | 0.0000 |
| 14 | 1345 | 30 | 1.4262 | 1.4240 | 0.0000 |

---

## Exploit candidates

States with n >= 10 observations and H >= 0.5 bits of response entropy. These are positions where AlphaQ has been observed enough times to estimate its response distribution and where that distribution is not strongly peaked on a single action.

**6 exploit candidate state(s) found.**

| Rank | State (E0..E14) | Grey | N obs | H (bits) | Distinct responses | Top response | Top fraction |
|------|-----------------|------|-------|----------|--------------------|--------------|--------------|
| 1 | `PPGPGGG--------` | 8 | 10 | 0.971 | 2 | E9P | 0.60 |
| 2 | `PGPGGGG--------` | 8 | 13 | 0.961 | 2 | E8G | 0.62 |
| 3 | `PPPPGGG--------` | 8 | 15 | 0.918 | 2 | E7G | 0.67 |
| 4 | `GPGPGGGP-PP-PGP` | 2 | 12 | 0.918 | 2 | E8P | 0.67 |
| 5 | `PGPGGGP--------` | 8 | 16 | 0.896 | 2 | E8G | 0.69 |
| 6 | `P-G-P---GGPGPPP` | 5 | 14 | 0.750 | 2 | E0P | 0.79 |

---

## Decision-boundary candidates

States with 2 <= n <= 6 observations where AlphaQ's response has differed across observations. These are low-confidence regions where AlphaQ may be near a policy decision boundary.

**2 decision-boundary candidate state(s) found.**

| Rank | State | Grey | N obs | Distinct responses |
|------|-------|------|-------|--------------------|
| 1 | `PGPGGGG-GG-P-P-` | 4 | 6 | 2 |
| 2 | `PPG-PG---GGGGPP` | 4 | 5 | 2 |

---

## Per-edge AlphaQ preferences

![Per-edge color distribution](../plots/investigation2_per_edge.png)

---

## Decision-gate verdict

**OPTIMISTIC** — AlphaQ shows pockets of high-entropy response (6 exploit candidates, 2 decision-boundary candidates). These are concrete positions where AlphaQ's policy is not strongly peaked. Phase 2 (predictive model) and Phase 4 (expected-value solver) should target these states as the primary search frontier.

---

## Generated artefacts

- `plots/investigation2_entropy_histogram.png`
- `plots/investigation2_entropy_vs_grey.png`
- `plots/investigation2_per_edge.png`
- `docs/INVESTIGATION_2_RESULTS.md` (this file)
