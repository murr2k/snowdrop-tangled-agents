# Empirical LUT Construction: Website-Calibrated Terminal Scoring

## Context

The oracle solver proved that AlphaQ is deterministic and routable, but the SA-derived LUT is not semantically equivalent to the platform's adjudicator. SA scores undergo nonlinear compression (66×) when mapped to website scores, making the formal threshold ε=0.0005 operationally meaningless. The operational win threshold is ~+2 in website score space.

**Goal:** Run diverse games against AlphaQ to collect (terminal_state → website_score) mappings empirically, then build a website-calibrated LUT to replace the SA-derived one. Re-run the oracle solver with this new LUT to find genuinely winning terminal states—if any exist.

**Key discovery:** A `calibration` table already exists in the DB schema (`stats/migrations.py:113-122`) and is actively populated during games (`play_tangled.py:2054-2062`). We likely already have data from 1,431 previous games.

## Implementation Steps

### Step 1: Mine existing calibration data

Create `tools/build_website_lut.py` — a script that:

1. Queries the existing `calibration` table: `SELECT terminal_state, website_score, predicted_score FROM calibration`
2. Groups by `terminal_state`, computes mean/std/count of website_score per terminal
3. Also queries `games` table joined with last move's `state_after` for any games missing from calibration
4. Reports: how many unique terminal states we have, score distribution, coverage vs the 32,768 possible states
5. Cross-references against the SA LUT (`oracle-solver/data/terminal_scores.bin`) to produce a SA→website calibration curve
6. Outputs a partial website LUT as `oracle-solver/data/website_scores.bin` (same format: 32768 × f32 LE, NaN for unobserved states)

**Critical files:** `stats/collector.py:346-373` (record_calibration), `tools/generate_terminal_lut.py` (encoding convention)

### Step 2: Add route cycling to OracleRouteStrategy

Modify `snowdrop_tangled_agents/strategy/oracle_route_strategy.py`:

- Add `route_mode` parameter: `'fixed'` (current behavior) or `'cycle'` (round-robin through all routes)
- In `end_game()`, when mode is `'cycle'`, advance `preferred_route_index` to next route
- Track which routes have been played and their outcomes in a simple log

Modify `play_tangled.py` strategy setup (line ~653):
- Add `--route-mode` argparse argument (`fixed` | `cycle`)
- Pass `route_mode` to `OracleRouteStrategy` constructor

This gives us 48 unique terminal states from one sweep of all oracle routes.

### Step 3: Add exploration strategy for terminal state discovery

Create `snowdrop_tangled_agents/strategy/terminal_explorer_strategy.py`:

A strategy designed to maximize terminal state diversity, not wins:

1. Maintains a set of `seen_terminal_states` (loaded from DB at init)
2. Uses MCTS for move selection but with a modified objective: prefer moves that lead toward **unseen** terminal states
3. Simple heuristic: for the first N moves, follow a randomized opening sequence (cycle through all 30 possible openings: 15 edges × 2 colors). For remaining moves, use standard MCTS.
4. Opening diversification: round-robin through all 30 openings, then cycle through random mid-game variations

Register as `--strategy terminal_explorer` in `play_tangled.py`.

This reaches terminal states BEYOND the 48 oracle routes.

### Step 4: Run game campaigns

**Campaign 1 — Oracle route sweep (48 games):**
```bash
poetry run python play_tangled.py --strategy oracle_route --route-mode cycle --games 48 --opponent alphaq
```
Plays each of the 48 oracle routes once. Each game records (terminal_state, website_score) in the calibration table.

**Campaign 2 — Exploration sweep (100-200 games):**
```bash
poetry run python play_tangled.py --strategy terminal_explorer --games 200 --opponent alphaq
```
Explores diverse openings and mid-game paths to discover new terminal states.

### Step 5: Build the website-calibrated LUT

Re-run `tools/build_website_lut.py` after campaigns:

1. Now has 200-300+ new data points plus ~1,431 existing ones
2. For observed terminal states: use empirical mean website score
3. For unobserved states: fit a regression model SA_score → website_score from observed pairs, use predictions as interpolated estimates
4. Output complete `oracle-solver/data/website_scores.bin` (32768 × f32 LE)
5. Report coverage statistics and the empirical SA→website calibration curve

### Step 6: Re-run oracle solver with website LUT

```bash
cd oracle-solver
cargo run --release -- --lut-path data/website_scores.bin --db-path ~/.tangled/game_stats.db --opponent alphaq --output output/website_oracle_routes.json
```

The solver now scores terminal states with website-calibrated values. Any routes with website_score > +2 (the empirical win threshold) are genuine win candidates.

### Step 7: Load new routes and test

Update `oracle_route_strategy.py` to accept a `--routes-file` parameter (default: `oracle_routes.json`, new: `website_oracle_routes.json`).

Run verification games with the top-scoring routes from the website-calibrated solver.

## Critical Files

| File | Action | Purpose |
|------|--------|---------|
| `tools/build_website_lut.py` | Create | Mine calibration data, build website LUT |
| `strategy/oracle_route_strategy.py` | Modify | Add route cycling mode |
| `strategy/terminal_explorer_strategy.py` | Create | Diversity-maximizing exploration strategy |
| `play_tangled.py` | Modify | Add `--route-mode`, `--routes-file`, register terminal_explorer |
| `oracle-solver/data/website_scores.bin` | Generated | Website-calibrated LUT output |
| `oracle-solver/output/website_oracle_routes.json` | Generated | Routes scored with website LUT |
| `stats/collector.py` | Read only | Calibration table already exists |
| `stats/migrations.py` | Read only | Schema already has calibration table |

## Verification

1. **Step 1 check:** Run `build_website_lut.py` on existing data. Expect 50-91 unique terminal states from 1,431 games. Verify SA→website correlation curve shows nonlinear compression.
2. **Step 2 check:** Run oracle_route with `--route-mode cycle --games 3`. Verify 3 different routes are played.
3. **Step 4 check:** After campaigns, re-run `build_website_lut.py`. Expect 150+ unique terminal states.
4. **Step 6 check:** Oracle solver with website LUT should find fewer "winning" routes (most SA-wins compress to draws). Any surviving wins with website_score > +2 are genuine candidates.
5. **Step 7 check:** Play verification games with top website-scored routes. If any produce wins, the approach succeeds.

## Risk Assessment

**Most likely outcome:** All reachable terminal states against AlphaQ have website scores in [-1, +1], confirming AlphaQ has converged to a zero-loss equilibrium. This is still valuable — it empirically proves the evaluator non-equivalence thesis and maps the actual terminal state landscape.

**Best case:** The website LUT reveals a terminal state with score > +2 that is reachable via oracle routing. First win against AlphaQ.

**Mitigation:** Even if no wins are found, the website-calibrated LUT is a reusable asset for future strategy development against any opponent.
