# Plan: Clean Re-run vs AlphaQ on Post-Fix Code

**Status:** Ready to execute
**Created:** 2026-05-08
**Resumable across sessions:** Yes — this doc is self-contained.
**Related:**
[WAYPOINT_2026-05-07_LUT_INVESTIGATION.md](WAYPOINT_2026-05-07_LUT_INVESTIGATION.md),
[PHASE_A_ORACLE_RESULTS.md](PHASE_A_ORACLE_RESULTS.md),
[POST_SA_LUT_STRATEGY_OPTIONS.md](POST_SA_LUT_STRATEGY_OPTIONS.md),
[EMPIRICAL_LUT_CONSTRUCTION_RESULTS.md](EMPIRICAL_LUT_CONSTRUCTION_RESULTS.md)

---

## Goal

Re-run a controlled batch of games against **AlphaQ Up** using the
post-fix codebase, then re-derive the WAYPOINT's quantitative claims
on this clean data to confirm whether the prior conclusions
(H1 rejected, H2 supported, H3 weakened) survive without the
contamination flagged in §0 of the waypoint.

If the clean numbers reproduce the tainted ones, the conclusions are
durable.  If they diverge meaningfully, the investigation needs to be
re-opened with the corrected baseline.

---

## Why this is needed

The waypoint relied on `~/.tangled/game_stats.db` records captured
before two fixes:

- `8421c76` — *Remove failed_edges blacklist that incorrectly skewed
  strategy*: the agent was avoiding edges marked "failed", biasing
  the distribution of board states AlphaQ was asked to respond to.
- `fa644f4` — *Fix interleaved move history: detect opponent moves at
  start of our turn*: opponent moves could be misattributed to wrong
  `state_after` values, corrupting the state→response mapping that
  the oracle and the forensic queries depend on.

Origin's parallel work (website-calibrated empirical LUT) reached the
same Nash conclusion via independent methodology, so the *high-level*
conclusions are durable.  But the specific figures in the waypoint
should not be cited until validated on clean data.

---

## Phase 1 — Pre-flight checks (must pass before Phase 2)

Run these in order in a fresh session.  Each must succeed before
proceeding.

### 1.1 Confirm git state
```bash
cd C:/Users/murr2/projects/snowdrop-tangled-agents
git status                           # working tree clean (or only the
                                     # known unrelated dirty files)
git rev-parse HEAD                   # note for later
git log --oneline | head -5
```

### 1.2 Confirm relevant fix commits are present
```bash
git log --oneline --all | grep -E "(8421c76|fa644f4)"
```
Expected: both commit hashes listed.  If missing, abort — we are not
on a post-fix branch.

### 1.3 Confirm environment
```bash
poetry run python -c "import scipy, h5py, numpy, tqdm; print('deps ok')"
poetry run python -c "from snowdrop_adjudicators import SchrodingerEquationAdjudicator; print('adj ok')"
poetry run playwright --version
```

`.env` must contain `TANGLED_USERNAME` and `TANGLED_PASSWORD`.

### 1.4 Confirm the LUT file is the Schrödinger one (not SA)
The live MCTS expects `terminal_scores.mat` to be Schrödinger.
```bash
poetry run python -c "
import h5py, numpy as np
with h5py.File('snowdrop_tangled_agents/matlab/rl/data/terminal_scores.mat','r') as f:
    s = np.array(f['scorer']).tobytes().decode('utf-16-le').rstrip('\x00')
    print('scorer:', s)
"
```
Expected: `scorer: schrodinger`.  If `simulated_annealing`, restore
from git: `git checkout HEAD -- snowdrop_tangled_agents/matlab/rl/data/terminal_scores.mat`.

### 1.5 Note the next free run_id
```bash
sqlite3 ~/.tangled/game_stats.db "SELECT MAX(run_id) FROM games;"
```
Add 1 — that becomes the run_id for this experiment.  Record it
below in §5.

---

## Phase 2 — Run the experiment

### 2.1 Default configuration (recommended for parity with Run 60–64)

| Parameter | Value | Rationale |
|---|---|---|
| Strategy | `alphaq_explorer` | Same as Run 60–64; enables direct comparison. Uses Thompson Sampling + ground-truth Schrödinger LUT + REINFORCE edge bias. |
| Opponent | `alphaq` | The investigation target. |
| Opening mode | `round_robin` | Covers all 30 openings (15 edges × 2 colors) systematically. Avoids opening-bias confound. |
| Games | **300** | 10 per opening on round-robin. ~5h at 60s/game. Enough for statistical power on the cross-tabulation. |
| MCTS iterations | `5000` | Matches Run 60–64 setting. |
| Notes/tag | `"postfix_rerun_v1"` | Captured in `games.notes` for filtering. |

### 2.2 Command

Run as a background task so the session can do other work:

```bash
cd C:/Users/murr2/projects/snowdrop-tangled-agents
poetry run python play_tangled.py \
    --strategy alphaq_explorer \
    --opponent alphaq \
    --games 300 \
    --mcts-iterations 5000 \
    --opening-mode round_robin \
    > /tmp/postfix_rerun.log 2>&1
```

(Use Bash with `run_in_background=true` and `timeout=21600000`.)

### 2.3 Variations to consider

If the user wants to take a different angle than parity with Run 60:

- **`oracle_route` strategy** — extends origin's `EMPIRICAL_LUT_CONSTRUCTION`
  campaign with cleaner data.  Use when goal is "add to the
  website-calibrated empirical LUT" rather than "re-validate forensic
  numbers".
- **`terminal_explorer` strategy** — maximises terminal-state
  diversity to expand coverage beyond the current ~120-state empirical LUT.
- **Smaller batch (60–100 games)** — if time-constrained, still gives
  a reasonable sample for cross-tabulation but lower statistical power.
- **Larger batch (500+ games)** — full Run 60 parity, ~10h.

### 2.4 Live monitoring (optional)
```bash
tail -c 500 /tmp/postfix_rerun.log | tr '\r' '\n' | tail -3
sqlite3 ~/.tangled/game_stats.db \
    "SELECT COUNT(*) FROM games WHERE run_id = <RUN_ID> ;"
```

---

## Phase 3 — Analysis

After the run completes, derive the post-fix versions of the
WAYPOINT §3, §6, §8 figures from the new data only.

### 3.1 Recompute the cross-tabulation (LUT sign vs result)

Using only games with `run_id = <RUN_ID>` from §1.5.  The
existing `_tmp_lut_vs_outcomes.py` pattern from the WAYPOINT
session works — adapt by adding a `WHERE run_id = ?` filter.

Compare to the waypoint's table:

| LUT sign | win | draw | loss |
|---|---:|---:|---:|
| POS (>0.1) | 334 | 364 | 194 |
| ZERO | 56 | 1,359 | 661 |
| NEG (<-0.1) | 0 | 0 | 13 |

Specifically check: does the **POS row vs AlphaQ specifically** still
show 0 wins / many draws / few losses?

### 3.2 Recompute the SA-vs-server sign-agreement

The waypoint's headline figure is *95.8% server-matches-SA in
disagreement cases*.  Re-derive from the new run only.  The
`_tmp_disagreement_search.py` pattern is the template.  Use the new
`terminal_scores_sa.mat` (already present in the repo).

### 3.3 Re-run the win-set intersection

```bash
poetry run python -m snowdrop_tangled_agents.oracle.find_sa_win_set
```
The reachable terminal set may grow if the new run reaches states
the old oracle didn't have.  In particular, look at:
- Number of unique terminal states in the new run
- Any **novel** SA-positive terminals (`sa_win_candidates.json`)
- Maximum SA score reached against AlphaQ in this run (compare to
  the waypoint's +0.891)

### 3.4 Cross-check against origin's empirical LUT

Origin's `EMPIRICAL_LUT_CONSTRUCTION_RESULTS.md` reports:
- Max website score achieved vs AlphaQ: **+0.861** (across 120 states)
- 0% win rate

If our new run reproduces these numbers, the Nash conclusion is
robust across both methodologies and the post-fix data.

---

## Phase 4 — Update the waypoint

Edit `docs/WAYPOINT_2026-05-07_LUT_INVESTIGATION.md` based on §3 results:

### If clean numbers reproduce the tainted ones
- Replace §0 caveat with a closing note that the data has been
  validated against post-fix runs.  Record the new run_id and game
  count for the validation.
- Bump the waypoint's `Status:` line to "validated".
- Mark this plan as complete in §5.

### If clean numbers diverge meaningfully
- Keep §0 caveat as-is.
- Add §10 "Post-fix re-run results" documenting the new figures
  alongside the old.
- If a novel SA-positive reachable terminal appears in §3.3, that
  is the new lead — propose a forced-game test to verify whether
  AlphaQ has an off-Nash deviation there.
- Re-open the H2 (Nash) conclusion: maybe AlphaQ wasn't actually at
  Nash; we just couldn't see clearly through the bug.

---

## Phase 5 — Bookkeeping

| Field | Value |
|---|---|
| Plan started session: | (fill in) |
| HEAD at start: | (fill in from §1.1) |
| Run ID for experiment: | (fill in from §1.5) |
| Games completed: | (fill in after §2) |
| Plan completed: | (fill in) |
| Outcome: | (fill in: "validated" / "diverged with X" / "in progress") |

---

## Resume protocol (for a fresh `/compact` or new session)

If you are reading this in a new session:

1. **Read this plan first.**  It is the source of truth.
2. **Check `memory/active_tasks.md`** for the latest in-progress
   state — it will say which Phase you are in.
3. **Check `git log --oneline -5`** to confirm the branch state.
4. Resume at the Phase indicated in `active_tasks.md`.

If `active_tasks.md` has nothing about this plan, start at §1.1.

---

## Estimated total wall-clock cost

| Phase | Cost |
|---|---|
| Phase 1 (pre-flight) | ~5 min |
| Phase 2 (300 games) | ~5 hours |
| Phase 3 (analysis) | ~30 min |
| Phase 4 (waypoint update) | ~15 min |
| **Total** | **~6 hours**, mostly unattended Phase 2 |
