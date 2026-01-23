# Tangled Game Analytics

A statistics analyzer and visualizer for tracking performance, discovering patterns, and measuring improvement in Tangled quantum game play.

## Overview

The analytics system collects data from every game played against the tangled-game.com website and provides tools to:

- **Track Progress** — Monitor win rate and score trends over time
- **Analyze Edges** — Discover which edge/color combinations are most effective
- **Study Openings** — Identify winning opening sequences
- **Model Opponents** — Learn opponent response patterns with Bayesian updating
- **Calibrate Evaluation** — Compare our predictions against actual outcomes

---

## Design Philosophy

This isn't just a collection of plots—it's an **instrumented research system** that closes the loop between play, learning, and diagnosis.

### Observability, Not Just Performance

Most game bots report W/L/D and stop. This system reports:

| Layer | What it measures | Why it matters |
|-------|------------------|----------------|
| **Trajectory** | Rolling win rate, rolling score | Detects trends, not just snapshots |
| **Mechanism** | Edge/color effectiveness, openings | Explains *why* performance changed |
| **Belief vs Reality** | Calibration error | Validates the evaluation function |
| **Opponent Structure** | Response-conditional probabilities | Reveals if learning is working |

When something changes, you can answer *why*, not just *what*.
This is the difference between "tuning" and **control**.

### Analytics Aligned to Architecture

Every metric maps to a subsystem lever:

| Plot | Subsystem | What you can adjust |
|------|-----------|---------------------|
| Progress | System behavior | Learning rate, policy version |
| Edge effectiveness | Heuristic priors | Bias terms, edge weights |
| Opening analysis | LUT-assisted phase | Opening book sequences |
| Calibration report | Annealer + evaluator | Adjudicator parameters |
| Opponent patterns | Bayesian model | Smoothing, alpha blending |

There are no vanity metrics. Each measurement corresponds to a lever you actually have.

### Future-Proofed Without Overengineering

Key design choices that enable scale:

- **Timestamped plots** → Historical comparison without overwriting truth
- **Query-first architecture** → Plots are views, not logic
- **SQLite schema stays simple** → No premature warehouse complexity
- **CLI-first tooling** → Reproducible, scriptable, automatable

This scales to thousands of games without turning into archaeology.

### Evaluating Non-Stationary Learners

A critically-damped, live-learning system can't be collapsed to a single scalar. Instead, track:

- Calibration error over time
- Loss rate vs draw rate divergence
- Score variance
- Opponent-response entropy

These are exactly the plots this system provides.
That's the correct epistemology for an adaptive system.

### Defensible to Outsiders

When challenged with:
- *"Are those wins just luck?"*
- *"Is the model actually learning?"*
- *"Did this change help or hurt?"*

You can answer with time-aligned plots, before/after comparisons, and opponent-specific breakdowns.

---

### Open Questions for Development

The hard part isn't building features—it's choosing *which question to ask next*, because now the system will actually answer it.

Current candidates:

1. **Is the opponent model helping rollouts?**
   A/B test: disable it, measure win rate delta

2. **Where do losses diverge from draws?**
   Score trajectory analysis by outcome

3. **Is there a learning ceiling?**
   Plot win rate vs opponent model entropy—does learning saturate?

4. **What's the cost of the opening book?**
   Does forcing E9→E10→E11 hurt late-game flexibility?

5. **Which edge mistakes are fatal?**
   Cluster losing games by the move where score collapsed

---

## Quick Start

```bash
# Generate all plots
python -m snowdrop_tangled_agents.tools.plot_progress --all

# Print text summary
python -c "from snowdrop_tangled_agents.stats import queries; queries.print_summary()"

# Check calibration accuracy
python -c "from snowdrop_tangled_agents.stats import queries; queries.print_calibration_report()"

# Live session stats
python -m snowdrop_tangled_agents.stats.session_stats
```

---

## Live Session Stats

Real-time statistics for the current gaming session. Automatically detects session boundaries and provides trend analysis.

### Usage

```bash
# One-shot report
python -m snowdrop_tangled_agents.stats.session_stats

# Watch mode (refreshes every 60s, press q to exit)
python -m snowdrop_tangled_agents.stats.session_stats --watch

# Custom refresh interval (30s)
python -m snowdrop_tangled_agents.stats.session_stats -w -i 30

# Custom session gap (10 minutes instead of default 30)
python -m snowdrop_tangled_agents.stats.session_stats --gap 10

# JSON output for scripting
python -m snowdrop_tangled_agents.stats.session_stats --json

# Clean up stale in-progress games (dry run)
python -m snowdrop_tangled_agents.stats.session_stats --cleanup

# Actually clean up stale games
python -m snowdrop_tangled_agents.stats.session_stats --cleanup --force
```

### Output Format

```
session_start = 2026-01-22 23:35
session_end = 2026-01-23 00:56 (est)
games = 49/50
wins = 5, 10.2%
draws = 22, 44.9%
losses = 22, 44.9%
avg_score = +0.556
median_score = +0.423
min_score = -2.428
max_score = +4.085
score_std = 1.006
avg_moves = 8.0
score_trend = -0.068
winrate_trend = +3.7%
recent_5 = DLLLL
```

### Session Detection

A **session** is a contiguous group of games. Sessions are separated by gaps of 30+ minutes (configurable via `--gap`).

- If a game is **in progress** (`result IS NULL`), the session includes all games from start to now
- If no game is running, shows the **most recent completed session**
- **Estimated end time** is calculated from play rate when games remain

### Statistics Explained

| Statistic | Description |
|-----------|-------------|
| `session_start` | First game timestamp (local time) |
| `session_end` | Last game or estimated completion (local time) |
| `games` | Completed/Total games in session |
| `wins/draws/losses` | Count and percentage |
| `avg_score` | Mean final score |
| `median_score` | Median final score (robust to outliers) |
| `min_score/max_score` | Score range |
| `score_std` | Score standard deviation |
| `avg_moves` | Average moves per game |
| `score_trend` | Second half avg minus first half avg |
| `winrate_trend` | Second half win rate minus first half |
| `recent_5` | Last 5 results (W/D/L)

### Timezone Handling

- **Database**: Timestamps stored in UTC
- **Display**: Converted to local time automatically

## Visualizations

### Progress Plot (`progress`)

Tracks performance over time with rolling averages.

```bash
python -m snowdrop_tangled_agents.tools.plot_progress -t progress
```

**Shows:**
- Rolling 20-game win rate (bold blue line)
- Cumulative win rate (faint blue line)
- Target win rate threshold (green dashed)
- Policy version markers (red dotted)
- Rolling average score (purple line)

**Use for:** Measuring improvement after code changes, detecting performance regressions.

---

### Edge Effectiveness Plot (`edge`)

Analyzes which edges and colors lead to better outcomes.

```bash
python -m snowdrop_tangled_agents.tools.plot_progress -t edge
```

**Shows:**
- Average score delta by edge/color
- Win rate by edge/color
- Green vs Purple comparison

**Use for:** Tuning heuristics, discovering edge priorities, validating strategy assumptions.

---

### Opening Analysis Plot (`opening`)

Evaluates opening move sequences.

```bash
python -m snowdrop_tangled_agents.tools.plot_progress -t opening
```

**Shows:**
- Win rate by opening sequence
- Average final score by opening
- Sample sizes for statistical significance

**Use for:** Opening book development, identifying strong/weak starts.

---

## Plot Naming Convention

Plots are saved to `plots/` with timestamps:

```
{plot_type}_{YYYYMMDD}_{HHMMSS}.png
```

**Examples:**
```
plots/progress_20260123_004342.png
plots/edge_20260123_004342.png
plots/opening_20260123_004343.png
```

This allows tracking how metrics evolve across multiple analysis sessions.

---

## Data Queries

The `snowdrop_tangled_agents.stats.queries` module provides programmatic access:

```python
from snowdrop_tangled_agents.stats import queries

# Edge effectiveness ranking
edges = queries.get_edge_effectiveness(min_games=5)
for e in edges[:5]:
    print(f"E{e.edge} {e.color}: delta={e.avg_delta:+.3f}, WR={e.win_rate:.1%}")

# Score progression by result
win_prog = queries.get_score_progression(result='win')
loss_prog = queries.get_score_progression(result='loss')

# Opening sequences
openings = queries.get_opening_sequences(num_moves=4, min_occurrences=3)

# Critical positions (big score swings)
critical = queries.get_critical_positions(score_swing_threshold=0.5)

# Opponent patterns
patterns = queries.get_opponent_patterns(opponent="melissa")
```

---

## Database Schema

Statistics are stored in SQLite at `~/.tangled/game_stats.db`:

### `games` table
| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | Unique game ID |
| timestamp | DATETIME | When game started |
| opponent | TEXT | Opponent name (e.g., "melissa") |
| result | TEXT | 'win', 'loss', 'draw' |
| final_score | REAL | Final game score |
| strategy | TEXT | Strategy used (e.g., "hybrid_solver") |

### `moves` table
| Column | Type | Description |
|--------|------|-------------|
| game_id | TEXT | Foreign key to games |
| move_number | INTEGER | Move sequence number |
| player | TEXT | 'us' or 'opponent' |
| edge | INTEGER | Edge index (0-14) |
| color | TEXT | 'G' (green) or 'P' (purple) |
| score_after | REAL | Score after this move |
| score_delta | REAL | Change from previous score |
| state_after | TEXT | 15-char board state |

### `calibration` table
| Column | Type | Description |
|--------|------|-------------|
| terminal_state | TEXT | Final board state |
| website_score | REAL | Score from tangled-game.com |
| predicted_score | REAL | Our evaluation |
| error | REAL | predicted - website |

---

## Opponent Modeling

The `OpponentModel` class learns opponent behavior using Bayesian updating:

```python
from snowdrop_tangled_agents.stats import get_opponent_model

model = get_opponent_model("melissa")
print(f"Games: {model.total_games}, Moves learned: {model.total_moves}")

# Predict opponent's response
probs = model.predict_response(
    our_last_move=(9, 'G'),
    available_moves=[(5, 'P'), (12, 'P'), (13, 'G')],
    grey_count=12,
    alpha=0.7
)
for move, prob in sorted(probs.items(), key=lambda x: -x[1])[:3]:
    print(f"  E{move[0]} {move[1]}: {prob:.1%}")
```

**Features:**
- Response-conditional: P(opp_move | our_last_move)
- Phase-conditional: P(opp_move | game_phase)
- Adaptive alpha blending based on confidence
- Laplace smoothing to handle sparse data
- Online learning after each game

---

## Extending the Analyzer

### Adding a New Plot Type

1. Add function to `plot_progress.py`:
```python
def plot_my_analysis(db_path: Path = None) -> Path:
    """My custom analysis plot."""
    # Query data
    # Create matplotlib figure
    # Save with: output_path = get_output_path("myanalysis")
    return output_path
```

2. Add to CLI in `main()`:
```python
parser.add_argument("--type", choices=[..., "myanalysis"])
# ...
elif args.type == "myanalysis":
    plot_my_analysis()
```

### Adding a New Query

Add to `snowdrop_tangled_agents/stats/queries.py`:
```python
def get_my_metric(db_path: Optional[Path] = None) -> dict:
    """Calculate my custom metric."""
    db_path = db_path or DEFAULT_DB_PATH
    with sqlite3.connect(db_path) as conn:
        # SQL query
        # Process results
        return results
```

---

## Future Ideas

- [ ] **Heat maps** — Board position value visualization
- [ ] **Move trees** — Interactive game tree exploration
- [ ] **Head-to-head** — Compare two policy versions directly
- [ ] **Time series** — Score by move number across games
- [ ] **Clustering** — Group games by play style or outcome pattern
- [ ] **Export** — CSV/JSON export for external analysis
- [ ] **Dashboard** — Real-time updating web dashboard
- [ ] **Annotations** — Mark significant games/moves for review

---

## Research Applications

This analytics system supports research into:

1. **Strategy Evolution** — How does performance change as the model learns?
2. **Quantum Game Theory** — Which classical heuristics transfer to quantum games?
3. **Opponent Adaptation** — How quickly can we learn opponent patterns?
4. **Opening Theory** — Are there dominant opening sequences?
5. **Endgame Analysis** — Where do games diverge from predicted outcomes?

Data collected here feeds back into strategy development, creating a virtuous cycle of play → analyze → improve → play.
