# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an agent development framework for the Tangled quantum game. Agents compete by coloring edges on graph structures, with outcomes determined by quantum adjudicators that simulate quantum annealing behavior.

## Build and Development Commands

```bash
# Install dependencies
poetry install

# Run tests with coverage
poetry run pytest

# Run a single test file
poetry run pytest snowdrop_tangled_agents/tests/test_file.py

# Run tournament simulation (100k games across X-Prize graphs)
poetry run python snowdrop_tangled_agents/playing_games/run_local_parallel_tournament.py
```

## Architecture

### Core Dependencies
- `snowdrop-tangled-game-engine`: Provides `Game`, `GameAgentBase`, `GamePlayerBase`, `LocalGamePlayer`, `GraphProperties`
- `snowdrop-adjudicators`: Provides `SimulatedAnnealingAdjudicator`, `SchrodingerEquationAdjudicator` for terminal state evaluation

### Agent Development Pattern
Agents inherit from `GameAgentBase` and implement `make_move(game: Game) -> tuple[int, int, int]`:
- Return format: `(move_type, move_index, move_state)`
- `move_type`: `Game.MoveType.EDGE` for edge moves, `Game.MoveType.QUIT` to quit
- `move_index`: Edge index (edges (i,j) where i<j are in lexical order)
- `move_state`: `Edge.State.ZERO` (grey), `Edge.State.FM` (green/ferromagnetic), `Edge.State.AFM` (purple/antiferromagnetic)

### Dynamic Agent Loading
Agents are loaded by string path via `import_agent()` in `utils/utilities.py`. Register new agents in the tournament by adding entries to the `competitors` dict with format:
```python
{'name': 'AgentName', 'agent_type': 'snowdrop_tangled_agents.YourAgentClass', 'kwargs': {...}}
```

### Tournament System
`run_local_parallel_tournament.py` runs parallel round-robin tournaments using `ProcessPoolExecutor`. Key configuration in `args` dict:
- `graph_number`: Which X-Prize graph (2, 11, 12, 18, 19, 20)
- `terminal_state_adjudicator`: `'simulated_annealing'` or `'schrodinger_equation'`
- `number_of_games_per_matchup`, `num_workers`: Parallelization settings

### Adjudicator Notes
Simulated annealing has known systematic errors on graphs 12 (Moser Spindle), 18 (3-Prism), and 19 (Barbell). Graphs 2, 11, 20 have matching adjudications between SA and ground truth. Schrödinger equation adjudicator is accurate but computationally expensive.

### Live Stats Dashboard

The game runner can publish real-time statistics to a WebSocket dashboard. Configure via environment variables:

```bash
export TANGLED_DASHBOARD_URL="wss://tangled-stats.fly.dev/ws/publish"
export TANGLED_DASHBOARD_API_KEY="your-api-key"
```

Or add to `.env` file. If not configured, publishing is silently disabled.

The publisher (`snowdrop_tangled_agents/stats/websocket_publisher.py`) sends:
- Session info (run_id, completed/planned games)
- Results (wins, draws, losses)
- Score statistics (avg, median, min, max, std)
- Trends (recent 5 results, score trend)
- Opponent model metrics (entropy, top-3 hit rate, prediction accuracy)
- ETA calculation based on game completion rate
