# Dependency Repositories

This document describes the three repositories that form the foundation of this project,
their contents, and how each contributes to snowdrop-tangled-agents.

---

## Dependency Architecture

```
snowdrop-tangled-agents  (this project — agents, solvers, tournaments)
├── snowdrop-tangled-game-engine  >=1.1.0   (game rules, types, agent base class)
└── snowdrop-adjudicators         >=0.1.0   (terminal-state scoring)
    └── snowdrop-tangled-game-engine >=1.1.0
```

`tangled-adjudicate` is not a declared dependency. It is Geordie Rose's original
adjudication library, now superseded by `snowdrop-adjudicators` for production use.
It remains valuable as a reference implementation and for its D-Wave quantum hardware
integration and lookup-table solver, neither of which exists in `snowdrop-adjudicators`.

---

## 1. snowdrop-tangled-game-engine

| Field | Value |
|-------|-------|
| Location | `C:\Users\murr2\projects\snowdrop-tangled-game-package` |
| Import | `snowdrop_tangled_game_engine` |
| Version | 1.1.0 |
| Role | Game rules, types, agent contract |

### What it provides

This is the foundational layer. Everything in this project ultimately builds on the
abstractions defined here. It answers three questions:

1. **What does a game state look like?** (`GameState` TypedDict, `Game` class)
2. **What does an agent need to implement?** (`GameAgentBase` abstract class)
3. **What graphs can be played on?** (`GraphProperties`, `GRAPH_DATABASE`)

### Key classes

**`Game`** — The central game-state manager. Owns the full mutable state of a single
match. Key methods used throughout this project:

| Method | Purpose |
|--------|---------|
| `create_game()` | Initialise a match on a specific graph with two players |
| `get_legal_moves(player_id)` | Returns list of `(move_type, move_index, move_state)` |
| `make_move()` | Apply a move, advance the turn |
| `is_game_over()` | True when all edges are coloured |
| `get_game_state()` | Export current state as a serialisable dict |
| `set_game_state()` | Restore state from a previously exported dict |

**`GameAgentBase`** — Abstract base class every agent in this project inherits from.
Requires exactly one method:

```python
def make_move(self, game: Game) -> tuple[int, int, int]
# Returns (move_type, move_index, move_state)
```

`move_type` is `Game.MoveType.EDGE` for normal play or `Game.MoveType.QUIT` to forfeit.

**`LocalGamePlayer`** — Orchestrates a full two-player match between two `GameAgentBase`
instances. Used by `run_local_parallel_tournament.py` to run every game in the
tournament. Handles turn alternation and game-over detection.

**`Edge` / `Vertex`** — Value types with IntEnum states:

| Edge State | Value | Ising J | Meaning |
|------------|-------|---------|---------|
| `NONE` | 0 | — | Not yet set |
| `ZERO` | 1 | 0 | Grey (no coupling) |
| `FM` | 2 | −1 | Green (ferromagnetic) |
| `AFM` | 3 | +1 | Purple (antiferromagnetic) |

**`GraphProperties`** — Loads graph definitions from the built-in database.
`ALLOWED_GRAPHS = [2, 5, 11, 12, 18, 19, 20, 24]` are the X-Prize submission graphs.
Each graph entry stores vertex count, edge list, vertex ownership, epsilon (draw
threshold), and anneal time.

### Graphs in the database

| ID | Name | Vertices | Edges | Notes |
|----|------|----------|-------|-------|
| 2 | K₃ | 3 | 3 | Triangle; used for smoke tests |
| 5 | Petersen | 10 | 15 | Classic snark graph |
| 11 | P₃ | 3 | 2 | Linear chain; unit-test fixture |
| 12 | Moser Spindle | 7 | 11 | SA adjudicator has known errors here |
| 18 | 3-Prism | 6 | 9 | SA adjudicator has known errors here |
| 19 | Barbell | 8 | 13 | SA adjudicator has known errors here |
| 20 | Diamond | 4 | 5 | |
| 24 | Mutant C60 | 60 | 90 | Too large for Schrödinger solver |

### Value to this project

- **Every agent** in this project is a `GameAgentBase` subclass.
- **Every tournament** uses `LocalGamePlayer` to run matches.
- **Every graph** played in development and competition is defined here.
- Without this package, there is no game to play and no contract for agents to fulfil.

---

## 2. snowdrop-adjudicators

| Field | Value |
|-------|-------|
| Location | `C:\Users\murr2\projects\snowdrop-adjudicators` |
| Import | `snowdrop_adjudicators` |
| Version | 0.1.0 |
| Role | Terminal-state scoring (determines the winner) |
| Depends on | `snowdrop-tangled-game-engine` >=1.1.0 |

### What it provides

After all edges are coloured, someone has to decide who won. That is the adjudicator's
job. It takes a terminal `GameState`, simulates a quantum annealing process, and returns
a winner, a score, and the full correlation matrix.

This package ships two adjudicators:

| Class | Method | Speed | Accuracy |
|-------|--------|-------|----------|
| `SimulatedAnnealingAdjudicator` | Classical Monte Carlo | Fast | Approximate — known errors on graphs 12, 18, 19 |
| `SchrodingerEquationAdjudicator` | Exact quantum simulation | Slow | Ground truth on all tested X-Prize graphs except Petersen and Mutant C60 |

### SimulatedAnnealingAdjudicator

Uses D-Wave's NEAL library to sample low-energy Ising configurations.

```python
from snowdrop_adjudicators import SimulatedAnnealingAdjudicator

adj = SimulatedAnnealingAdjudicator()
adj.setup(epsilon=0.5, num_reads=10000)
result = adj.adjudicate(game_state)
# result['winner'] -> 'red' | 'blue' | 'draw'
# result['score']  -> float (influence difference)
```

**Setup parameters:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `epsilon` | (required) | Score threshold for draw vs win |
| `num_reads` | 10 000 | Number of annealing samples |
| `num_sweeps` | 16 | Sweeps per read |
| `beta_max` | 3.0 | Maximum inverse temperature |

**Known systematic errors:** On graphs with frustrated ground states (Moser Spindle,
3-Prism, Barbell), simulated annealing does not account for order-by-disorder effects
and can return the wrong winner. See `docs/SCORE_OUTCOME_DISCREPANCY.md` for the full
analysis and its impact on agent training.

### SchrodingerEquationAdjudicator

Directly solves the time-dependent Schrödinger equation using the D-Wave Advantage2
annealing schedule. Produces ground-truth results but scales exponentially with qubit
count — practical only for graphs up to ~7 qubits.

```python
from snowdrop_adjudicators import SchrodingerEquationAdjudicator

adj = SchrodingerEquationAdjudicator()
adj.setup(epsilon=0.5, anneal_time=40.0)
result = adj.adjudicate(game_state)
```

**Setup parameters:**

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `epsilon` | (required) | Score threshold for draw vs win |
| `anneal_time` | (required) | Annealing duration in nanoseconds |
| `s_min` | 0.001 | Start of anneal window |
| `s_max` | 0.999 | End of anneal window |

### Shared output format

Both adjudicators return an `AdjudicationResult` TypedDict:

```python
{
    'game_state':         GameState,          # echo of input
    'adjudicator':        str,                # 'simulated_annealing' or 'schrodinger_equation'
    'winner':             str | None,         # 'red' | 'blue' | 'draw' | None
    'score':              float | None,       # red_influence - blue_influence
    'influence_vector':   ndarray | None,     # per-vertex influence
    'correlation_matrix': ndarray | None,     # spin-spin correlations
    'parameters':         dict                # config used
}
```

### Internal pipeline

1. Validate game state structure
2. Convert `GameState` → `IsingModel` (h fields, J couplings)
3. Compute correlation matrix (via NEAL or Schrödinger evolution)
4. Influence vector = row sums of correlation matrix
5. Score = influence[red_vertex] − influence[blue_vertex]
6. Winner = `'red'` if score > epsilon, `'blue'` if score < −epsilon, else `'draw'`

### Test coverage

Tests live in `snowdrop_adjudicators/tests/`. Seven fixture game states cover all X-Prize
graphs. The simulated annealing tests use 1 M reads for deterministic validation and
explicitly mark graphs 12, 18, 19 as expected failures. Schrödinger tests skip Petersen
(too slow) and validate scores within ±0.15 tolerance.

### Value to this project

- **Terminal evaluation in every agent.** MCTS rollouts, LUT lookups, and P(win) calibration
  all bottom out in a call to `SimulatedAnnealingAdjudicator.adjudicate()`.
- **Ground truth for calibration.** `SchrodingerEquationAdjudicator` was used to generate
  the P(win) calibration curve (`calibration_pwin.mat`) that corrects SA's systematic bias.
- **Tournament scoring.** `run_local_parallel_tournament.py` uses SA adjudication to
  determine the winner of every game.

---

## 3. tangled-adjudicate

| Field | Value |
|-------|-------|
| Location | `C:\Users\murr2\projects\tangled-adjudicate` |
| Import | `tangled_adjudicate` |
| Version | 0.0.1 |
| Role | Reference implementation; D-Wave hardware; lookup tables |
| Author | Geordie Rose (Snowdrop Quantum Applications Corporation) |

### Relationship to snowdrop-adjudicators

`tangled-adjudicate` is Geordie Rose's original adjudication library. `snowdrop-adjudicators`
is the modernised, production-hardened version used as a declared dependency. The two share
the same core algorithms (simulated annealing via NEAL, Schrödinger equation solver) but
differ in structure and scope:

| Feature | tangled-adjudicate | snowdrop-adjudicators |
|---------|--------------------|-----------------------|
| Simulated Annealing | Yes | Yes |
| Schrödinger Equation | Yes | Yes |
| **D-Wave Quantum Hardware** | **Yes** | No |
| **Lookup Table Solver** | **Yes** | No |
| Abstract base class | No | Yes (`Adjudicator` ABC) |
| TypedDict result type | No | Yes (`AdjudicationResult`) |
| Test suite | No | Yes (7 fixtures, parametrised) |

### What it provides

A single `Adjudicator` class with four solver methods:

**`simulated_annealing(game_state)`** — Neal-based classical approximation. Functionally
equivalent to `SimulatedAnnealingAdjudicator` in the newer package.

**`schrodinger_equation(game_state)`** — Numerical quantum simulation. Equivalent to
`SchrodingerEquationAdjudicator`.

**`quantum_annealing(game_state)`** — Submits the Ising problem to real D-Wave quantum
hardware. This is the only path to actual quantum annealing in the entire repository set.
The implementation handles:
- Multiple non-overlapping embeddings into Zephyr topology (via minorminer)
- Graph automorphism selection for noise averaging
- Gauge transforms (random sign flips for bias cancellation)
- Shimming (iterative flux-bias correction)
- Parallel chip runs across multiple embeddings

**`lookup_table(game_state)`** — Pre-computed exhaustive results for K₃ and K₄. Downloads
cached tables from Google Drive on first use. Fastest possible adjudication for small graphs.

### Graph definitions

`tangled-adjudicate` ships its own `GraphProperties` class defining 11 graphs (numbered
1–11), including several not in the game engine's database:

| ID | Name | Vertices | Edges |
|----|------|----------|-------|
| 1 | K₂ | 2 | 1 |
| 2 | K₃ | 3 | 3 |
| 3 | K₄ | 4 | 6 |
| 4 | Hexagon | 6 | 6 |
| 5 | Petersen | 10 | 15 |
| 6 | Tesseract | 16 | 32 |
| 7–10 | Zephyr Z(1,1)–Z(1,4) | 12–48 | 22–280 |
| 11 | Linear 3-vertex | 3 | 2 |

Graphs 6–10 are D-Wave Zephyr-topology graphs, useful for benchmarking hardware
embedding but not part of the X-Prize competition set.

### Configuration

All behaviour is controlled through a `Params` class:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `GRAPH_NUMBER` | 3 | Which graph to use |
| `EPSILON` | 0.5 | Draw threshold |
| `NUM_READS_SA` | 10 000 | SA sample count |
| `USE_QC` | False | Enable D-Wave hardware |
| `QC_SOLVER_TO_USE` | Advantage2_prototype2.6 | Target chip |
| `NUMBER_OF_CHIP_RUNS` | 1 | Parallel embedding runs |
| `NUM_READS_QC` | 1 000 | Samples per chip run |
| `ANNEAL_TIME_IN_NS` | 5 | Quantum anneal duration |
| `USE_GAUGE_TRANSFORM` | False | Random gauge flip |
| `USE_SHIM` | False | Flux-bias shimming |

### Value to this project

- **D-Wave hardware path.** If quantum annealing experiments are ever resumed, this is
  the only implementation that can submit problems to real hardware. The embedding,
  automorphism, gauge-transform, and shimming logic is non-trivial and not replicated
  elsewhere.
- **Lookup-table solver.** Pre-computed exhaustive ground-truth results for K₃ and K₄
  can serve as fast validation fixtures without running either SA or Schrödinger.
- **Reference implementation.** When the behaviour of `snowdrop-adjudicators` is
  unclear or suspected incorrect, `tangled-adjudicate` is the canonical source to
  compare against. It is Geordie Rose's original code.
- **Additional graph definitions.** The Zephyr-topology graphs (7–10) and Tesseract (6)
  are not available anywhere else in the project and may be useful for hardware
  embedding research.

---

## Quick reference: which adjudicator to use

| Situation | Use |
|-----------|-----|
| Tournament games (fast, approximate) | `SimulatedAnnealingAdjudicator` |
| Generating ground truth for calibration | `SchrodingerEquationAdjudicator` |
| Validating against known-correct results | `tangled-adjudicate` lookup table (K₃/K₄) |
| Submitting to D-Wave quantum hardware | `tangled-adjudicate` quantum annealing |
| Graphs 12, 18, 19 accurate scoring | `SchrodingerEquationAdjudicator` (SA is wrong here) |
| Graphs too large for Schrödinger | `SimulatedAnnealingAdjudicator` (accept SA errors) |

---

## Ising model mapping (shared across all three packages)

All adjudicators convert game edges to an Ising spin-glass problem using the same mapping:

| Edge State | Ising J | Physical Meaning |
|------------|---------|------------------|
| Grey (ZERO) | 0 | No coupling |
| Green (FM) | −1 | Ferromagnetic (spins align) |
| Purple (AFM) | +1 | Antiferromagnetic (spins anti-align) |

Uncoloured edges (state 0) also map to J = 0. The local fields h are all zero; the
problem is a pure spin-glass determined entirely by the edge couplings.
