"""
Generate Terminal State Lookup Table (LUT) for MATLAB MCTS.

Enumerates all 2^E possible terminal states for a given graph and evaluates
each one.  Results are saved as a MATLAB .mat file for O(1) lookup during
MCTS rollouts.

Scorer selection
----------------
SchrodingerEquationAdjudicator produces ground-truth scores but scales
super-exponentially with vertex count (~7s at 4 vertices, ~108s at 7
vertices, >3 min at 10 vertices).  SimulatedAnnealingAdjudicator is fast
(~50ms per state at 10K reads) but has documented systematic errors on
frustrated ground states — it can return the wrong winner on graphs 12, 18,
and 19, and returns inaccurate scores on graph 5 (Petersen).

The script auto-selects the scorer based on vertex count:

    vertices <= 5  →  Schrödinger  (sequential, finishes in minutes)
    vertices <= 7  →  Schrödinger  (multiprocessing, finishes in hours)
    vertices >= 8  →  SA at 100K reads + optional Schrödinger validation sample

Use --scorer to override the automatic selection.

Usage
-----
    # Auto-select scorer, generate LUT for graph 5 (Petersen)
    python snowdrop_tangled_agents/tools/generate_terminal_lut.py --graph 5

    # Force Schrödinger on a small graph (ground truth)
    python snowdrop_tangled_agents/tools/generate_terminal_lut.py --graph 20 --scorer schrodinger

    # SA LUT for Petersen with a 50-state Schrödinger validation sample
    python snowdrop_tangled_agents/tools/generate_terminal_lut.py --graph 5 --validate 50

    # Dry run: print what would be generated without running
    python snowdrop_tangled_agents/tools/generate_terminal_lut.py --graph 5 --dry-run

Output
------
    snowdrop_tangled_agents/matlab/rl/data/terminal_scores.mat

    Fields in the .mat file:
        terminal_scores   — float32 array of length 2^E, indexed by bit-packed state
        num_states        — total number of terminal states (2^E)
        num_edges         — number of edges E
        graph_id          — graph number used
        scorer            — 'schrodinger' or 'simulated_annealing'
        num_reads         — SA num_reads (0 if Schrödinger was used)
        generation_time_sec
        description       — human-readable summary
"""

import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import savemat
from tqdm import tqdm

from snowdrop_adjudicators import (
    SchrodingerEquationAdjudicator,
    SimulatedAnnealingAdjudicator,
)
from snowdrop_tangled_game_engine import GraphProperties
from snowdrop_tangled_game_engine.game import Edge


# ---------------------------------------------------------------------------
# Scorer selection thresholds
# ---------------------------------------------------------------------------
# Schrödinger is ground truth but impractical above this vertex count for a
# full LUT *using this Python solver*.  At 7 vertices (~108s/state, 2048
# states) multiprocessing makes it feasible in hours.  At 10 vertices the
# Python eigh-based solver is too slow; use the MATLAB split-operator solver
# instead (generate_petersen_lut_schrodinger.m, ~0.7 s/state).
SCHRODINGER_FAST_VERTEX_LIMIT = 5    # sequential, minutes
SCHRODINGER_SLOW_VERTEX_LIMIT = 7    # multiprocessing, hours
SA_NUM_READS = 100_000               # high read count for stable SA scores


# ---------------------------------------------------------------------------
# State encoding  (shared with MATLAB side)
# ---------------------------------------------------------------------------
def idx_to_state(idx: int, num_edges: int) -> str:
    """
    Convert a 0-based index to an E-char state string.

    Bit j of idx: 1 → 'G' (green / FM), 0 → 'P' (purple / AFM).
    """
    return ''.join('G' if (idx >> j) & 1 else 'P' for j in range(num_edges))


def state_to_idx(state: str) -> int:
    """Convert an E-char state string back to a 0-based index."""
    idx = 0
    for j, ch in enumerate(state):
        if ch == 'G':
            idx |= (1 << j)
    return idx


# ---------------------------------------------------------------------------
# Game-state builder  (shared by both scorers)
# ---------------------------------------------------------------------------
def build_game_state(state_str: str, edge_list, num_vertices: int,
                     graph_id: int, p1_node: int, p2_node: int) -> dict:
    """
    Convert a terminal state string to a GameState dict accepted by any
    adjudicator.
    """
    edges = []
    for i, (v1, v2) in enumerate(edge_list):
        edge_state = Edge.State.FM.value if state_str[i] == 'G' else Edge.State.AFM.value
        edges.append((v1, v2, edge_state))

    return {
        'num_nodes':            num_vertices,
        'edges':                edges,
        'graph_id':             graph_id,
        'player1_id':           'p1',
        'player2_id':           'p2',
        'turn_count':           len(edge_list),
        'current_player_index': 2,
        'player1_node':         p1_node,
        'player2_node':         p2_node,
    }


# ---------------------------------------------------------------------------
# Per-state evaluation functions  (top-level so they are picklable for
# multiprocessing)
# ---------------------------------------------------------------------------
def _eval_schrodinger(args):
    """Evaluate one state with Schrödinger.  Designed for ProcessPoolExecutor."""
    idx, state_str, edge_list, num_vertices, graph_id, p1_node, p2_node, epsilon, anneal_time = args

    adj = SchrodingerEquationAdjudicator()
    adj.setup(epsilon=epsilon, anneal_time=anneal_time)

    game_state = build_game_state(state_str, edge_list, num_vertices,
                                  graph_id, p1_node, p2_node)
    result = adj.adjudicate(game_state)
    return idx, float(result['score'])


def _eval_sa(args):
    """Evaluate one state with SA.  Designed for ProcessPoolExecutor."""
    idx, state_str, edge_list, num_vertices, graph_id, p1_node, p2_node, num_reads = args

    adj = SimulatedAnnealingAdjudicator()
    adj.setup(epsilon=0.0, num_reads=num_reads)

    game_state = build_game_state(state_str, edge_list, num_vertices,
                                  graph_id, p1_node, p2_node)
    result = adj.adjudicate(game_state)
    return idx, float(result['score'])


# ---------------------------------------------------------------------------
# LUT generation  — Schrödinger path
# ---------------------------------------------------------------------------
def generate_lut_schrodinger(num_edges: int, edge_list, num_vertices: int,
                             graph_id: int, p1_node: int, p2_node: int,
                             epsilon: float, anneal_time: float,
                             num_workers: int) -> np.ndarray:
    """Generate the full LUT using SchrodingerEquationAdjudicator."""
    total_states = 2 ** num_edges
    scores = np.zeros(total_states, dtype=np.float32)

    use_pool = (num_vertices > SCHRODINGER_FAST_VERTEX_LIMIT)

    if use_pool:
        print(f"  Schrödinger with {num_workers} workers (estimated hours)...")
        work = [
            (idx,
             idx_to_state(idx, num_edges),
             edge_list, num_vertices, graph_id, p1_node, p2_node,
             epsilon, anneal_time)
            for idx in range(total_states)
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_eval_schrodinger, w): w[0] for w in work}
            for future in tqdm(as_completed(futures), total=total_states,
                               desc="Schrödinger (parallel)"):
                idx, score = future.result()
                scores[idx] = score
    else:
        print(f"  Schrödinger sequential ({total_states} states)...")
        adj = SchrodingerEquationAdjudicator()
        adj.setup(epsilon=epsilon, anneal_time=anneal_time)
        for idx in tqdm(range(total_states), desc="Schrödinger"):
            state_str = idx_to_state(idx, num_edges)
            game_state = build_game_state(state_str, edge_list, num_vertices,
                                          graph_id, p1_node, p2_node)
            result = adj.adjudicate(game_state)
            scores[idx] = float(result['score'])

    return scores


# ---------------------------------------------------------------------------
# LUT generation  — SA path
# ---------------------------------------------------------------------------
def generate_lut_sa(num_edges: int, edge_list, num_vertices: int,
                    graph_id: int, p1_node: int, p2_node: int,
                    num_reads: int, num_workers: int) -> np.ndarray:
    """Generate the full LUT using SimulatedAnnealingAdjudicator."""
    total_states = 2 ** num_edges
    scores = np.zeros(total_states, dtype=np.float32)

    print(f"  SA with {num_reads:,} reads, {num_workers} workers...")
    work = [
        (idx,
         idx_to_state(idx, num_edges),
         edge_list, num_vertices, graph_id, p1_node, p2_node,
         num_reads)
        for idx in range(total_states)
    ]
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_eval_sa, w): w[0] for w in work}
        for future in tqdm(as_completed(futures), total=total_states,
                           desc="SA (parallel)"):
            idx, score = future.result()
            scores[idx] = score

    return scores


# ---------------------------------------------------------------------------
# Schrödinger validation sample  — spot-checks SA scores against ground truth
# ---------------------------------------------------------------------------
def run_validation_sample(n_samples: int, scores_sa: np.ndarray,
                          num_edges: int, edge_list, num_vertices: int,
                          graph_id: int, p1_node: int, p2_node: int,
                          epsilon: float, anneal_time: float) -> dict:
    """
    Pick n_samples random terminal states, score them with Schrödinger, and
    compare against the SA scores already in scores_sa.

    Returns a dict with:
        indices         — which states were sampled
        sa_scores       — SA scores for those states
        schrodinger_scores — Schrödinger scores for those states
        winner_flips    — count of states where SA and Schrödinger disagree on winner
        mean_abs_error  — mean |SA − Schrödinger| across the sample
    """
    total_states = 2 ** num_edges
    sample_indices = sorted(random.sample(range(total_states), n_samples))

    print(f"  Running Schrödinger validation on {n_samples} sampled states...")
    adj = SchrodingerEquationAdjudicator()
    adj.setup(epsilon=epsilon, anneal_time=anneal_time)

    sa_scores = []
    se_scores = []
    winner_flips = 0

    for i, idx in enumerate(tqdm(sample_indices, desc="Validation")):
        state_str = idx_to_state(idx, num_edges)
        game_state = build_game_state(state_str, edge_list, num_vertices,
                                      graph_id, p1_node, p2_node)

        result = adj.adjudicate(game_state)
        se_score = float(result['score'])
        sa_score = float(scores_sa[idx])

        sa_scores.append(sa_score)
        se_scores.append(se_score)

        # Winner flip: signs disagree beyond epsilon on both sides
        sa_winner = 'red' if sa_score > epsilon else ('blue' if sa_score < -epsilon else 'draw')
        se_winner = 'red' if se_score > epsilon else ('blue' if se_score < -epsilon else 'draw')
        if sa_winner != se_winner:
            winner_flips += 1
            print(f"    FLIP idx={idx}: SA={sa_score:+.4f} ({sa_winner}) "
                  f"vs SE={se_score:+.4f} ({se_winner})")

    sa_arr = np.array(sa_scores)
    se_arr = np.array(se_scores)

    return {
        'indices':            np.array(sample_indices),
        'sa_scores':          sa_arr,
        'schrodinger_scores': se_arr,
        'winner_flips':       winner_flips,
        'mean_abs_error':     float(np.mean(np.abs(sa_arr - se_arr))),
        'max_abs_error':      float(np.max(np.abs(sa_arr - se_arr))),
        'flip_rate':          winner_flips / n_samples,
    }


# ---------------------------------------------------------------------------
# Statistics and save
# ---------------------------------------------------------------------------
def print_statistics(scores: np.ndarray, epsilon: float, scorer: str):
    """Print summary statistics for the generated LUT."""
    total = len(scores)
    wins   = int(np.sum(scores >  epsilon))
    losses = int(np.sum(scores < -epsilon))
    draws  = total - wins - losses

    print(f"\n  Statistics ({scorer}):")
    print(f"    Min score:  {scores.min():+.4f}")
    print(f"    Max score:  {scores.max():+.4f}")
    print(f"    Mean score: {scores.mean():+.4f}")
    print(f"    Std dev:    {scores.std():.4f}")
    print(f"    P1 wins (score > {epsilon}):   {wins:>6,} ({100*wins/total:.1f}%)")
    print(f"    P2 wins (score < -{epsilon}):  {losses:>6,} ({100*losses/total:.1f}%)")
    print(f"    Draws:                          {draws:>6,} ({100*draws/total:.1f}%)")


def save_lut(scores: np.ndarray, output_path: Path, graph_id: int,
             num_edges: int, scorer: str, num_reads: int, elapsed: float,
             validation: dict | None = None):
    """Save the LUT and metadata as a MATLAB .mat file."""
    mat_data = {
        'terminal_scores':      scores,
        'num_states':           len(scores),
        'num_edges':            num_edges,
        'graph_id':             graph_id,
        'scorer':               scorer,
        'num_reads':            num_reads,
        'generation_time_sec':  elapsed,
        'generated_at':         datetime.now(timezone.utc).isoformat(),
        'description': (
            f"Terminal state scores for graph {graph_id} ({scorer}). "
            f"Index i: bit j=1 means edge j is G (green/FM), 0 means P (purple/AFM). "
            f"Scores are from Player 1 perspective."
        ),
    }

    if validation is not None:
        mat_data['validation_indices']            = validation['indices']
        mat_data['validation_sa_scores']          = validation['sa_scores']
        mat_data['validation_schrodinger_scores'] = validation['schrodinger_scores']
        mat_data['validation_winner_flips']       = validation['winner_flips']
        mat_data['validation_mean_abs_error']     = validation['mean_abs_error']
        mat_data['validation_max_abs_error']      = validation['max_abs_error']
        mat_data['validation_flip_rate']          = validation['flip_rate']

    savemat(output_path, mat_data)
    print(f"\n  Saved LUT to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate terminal-state LUT for MATLAB MCTS solver."
    )
    parser.add_argument(
        '--graph', type=int, default=5,
        help='Graph number from GraphProperties (default: 5, Petersen).'
    )
    parser.add_argument(
        '--scorer', choices=['auto', 'schrodinger', 'simulated_annealing'],
        default='auto',
        help='Scorer to use.  "auto" selects based on vertex count.'
    )
    parser.add_argument(
        '--validate', type=int, default=0, metavar='N',
        help='Run Schrödinger on N randomly sampled states to validate SA scores. '
             'Only meaningful when scorer is SA.'
    )
    parser.add_argument(
        '--workers', type=int, default=8,
        help='Number of parallel workers (default: 8).'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print plan without running any adjudication.'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Output filename (relative to data dir or absolute). '
             'Default: terminal_scores.mat'
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------------------
    # Load graph metadata from GraphProperties
    # ---------------------------------------------------------------------------
    gp = GraphProperties()

    if args.graph not in gp.graph_database:
        print(f"ERROR: graph {args.graph} not in graph_database. "
              f"Available: {list(gp.graph_database.keys())}")
        return

    g = gp.graph_database[args.graph]
    edge_list     = g['edge_list']
    num_vertices  = g['num_nodes']
    num_edges     = len(edge_list)
    p1_node       = g['player1_node']
    p2_node       = g['player2_node']
    total_states  = 2 ** num_edges

    # epsilon and anneal_time are parallel lists indexed by position in allowed_graphs
    if args.graph in gp.allowed_graphs:
        idx       = gp.allowed_graphs.index(args.graph)
        epsilon   = gp.epsilon_values[idx]
        anneal_time = gp.anneal_times[idx]
    else:
        epsilon     = 0.5       # safe default
        anneal_time = 350.0     # safe default

    # ---------------------------------------------------------------------------
    # Scorer selection
    # ---------------------------------------------------------------------------
    if args.scorer == 'auto':
        if num_vertices <= SCHRODINGER_SLOW_VERTEX_LIMIT:
            scorer = 'schrodinger'
        else:
            scorer = 'simulated_annealing'
    else:
        scorer = args.scorer

    num_reads = SA_NUM_READS if scorer == 'simulated_annealing' else 0

    # ---------------------------------------------------------------------------
    # Plan
    # ---------------------------------------------------------------------------
    print("=" * 70)
    print("TERMINAL STATE LUT GENERATOR")
    print("=" * 70)
    print(f"  Graph:            {args.graph}")
    print(f"  Vertices:         {num_vertices}")
    print(f"  Edges:            {num_edges}")
    print(f"  Terminal states:  {total_states:,}")
    print(f"  P1 vertex:        {p1_node}")
    print(f"  P2 vertex:        {p2_node}")
    print(f"  Epsilon:          {epsilon}")
    print(f"  Anneal time:      {anneal_time} ns")
    print(f"  Scorer:           {scorer}")
    if scorer == 'simulated_annealing':
        print(f"  SA num_reads:     {num_reads:,}")
    print(f"  Workers:          {args.workers}")
    if args.validate > 0:
        print(f"  Validation sample:{args.validate} states (Schrödinger)")
    print()

    if args.dry_run:
        print("  [DRY RUN — no adjudication will be performed]")
        return

    # ---------------------------------------------------------------------------
    # Output path
    # ---------------------------------------------------------------------------
    script_dir   = Path(__file__).parent
    project_root = script_dir.parent.parent
    data_dir     = (project_root / "snowdrop_tangled_agents"
                    / "matlab" / "rl" / "data")
    if args.output:
        out_arg = Path(args.output)
        output_path = out_arg if out_arg.is_absolute() else (data_dir / out_arg)
    else:
        output_path = data_dir / "terminal_scores.mat"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Generate
    # ---------------------------------------------------------------------------
    print("Generating LUT...")
    start_time = time.time()

    if scorer == 'schrodinger':
        scores = generate_lut_schrodinger(
            num_edges, edge_list, num_vertices, args.graph,
            p1_node, p2_node, epsilon, anneal_time, args.workers
        )
    else:
        scores = generate_lut_sa(
            num_edges, edge_list, num_vertices, args.graph,
            p1_node, p2_node, num_reads, args.workers
        )

    elapsed = time.time() - start_time
    print(f"  Generation complete in {elapsed:.1f}s "
          f"({elapsed / total_states * 1000:.2f} ms/state)")

    # ---------------------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------------------
    print_statistics(scores, epsilon, scorer)

    # ---------------------------------------------------------------------------
    # Validation sample (SA path only)
    # ---------------------------------------------------------------------------
    validation = None
    if scorer == 'simulated_annealing' and args.validate > 0:
        print(f"\nRunning Schrödinger validation ({args.validate} states)...")
        val_start = time.time()
        validation = run_validation_sample(
            args.validate, scores,
            num_edges, edge_list, num_vertices, args.graph,
            p1_node, p2_node, epsilon, anneal_time
        )
        val_elapsed = time.time() - val_start

        print(f"\n  Validation results ({val_elapsed:.0f}s):")
        print(f"    Mean |SA - SE| error: {validation['mean_abs_error']:.4f}")
        print(f"    Max  |SA - SE| error: {validation['max_abs_error']:.4f}")
        print(f"    Winner flips:         {validation['winner_flips']} / {args.validate} "
              f"({100 * validation['flip_rate']:.1f}%)")

    # ---------------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------------
    save_lut(scores, output_path, args.graph, num_edges, scorer,
             num_reads, elapsed, validation)


if __name__ == "__main__":
    main()
