"""
Generate Terminal State Lookup Table (LUT) for MATLAB MCTS.

Enumerates all 2^15 = 32,768 possible terminal states and evaluates each
using the SimulatedAnnealingAdjudicator. Results are saved as a MATLAB .mat
file for O(1) lookup during MCTS rollouts.

Usage:
    python snowdrop_tangled_agents/tools/generate_terminal_lut.py

Output:
    snowdrop_tangled_agents/matlab/rl/data/terminal_scores.mat
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import savemat
from tqdm import tqdm

from snowdrop_adjudicators import SimulatedAnnealingAdjudicator
from snowdrop_tangled_game_engine.game import Edge


# Petersen graph structure
PETERSEN_EDGES = [
    (0, 2), (0, 3), (0, 6), (1, 3), (1, 4),
    (1, 7), (2, 4), (2, 8), (3, 9), (4, 5),
    (5, 6), (5, 9), (6, 7), (7, 8), (8, 9),
]

NUM_VERTICES = 10
NUM_EDGES = 15
MY_VERTEX = 5
OPP_VERTEX = 7


def idx_to_state(idx: int) -> str:
    """
    Convert a 0-based index to a 15-char state string.

    Index i: bit j = 1 means edge j is 'G' (green/FM), 0 means 'P' (purple/AFM)

    Args:
        idx: Integer in range [0, 32767]

    Returns:
        15-char string of 'G' and 'P'
    """
    state = []
    for j in range(NUM_EDGES):
        if (idx >> j) & 1:
            state.append('G')
        else:
            state.append('P')
    return ''.join(state)


def state_to_idx(state: str) -> int:
    """
    Convert a 15-char state string to a 0-based index.

    Args:
        state: 15-char string of 'G' and 'P'

    Returns:
        Integer in range [0, 32767]
    """
    idx = 0
    for j in range(NUM_EDGES):
        if state[j] == 'G':
            idx |= (1 << j)
    return idx


def evaluate_state(state: str, adj: SimulatedAnnealingAdjudicator) -> float:
    """
    Evaluate a terminal state using the SimulatedAnnealingAdjudicator.

    Args:
        state: 15-char string of 'G' and 'P'
        adj: Pre-initialized adjudicator instance

    Returns:
        Score from Player 1's perspective
    """
    # Build edge list for adjudicator
    edges = []
    for i, (v1, v2) in enumerate(PETERSEN_EDGES):
        color = state[i]
        edge_state = Edge.State.FM.value if color == 'G' else Edge.State.AFM.value
        edges.append((v1, v2, edge_state))

    # Create game state dict
    game_state = {
        'num_nodes': NUM_VERTICES,
        'edges': edges,
        'graph_id': 11,  # Petersen graph
        'player1_id': 'p1',
        'player2_id': 'p2',
        'turn_count': NUM_EDGES,
        'current_player_index': 2,
        'player1_node': MY_VERTEX,
        'player2_node': OPP_VERTEX
    }

    result = adj.adjudicate(game_state)
    return float(result['score'])


def generate_lut(output_path: Path, num_reads: int = 10000) -> np.ndarray:
    """
    Generate the full terminal state LUT.

    Args:
        output_path: Path to save the .mat file
        num_reads: Number of SA reads for more stable results

    Returns:
        Array of 32768 scores
    """
    total_states = 2 ** NUM_EDGES  # 32768
    scores = np.zeros(total_states, dtype=np.float32)

    # Initialize adjudicator once
    adj = SimulatedAnnealingAdjudicator()
    adj.setup(epsilon=0.0, num_reads=num_reads)

    print(f"Generating LUT for {total_states:,} terminal states...")
    print(f"Using SimulatedAnnealingAdjudicator with num_reads={num_reads}")

    start_time = time.time()

    for idx in tqdm(range(total_states), desc="Evaluating states"):
        state = idx_to_state(idx)
        scores[idx] = evaluate_state(state, adj)

    elapsed = time.time() - start_time
    print(f"Generation complete in {elapsed:.1f}s ({elapsed/total_states*1000:.2f}ms per state)")

    # Save as MATLAB .mat file
    # Note: MATLAB uses 1-based indexing, but we save with 0-based values
    # MATLAB will load and add 1 to indices when looking up
    mat_data = {
        'terminal_scores': scores,
        'num_states': total_states,
        'num_edges': NUM_EDGES,
        'generation_time_sec': elapsed,
        'num_reads': num_reads,
        'description': 'Terminal state scores for Petersen graph (graph 11). '
                       'Index i corresponds to state where bit j=1 means edge j is G (green/FM). '
                       'Scores are from Player 1 perspective.'
    }

    savemat(output_path, mat_data)
    print(f"Saved LUT to {output_path}")

    # Print some statistics
    print(f"\nStatistics:")
    print(f"  Min score: {scores.min():.3f}")
    print(f"  Max score: {scores.max():.3f}")
    print(f"  Mean score: {scores.mean():.3f}")
    print(f"  Std dev: {scores.std():.3f}")

    # Count win/loss/draw states
    wins = np.sum(scores > 0.5)
    losses = np.sum(scores < -0.5)
    draws = total_states - wins - losses
    print(f"\n  States favorable to P1 (score > 0.5): {wins:,} ({wins/total_states*100:.1f}%)")
    print(f"  States favorable to P2 (score < -0.5): {losses:,} ({losses/total_states*100:.1f}%)")
    print(f"  Balanced states: {draws:,} ({draws/total_states*100:.1f}%)")

    return scores


def main():
    """Main entry point."""
    # Determine output path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    output_path = project_root / "snowdrop_tangled_agents" / "matlab" / "rl" / "data" / "terminal_scores.mat"

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate LUT
    scores = generate_lut(output_path)

    # Verify a few states
    print("\nVerification (sample states):")
    test_indices = [0, 32767, 16384, 12345]
    adj = SimulatedAnnealingAdjudicator()
    adj.setup(epsilon=0.0)

    for idx in test_indices:
        state = idx_to_state(idx)
        lut_score = scores[idx]
        live_score = evaluate_state(state, adj)
        print(f"  idx={idx:5d} state={state} LUT={lut_score:+.3f} live={live_score:+.3f} diff={abs(lut_score-live_score):.3f}")


if __name__ == "__main__":
    main()
