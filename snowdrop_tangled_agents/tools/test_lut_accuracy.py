"""
Test LUT Accuracy Against Live Adjudicator.

Validates that the pre-computed terminal state LUT matches the
SimulatedAnnealingAdjudicator within acceptable error bounds.

Usage:
    python snowdrop_tangled_agents/tools/test_lut_accuracy.py
"""

import random
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

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
    """Convert index to state string."""
    state = []
    for j in range(NUM_EDGES):
        if (idx >> j) & 1:
            state.append('G')
        else:
            state.append('P')
    return ''.join(state)


def state_to_idx(state: str) -> int:
    """Convert state string to index."""
    idx = 0
    for j in range(NUM_EDGES):
        if state[j] == 'G':
            idx |= (1 << j)
    return idx


def evaluate_state_live(state: str, adj: SimulatedAnnealingAdjudicator) -> float:
    """Evaluate state using live adjudicator."""
    edges = []
    for i, (v1, v2) in enumerate(PETERSEN_EDGES):
        color = state[i]
        edge_state = Edge.State.FM.value if color == 'G' else Edge.State.AFM.value
        edges.append((v1, v2, edge_state))

    game_state = {
        'num_nodes': NUM_VERTICES,
        'edges': edges,
        'graph_id': 11,
        'player1_id': 'p1',
        'player2_id': 'p2',
        'turn_count': NUM_EDGES,
        'current_player_index': 2,
        'player1_node': MY_VERTEX,
        'player2_node': OPP_VERTEX
    }

    result = adj.adjudicate(game_state)
    return float(result['score'])


def test_lut_accuracy(lut_path: Path, num_samples: int = 1000, tolerance: float = 0.5):
    """
    Test LUT accuracy against live adjudicator.

    Args:
        lut_path: Path to the terminal_scores.mat file
        num_samples: Number of random states to test
        tolerance: Maximum acceptable mean absolute error

    Returns:
        True if all tests pass
    """
    print(f"Loading LUT from {lut_path}...")
    data = loadmat(str(lut_path))
    lut = data['terminal_scores'].flatten()

    if len(lut) != 32768:
        print(f"ERROR: LUT has {len(lut)} entries, expected 32768")
        return False

    print(f"LUT loaded: {len(lut)} entries")
    print(f"  Min: {lut.min():.3f}, Max: {lut.max():.3f}, Mean: {lut.mean():.3f}")

    # Initialize adjudicator
    adj = SimulatedAnnealingAdjudicator()
    adj.setup(epsilon=0.0)

    # Test random samples
    print(f"\nTesting {num_samples} random states against live adjudicator...")
    errors = []
    test_indices = random.sample(range(32768), num_samples)

    for i, idx in enumerate(test_indices):
        state = idx_to_state(idx)
        lut_score = lut[idx]
        live_score = evaluate_state_live(state, adj)
        error = abs(lut_score - live_score)
        errors.append(error)

        if (i + 1) % 100 == 0:
            print(f"  Tested {i + 1}/{num_samples}...")

    errors = np.array(errors)
    mae = errors.mean()
    max_error = errors.max()
    within_tolerance = np.sum(errors <= tolerance) / len(errors) * 100

    print(f"\nResults:")
    print(f"  Mean Absolute Error: {mae:.4f}")
    print(f"  Max Error: {max_error:.4f}")
    print(f"  % within tolerance ({tolerance}): {within_tolerance:.1f}%")

    # Test specific edge cases
    print("\nEdge case tests:")
    edge_cases = [
        ('P' * 15, "All Purple"),
        ('G' * 15, "All Green"),
        ('G' * 7 + 'P' * 8, "Balanced"),
        ('GPGPGPGPGPGPGPG', "Alternating"),
    ]

    for state, name in edge_cases:
        idx = state_to_idx(state)
        lut_score = lut[idx]
        live_score = evaluate_state_live(state, adj)
        error = abs(lut_score - live_score)
        status = "OK" if error <= tolerance else "WARN"
        print(f"  {name}: LUT={lut_score:+.3f}, live={live_score:+.3f}, err={error:.3f} [{status}]")

    # Test index round-trip
    print("\nIndex conversion tests:")
    all_pass = True
    for idx in [0, 1, 32767, 16384, 12345]:
        state = idx_to_state(idx)
        roundtrip_idx = state_to_idx(state)
        if roundtrip_idx != idx:
            print(f"  FAIL: idx={idx}, roundtrip={roundtrip_idx}")
            all_pass = False
        else:
            print(f"  OK: idx={idx} -> state={state} -> idx={roundtrip_idx}")

    # Overall result
    success = mae < tolerance and all_pass
    print(f"\n{'PASS' if success else 'FAIL'}: MAE={mae:.4f}, tolerance={tolerance}")
    return success


def main():
    """Main entry point."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    lut_path = project_root / "snowdrop_tangled_agents" / "matlab" / "rl" / "data" / "terminal_scores.mat"

    if not lut_path.exists():
        print(f"ERROR: LUT file not found at {lut_path}")
        print("Run generate_terminal_lut.py first to create the LUT.")
        sys.exit(1)

    success = test_lut_accuracy(lut_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
