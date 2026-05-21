"""
Pure Python switchback move selection for the Petersen graph.

Implements the algorithm documented in docs/SWITCHBACK_ALGORITHM.md.
No MATLAB dependency.
"""

import math
from typing import Optional, Tuple

# ── Graph constants ──────────────────────────────────────────────────────────

# 12 five-cycles of the Petersen graph (0-indexed edge indices).
# Verified against networkx enumeration of GraphProperties().graph_database[5].
_CYCLES: tuple[tuple[int, ...], ...] = (
    (0,  1,  3,  4,  6),
    (0,  1,  7,  8, 14),
    (0,  2,  6,  9, 10),
    (0,  2,  7, 12, 13),
    (1,  2,  3,  5, 12),
    (1,  2,  8, 10, 11),
    (3,  4,  8,  9, 11),
    (3,  5,  8, 13, 14),
    (4,  5,  6,  7, 13),
    (4,  5,  9, 10, 12),
    (6,  7,  9, 11, 14),
    (10, 11, 12, 13, 14),
)

# Edge-orbit mapping under Stab(p1=5, p2=7) ≤ Aut(Petersen).
# _ORBIT[e] = orbit id (0-indexed) for edge e.
# 9 orbits: 6 swap-pairs + 3 fixed points.
_ORBIT: tuple[int, ...] = (
    0, 0,   # edges 0,1  — orbit 0 (swap pair)
    1,      # edge  2    — orbit 1 (fixed)
    2, 3,   # edges 3,4  — orbits 2,3 (each starts a pair)
    4,      # edge  5    — orbit 4
    5,      # edge  6    — orbit 5
    2,      # edge  7    — orbit 2 (pairs with edge 3)
    5,      # edge  8    — orbit 5 (pairs with edge 6)
    6,      # edge  9    — orbit 6
    7,      # edge  10   — orbit 7 (fixed)
    6,      # edge  11   — orbit 6 (pairs with edge 9)
    8,      # edge  12   — orbit 8 (fixed)
    4,      # edge  13   — orbit 4 (pairs with edge 5)
    3,      # edge  14   — orbit 3 (pairs with edge 4)
)

_NUM_ORBITS = 9
_WEIGHTS = (1.0, -2.0, 0.3, 1.0)  # w1 sat, w2 frust, w3 undec, w4 balance


# ── Feature functions ────────────────────────────────────────────────────────

def _cycle_features(state: str) -> tuple[int, int, int]:
    """Return (satisfied_count, frustrated_count, undecided_count)."""
    sat = frust = undec = 0
    for cycle in _CYCLES:
        has_grey = False
        p_count = 0
        for e in cycle:
            c = state[e]
            if c == '-':
                has_grey = True
                break
            if c == 'P':
                p_count += 1
        if has_grey:
            undec += 1
        elif p_count % 2 == 0:
            sat += 1
        else:
            frust += 1
    return sat, frust, undec


def _orbit_balance(state: str) -> float:
    """
    Penalise uneven coloring coverage and G/P imbalance across the 9 orbits.

    Returns -(std of coverage + std of |net|), so higher = more balanced.
    """
    cov = [0] * _NUM_ORBITS   # colored edges per orbit
    net = [0] * _NUM_ORBITS   # (G count - P count) per orbit

    for e, c in enumerate(state):
        if c == 'G':
            cov[_ORBIT[e]] += 1
            net[_ORBIT[e]] += 1
        elif c == 'P':
            cov[_ORBIT[e]] += 1
            net[_ORBIT[e]] -= 1

    abs_net = [abs(n) for n in net]
    mean_cov = sum(cov) / _NUM_ORBITS
    mean_net = sum(abs_net) / _NUM_ORBITS
    std_cov = math.sqrt(sum((x - mean_cov) ** 2 for x in cov) / _NUM_ORBITS)
    std_net = math.sqrt(sum((x - mean_net) ** 2 for x in abs_net) / _NUM_ORBITS)
    return -(std_cov + std_net)


def switchback_score(state: str) -> float:
    """
    Scalar score for a game state under the switchback criterion.

    S(σ) = +1.0·sat(σ)  −2.0·frust(σ)  +0.3·undec(σ)  +1.0·balance(σ)
    """
    sat, frust, undec = _cycle_features(state)
    balance = _orbit_balance(state)
    w1, w2, w3, w4 = _WEIGHTS
    return w1 * sat + w2 * frust + w3 * undec + w4 * balance


# ── Move selection ───────────────────────────────────────────────────────────

def best_move(state: str) -> Optional[Tuple[int, str]]:
    """
    Greedy 1-step lookahead under switchback_score.

    Returns (edge_index, color) where color is 'G' or 'P',
    or None if the board is fully colored.

    Tie-breaking: lowest edge index, 'G' before 'P'.
    """
    best_score = -math.inf
    best_edge: Optional[int] = None
    best_color: Optional[str] = None

    state_list = list(state)
    for e, c in enumerate(state):
        if c != '-':
            continue
        for color in ('G', 'P'):
            state_list[e] = color
            score = switchback_score(''.join(state_list))
            if score > best_score:
                best_score = score
                best_edge = e
                best_color = color
        state_list[e] = '-'

    if best_edge is None:
        return None
    return best_edge, best_color


# ── Strategy wrapper (matches HybridSolverStrategy interface) ────────────────

class SwitchbackStrategy:
    """
    Drop-in strategy using pure Python switchback move selection.

    Implements the same initialize / calculate_move / record_move /
    end_game / get_stats interface as HybridSolverStrategy so
    play_tangled.py needs no special casing beyond construction.
    """

    def __init__(self, player: int = 1, move_overrides: Optional[dict] = None):
        self.player = player
        # move_overrides: {grey_count: (edge_index, color)} — same format as
        # HybridSolverStrategy's move_overrides / oracle_overrides.
        # Used to force a specific opening (e.g. {15: (7, 'G')} for E7G).
        self._move_overrides: dict = move_overrides or {}
        self.moves_calculated = 0
        self.total_time = 0.0
        self.last_score = 0.0
        self.last_strategy = 'switchback'

    def initialize(self, opponent: str = '') -> bool:
        return True

    def calculate_move(
        self,
        state: str,
        score: float = 0.0,
        score_history: list = None,
    ) -> Optional[Tuple[int, str, dict]]:
        import time
        start = time.time()

        grey_count = state.count('-')
        if grey_count in self._move_overrides:
            edge, color = self._move_overrides[grey_count]
            elapsed = time.time() - start
            self.moves_calculated += 1
            self.total_time += elapsed
            stats = {
                'strategy': 'switchback/override',
                'score': switchback_score(state),
                'num_grey': grey_count,
                'time': elapsed,
            }
            return edge, color, stats

        result = best_move(state)
        if result is None:
            return None

        edge, color = result
        elapsed = time.time() - start
        self.moves_calculated += 1
        self.total_time += elapsed
        self.last_score = switchback_score(state)

        stats = {
            'strategy': 'switchback',
            'score': self.last_score,
            'num_grey': state.count('-'),
            'time': elapsed,
        }
        return edge, color, stats

    def record_move(self, edge: int, color: str, score_after: float) -> None:
        pass

    def end_game(self, result: str, final_score: float) -> None:
        pass

    def get_stats(self) -> dict:
        avg = self.total_time / self.moves_calculated if self.moves_calculated else 0.0
        return {
            'moves_calculated': self.moves_calculated,
            'avg_move_time': avg,
            'backend': 'python',
        }
