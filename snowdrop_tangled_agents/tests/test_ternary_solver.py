"""Ternary solver checks against brute-force minimax on small subgames."""

import numpy as np
import pytest

from snowdrop_tangled_agents.strategy import ternary_model as tm
from snowdrop_tangled_agents.strategy.ternary_strategy import TOL, Solution


def brute(state: str, lut: np.ndarray, us: int) -> tuple:
    """(value, tiebreak) by plain recursion, same rules as Solution."""
    free = [e for e, c in enumerate(state) if c == '-']
    if not free:
        v = float(lut[tm.terminal_index(state)]) * (1 if us == 1 else -1)
        return v, v
    kids = [brute(state[:e] + c + state[e + 1:], lut, us) for e in free for c in 'ZGP']
    our_turn = ((tm.NUM_EDGES - len(free)) % 2 == 0) == (us == 1)
    if not our_turn:
        return min(v for v, _ in kids), sum(w for _, w in kids) / len(kids)
    best = max(v for v, _ in kids)
    return best, max(w for v, w in kids if v >= best - TOL)


def random_position(rng, free: int) -> str:
    board = [tm.COLORS[d] for d in rng.integers(0, 3, tm.NUM_EDGES)]
    for e in rng.choice(tm.NUM_EDGES, free, replace=False):
        board[e] = '-'
    return ''.join(board)


@pytest.fixture(scope="module")
def lut():
    return tm.build_lut(4.0) if not tm.lut_path(4.0).exists() else tm.load_lut(4.0)


@pytest.mark.parametrize("us", [1, 2])
@pytest.mark.parametrize("free", [1, 2, 3, 4])
def test_solution_matches_brute_force(lut, us, free):
    rng = np.random.default_rng(100 * us + free)
    for _ in range(4):
        state = random_position(rng, free)
        s = Solution(state, lut, us)
        v, w = brute(state, lut, us)
        assert abs(s.V[0] - v) < 1e-5 and abs(s.W[0] - w) < 1e-5, state


@pytest.mark.parametrize("us", [1, 2])
def test_layer_store_matches_solution(lut, us):
    from snowdrop_tangled_agents.strategy import line_planner as lp
    if not lp.layer_store_path(4.0, us).exists():
        pytest.skip("layer store not built (solve_ternary_game --player N)")
    oracle = lp.ValueOracle(4.0, us)
    rng = np.random.default_rng(us)
    for k in (5, 6):
        for _ in range(3):
            state = random_position(rng, tm.NUM_EDGES - k)
            s = Solution(state, lut, us)
            v, w = oracle.value(state)
            assert abs(s.V[0] - v) < 1e-5 and abs(s.W[0] - w) < 1e-4, state
