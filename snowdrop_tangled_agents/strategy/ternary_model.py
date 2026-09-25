"""
Three-color terminal model for the Petersen graph.

tangled-game.com lets a player color an edge grey (zero coupling), green
(ferromagnetic) or purple (antiferromagnetic). Board strings here use
'-' uncolored, 'Z' grey, 'G' green, 'P' purple; the site's edge labels are
grey=1, green=2, purple=3.

The site adjudicates terminal states from a precomputed D-Wave lookup table
that is not available offline. This module approximates it by exact
enumeration of every spin configuration, averaging the adjudicator's influence
score over a Boltzmann distribution. Couplings, energy and score follow
snowdrop-adjudicators: J = 0 (grey), -1 (green), +1 (purple);
E = sum J_ij s_i s_j; score = influence[p1] - influence[p2] with
influence_v = sum_{j != v} <s_v s_j>.

beta=None is the zero-temperature limit: a uniform average over the
degenerate ground states. It declares 41% of terminals exact draws (the site
reports ~14% for this graph), because symmetric ground-state manifolds cancel
exactly where the hardware's non-uniform sampling does not. A finite beta
breaks those ties toward high-entropy states while keeping every decisive
ground-state outcome; beta=4 gives a 12.7% draw rate and is the default until
enough captured (board, lookup-table score) pairs exist to fit it properly.

All 3^15 terminal scores fit in a 57 MB float32 table indexed by
terminal_index(); build_lut() computes it in under a minute and load_lut()
caches it under ~/.tangled/.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

EDGES = ((0, 2), (0, 3), (0, 6), (1, 3), (1, 4), (1, 7), (2, 4), (2, 8),
         (3, 9), (4, 5), (5, 6), (5, 9), (6, 7), (7, 8), (8, 9))
NUM_EDGES = 15
NUM_VERTICES = 10
P1_NODE = 5
P2_NODE = 7

COLORS = 'ZGP'                       # terminal digit order: Z=0, G=1, P=2
COUPLING = {'Z': 0.0, 'G': -1.0, 'P': 1.0}
NUM_TERMINALS = 3 ** NUM_EDGES       # 14,348,907
DRAW_EPSILON = 0.0005                # site draw band for the Petersen graph
DEFAULT_BETA = 4.0

LUT_DIR = Path.home() / ".tangled"

_POW3 = np.array([3 ** e for e in range(NUM_EDGES)], dtype=np.int64)


def _spin_tables() -> tuple:
    """Edge products and score weights for the 512 configurations with s_0 = +1.

    With no local fields the energy and score are invariant under a global
    spin flip, so half the 1024 configurations suffice.
    """
    configs = np.array([[1] + [1 - 2 * ((k >> b) & 1) for b in range(NUM_VERTICES - 1)]
                        for k in range(2 ** (NUM_VERTICES - 1))], dtype=np.float32)
    products = np.stack([configs[:, i] * configs[:, j] for i, j in EDGES], axis=1)
    # influence[p1] - influence[p2] = <(s_p1 - s_p2) * sum_j s_j>; the
    # self-correlation terms cancel.
    weights = (configs[:, P1_NODE] - configs[:, P2_NODE]) * configs.sum(axis=1)
    return products, weights.astype(np.float32)


_PRODUCTS, _WEIGHTS = _spin_tables()


def score_couplings(couplings: np.ndarray, beta: Optional[float] = DEFAULT_BETA) -> np.ndarray:
    """Model score for a batch of coupling vectors, shape (N, 15) -> (N,).

    beta=None averages uniformly over the ground states; a finite beta uses
    Boltzmann weights exp(-beta * (E - E_min)).
    """
    energy = np.asarray(couplings, dtype=np.float32) @ _PRODUCTS.T
    excess = energy - energy.min(axis=1, keepdims=True)
    weight = (excess < 1e-3).astype(np.float32) if beta is None else np.exp(-beta * excess)
    return (weight @ _WEIGHTS) / weight.sum(axis=1)


def couplings_of(state: str) -> np.ndarray:
    """Coupling vector for a board; uncolored edges count as zero coupling."""
    return np.array([COUPLING.get(c, 0.0) for c in state], dtype=np.float32)


def score_state(state: str, beta: Optional[float] = DEFAULT_BETA) -> float:
    """Model score of one board (uncolored edges treated as grey)."""
    return float(score_couplings(couplings_of(state)[None, :], beta)[0])


def terminal_index(state: str) -> int:
    """Index of a fully colored board in the terminal table."""
    return sum(COLORS.index(c) * 3 ** e for e, c in enumerate(state))


def build_lut(beta: Optional[float] = DEFAULT_BETA, chunk: int = 250_000) -> np.ndarray:
    """Model score of every terminal board, indexed by terminal_index()."""
    coupling_of_digit = np.array([COUPLING[c] for c in COLORS], dtype=np.float32)
    lut = np.empty(NUM_TERMINALS, dtype=np.float32)
    for start in range(0, NUM_TERMINALS, chunk):
        idx = np.arange(start, min(start + chunk, NUM_TERMINALS), dtype=np.int64)
        digits = (idx[:, None] // _POW3[None, :]) % 3
        lut[start:start + len(idx)] = score_couplings(coupling_of_digit[digits], beta)
    return lut


def lut_path(beta: Optional[float] = DEFAULT_BETA) -> Path:
    return LUT_DIR / ("ternary_gs_lut.npy" if beta is None else f"ternary_b{beta:g}_lut.npy")


_LUTS: dict = {}


def load_lut(beta: Optional[float] = DEFAULT_BETA) -> np.ndarray:
    """Terminal table for this beta, built and cached on first use."""
    if beta not in _LUTS:
        path = lut_path(beta)
        if path.exists():
            _LUTS[beta] = np.load(path)
        else:
            logger.info(f"Building ternary terminal table (beta={beta}, one-time, <1 min)...")
            _LUTS[beta] = build_lut(beta)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, _LUTS[beta])
            logger.info(f"Saved {path} ({_LUTS[beta].nbytes / 1e6:.0f} MB)")
    return _LUTS[beta]
