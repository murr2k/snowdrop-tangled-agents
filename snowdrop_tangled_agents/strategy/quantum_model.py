"""
Quantum-annealing terminal model for three-color Tangled on the Petersen graph.

The site adjudicates from a lookup table made on annealing hardware. On
terminals with degenerate ground states the hardware does not sample the
ground states uniformly, so the classical Boltzmann model (ternary_model) can
get the sign wrong: `ZGPGPGGGGGZGGPP` models at +0.572 and the table says
-1.2275. This module computes the classical emulation Snowdrop ships with
snowdrop-adjudicators (SchrodingerEquationAdjudicator): closed-system
Schrodinger evolution of

    H(s) = -Delta(s) sum_n sx_n + E(s) sum_(n<m) J_nm sz_n sz_m,   s = t / tf,

on the Advantage2 schedule shipped with that package (Delta = A/4, E = B/2 in
GHz, as the package scales them), tf = 40 ns for graph 5, starting in the
ground state of H(s_min), and scores the final state with the package's rule:
connected correlations C_nm, influence_n = sum_m C_nm, score = influence[5] -
influence[7].

The package diagonalizes H at every step (minutes per board). Here the state
is propagated with a second-order split operator (diagonal Ising phase, then
a product of single-qubit x rotations), batched over boards.

Only 53,568 coupling patterns are distinct up to gauge (flipping spin i flips
the sign of J on its edges and of every correlation involving i, and leaves
the transverse field alone), and 824 up to the graph's 120 automorphisms as
well. build_quantum_lut() evolves those 824 representatives and maps every
one of the 3^15 terminals onto them, so a full table takes one batched run.
"""

import itertools
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from snowdrop_tangled_agents.strategy import ternary_model as tm

logger = logging.getLogger(__name__)

N = tm.NUM_VERTICES
DIM = 1 << N
_BITS = ((np.arange(DIM)[:, None] >> np.arange(N)[None, :]) & 1)
Z = (1 - 2 * _BITS).astype(np.float64)                  # (1024, 10) spin value per basis state
ZZ = np.stack([Z[:, a] * Z[:, b] for a, b in tm.EDGES], axis=1)   # (1024, 15)


@lru_cache(maxsize=1)
def schedule() -> tuple:
    """(s, Delta(s), E(s)) in GHz on a 1001-point grid, scaled as the package does."""
    import os
    import snowdrop_adjudicators.schrodinger as pkg
    data = np.loadtxt(os.path.join(os.path.dirname(pkg.__file__), "advantage2.1.3.txt"))
    return data[:, 0], data[:, 1] / 2 * 0.5, data[:, 2] * 0.5


def _interp(s: float) -> tuple:
    grid, delta, big_e = schedule()
    return float(np.interp(s, grid, delta)), float(np.interp(s, grid, big_e))


def _x_rotation(psi: np.ndarray, c, sn) -> np.ndarray:
    """Apply prod_n (c + sn * sx_n) to psi (B, 1024); c, sn scalars (complex or real)."""
    B = psi.shape[0]
    for n in range(N):
        v = psi.reshape(B, DIM >> (n + 1), 2, 1 << n)
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        a2 = c * a + sn * b
        b2 = c * b + sn * a
        v[:, :, 0, :] = a2
        v[:, :, 1, :] = b2
    return psi


def ground_state(diag: np.ndarray, s: float, steps: int = 200) -> np.ndarray:
    """Ground state of H(s) for each board (imaginary-time split operator from |+...+>)."""
    delta, big_e = _interp(s)
    B = diag.shape[0]
    psi = np.full((B, DIM), 1 / np.sqrt(DIM), dtype=np.complex128)
    for i in range(steps):
        tau = 0.3 / max(delta, 1e-9) if i < steps // 2 else 0.03 / max(delta, 1e-9)
        half = np.exp(-0.5 * tau * big_e * diag)
        psi *= half
        _x_rotation(psi, np.cosh(tau * delta), np.sinh(tau * delta))
        psi *= half
        psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    return psi


def evolve(J: np.ndarray, tf: float = 40.0, s_min: float = 0.001, s_stop: float = 0.8,
           ds: float = 1e-4) -> np.ndarray:
    """Final z-basis probabilities (B, 1024) for coupling vectors J (B, 15).

    Past s ~ 0.75 the transverse field is below 1e-3 GHz and the populations
    are frozen, so the evolution stops at s_stop (the package runs to 0.999).
    """
    J = np.asarray(J, dtype=np.float64)
    diag = J @ ZZ.T                                      # (B, 1024) Ising energy per basis state (units of E)
    psi = ground_state(diag, s_min)
    w = 2 * np.pi * tf                                   # phase per unit s per GHz
    s = s_min
    while s < s_stop - 1e-12:
        h = min(ds, s_stop - s)
        delta, big_e = _interp(s + h / 2)
        half = np.exp(-0.5j * w * h * big_e * diag)
        psi *= half
        _x_rotation(psi, np.cos(w * h * delta), 1j * np.sin(w * h * delta))
        psi *= half
        s += h
    return np.abs(psi) ** 2


def correlations(prob: np.ndarray) -> np.ndarray:
    """Connected correlation matrices (B, 10, 10), zero diagonal, from z-basis probabilities."""
    m = prob @ Z                                         # (B, 10)
    zz = np.einsum('bk,kn,km->bnm', prob, Z, Z)
    C = zz - m[:, :, None] * m[:, None, :]
    C[:, np.arange(N), np.arange(N)] = 0.0
    return C


def score_from_correlations(C: np.ndarray) -> np.ndarray:
    infl = C.sum(axis=1)
    return infl[:, tm.P1_NODE] - infl[:, tm.P2_NODE]


def score_terminals(terminals: list, **kw) -> np.ndarray:
    """Direct model score of explicit boards (no symmetry reduction)."""
    J = np.stack([tm.couplings_of(t) for t in terminals])
    return score_from_correlations(correlations(evolve(J, **kw)))


# ---------------------------------------------------------------- symmetry reduction

@lru_cache(maxsize=1)
def automorphisms() -> list:
    """The 120 vertex permutations of the Petersen graph, as tuples p with p[v] = image of v."""
    adj = {frozenset(e) for e in tm.EDGES}
    out = []

    def extend(m: dict):
        v = len(m)
        if v == N:
            out.append(tuple(m[i] for i in range(N)))
            return
        for w in range(N):
            if w in m.values():
                continue
            if all((frozenset((m[u], w)) in adj) == (frozenset((u, v)) in adj) for u in m):
                m[v] = w
                extend(m)
                del m[v]

    extend({})
    return out


def _edge_perm(p: tuple) -> tuple:
    eidx = {frozenset(e): i for i, e in enumerate(tm.EDGES)}
    return tuple(eidx[frozenset((p[a], p[b]))] for a, b in tm.EDGES)


def _forest(mask: int) -> list:
    """Spanning-forest edges of the colored subgraph, in BFS order: (edge, parent_vertex, child_vertex)."""
    nbr = {v: [] for v in range(N)}
    for e, (a, b) in enumerate(tm.EDGES):
        if mask >> e & 1:
            nbr[a].append((e, b))
            nbr[b].append((e, a))
    seen, order = set(), []
    for root in range(N):
        if root in seen:
            continue
        seen.add(root)
        queue = [root]
        while queue:
            u = queue.pop(0)
            for e, w in nbr[u]:
                if w not in seen:
                    seen.add(w)
                    order.append((e, u, w))
                    queue.append(w)
    return order


@lru_cache(maxsize=1)
def _sx_sum() -> np.ndarray:
    X = np.zeros((DIM, DIM))
    idx = np.arange(DIM)
    for n in range(N):
        X[idx, idx ^ (1 << n)] += 1.0
    return X


def thermal_probs(J: np.ndarray, s_star: float = 0.5, beta: float = 40.0, batch: int = 8) -> np.ndarray:
    """Quantum Boltzmann model: z-basis probabilities of exp(-beta H(s_star)) / Z, beta in 1/GHz.

    The standard freeze-out picture of annealer output: the transverse field at
    s_star reweights degenerate ground states, beta sets the thermal excitations.
    """
    delta, big_e = _interp(s_star)
    diag = np.asarray(J, dtype=np.float64) @ ZZ.T
    X = _sx_sum()
    out = np.empty((len(diag), DIM))
    for i in range(0, len(diag), batch):
        d = diag[i:i + batch]
        H = -delta * X[None] + np.einsum('bk,kl->bkl', big_e * d, np.eye(DIM))
        w, V = np.linalg.eigh(H)
        b = np.exp(-beta * (w - w[:, :1]))
        out[i:i + batch] = np.einsum('bkn,bn->bk', V ** 2, b) / b.sum(axis=1, keepdims=True)
    return out


def model_probs(J: np.ndarray, kw: dict) -> np.ndarray:
    """Probabilities under the model named by kw: mode 'anneal' (tf, ds) or 'thermal' (s_star, beta)."""
    kw = dict(kw)
    mode = kw.pop('mode', 'anneal')
    return thermal_probs(J, **kw) if mode == 'thermal' else evolve(J, **kw)


def build_quantum_lut(tf: float = 40.0, ds: float = 1e-4, workers: int = 1, **kw) -> np.ndarray:
    """Model score of every terminal, indexed by ternary_model.terminal_index().

    kw mode='thermal' with s_star, beta builds the quantum Boltzmann table instead of the anneal one.
    """
    if kw.get('mode') == 'thermal':
        params = dict(kw)
    else:
        params = dict(tf=tf, ds=ds, **kw)
    eperms = [_edge_perm(p) for p in automorphisms()]
    vperms = automorphisms()
    full = (1 << tm.NUM_EDGES) - 1

    # Orbit representative of every colored-edge mask, and one automorphism mapping rep -> mask.
    rep_of = np.full(1 << tm.NUM_EDGES, -1, dtype=np.int64)
    map_of = np.zeros(1 << tm.NUM_EDGES, dtype=np.int64)
    reps = []
    for m in range(full + 1):
        if rep_of[m] >= 0:
            continue
        reps.append(m)
        for k, ep in enumerate(eperms):
            img = sum(1 << ep[i] for i in range(tm.NUM_EDGES) if m >> i & 1)
            if rep_of[img] < 0:
                rep_of[img] = m
                map_of[img] = k

    # Frustration classes of each representative: forest edges ferromagnetic (J=-1), non-tree edges free.
    base, jobs = {}, []
    for m in reps:
        forest = _forest(m)
        tree = {e for e, _, _ in forest}
        cotree = [e for e in range(tm.NUM_EDGES) if m >> e & 1 and e not in tree]
        base[m] = len(jobs)                               # class index = base + cotree signs as binary (+1 -> 1)
        for signs in itertools.product((-1.0, 1.0), repeat=len(cotree)):
            J = np.zeros(tm.NUM_EDGES)
            J[list(tree)] = -1.0
            J[cotree] = signs
            jobs.append(J)
    jobs = np.array(jobs)
    logger.info(f"quantum LUT: {len(reps)} mask orbits, {len(jobs)} classes to evolve (tf={tf} ns)")

    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        chunks = np.array_split(jobs, workers)
        with ProcessPoolExecutor(workers) as pool:
            probs = list(pool.map(model_probs, chunks, [params] * len(chunks)))
        C_rep = np.concatenate([correlations(p) for p in probs])
    else:
        C_rep = correlations(model_probs(jobs, params))

    # Map every terminal onto its class. A terminal's coupling on edge e of the
    # representative is J(ep[e]); a gauge g (spins) turns the forest ferromagnetic,
    # then C_term[p[i], p[j]] = g_i g_j C_class[i, j].
    lut = np.empty(tm.NUM_TERMINALS, dtype=np.float32)
    coupling_of_digit = np.array([tm.COUPLING[c] for c in tm.COLORS])
    pow3 = np.array([3 ** e for e in range(tm.NUM_EDGES)], dtype=np.int64)
    for mask in range(full + 1):
        rep, k = int(rep_of[mask]), int(map_of[mask])
        ep, vp = eperms[k], vperms[k]
        colored = [e for e in range(tm.NUM_EDGES) if mask >> e & 1]
        grey = [e for e in range(tm.NUM_EDGES) if not mask >> e & 1]
        signs = np.array(list(itertools.product((-1.0, 1.0), repeat=len(colored))))   # (M, |colored|) J on mask edges
        if len(colored) == 0:
            signs = np.zeros((1, 0))
        Jt = np.zeros((len(signs), tm.NUM_EDGES))
        Jt[:, colored] = signs
        Jr = Jt[:, list(ep)]                              # coupling on each representative edge
        forest = _forest(rep)
        tree = {e for e, _, _ in forest}
        cotree = [e for e in range(tm.NUM_EDGES) if rep >> e & 1 and e not in tree]
        g = np.ones((len(signs), N))
        for e, u, w in forest:                            # g_u g_w J = -1  =>  g_w = -g_u J
            g[:, w] = -g[:, u] * Jr[:, e]
        a = np.array([tm.EDGES[e][0] for e in cotree], dtype=int)
        b = np.array([tm.EDGES[e][1] for e in cotree], dtype=int)
        cls_signs = (g[:, a] * g[:, b] * Jr[:, cotree]) if cotree else np.zeros((len(signs), 0))
        place = 2 ** np.arange(len(cotree) - 1, -1, -1)
        idx = base[rep] + ((cls_signs > 0).astype(np.int64) @ place if cotree else 0)
        C = C_rep[idx] * g[:, :, None] * g[:, None, :]    # correlations in representative labels
        inv = np.argsort(vp)                              # representative vertex mapped to 5 and 7
        infl = C.sum(axis=1)
        score = infl[:, inv[tm.P1_NODE]] - infl[:, inv[tm.P2_NODE]]
        # terminal index: colored edges get digit 1 (G, J=-1) or 2 (P, J=+1), grey edges 0
        digits = np.zeros((len(signs), tm.NUM_EDGES), dtype=np.int64)
        digits[:, colored] = np.where(signs > 0, 2, 1)
        lut[digits @ pow3] = score
    return lut




def lut_name(tf: float = 40.0) -> str:
    return f"q{tf:g}"


def main():
    import argparse
    import time
    ap = argparse.ArgumentParser(description="Build the quantum-annealing terminal table")
    ap.add_argument("--tf", type=float, default=40.0, help="anneal time in ns (default %(default)s)")
    ap.add_argument("--ds", type=float, default=1e-4, help="split-operator step in s (default %(default)s)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--check", type=int, default=24, help="verify N random terminals against direct evolution")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    t = time.time()
    lut = build_quantum_lut(tf=args.tf, ds=args.ds, workers=args.workers)
    name = lut_name(args.tf)
    path = tm.lut_path(name)
    np.save(path, lut)
    logger.info(f"saved {path} in {time.time() - t:.0f} s; draws (|score| <= eps): "
                f"{np.mean(np.abs(lut) <= tm.DRAW_EPSILON):.2%}")
    if args.check:
        rng = np.random.default_rng(0)
        boards = [''.join(tm.COLORS[d] for d in rng.integers(0, 3, tm.NUM_EDGES)) for _ in range(args.check)]
        direct = score_terminals(boards, tf=args.tf, ds=args.ds)
        table = np.array([lut[tm.terminal_index(b)] for b in boards])
        logger.info(f"symmetry-mapped vs direct, max |diff| over {args.check} boards: "
                    f"{np.max(np.abs(direct - table)):.2e}")


if __name__ == "__main__":
    main()
