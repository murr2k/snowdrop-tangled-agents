"""Phase 5A.3 — eigsh independent reference.

For a sample of terminal boards, compute the EXACT GROUND STATE of the
quantum-annealing Hamiltonian H(s_max) — bypassing time evolution entirely
— and compute the resulting score via the same correlation-matrix recipe
the Schrödinger adjudicator uses.

This distinguishes three failure modes:

(1) Ground state at s_max matches calib → time evolution converged; the
    calib oracle is essentially the adiabatic-limit result. Anneal-time
    tuning won't help; the model itself is right or wrong, not the dynamics.

(2) Ground state at s_max disagrees with calib → calib's time evolution is
    NOT converged. The calib wavefunction is far from the s_max ground state
    (consistent with the very short 1.85 ns anneal time). In that case the
    Phase 5A.2 sweep should test LONGER anneal times to see if a slower
    anneal closes the website gap.

(3) Ground state at s_max matches the website better than calib →
    confirms (2) AND points the parameter search at the adiabatic regime.
    Ground state at s_max matches calib but neither matches the website →
    points away from the Schrödinger TFIM model entirely.

Usage:
    poetry run python scripts/calib_eigsh_reference.py [--n 50]
"""

from __future__ import annotations

import argparse
import io
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from snowdrop_tangled_game_engine import GraphProperties
from snowdrop_adjudicators.schrodinger.sparse_matrices import (
    create_pauli_matrices_for_full_size_hamiltonian,
    create_sparse_hamiltonian,
    load_schedule_data,
)

DB = Path.home() / ".tangled" / "game_stats.db"
LUT = Path("snowdrop_tangled_agents/matlab/rl/data/terminal_scores.mat")

# Match the adjudicator defaults
SCHEDULE_REDUCTION = 0.5  # see schrodinger_functions.py line 80

# Multiple s values: at s_max=0.999 the schedule sets delta=0 (pure classical
# Ising, massively degenerate ground state, score becomes arbitrary). Smaller
# s values keep some transverse field, lifting the degeneracy.
S_VALUES = [0.7, 0.8, 0.9, 0.95, 0.99, 0.999]


@dataclass
class Stats:
    n: int
    r2: float
    rmse: float
    mae: float
    bias: float

    def fmt(self) -> str:
        return (
            f"n={self.n:4d}  R²={self.r2:+.4f}  RMSE={self.rmse:.4f}  "
            f"MAE={self.mae:.4f}  bias={self.bias:+.4f}"
        )


def compute_stats(target: np.ndarray, predicted: np.ndarray) -> Stats:
    if len(target) == 0 or target.std() == 0:
        return Stats(len(target), float("nan"), float("nan"), float("nan"), float("nan"))
    r = target - predicted
    ss_res = float(np.sum(r**2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    return Stats(
        n=len(target),
        r2=1.0 - ss_res / ss_tot,
        rmse=float(np.sqrt(ss_res / len(target))),
        mae=float(np.mean(np.abs(r))),
        bias=float(np.mean(r)),
    )


def board_to_index(board: str) -> int:
    return sum(1 << j for j, c in enumerate(board) if c == "G")


def board_to_ising(board: str, edges: list[tuple[int, int]], num_nodes: int) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Translate a 15-char G/P board into Ising (h, J) per adjudicator convention:
       G = FM = J=-1, P = AFM = J=+1, no local fields."""
    h: dict[int, float] = {i: 0.0 for i in range(num_nodes)}
    j: dict[tuple[int, int], float] = {}
    for e_idx, (v1, v2) in enumerate(edges):
        v1, v2 = (v1, v2) if v1 < v2 else (v2, v1)
        c = board[e_idx]
        if c == "G":
            j[(v1, v2)] = -1.0
        elif c == "P":
            j[(v1, v2)] = +1.0
        else:
            raise ValueError(f"bad char {c!r} in board {board!r}")
    return h, j


def ground_state_score_and_gap(
    board: str,
    edges: list[tuple[int, int]],
    num_nodes: int,
    player1_node: int,
    player2_node: int,
    sx: dict,
    sz: dict,
    big_e_s: float,
    delta_s: float,
) -> tuple[float, float]:
    """Build H(s), find ground state via eigh, compute score and the gap to
       the first excited state (small gap → ground state is near-degenerate
       and the choice of eigenvector is arbitrary)."""
    h, jay = board_to_ising(board, edges, num_nodes)
    H = create_sparse_hamiltonian(num_nodes, h, jay, big_e_s, delta_s, sx, sz)
    H_dense = H.toarray()
    evals, evecs = np.linalg.eigh(H_dense)
    psi = evecs[:, 0]
    gap = float(evals[1] - evals[0])

    sz_exp = np.array([
        float(np.real(np.vdot(psi, sz[i].dot(psi))))
        for i in range(num_nodes)
    ])
    C = np.zeros((num_nodes, num_nodes))
    for n in range(num_nodes - 1):
        for m in range(n + 1, num_nodes):
            corr_nm = float(np.real(np.vdot(psi, sz[m].dot(sz[n].dot(psi)))))
            C[n, m] = corr_nm - sz_exp[m] * sz_exp[n]
    C = C + C.T
    influence = np.sum(C, axis=0)
    return float(influence[player1_node] - influence[player2_node]), gap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of boards to sample")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 78)
    print("Phase 5A.3 — eigsh independent reference vs calib vs website")
    print("=" * 78)

    g = GraphProperties().graph_database[5]
    edges = list(g["edge_list"])
    num_nodes = g["num_nodes"]
    p1 = g["player1_node"]
    p2 = g["player2_node"]
    num_edges = len(edges)
    print(f"\nGraph 5 (Petersen): {num_nodes} nodes, {num_edges} edges, P1={p1}, P2={p2}")

    print("\nBuilding Pauli operators (one-time)...")
    sx, sy, sz = create_pauli_matrices_for_full_size_hamiltonian(n_qubits=num_nodes)
    print(f"  built {len(sx)} σx, {len(sz)} σz operators on {2**num_nodes}-dim Hilbert space")

    print("\nLoading annealing schedule...")
    delta_qubit, big_e_qubit = load_schedule_data()
    delta_qubit = delta_qubit * SCHEDULE_REDUCTION
    big_e_qubit = big_e_qubit * SCHEDULE_REDUCTION
    print(f"  schedule has {len(delta_qubit)} samples")
    print(f"  s value:        delta(s)        big_e(s)        ratio")
    s_params: list[tuple[float, float, float]] = []  # (s, big_e, delta)
    for s_val in S_VALUES:
        s_idx = int(round(s_val * (len(delta_qubit) - 1)))
        delta_s = float(delta_qubit[s_idx])
        big_e_s = float(big_e_qubit[s_idx])
        ratio = delta_s / big_e_s if big_e_s > 0 else float("inf")
        s_params.append((s_val, big_e_s, delta_s))
        print(f"  s={s_val:<7.3f}    {delta_s:>10.6f}    {big_e_s:>10.6f}    {ratio:.6f}")

    print(f"\nLoading calib LUT from {LUT.name}...")
    with h5py.File(LUT, "r") as f:
        calib_lut = f["terminal_scores"][:].flatten().astype(np.float64)
    print(f"  {calib_lut.size} entries")

    print(f"\nSampling {args.n} distinct boards from calibration table (seed={args.seed})...")
    conn = sqlite3.connect(str(DB))
    rows = conn.execute(
        "SELECT terminal_state, AVG(website_score), COUNT(*) "
        "FROM calibration GROUP BY terminal_state"
    ).fetchall()
    conn.close()
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(rows), size=min(args.n, len(rows)), replace=False)
    sample = [rows[i] for i in idx]

    print(f"\nComputing ground-state scores at {len(s_params)} s-values × {len(sample)} boards...")
    boards: list[str] = []
    website: list[float] = []
    calib: list[float] = []
    eigsh_scores: list[list[float]] = [[] for _ in s_params]  # parallel per s
    eigsh_gaps: list[list[float]] = [[] for _ in s_params]
    t0 = time.time()
    for k, (state, ws, cnt) in enumerate(sample):
        try:
            board_idx = board_to_index(state)
        except (ValueError, TypeError):
            continue
        boards.append(state)
        website.append(float(ws))
        calib.append(float(calib_lut[board_idx]))
        for si, (s_val, big_e_s, delta_s) in enumerate(s_params):
            sc, gap = ground_state_score_and_gap(state, edges, num_nodes, p1, p2, sx, sz, big_e_s, delta_s)
            eigsh_scores[si].append(sc)
            eigsh_gaps[si].append(gap)
        if (k + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {k+1}/{len(sample)} boards  ({elapsed:.1f}s)")
    print(f"  done in {time.time()-t0:.1f}s")

    website_arr = np.array(website)
    calib_arr = np.array(calib)

    print("\n--- Fit to website, by s-value (ground state at that s) ---")
    print(f"  {'s':<7s}  {'big_e':>9s}  {'delta':>9s}  {'mean gap':>10s}  comparison")
    for si, (s_val, big_e_s, delta_s) in enumerate(s_params):
        eig_arr = np.array(eigsh_scores[si])
        mean_gap = float(np.mean(eigsh_gaps[si]))
        st = compute_stats(website_arr, eig_arr)
        print(f"  s={s_val:<5.3f}  {big_e_s:>9.4f}  {delta_s:>9.4f}  {mean_gap:>10.4f}  {st.fmt()}")
    print(f"\n  reference:  calib (anneal=1.85ns) vs website:  {compute_stats(website_arr, calib_arr).fmt()}")

    print("\n--- Score variance by s-value (does the ground state actually vary across boards?) ---")
    print(f"  website variance:  {website_arr.var():.4f}, std: {website_arr.std():.4f}")
    print(f"  calib   variance:  {calib_arr.var():.4f}, std: {calib_arr.std():.4f}")
    for si, (s_val, _, _) in enumerate(s_params):
        eig_arr = np.array(eigsh_scores[si])
        print(f"  s={s_val:<5.3f}  variance: {eig_arr.var():.4f}, std: {eig_arr.std():.4f}, "
              f"unique values: {len(np.unique(np.round(eig_arr, 4)))}/{len(eig_arr)}")

    # Find best-fitting s
    best_si = max(range(len(s_params)), key=lambda i: compute_stats(website_arr, np.array(eigsh_scores[i])).r2)
    best_s = s_params[best_si][0]
    best_r2 = compute_stats(website_arr, np.array(eigsh_scores[best_si])).r2
    calib_r2 = compute_stats(website_arr, calib_arr).r2
    print(f"\n--- Best ground-state s: {best_s} (R² = {best_r2:+.4f}) ---")
    print(f"--- Calib (1.85 ns evolution) R² on same sample: {calib_r2:+.4f} ---")
    if best_r2 > calib_r2:
        print(f"  → Adiabatic ground state at s={best_s} fits BETTER than calib's 1.85 ns evolution.")
        print(f"    Phase 5A.2 should test LONGER anneal times to approach the adiabatic limit.")
    else:
        print(f"  → Calib's short-time evolution fits BETTER than any adiabatic ground state.")
        print(f"    The website's behaviour is not adiabatic; either uses a very short anneal,")
        print(f"    or a non-Schrödinger model. Phase 5A.2 may need to revisit the model itself.")


if __name__ == "__main__":
    main()
