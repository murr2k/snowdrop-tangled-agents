#!/usr/bin/env python3
"""
calibrate_adjudicator.py

Fits Schrodinger adjudicator parameters (anneal_time, epsilon) to match
the website's adjudicator scores on observed terminal boards.

ARCHITECTURE NOTE
-----------------
The Python SchrodingerEquationAdjudicator solves a 1024x1024 eigenvalue
problem ~2000 times per board evaluation (~2-5 min/board on 10 qubits).
Calibrating over 1000+ boards is infeasible in pure Python.

This script takes a two-track approach:

  Track A (MATLAB, fast):   Export calibration boards to .mat, then run
                             MATLAB scripts that use the split-operator
                             Schrodinger solver (~0.7 s/board). See
                             --export-mat and snowdrop_tangled_agents/matlab/rl/
                             calibrate_schrodinger.m

  Track B (Python, slow):   Direct Python evaluation. Useful for small
                             samples (<20 boards) or when MATLAB is unavailable.
                             Enable with --python-eval.

  Track C (SA proxy):       Linear regression on SA terminal values vs
                             website scores. No new evaluations needed.
                             Fast sanity check. Enable with --sa-proxy.

USAGE
-----
    # Export calibration data for MATLAB processing (recommended first step)
    python scripts/calibrate_adjudicator.py --export-mat

    # Then run in MATLAB:
    #   cd snowdrop_tangled_agents/matlab/rl
    #   calibrate_schrodinger('../../data/calibration_boards.mat')

    # SA proxy calibration (instant baseline)
    python scripts/calibrate_adjudicator.py --sa-proxy

    # Python evaluation (slow, for small sample only)
    python scripts/calibrate_adjudicator.py --python-eval --sample 10 --workers 4

    # Load MATLAB calibration results and fit epsilon
    python scripts/calibrate_adjudicator.py --load-matlab-results data/matlab_calib_results.mat
"""

import argparse
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "snowdrop_tangled_agents" / "matlab" / "rl" / "data"
DEFAULT_DB = Path.home() / ".tangled" / "game_stats.db"

GRAPH_ID = 5  # Petersen graph


# ---------------------------------------------------------------------------
# DB access
# ---------------------------------------------------------------------------
def load_calibration_data(db_path: Path) -> tuple[list[str], np.ndarray]:
    """Return deduplicated (states, mean_website_scores) from calibration table."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(
        "SELECT terminal_state, website_score FROM calibration "
        "WHERE terminal_state IS NOT NULL AND website_score IS NOT NULL"
    )
    rows = cur.fetchall()
    conn.close()

    state_scores: dict[str, list[float]] = defaultdict(list)
    for state, ws in rows:
        state_scores[state].append(ws)

    states = sorted(state_scores.keys())
    website_scores = np.array([np.mean(state_scores[s]) for s in states], dtype=np.float64)
    return states, website_scores


def state_str_to_base_idx(state: str) -> int:
    """Convert 15-char G/P state to 0-indexed base integer (bit j=1 if G)."""
    idx = 0
    for j, c in enumerate(state):
        if c == 'G':
            idx |= (1 << j)
    return idx


# ---------------------------------------------------------------------------
# Track A: Export calibration data for MATLAB processing
# ---------------------------------------------------------------------------
def export_for_matlab(states: list[str], website_scores: np.ndarray,
                      output_path: Path) -> None:
    """Export calibration boards + website scores to a .mat file for MATLAB processing."""

    # Encode board states as 15-bit integers (bit j=1 if G, 0 if P)
    board_indices = np.array([state_str_to_base_idx(s) for s in states], dtype=np.int32)

    # Also export as character matrix for MATLAB convenience
    state_matrix = np.array([[ord(c) for c in s] for s in states], dtype=np.uint8)

    scipy.io.savemat(str(output_path), {
        'board_indices':  board_indices,      # (N,) 0-indexed base integers
        'state_strings':  state_matrix,       # (N, 15) ASCII codes: G=71, P=80
        'website_scores': website_scores,     # (N,) website adjudicator scores
        'n_boards':       len(states),
        'n_edges':        15,
        'graph_id':       GRAPH_ID,
    }, do_compression=False)

    print(f"Exported {len(states)} calibration boards to {output_path}")
    print()
    print("Next step -- run in MATLAB:")
    print(f"  cd {PROJECT_ROOT / 'snowdrop_tangled_agents' / 'matlab' / 'rl'}")
    print("  calibrate_schrodinger('../../data/calibration_boards.mat')")
    print()
    print("This generates: data/matlab_calib_results.mat")
    print("Then load results:")
    print("  python scripts/calibrate_adjudicator.py --load-matlab-results "
          "snowdrop_tangled_agents/matlab/rl/data/matlab_calib_results.mat")


# ---------------------------------------------------------------------------
# Track B: Direct Python evaluation (slow)
# ---------------------------------------------------------------------------
def _eval_one_python(args: tuple) -> tuple[int, float]:
    """Worker: evaluate one terminal state. Top-level for picklability."""
    idx, state_str, anneal_time = args
    from snowdrop_tangled_game_engine import GraphProperties
    from snowdrop_tangled_game_engine.game import Edge
    from snowdrop_adjudicators import SchrodingerEquationAdjudicator

    gp = GraphProperties()
    g = gp.graph_database[GRAPH_ID]
    el = g['edge_list']

    edges = [
        (v1, v2, Edge.State.FM.value if state_str[i] == 'G' else Edge.State.AFM.value)
        for i, (v1, v2) in enumerate(el)
    ]
    gs = {
        'num_nodes': g['num_nodes'], 'edges': edges, 'graph_id': GRAPH_ID,
        'player1_id': 'p1', 'player2_id': 'p2',
        'turn_count': len(el), 'current_player_index': 2,
        'player1_node': g['player1_node'], 'player2_node': g['player2_node'],
    }
    adj = SchrodingerEquationAdjudicator()
    adj.setup(epsilon=0.0, anneal_time=float(anneal_time))
    result = adj.adjudicate(gs)
    return idx, float(result['score'])


def python_eval_batch(states: list[str], anneal_time: float, workers: int = 4) -> np.ndarray:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    scores = np.zeros(len(states))
    work = [(i, s, anneal_time) for i, s in enumerate(states)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_eval_one_python, w): w[0] for w in work}
        for future in as_completed(futures):
            i, sc = future.result()
            scores[i] = sc
    return scores


def python_calibrate(states: list[str], website_scores: np.ndarray,
                     workers: int, sample_size: int) -> None:
    """Grid search + refinement using Python Schrodinger solver."""
    import warnings
    # Time one evaluation first
    print("Timing one Schrodinger evaluation (may take several minutes on 10 qubits)...")
    t0 = time.perf_counter()
    _, score_one = _eval_one_python((0, states[0], 40.0))
    elapsed = time.perf_counter() - t0
    print(f"  One evaluation: {elapsed:.1f}s  score={score_one:.4f}")

    if elapsed > 30:
        print(f"\nWARNING: {elapsed:.0f}s per board is too slow for full calibration.")
        print("         Grid search on 15 anneal_times x {sample_size} boards")
        print(f"         would take ~{15 * sample_size * elapsed / 3600:.1f} hours.")
        print()
        print("Recommendation: use --export-mat and run calibrate_schrodinger.m in MATLAB")
        print("                (MATLAB solver: ~0.7 s/board, 14x faster)")
        print()
        cont = input("Continue anyway? [y/N] ").strip().lower()
        if cont != 'y':
            return

    # Sample
    rng = np.random.default_rng(42)
    n = min(sample_size, len(states))
    idx = rng.choice(len(states), size=n, replace=False)
    sample_states = [states[i] for i in idx]
    sample_scores = website_scores[idx]

    # Grid search
    candidates = np.logspace(np.log10(5), np.log10(20000), 12).tolist()
    print(f"\nGrid search ({len(candidates)} candidates x {n} boards, "
          f"ETA {len(candidates)*n*elapsed/60:.0f} min):")
    print(f"  {'anneal_time':>12}  {'R^2':>8}  {'MAE':>8}")
    print("  " + "-" * 35)

    best_at, best_r2 = candidates[0], -np.inf
    for at in candidates:
        local = python_eval_batch(sample_states, at, workers)
        r2 = r_squared(sample_scores, local)
        mae = np.mean(np.abs(sample_scores - local))
        print(f"  {at:>12.1f}  {r2:>8.4f}  {mae:>8.4f}")
        if r2 > best_r2:
            best_r2, best_at = r2, at

    print(f"\nBest: anneal_time={best_at:.1f} ns  R^2={best_r2:.4f}")
    print("Run --load-matlab-results after MATLAB refinement for full validation.")


# ---------------------------------------------------------------------------
# Track C: SA proxy calibration
# ---------------------------------------------------------------------------
def sa_proxy_calibrate(states: list[str], website_scores: np.ndarray,
                       sa_lut_path: Path) -> None:
    """Use existing SA terminal LUT values as proxy; fit linear scaling to website scores."""
    import h5py

    print(f"\nLoading SA terminal LUT from {sa_lut_path.name}...")
    with h5py.File(str(sa_lut_path), 'r') as f:
        sa_terminal = f['terminalLUT'][()].flatten().astype(np.float64)

    base_indices = np.array([state_str_to_base_idx(s) for s in states])
    sa_scores = sa_terminal[base_indices]

    # Linear fit: website ≈ a * SA + b
    A = np.column_stack([sa_scores, np.ones(len(sa_scores))])
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, website_scores, rcond=None)
    a, b = coeffs
    fitted = a * sa_scores + b

    r2 = r_squared(website_scores, fitted)
    mae = np.mean(np.abs(website_scores - fitted))

    print(f"\n  SA proxy calibration results:")
    print(f"  website_score ~= {a:.4f} * SA_score + {b:.4f}")
    print(f"  R^2  : {r2:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  SA R^2 (raw, no transform) : {r_squared(website_scores, sa_scores):.4f}")
    print()
    if r2 < 0.9:
        print("  Note: SA proxy R^2 < 0.9. The SA adjudicator has systematic errors on")
        print("  the Petersen graph. Calibrated Schrodinger (Track A) is needed for")
        print("  a reliable oracle. Use --export-mat to generate MATLAB calibration data.")
    else:
        print("  SA proxy is well calibrated. Consider using it as an interim oracle.")


# ---------------------------------------------------------------------------
# Track D: Load MATLAB calibration results
# ---------------------------------------------------------------------------
def load_matlab_results(results_path: Path, states: list[str],
                        website_scores: np.ndarray) -> None:
    """Load MATLAB-computed scores and fit epsilon, then print recommendations."""
    import h5py

    print(f"Loading MATLAB calibration results from {results_path}...")
    try:
        data = scipy.io.loadmat(str(results_path))
    except Exception:
        with h5py.File(str(results_path), 'r') as f:
            data = {k: f[k][()] for k in f.keys() if not k.startswith('#')}

    best_at = float(np.asarray(data['best_anneal_time']).flat[0])
    local_scores = np.asarray(data['best_local_scores']).flatten()
    r2_grid = data.get('r2_grid', None)

    r2 = r_squared(website_scores, local_scores)
    mae = float(np.mean(np.abs(website_scores - local_scores)))

    print(f"\n  Best anneal_time  : {best_at:.2f} ns")
    print(f"  Full dataset R^2  : {r2:.4f}")
    print(f"  Full dataset MAE  : {mae:.4f}")

    # Fit epsilon
    best_eps, eps_acc = fit_epsilon(local_scores, website_scores)
    print(f"\n  Best epsilon      : {best_eps:.6f}")
    print(f"  Classification acc: {eps_acc:.4f}")

    print()
    print("=" * 60)
    if r2 > 0.9:
        print("  SUCCESS: R^2 > 0.9 — oracle well calibrated.")
    elif r2 > 0.5:
        print("  PARTIAL: R^2 in 0.5-0.9 — meaningful but imperfect.")
    else:
        print("  POOR: R^2 < 0.5 — website may use different physics.")

    print(f"\n  To rebuild the oracle with calibrated parameters:")
    print(f"    In MATLAB: generate_petersen_lut_schrodinger(...")
    print(f"      'epsilon', {best_eps:.6f}, 'anneal_time', {best_at:.2f})")
    print(f"    Or via Python generate_terminal_lut.py once --epsilon/--anneal-time")
    print(f"    CLI args are added to that script.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def fit_epsilon(local_scores: np.ndarray, website_scores: np.ndarray
                ) -> tuple[float, float]:
    """Find epsilon that maximises win/draw/loss classification accuracy."""
    candidates = np.sort(np.abs(website_scores))
    best_eps, best_acc = candidates[0], 0.0
    true_labels = np.sign(website_scores)
    for eps in candidates:
        pred = np.where(local_scores > eps, 1, np.where(local_scores < -eps, -1, 0))
        acc = float(np.mean(pred == true_labels))
        if acc > best_acc:
            best_acc, best_eps = acc, float(eps)
    return best_eps, best_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=str(DEFAULT_DB))

    track = parser.add_mutually_exclusive_group(required=True)
    track.add_argument("--export-mat", action="store_true",
                       help="Track A: export calibration boards for MATLAB processing")
    track.add_argument("--python-eval", action="store_true",
                       help="Track B: direct Python Schrodinger evaluation (slow)")
    track.add_argument("--sa-proxy", action="store_true",
                       help="Track C: instant calibration using SA terminal values")
    track.add_argument("--load-matlab-results", metavar="PATH",
                       help="Track D: load MATLAB-computed results and fit epsilon")

    parser.add_argument("--output", default=str(DATA_DIR / "calibration_boards.mat"),
                        help="Output path for --export-mat (default: data/calibration_boards.mat)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Workers for --python-eval (default: 4)")
    parser.add_argument("--sample", type=int, default=10,
                        help="Boards for --python-eval grid search (default: 10)")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading calibration data from {db_path.name}...")
    states, website_scores = load_calibration_data(db_path)
    print(f"  {len(states)} distinct terminal boards")
    print(f"  Website scores: [{website_scores.min():.3f}, {website_scores.max():.3f}]")

    if args.export_mat:
        export_for_matlab(states, website_scores, Path(args.output))

    elif args.python_eval:
        python_calibrate(states, website_scores, args.workers, args.sample)

    elif args.sa_proxy:
        sa_mat = DATA_DIR / "expanded_lut_sa.mat"
        if not sa_mat.exists():
            print(f"ERROR: SA MAT not found: {sa_mat}", file=sys.stderr)
            sys.exit(1)
        sa_proxy_calibrate(states, website_scores, sa_mat)

    elif args.load_matlab_results:
        load_matlab_results(Path(args.load_matlab_results), states, website_scores)


if __name__ == "__main__":
    main()
