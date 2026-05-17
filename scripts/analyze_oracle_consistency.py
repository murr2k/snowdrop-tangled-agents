#!/usr/bin/env python3
"""
analyze_oracle_consistency.py

Quantifies internal consistency of the SA and Schrodinger oracle LUTs by
measuring round-to-round value swings across AlphaQ games in the DB.

For each game, reconstructs the board state at each grey level from
moves.state_after, looks up both LUT values, and reports the volatility profile.

USAGE
-----
    python scripts/analyze_oracle_consistency.py [--db PATH] [--opponent alphaq] [--plot]
"""

import argparse
import itertools
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "snowdrop_tangled_agents" / "matlab" / "rl" / "data"

SA_MAT = DATA_DIR / "expanded_lut_sa.mat"
SCHR_MAT = DATA_DIR / "expanded_lut_schr.mat"
CALIB_MAT = DATA_DIR / "expanded_lut_calib.mat"
DEFAULT_DB = Path.home() / ".tangled" / "game_stats.db"

NUM_EDGES = 15

# MAT file key names for each grey level
LEVEL_KEYS = {
    0:  'terminalLUT',
    1:  'oneGreyScores',
    2:  'twoGreyScores',
    3:  'threeGreyScores',
    4:  'fourGreyScores',
    5:  'fiveGreyScores',
    6:  'sixGreyScores',
    7:  'sevenGreyScores',
    8:  'eightGreyScores',
    9:  'nineGreyScores',
    10: 'tenGreyScores',
    11: 'elevenGreyScores',
    12: 'twelveGreyScores',
    13: 'thirteenGreyScores',
    14: 'fourteenGreyScores',
    15: 'fifteenGreyScores',
}

C15 = {k: len(list(itertools.combinations(range(15), k))) for k in range(16)}


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def state_to_flat_idx(state_str: str) -> int:
    """Convert a 15-char state string (G/P only, no grey) to terminal flat index."""
    idx = 0
    for j, c in enumerate(state_str):
        if c == 'G':
            idx |= (1 << j)
    return idx


def state_to_level_flat_idx(state_str: str) -> tuple[int, int] | None:
    """Convert a 15-char state string (may contain '-' for grey) to (grey_level, flat_idx).

    flat_idx = base_idx * C(15,k) + combo_idx
    where k = number of grey ('-') edges, base_idx encodes non-grey G/P edges,
    combo_idx = position of grey edge set in combinations(range(15), k).

    Returns None if state is invalid.
    """
    grey_positions = []
    base_idx = 0
    for j, c in enumerate(state_str):
        if c == 'G':
            base_idx |= (1 << j)
        elif c == '-':
            grey_positions.append(j)

    k = len(grey_positions)
    if k == 0:
        return 0, base_idx

    grey_tuple = tuple(grey_positions)  # already sorted (enumerate order)
    # Look up combo_idx in C(15, k)
    combo_idx = _COMBO_MAPS[k].get(grey_tuple)
    if combo_idx is None:
        return None
    return k, base_idx * C15[k] + combo_idx


# Pre-build combo maps for all levels (fast O(1) lookup)
_COMBO_MAPS: dict[int, dict[tuple, int]] = {
    k: {c: i for i, c in enumerate(itertools.combinations(range(NUM_EDGES), k))}
    for k in range(1, 16)
}


# ---------------------------------------------------------------------------
# Batched LUT lookup from MAT file
# ---------------------------------------------------------------------------

def batch_lookup(mat_path: Path, level: int, flat_indices: np.ndarray) -> np.ndarray | None:
    """Load specific flat_indices from a MAT file level array.

    Returns float32 array of same length as flat_indices, or None on failure.
    """
    if not HAS_H5PY:
        return None
    key = LEVEL_KEYS.get(level)
    if key is None:
        return None
    try:
        with h5py.File(str(mat_path), 'r') as f:
            if key not in f:
                return None
            ds = f[key]
            # h5py fancy indexing requires sorted unique indices
            sorted_idx, inverse = np.unique(flat_indices, return_inverse=True)
            values = ds[sorted_idx.tolist()]  # h5py reads in order
            return values[inverse].astype(np.float32)
    except Exception as e:
        print(f"  WARNING: lookup failed for level {level} in {mat_path.name}: {e}")
        return None


# ---------------------------------------------------------------------------
# DB queries
# ---------------------------------------------------------------------------

def get_game_state_sequences(conn: sqlite3.Connection,
                              opponent: str) -> list[list[tuple[int, str, float | None]]]:
    """Return list of games; each game is a list of (grey_after_our_move, state_str, db_score).

    Only includes states after OUR moves (player='us').
    grey_after_our_move = number of '-' chars in state_str after our move.
    """
    cur = conn.execute("""
        SELECT g.id, m.move_number, m.state_after, m.score_after
        FROM games g
        JOIN moves m ON m.game_id = g.id
        WHERE g.opponent = ? AND m.player = 'us' AND m.state_after IS NOT NULL
        ORDER BY g.id, m.move_number
    """, (opponent,))
    rows = cur.fetchall()

    games: dict[str, list] = defaultdict(list)
    for game_id, move_num, state_after, score_after in rows:
        grey_count = state_after.count('-')
        games[game_id].append((grey_count, state_after, score_after))

    return [seq for seq in games.values() if seq]


def get_calibration_pairs(conn: sqlite3.Connection) -> list[tuple[str, float]]:
    """Return (terminal_state, website_score) pairs from calibration table."""
    try:
        cur = conn.execute(
            "SELECT terminal_state, website_score FROM calibration WHERE website_score IS NOT NULL"
        )
        return [(r[0], r[1]) for r in cur.fetchall() if r[0]]
    except sqlite3.OperationalError:
        return []


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyse(game_sequences: list, sa_mat: Path, schr_mat: Path,
            calib_mat: Path | None = None) -> dict:
    """Compute per-grey-level LUT values and round-to-round swings.

    Returns dict with keys sa/schr/calib/db and sa_swings/schr_swings/calib_swings/db_swings.
    """
    # Collect all (game_idx, round_idx, grey, state_str, db_score) per game
    all_states: dict[int, list[tuple[int, str, float | None]]] = {}
    for gi, seq in enumerate(game_sequences):
        all_states[gi] = seq  # [(grey, state, db_score), ...]

    # Group state strings by grey level and collect flat indices
    level_states: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    # (gi, ri, state_str) per level
    for gi, seq in all_states.items():
        for ri, (grey, state, _) in enumerate(seq):
            level_states[grey].append((gi, ri, state))

    print(f"\n  State counts by grey level:")
    for grey in sorted(level_states.keys(), reverse=True):
        print(f"    grey={grey}: {len(level_states[grey])} states")

    # Batch lookup from MAT files
    sa_lookup: dict[tuple[int, int], float] = {}   # (gi, ri) -> value
    schr_lookup: dict[tuple[int, int], float] = {}
    calib_lookup: dict[tuple[int, int], float] = {}

    for grey, entries in level_states.items():
        flat_indices = []
        key_pairs = []
        for gi, ri, state in entries:
            result = state_to_level_flat_idx(state)
            if result is None:
                continue
            k, flat = result
            flat_indices.append(flat)
            key_pairs.append((gi, ri))

        if not flat_indices:
            continue

        flat_arr = np.array(flat_indices, dtype=np.int64)

        sa_vals = batch_lookup(sa_mat, grey, flat_arr)
        schr_vals = batch_lookup(schr_mat, grey, flat_arr)
        calib_vals = batch_lookup(calib_mat, grey, flat_arr) if calib_mat else None

        for i, (gi, ri) in enumerate(key_pairs):
            if sa_vals is not None:
                sa_lookup[(gi, ri)] = float(sa_vals[i])
            if schr_vals is not None:
                schr_lookup[(gi, ri)] = float(schr_vals[i])
            if calib_vals is not None:
                calib_lookup[(gi, ri)] = float(calib_vals[i])

    # Compute per-game value sequences and round-to-round swings
    sa_by_grey: dict[int, list[float]] = defaultdict(list)
    schr_by_grey: dict[int, list[float]] = defaultdict(list)
    calib_by_grey: dict[int, list[float]] = defaultdict(list)
    db_by_grey: dict[int, list[float]] = defaultdict(list)
    sa_swings: dict[int, list[float]] = defaultdict(list)
    schr_swings: dict[int, list[float]] = defaultdict(list)
    calib_swings: dict[int, list[float]] = defaultdict(list)
    db_swings: dict[int, list[float]] = defaultdict(list)

    for gi, seq in all_states.items():
        prev_sa = prev_schr = prev_calib = prev_db = None

        for ri, (grey, state, db_score) in enumerate(seq):
            sa_v = sa_lookup.get((gi, ri))
            schr_v = schr_lookup.get((gi, ri))
            calib_v = calib_lookup.get((gi, ri))
            db_v = db_score

            if sa_v is not None:
                sa_by_grey[grey].append(sa_v)
            if schr_v is not None:
                schr_by_grey[grey].append(schr_v)
            if calib_v is not None:
                calib_by_grey[grey].append(calib_v)
            if db_v is not None:
                db_by_grey[grey].append(db_v)

            if prev_sa is not None and sa_v is not None:
                sa_swings[grey].append(abs(sa_v - prev_sa))
            if prev_schr is not None and schr_v is not None:
                schr_swings[grey].append(abs(schr_v - prev_schr))
            if prev_calib is not None and calib_v is not None:
                calib_swings[grey].append(abs(calib_v - prev_calib))
            if prev_db is not None and db_v is not None:
                db_swings[grey].append(abs(db_v - prev_db))

            prev_sa = sa_v
            prev_schr = schr_v
            prev_calib = calib_v
            prev_db = db_v

    return {
        'sa': sa_by_grey,
        'schr': schr_by_grey,
        'calib': calib_by_grey,
        'db': db_by_grey,
        'sa_swings': sa_swings,
        'schr_swings': schr_swings,
        'calib_swings': calib_swings,
        'db_swings': db_swings,
    }


def print_report(results: dict, calibration_pairs: list):
    print()
    print("=" * 88)
    print("  Oracle Consistency Report -- Mean |delta-value| between consecutive rounds")
    print("  (P1 game: grey 14->12->10->8->6->4->2->0, each step = 1 round)")
    print("=" * 88)
    print(f"  {'Grey':>6}  {'SA |dv|':>10}  {'Schr |dv|':>10}  {'Calib |dv|':>11}  {'DB |dv|':>10}  {'N':>6}")
    print("  " + "-" * 68)

    all_grey = sorted(
        set(results['sa_swings']) | set(results['schr_swings']) |
        set(results.get('calib_swings', {})) | set(results['db_swings']),
        reverse=True
    )
    for g in all_grey:
        sa = results['sa_swings'].get(g, [])
        sc = results['schr_swings'].get(g, [])
        ca = results.get('calib_swings', {}).get(g, [])
        db = results['db_swings'].get(g, [])
        sa_s  = f"{np.mean(sa):.4f}" if sa else "     -"
        sc_s  = f"{np.mean(sc):.4f}" if sc else "     -"
        ca_s  = f"{np.mean(ca):.4f}" if ca else "      -"
        db_s  = f"{np.mean(db):.4f}" if db else "     -"
        n = max(len(sa), len(sc), len(ca), len(db))
        print(f"  {g:>6}  {sa_s:>10}  {sc_s:>10}  {ca_s:>11}  {db_s:>10}  {n:>6}")

    print()
    print("  Overall mean |dv| (all grey levels combined):")
    all_sa = [v for vs in results['sa_swings'].values() for v in vs]
    all_sc = [v for vs in results['schr_swings'].values() for v in vs]
    all_ca = [v for vs in results.get('calib_swings', {}).values() for v in vs]
    all_db = [v for vs in results['db_swings'].values() for v in vs]
    if all_sa:
        print(f"    SA    oracle: {np.mean(all_sa):.4f}  (median {np.median(all_sa):.4f})")
    if all_sc:
        print(f"    Schr  oracle: {np.mean(all_sc):.4f}  (median {np.median(all_sc):.4f})")
    if all_ca:
        print(f"    Calib oracle: {np.mean(all_ca):.4f}  (median {np.median(all_ca):.4f})")
    if all_db:
        print(f"    DB    scores: {np.mean(all_db):.4f}  (median {np.median(all_db):.4f})")

    if calibration_pairs:
        print()
        print("  Terminal accuracy vs website scores (sample of calibration pairs):")
        sample = calibration_pairs[:500]
        flat_indices = []
        for state, _ in sample:
            res = state_to_level_flat_idx(state)
            flat_indices.append(res[1] if res else 0)
        flat_arr = np.array(flat_indices, dtype=np.int64)
        website = np.array([ws for _, ws in sample], dtype=np.float32)

        sa_term    = batch_lookup(SA_MAT, 0, flat_arr)
        sc_term    = batch_lookup(SCHR_MAT, 0, flat_arr)
        calib_term = batch_lookup(CALIB_MAT, 0, flat_arr)

        for label, vals in [("SA   ", sa_term), ("Schr ", sc_term), ("Calib", calib_term)]:
            if vals is not None:
                mae = np.mean(np.abs(vals - website))
                r2 = 1 - np.var(vals - website) / np.var(website)
                print(f"    {label} terminal: MAE={mae:.4f}  R2={r2:.4f}  (n={len(sample)})")

    print("=" * 76)


def plot_report(results: dict, output_path: Path):
    if not HAS_MATPLOTLIB:
        print("(matplotlib not available -- skipping plot)")
        return

    all_grey = sorted(
        set(results['sa_swings']) | set(results['schr_swings']) |
        set(results.get('calib_swings', {})),
        reverse=True
    )
    sa_means    = [np.mean(results['sa_swings'].get(g, [np.nan])) for g in all_grey]
    sc_means    = [np.mean(results['schr_swings'].get(g, [np.nan])) for g in all_grey]
    calib_means = [np.mean(results.get('calib_swings', {}).get(g, [np.nan])) for g in all_grey]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(all_grey, sa_means,    'o-', label='SA oracle',             color='tab:orange', linewidth=2)
    ax.plot(all_grey, sc_means,    's-', label='Schrodinger (40ns)',    color='tab:blue',   linewidth=2)
    ax.plot(all_grey, calib_means, '^-', label='Calib Schrodinger (1.85ns)', color='tab:green', linewidth=2)
    ax.set_xlabel("Grey level after our move (high = early game, low = late game)")
    ax.set_ylabel("Mean |delta-value| between rounds")
    ax.set_title(
        "Oracle Value Volatility by Grey Level\n"
        "(lower = more internally consistent; ideal oracle is flat near 0)"
    )
    ax.invert_xaxis()
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    print(f"\nPlot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--opponent", default="alphaq")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to N games (0 = all)")
    parser.add_argument("--plot", action="store_true",
                        help="Save volatility plot to oracle_consistency.png")
    args = parser.parse_args()

    if not HAS_H5PY:
        print("ERROR: h5py is required. Install with: pip install h5py", file=sys.stderr)
        sys.exit(1)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.opponent} game sequences from {db_path.name}...")
    conn = sqlite3.connect(str(db_path))
    sequences = get_game_state_sequences(conn, args.opponent)
    if args.limit:
        sequences = sequences[:args.limit]
    print(f"  {len(sequences)} games with move data")

    calibration_pairs = get_calibration_pairs(conn)
    if calibration_pairs:
        print(f"  {len(calibration_pairs)} calibration pairs available")

    conn.close()

    calib_mat = CALIB_MAT if CALIB_MAT.exists() else None
    print("\nBatch-looking up SA, Schrodinger, and Calib oracle values...")
    if calib_mat is None:
        print("  (expanded_lut_calib.mat not found -- skipping calib column)")
    results = analyse(sequences, SA_MAT, SCHR_MAT, calib_mat)

    print_report(results, calibration_pairs)

    if args.plot:
        plot_report(results, Path("oracle_consistency.png"))


if __name__ == "__main__":
    main()
