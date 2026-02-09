"""
Build a website-calibrated terminal state LUT from empirical game data.

Mines the calibration table in game_stats.db for (terminal_state, website_score)
pairs observed during live play on tangled-game.com.  Cross-references against
the SA-derived LUT to quantify the nonlinear compression between SA and website
scores.  Outputs a partial LUT (NaN for unobserved states) in the same binary
format as the oracle-solver's terminal_scores.bin (32768 x f32 LE).

Usage:
    poetry run python -m snowdrop_tangled_agents.tools.build_website_lut
    poetry run python -m snowdrop_tangled_agents.tools.build_website_lut --opponent alphaq
    poetry run python -m snowdrop_tangled_agents.tools.build_website_lut --report-only

Output:
    oracle-solver/data/website_scores.bin   (32768 x f32 LE, NaN for unobserved)
"""

import argparse
import math
import os
import sqlite3
import struct
from pathlib import Path

import numpy as np


def state_to_idx(state: str) -> int:
    """Convert a 15-char terminal state string to a 0-based LUT index.

    Bit j of index: 1 = Green (FM), 0 = Purple (AFM).
    Matches the convention in generate_terminal_lut.py and oracle-solver/src/types.rs.
    """
    idx = 0
    for j, ch in enumerate(state):
        if ch == 'G':
            idx |= (1 << j)
    return idx


def idx_to_state(idx: int, num_edges: int = 15) -> str:
    """Convert a LUT index back to a terminal state string."""
    return ''.join('G' if (idx >> j) & 1 else 'P' for j in range(num_edges))


def load_sa_lut(path: str) -> np.ndarray:
    """Load the SA-derived LUT (32768 x f32 LE)."""
    with open(path, "rb") as f:
        data = f.read()
    return np.array(struct.unpack(f"<{len(data)//4}f", data), dtype=np.float32)


def query_calibration(db_path: str, opponent: str = None):
    """Query calibration table for (terminal_state, website_score) pairs.

    Returns list of (terminal_state, avg_website_score, count, std, min, max).
    """
    conn = sqlite3.connect(db_path)

    if opponent:
        rows = conn.execute("""
            SELECT c.terminal_state,
                   AVG(c.website_score) as avg_ws,
                   COUNT(*) as cnt,
                   GROUP_CONCAT(c.website_score) as scores
            FROM calibration c
            JOIN games g ON c.game_id = g.id
            WHERE g.opponent = ?
            GROUP BY c.terminal_state
            ORDER BY avg_ws DESC
        """, (opponent,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT c.terminal_state,
                   AVG(c.website_score) as avg_ws,
                   COUNT(*) as cnt,
                   GROUP_CONCAT(c.website_score) as scores
            FROM calibration c
            GROUP BY c.terminal_state
            ORDER BY avg_ws DESC
        """).fetchall()

    conn.close()

    results = []
    for term, avg_ws, cnt, scores_str in rows:
        scores = [float(s) for s in scores_str.split(",")]
        std = np.std(scores) if len(scores) > 1 else 0.0
        results.append({
            "terminal_state": term,
            "avg_website_score": avg_ws,
            "count": cnt,
            "std": float(std),
            "min": min(scores),
            "max": max(scores),
        })

    return results


def query_game_results(db_path: str, opponent: str = None):
    """Query game results for terminal states, including win/loss/draw."""
    conn = sqlite3.connect(db_path)

    if opponent:
        rows = conn.execute("""
            SELECT c.terminal_state, g.result, COUNT(*) as cnt
            FROM calibration c
            JOIN games g ON c.game_id = g.id
            WHERE g.opponent = ?
            GROUP BY c.terminal_state, g.result
        """, (opponent,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT c.terminal_state, g.result, COUNT(*) as cnt
            FROM calibration c
            JOIN games g ON c.game_id = g.id
            GROUP BY c.terminal_state, g.result
        """).fetchall()

    conn.close()

    # Organize by terminal state
    results = {}
    for term, result, cnt in rows:
        if term not in results:
            results[term] = {"win": 0, "loss": 0, "draw": 0}
        if result in results[term]:
            results[term][result] = cnt

    return results


def build_website_lut(calibration_data, total_states: int = 32768) -> np.ndarray:
    """Build a website-score LUT from calibration data.

    Unobserved states are filled with NaN.
    """
    lut = np.full(total_states, float('nan'), dtype=np.float32)

    for entry in calibration_data:
        term = entry["terminal_state"]
        if len(term) == 15 and '-' not in term:
            idx = state_to_idx(term)
            lut[idx] = entry["avg_website_score"]

    return lut


def save_lut_binary(lut: np.ndarray, path: str):
    """Save LUT as 32768 x f32 little-endian binary."""
    with open(path, "wb") as f:
        f.write(lut.tobytes())


def print_report(calibration_data, sa_lut, game_results, opponent: str = None):
    """Print detailed analysis report."""
    label = f" (opponent={opponent})" if opponent else " (all opponents)"
    print(f"\n{'='*70}")
    print(f"WEBSITE-CALIBRATED LUT REPORT{label}")
    print(f"{'='*70}")

    n_unique = len(calibration_data)
    n_total = sum(e["count"] for e in calibration_data)
    print(f"\n  Unique terminal states observed: {n_unique}")
    print(f"  Total game observations:         {n_total}")
    print(f"  Coverage of 32768 possible:      {100*n_unique/32768:.2f}%")

    if not calibration_data:
        print("  No calibration data found.")
        return

    # Score distribution
    ws_scores = [e["avg_website_score"] for e in calibration_data]
    print(f"\n  Website score statistics:")
    print(f"    Min:    {min(ws_scores):+.4f}")
    print(f"    Max:    {max(ws_scores):+.4f}")
    print(f"    Mean:   {np.mean(ws_scores):+.4f}")
    print(f"    Median: {np.median(ws_scores):+.4f}")
    print(f"    Std:    {np.std(ws_scores):.4f}")

    # Win threshold analysis
    above_2 = [e for e in calibration_data if e["avg_website_score"] > 2]
    above_1 = [e for e in calibration_data if e["avg_website_score"] > 1]
    print(f"\n  Terminal states with avg website score > +2: {len(above_2)}")
    print(f"  Terminal states with avg website score > +1: {len(above_1)}")

    # Score buckets
    print(f"\n  Website score distribution:")
    buckets = [(-99, -2), (-2, -1), (-1, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 99)]
    bucket_labels = ["< -2", "[-2, -1)", "[-1, 0)", "[0, 0.5)", "[0.5, 1)", "[1, 2)", ">= 2"]
    for (lo, hi), label in zip(buckets, bucket_labels):
        count = sum(1 for e in calibration_data if lo <= e["avg_website_score"] < hi)
        print(f"    {label:>12s}: {count:4d} unique terminals")

    # SA vs Website correlation
    if sa_lut is not None:
        sa_vals = []
        ws_vals = []
        for e in calibration_data:
            term = e["terminal_state"]
            if len(term) == 15 and '-' not in term:
                idx = state_to_idx(term)
                sa_vals.append(float(sa_lut[idx]))
                ws_vals.append(e["avg_website_score"])

        if sa_vals:
            sa_arr = np.array(sa_vals)
            ws_arr = np.array(ws_vals)
            corr = np.corrcoef(sa_arr, ws_arr)[0, 1]

            print(f"\n  SA vs Website score correlation:")
            print(f"    Pearson r:       {corr:.4f}")
            print(f"    SA score range:  [{sa_arr.min():+.4f}, {sa_arr.max():+.4f}]")
            print(f"    WS score range:  [{ws_arr.min():+.4f}, {ws_arr.max():+.4f}]")

            # Compression analysis
            print(f"\n  Nonlinear compression (SA -> Website):")
            for lo, hi in [(0.5, 2), (2, 5), (5, 10), (10, 20)]:
                mask = (sa_arr >= lo) & (sa_arr < hi)
                if mask.any():
                    ws_sub = ws_arr[mask]
                    sa_sub = sa_arr[mask]
                    ratio = sa_sub.mean() / ws_sub.mean() if ws_sub.mean() != 0 else float('inf')
                    print(f"    SA [{lo:+5.1f}, {hi:+5.1f}): n={mask.sum():4d}, "
                          f"ws_avg={ws_sub.mean():+.4f}, compression={ratio:.1f}x")

    # Game outcomes by terminal state
    if game_results:
        wins_possible = [t for t, r in game_results.items() if r.get("win", 0) > 0]
        print(f"\n  Terminal states that have produced wins: {len(wins_possible)}")
        if wins_possible:
            for t in wins_possible[:10]:
                r = game_results[t]
                ws = next((e["avg_website_score"] for e in calibration_data
                          if e["terminal_state"] == t), None)
                print(f"    {t}  W={r['win']} D={r.get('draw',0)} L={r.get('loss',0)}  ws={ws:+.4f}" if ws else
                      f"    {t}  W={r['win']} D={r.get('draw',0)} L={r.get('loss',0)}")

    # Top terminals by website score
    print(f"\n  Top 15 terminal states by website score:")
    for e in calibration_data[:15]:
        sa_val = float(sa_lut[state_to_idx(e["terminal_state"])]) if sa_lut is not None else 0
        r = game_results.get(e["terminal_state"], {})
        result_str = f"W={r.get('win',0)} D={r.get('draw',0)} L={r.get('loss',0)}"
        print(f"    {e['terminal_state']}  ws={e['avg_website_score']:+7.4f}  "
              f"sa={sa_val:+7.3f}  n={e['count']:3d}  {result_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Build website-calibrated terminal state LUT from game data."
    )
    parser.add_argument(
        "--opponent", type=str, default=None,
        help="Filter calibration data by opponent (e.g., 'alphaq', 'melissa'). "
             "Default: all opponents."
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Print report without writing LUT file."
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Path to game_stats.db. Default: ~/.tangled/game_stats.db"
    )
    parser.add_argument(
        "--sa-lut-path", type=str, default=None,
        help="Path to SA-derived terminal_scores.bin for cross-reference."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for website LUT binary. Default: oracle-solver/data/website_scores.bin"
    )
    args = parser.parse_args()

    # Locate database
    if args.db_path:
        db_path = args.db_path
    else:
        db_candidates = [
            os.path.expanduser("~/.tangled/game_stats.db"),
            "game_stats.db",
        ]
        db_path = next((p for p in db_candidates if os.path.exists(p)), None)
        if db_path is None:
            raise FileNotFoundError("game_stats.db not found")

    # Locate SA LUT
    sa_lut = None
    if args.sa_lut_path:
        sa_lut = load_sa_lut(args.sa_lut_path)
    else:
        project_root = Path(__file__).resolve().parent.parent.parent
        sa_path = project_root / "oracle-solver" / "data" / "terminal_scores.bin"
        if sa_path.exists():
            sa_lut = load_sa_lut(str(sa_path))

    # Query data
    print(f"Loading calibration data from {db_path}")
    calibration_data = query_calibration(db_path, args.opponent)
    game_results = query_game_results(db_path, args.opponent)

    # Report
    print_report(calibration_data, sa_lut, game_results, args.opponent)

    # Build and save LUT
    if not args.report_only and calibration_data:
        lut = build_website_lut(calibration_data)

        observed = np.sum(~np.isnan(lut))
        print(f"\n  Website LUT: {observed} observed / {len(lut)} total states")

        # Output path
        if args.output:
            output_path = args.output
        else:
            project_root = Path(__file__).resolve().parent.parent.parent
            output_path = str(project_root / "oracle-solver" / "data" / "website_scores.bin")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_lut_binary(lut, output_path)
        print(f"  Saved website LUT to {output_path}")


if __name__ == "__main__":
    main()
