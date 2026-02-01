"""
Generate P(win) calibration curve from game history.

Reads the calibration table (SA predicted scores paired with actual game
results) and fits a monotonic P(win) curve via quantile-binned isotonic
regression.  The curve is saved as calibration_pwin.mat so TangledMCTS can
load it with a plain ``load()`` call and interpolate at runtime with
``interp1``.

The calibration addresses a fundamental mismatch: SimulatedAnnealing scores
do not reliably predict the winner.  SA scores in [+2, +5] only win ~71 %
of the time, yet the website score in that same range wins ~98.5 %.  By
mapping SA scores through the empirical P(win) curve the MCTS terminal
evaluation becomes "what is the probability we actually win from this state"
rather than "what does SA think the score is".

Usage:
    poetry run python -m snowdrop_tangled_agents.tools.generate_calibration

Output:
    snowdrop_tangled_agents/matlab/rl/data/calibration_pwin.mat
        Fields:
            scores  — Nx1 double, sorted calibration knot points
            pwin    — Nx1 double, monotonic P(win) at each knot
            n_games — scalar, number of calibration samples used
            n_wins  — scalar, number of wins in the sample
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.io import savemat

N_BINS = 30  # quantile bins; ~50 samples each at 1 500 games


def load_calibration_data(db_path: str) -> list[tuple[float, int]]:
    """Return (predicted_score, is_win) for every calibrated game.

    Draws and losses both map to is_win = 0.  The calibration curve
    therefore estimates P(win), not P(not-loss).
    """
    conn = sqlite3.connect(db_path)
    cur = conn.execute("""
        SELECT c.predicted_score,
               CASE WHEN g.result = 'win' THEN 1 ELSE 0 END
        FROM   calibration c
        JOIN   games g ON c.game_id = g.id
        WHERE  g.result IN ('win', 'loss', 'draw')
    """)
    data = [(float(row[0]), int(row[1])) for row in cur.fetchall()]
    conn.close()
    return data


def isotonic_regression(y: list[float]) -> list[float]:
    """Pool Adjacent Violators — enforce non-decreasing constraint.

    Each element starts as its own block.  Whenever a block's mean exceeds
    the next block's mean the two are merged and their combined mean is
    checked against the previous block, and so on.
    """
    # blocks: list of [sum_y, count]
    blocks: list[list[float]] = [[v, 1.0] for v in y]

    i = 0
    while i < len(blocks) - 1:
        mean_i = blocks[i][0] / blocks[i][1]
        mean_next = blocks[i + 1][0] / blocks[i + 1][1]
        if mean_i > mean_next:
            # merge
            blocks[i] = [blocks[i][0] + blocks[i + 1][0],
                         blocks[i][1] + blocks[i + 1][1]]
            blocks.pop(i + 1)
            if i > 0:
                i -= 1          # re-check against previous block
        else:
            i += 1

    # expand back to full-length vector
    result: list[float] = []
    for s, c in blocks:
        result.extend([s / c] * int(c))
    return result


def compute_calibration_curve(
    data: list[tuple[float, int]],
    n_bins: int = N_BINS,
) -> tuple[list[float], list[float]]:
    """Quantile-bin the data, smooth with Laplace, enforce monotonicity.

    Returns (scores, pwin) where scores are sorted knot points and pwin is
    the monotonically non-decreasing P(win) at each knot.  Sentinel points
    at ±100 anchor the extrapolation to 0 and 1.
    """
    data.sort(key=lambda x: x[0])
    n = len(data)
    bin_size = n // n_bins

    scores: list[float] = []
    pwin: list[float] = []

    for i in range(n_bins):
        start = i * bin_size
        end = (start + bin_size) if i < n_bins - 1 else n

        bin_scores = [d[0] for d in data[start:end]]
        bin_wins = sum(d[1] for d in data[start:end])
        bin_total = len(bin_scores)

        # median score as knot position
        mid = len(bin_scores) // 2
        scores.append(bin_scores[mid])

        # Laplace-smoothed win rate
        pwin.append((bin_wins + 0.5) / (bin_total + 1.0))

    # Sentinel extrapolation points
    scores = [-100.0] + scores + [100.0]
    pwin = [0.0] + pwin + [1.0]

    # Enforce monotonicity
    pwin = isotonic_regression(pwin)

    return scores, pwin


def main() -> None:
    # locate database
    db_candidates = [
        os.path.expanduser("~/.tangled/game_stats.db"),
        "game_stats.db",
    ]
    db_path = next((p for p in db_candidates if os.path.exists(p)), None)
    if db_path is None:
        raise FileNotFoundError("game_stats.db not found")

    print(f"Loading calibration data from {db_path}")
    data = load_calibration_data(db_path)
    n_wins = sum(d[1] for d in data)
    print(f"  {len(data)} samples, {n_wins} wins, {len(data) - n_wins} non-wins")

    scores, pwin = compute_calibration_curve(data)

    print(f"\nCalibration curve ({len(scores)} knots, including sentinels):")
    print(f"  {'Score':>8}  {'P(win)':>6}")
    print(f"  {'------':>8}  {'------':>6}")
    for s, p in zip(scores, pwin):
        if -50 < s < 50:          # skip sentinels in printout
            print(f"  {s:>+8.3f}  {p:>6.3f}")

    # Save as .mat
    output_dir = Path(__file__).resolve().parent.parent / "matlab" / "rl" / "data"
    output_path = output_dir / "calibration_pwin.mat"
    output_dir.mkdir(parents=True, exist_ok=True)

    savemat(str(output_path), {
        "scores":  np.array(scores, dtype=np.float64),
        "pwin":    np.array(pwin,   dtype=np.float64),
        "n_games": np.float64(len(data)),
        "n_wins":  np.float64(n_wins),
    })
    print(f"\nSaved calibration to {output_path}")


if __name__ == "__main__":
    main()
