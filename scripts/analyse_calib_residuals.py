"""Phase 5A.1 — Residual analysis: calibrated oracle vs website scores.

For every distinct terminal board in the calibration table:
- Look up the current calib oracle's score from terminal_scores.mat
- Compute residual = website_score - calib_score
- Bin residuals by structure (Green count, 5-cycle frustration), magnitude, sign
- Report R², RMSE, MAE, structural patterns
- Save plot

Tells us whether the R²=0.60 ceiling is parameter error (structured residuals
that can be re-fit) or model error (white-noise residuals — the Schrödinger
TFIM model is fundamentally not what the website uses).

Usage:
    poetry run python scripts/analyse_calib_residuals.py
"""

from __future__ import annotations

import io
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

# Windows cp1252 console can't render non-ASCII; force UTF-8.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from snowdrop_tangled_game_engine import GraphProperties

DB = Path.home() / ".tangled" / "game_stats.db"
LUT = Path("snowdrop_tangled_agents/matlab/rl/data/terminal_scores.mat")
PLOT_OUT = Path("plots/phase5a_calib_residuals.png")


def get_petersen_edges() -> list[tuple[int, int]]:
    return list(GraphProperties().graph_database[5]["edge_list"])


def get_petersen_5cycles(edges: list[tuple[int, int]]) -> list[tuple[int, ...]]:
    """Return 5-cycles as sorted tuples of edge indices. Petersen has exactly 12."""
    import networkx as nx
    g = nx.Graph()
    g.add_edges_from(edges)
    edge_idx = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    cycles_set: set[tuple[int, ...]] = set()
    for cycle in nx.simple_cycles(g, length_bound=5):
        if len(cycle) != 5:
            continue
        c_edges = []
        for k in range(5):
            a, b = cycle[k], cycle[(k + 1) % 5]
            c_edges.append(edge_idx[(min(a, b), max(a, b))])
        cycles_set.add(tuple(sorted(c_edges)))
    return sorted(cycles_set)


def board_to_index(board: str, num_edges: int) -> int:
    """'GPPGP...' -> integer where bit j is 1 iff position j is 'G'."""
    if len(board) != num_edges or any(c not in "GP" for c in board):
        raise ValueError(f"bad board {board!r}")
    return sum(1 << j for j, c in enumerate(board) if c == "G")


def load_calib_lut() -> np.ndarray:
    with h5py.File(LUT, "r") as f:
        return f["terminal_scores"][:].flatten().astype(np.float64)


@dataclass
class Stats:
    n: int
    r2: float
    rmse: float
    mae: float
    bias: float

    def fmt(self) -> str:
        return (
            f"n={self.n:5d}  R²={self.r2:+.4f}  RMSE={self.rmse:.4f}  "
            f"MAE={self.mae:.4f}  bias={self.bias:+.4f}"
        )


def compute_stats(website: np.ndarray, predicted: np.ndarray) -> Stats:
    n = len(website)
    if n == 0 or website.std() == 0:
        return Stats(n, float("nan"), float("nan"), float("nan"), float("nan"))
    residual = website - predicted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((website - website.mean()) ** 2))
    return Stats(
        n=n,
        r2=1.0 - ss_res / ss_tot,
        rmse=float(np.sqrt(ss_res / n)),
        mae=float(np.mean(np.abs(residual))),
        bias=float(np.mean(residual)),
    )


def main() -> None:
    print("=" * 78)
    print("Phase 5A.1 — Calib oracle vs website residual analysis")
    print("=" * 78)

    edges = get_petersen_edges()
    num_edges = len(edges)
    print(f"\nPetersen edges ({num_edges}): {edges}")

    cycles = get_petersen_5cycles(edges)
    print(f"Petersen 5-cycles found: {len(cycles)}")
    if len(cycles) != 12:
        print(f"  WARNING: expected 12, skipping 5-cycle analysis")
        cycles = []

    print(f"\nLoading calib LUT from {LUT.name}...")
    lut = load_calib_lut()
    print(f"  {lut.size} entries, range [{lut.min():.3f}, {lut.max():.3f}], "
          f"mean={lut.mean():.3f}, std={lut.std():.3f}")

    print(f"\nQuerying {DB.name} calibration table...")
    conn = sqlite3.connect(str(DB))
    rows = conn.execute(
        "SELECT terminal_state, AVG(website_score), COUNT(*) FROM calibration "
        "GROUP BY terminal_state"
    ).fetchall()
    conn.close()

    boards: list[str] = []
    website: list[float] = []
    calib: list[float] = []
    counts: list[int] = []
    skipped = 0
    for state, score, cnt in rows:
        try:
            idx = board_to_index(state, num_edges)
        except ValueError:
            skipped += 1
            continue
        boards.append(state)
        website.append(float(score))
        calib.append(float(lut[idx]))
        counts.append(int(cnt))
    if skipped:
        print(f"  skipped {skipped} malformed boards")
    print(f"  {len(boards)} distinct boards, {sum(counts)} total observations")

    website_arr = np.array(website)
    calib_arr = np.array(calib)
    counts_arr = np.array(counts)
    residual = website_arr - calib_arr

    overall = compute_stats(website_arr, calib_arr)
    print("\n--- Overall fit ---")
    print(f"  {overall.fmt()}")

    print("\n--- By Green count (board structure) ---")
    g_counts = np.array([b.count("G") for b in boards])
    for g in range(0, 16, 2):
        mask = (g_counts >= g) & (g_counts < min(g + 2, 16))
        if mask.sum() < 5:
            continue
        print(f"  G in [{g:2d}, {min(g+2, 16):2d}): {compute_stats(website_arr[mask], calib_arr[mask]).fmt()}")

    print("\n--- By |website_score| magnitude ---")
    abs_w = np.abs(website_arr)
    bins = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 100.0]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (abs_w >= lo) & (abs_w < hi)
        if mask.sum() < 5:
            continue
        print(f"  |website| ∈ [{lo:.1f}, {hi:.1f}): {compute_stats(website_arr[mask], calib_arr[mask]).fmt()}")

    if cycles:
        print("\n--- By 5-cycle frustration count ---")
        def frust(b: str) -> int:
            return sum(1 for c in cycles if sum(1 for e in c if b[e] == "P") % 2 == 1)
        f_arr = np.array([frust(b) for b in boards])
        for f in range(0, 13):
            mask = f_arr == f
            if mask.sum() < 5:
                continue
            print(f"  frust={f:2d}: {compute_stats(website_arr[mask], calib_arr[mask]).fmt()}")

    print("\n--- Residual sign bias by website-score sign ---")
    masks = [
        ("website > +0.1", website_arr > 0.1),
        ("website < -0.1", website_arr < -0.1),
        ("|website| ≤ 0.1", np.abs(website_arr) <= 0.1),
    ]
    for label, m in masks:
        if m.sum() == 0:
            continue
        r = residual[m]
        print(f"  {label:18s} ({m.sum():4d}): mean res = {r.mean():+.4f}  std = {r.std():.4f}  median = {np.median(r):+.4f}")

    print("\n--- Top 15 absolute residuals ---")
    top = np.argsort(np.abs(residual))[-15:][::-1]
    for i in top:
        g = boards[i].count("G")
        print(f"  {boards[i]}  website={website_arr[i]:+8.3f}  calib={calib_arr[i]:+8.3f}  res={residual[i]:+7.3f}  G={g:2d}  n={counts_arr[i]}")

    print("\n--- Score-magnitude scaling check ---")
    slope, intercept = np.polyfit(calib_arr, website_arr, 1)
    rescaled = slope * calib_arr + intercept
    print(f"  Linear fit website = {slope:+.4f} * calib + {intercept:+.4f}")
    print(f"  After rescale:    {compute_stats(website_arr, rescaled).fmt()}")
    print(f"  Δ R² from rescale: {compute_stats(website_arr, rescaled).r2 - overall.r2:+.4f}")

    print("\n--- Correlation structure of residual vs calib prediction ---")
    pearson = np.corrcoef(calib_arr, residual)[0, 1]
    print(f"  Pearson corr(calib, residual) = {pearson:+.4f}")
    print(f"  (large |corr| → systematic over/underprediction with score; small → unstructured residual)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        ax = axes[0]
        ax.scatter(calib_arr, website_arr, s=8, alpha=0.35)
        lim = max(np.abs(calib_arr).max(), np.abs(website_arr).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="y = x")
        xs = np.linspace(-lim, lim, 100)
        ax.plot(xs, slope * xs + intercept, "r-", lw=1.0,
                label=f"linear fit: y = {slope:+.3f}x{intercept:+.3f}")
        ax.set_xlabel("calib oracle score")
        ax.set_ylabel("website_score")
        ax.set_title(f"Calib vs website (n={len(boards)}, R²={overall.r2:+.3f})")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.scatter(calib_arr, residual, s=8, alpha=0.35)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("calib oracle score")
        ax.set_ylabel("residual (website − calib)")
        ax.set_title(f"Residual vs predicted (corr={pearson:+.3f})")
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.hist(residual, bins=60, alpha=0.75, color="steelblue", edgecolor="black")
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(residual.mean(), color="red", lw=1, ls="--",
                   label=f"mean = {residual.mean():+.3f}")
        ax.set_xlabel("residual")
        ax.set_ylabel("count")
        ax.set_title(f"Residual histogram (std = {residual.std():.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(PLOT_OUT, dpi=120)
        print(f"\nPlot saved to {PLOT_OUT}")
    except ImportError:
        print("\n(matplotlib not available — skipping plot)")


if __name__ == "__main__":
    main()
