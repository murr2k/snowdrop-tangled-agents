"""
Refit the ternary terminal model to captured lookup-table scores.

Every adjudicated game adds one true (terminal, lookup-table score) pair. This
fits the Boltzmann model's beta and an overall score scale a (real ~ a *
model(beta)) by least squares, and reports how well each candidate beta
classifies the pairs into win/draw/loss with the site's 0.0005 draw band.
The scale matters for the planner: a model margin m only clears the draw band
when a * m > 0.0005.

With --build the chosen beta's terminal table is built (under a minute) and
both full passes run (opening book, reply book, layer stores; ~45 s each), so
`--ternary-beta <beta>` and --plan-lines use it.

Usage:
    python -m snowdrop_tangled_agents.tools.refit_ternary_model [--build]
"""

import argparse
import subprocess
import sys

import numpy as np

from snowdrop_tangled_agents.strategy import ternary_model as tm
from snowdrop_tangled_agents.tools import alphaq_captures as ac

BETAS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 16.0, None)


def outcome(x: float, eps: float = tm.DRAW_EPSILON) -> int:
    return 1 if x > eps else -1 if x < -eps else 0


def fit(pairs: list) -> list:
    """Per beta: (beta, scale, rmse, class agreement)."""
    terminals = [t for t, _ in pairs]
    real = np.array([s for _, s in pairs], dtype=np.float64)
    coup = np.stack([tm.couplings_of(t) for t in terminals])
    rows = []
    for beta in BETAS:
        model = tm.score_couplings(coup, beta).astype(np.float64)
        denom = float(model @ model)
        # A negative scale would invert every decision; clamp it (a ~ 0 means no predictive power).
        a = max(float(model @ real) / denom, 0.0) if denom > 0 else 1.0
        rmse = float(np.sqrt(np.mean((a * model - real) ** 2)))
        agree = float(np.mean([outcome(a * m) == outcome(r) for m, r in zip(model, real)]))
        rows.append((beta, a, rmse, agree, model))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build", action="store_true", help="build the best beta's table, books and layer stores")
    ap.add_argument("--beta", type=tm.parse_model, default=None, help="build this beta instead of the best fit")
    ap.add_argument("--show", type=int, default=15, help="list the N largest-|score| pairs")
    args = ap.parse_args()

    known = ac.known_terminals(ac.load_games())
    pairs = sorted(known.items(), key=lambda kv: -abs(kv[1]))
    print(f"{len(pairs)} distinct adjudicated terminals")
    rows = fit(pairs)
    for beta, a, rmse, agree, _ in rows:
        print(f"beta={tm.model_name(beta):>4s} scale={a:+.4f} rmse={rmse:.6f} W/D/L agreement={agree:.2%}")
    best = min(rows, key=lambda r: (-r[3], r[2]))
    print(f"best: beta={tm.model_name(best[0])} scale={best[1]:+.4f}")
    b4 = next(r for r in rows if r[0] == 4.0)
    for (t, s), mb, m4 in list(zip(pairs, best[4], b4[4]))[:args.show]:
        print(f"  {t} real={s:+.6f} best={best[1] * mb:+.6f} b4={m4:+.6f}")

    if args.build:
        beta = args.beta if args.beta is not None else best[0]
        tm.load_lut(beta)      # builds and caches the table
        for player in (1, 2):
            subprocess.run([sys.executable, "-m", "snowdrop_tangled_agents.tools.solve_ternary_game",
                            "--player", str(player), "--beta", tm.model_name(beta).lstrip('b')
                            if beta is not None else "inf"], check=True)


if __name__ == "__main__":
    main()
