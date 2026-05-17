"""Phase 4 decision-gate analysis.

Pulls the just-finished 50-game run (run 141) and compares its score
distribution to the prior minimax-mode P1-vs-AlphaQ-calib runs in the
DB. The gate (per docs/PROJECT_PLAN_ALPHAQ_TARGETED_INVESTIGATION.md
Phase 4):
    Any wins -> scale.
    Zero wins, mean score significantly higher -> continue tuning.
    Zero wins, mean score unchanged -> trigger Phase 5 pivot."""

import sqlite3
import statistics
import sys
from pathlib import Path

DB_PATH = Path.home() / ".tangled" / "game_stats.db"
PHASE4_RUN_ID = 141


def discover_schema(conn):
    cur = conn.cursor()
    print("=== schema ===")
    for tbl in ("runs", "games", "moves"):
        cur.execute(f"PRAGMA table_info({tbl})")
        cols = [(c[1], c[2]) for c in cur.fetchall()]
        print(f"  {tbl}: {cols}")
    print()


def summarise(name, scores, wins, losses, draws):
    n = len(scores)
    if n == 0:
        print(f"{name}: NO GAMES")
        return
    print(f"=== {name} (n={n}) ===")
    print(f"  W/D/L: {wins}/{draws}/{losses}")
    print(f"  score: mean={statistics.mean(scores):+.4f}  "
          f"median={statistics.median(scores):+.4f}  "
          f"stdev={statistics.stdev(scores) if n > 1 else 0:.4f}")
    print(f"  range: [{min(scores):+.4f}, {max(scores):+.4f}]")
    # Histogram of score bins
    bins = [(-1.0, -0.7), (-0.7, -0.5), (-0.5, -0.2), (-0.2, 0.2),
            (0.2, 0.5), (0.5, 0.7), (0.7, 1.0)]
    for lo, hi in bins:
        cnt = sum(1 for s in scores if lo <= s < hi)
        bar = "#" * cnt
        print(f"  [{lo:+.1f}, {hi:+.1f}): {cnt:3d} {bar}")
    print()


def main() -> int:
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(str(DB_PATH))
    discover_schema(conn)
    cur = conn.cursor()

    # Pull all games in run 141 (the Phase 4 run we just finished)
    print(f"=== Phase 4 run ({PHASE4_RUN_ID}) details")
    cur.execute(
        "SELECT g.id, g.opponent, g.result, g.final_score, r.seat, r.lut_variant "
        "FROM games g JOIN runs r ON g.run_id = r.id WHERE g.run_id = ?",
        (PHASE4_RUN_ID,))
    rows = cur.fetchall()
    print(f"  games in run: {len(rows)}")
    if rows:
        print(f"  opponent set: {set(r[1] for r in rows)}")
        print(f"  seat set: {set(r[4] for r in rows)}")
        print(f"  lut_variant set: {set(r[5] for r in rows)}")

    # Phase 4 stats: this run only
    p4_scores, p4_w, p4_l, p4_d = [], 0, 0, 0
    for _, _, result, score, *_ in rows:
        p4_scores.append(float(score))
        if result == 'win':
            p4_w += 1
        elif result == 'loss':
            p4_l += 1
        else:
            p4_d += 1
    summarise("Phase 4 (run 141, --solver-adversary expected, calib LUT, P1)",
              p4_scores, p4_w, p4_l, p4_d)

    # Comparable minimax-mode baseline: the most recent prior P1 vs AlphaQ
    # hybrid_solver run before Phase 4 (run 130). Note: the lut_variant column
    # was added recently so earlier runs report it as NULL; run 130 itself was
    # before the calib variant was wired into runs metadata. It's the closest
    # baseline we have (10 games, hours before Phase 4, same strategy + seat).
    BASELINE_RUN_ID = 130
    cur.execute(
        "SELECT g.id, g.run_id, g.result, g.final_score FROM games g "
        "WHERE g.run_id = ?",
        (BASELINE_RUN_ID,))
    base_rows = cur.fetchall()
    base_scores, base_w, base_l, base_d = [], 0, 0, 0
    for _, _, result, score in base_rows:
        if score is None:
            continue
        base_scores.append(float(score))
        if result == 'win':
            base_w += 1
        elif result == 'loss':
            base_l += 1
        else:
            base_d += 1
    summarise("Minimax baseline (run 130, P1 vs AlphaQ, hybrid_solver, 2026-05-17 06:38)",
              base_scores, base_w, base_l, base_d)

    # Also show ANY AlphaQ P1 game regardless of LUT (broader baseline)
    cur.execute(
        "SELECT g.result, g.final_score, r.lut_variant FROM games g "
        "JOIN runs r ON g.run_id = r.id "
        "WHERE g.opponent = 'alphaq' AND r.seat = 1 AND g.run_id != ?",
        (PHASE4_RUN_ID,))
    all_rows = cur.fetchall()
    by_lut = {}
    for result, score, lut in all_rows:
        if score is None:
            continue
        by_lut.setdefault(lut or '(null)', []).append((result, float(score)))
    print("=== Broader P1-vs-AlphaQ baseline by LUT variant ===")
    for lut, items in sorted(by_lut.items()):
        sc = [s for _, s in items]
        w = sum(1 for r, _ in items if r == 'win')
        l = sum(1 for r, _ in items if r == 'loss')
        d = sum(1 for r, _ in items if r == 'draw')
        if not sc:
            continue
        print(f"  lut={lut!s:>10}: n={len(sc):4d}  W/D/L={w}/{d}/{l}  "
              f"mean={statistics.mean(sc):+.4f}  "
              f"median={statistics.median(sc):+.4f}  "
              f"min={min(sc):+.4f}  max={max(sc):+.4f}")
    print()

    # Welch's t-test between Phase 4 and minimax baseline scores
    if len(p4_scores) > 1 and len(base_scores) > 1:
        try:
            from scipy.stats import ttest_ind, mannwhitneyu
            t_stat, t_p = ttest_ind(p4_scores, base_scores, equal_var=False)
            u_stat, u_p = mannwhitneyu(p4_scores, base_scores, alternative='two-sided')
            print("=== Phase 4 vs minimax-calib baseline ===")
            print(f"  mean_diff = {statistics.mean(p4_scores) - statistics.mean(base_scores):+.4f}")
            print(f"  Welch t-test: t={t_stat:.3f}  p={t_p:.4f}")
            print(f"  Mann-Whitney U: U={u_stat:.1f}  p={u_p:.4f}")
            print()
        except Exception as e:
            print(f"  (stats package error: {e})")

    # Decision
    print("=== DECISION GATE ===")
    if p4_w > 0:
        print(f"  Phase 4 produced {p4_w} win(s) -> SCALE to 500-game campaign")
    elif len(base_scores) > 1 and len(p4_scores) > 1:
        from scipy.stats import ttest_ind
        _, p = ttest_ind(p4_scores, base_scores, equal_var=False)
        p4_mean = statistics.mean(p4_scores)
        base_mean = statistics.mean(base_scores)
        mean_diff = p4_mean - base_mean
        if p < 0.05 and mean_diff > 0:
            print(f"  Zero wins, but Phase 4 mean ({p4_mean:+.4f}) significantly HIGHER "
                  f"than baseline ({base_mean:+.4f}), p={p:.4g} -> CONTINUE TUNING")
        elif p < 0.05 and mean_diff < 0:
            print(f"  Zero wins, Phase 4 mean ({p4_mean:+.4f}) significantly LOWER "
                  f"than baseline ({base_mean:+.4f}), p={p:.4g}, shift {mean_diff:+.4f}")
            print(f"  --> Expected-value reformulation actively damaged performance.")
            print(f"  --> PHASE 5 PIVOT to tensor networks.")
        else:
            print(f"  Zero wins, Phase 4 mean ({p4_mean:+.4f}) statistically indistinguishable "
                  f"from baseline ({base_mean:+.4f}), p={p:.4g} -> PHASE 5 PIVOT.")
    else:
        print(f"  Zero wins, no comparable minimax baseline in DB. "
              f"Default to PHASE 5 PIVOT.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
