"""Full historical breakdown of P1-vs-AlphaQ games by strategy, to test
whether the run-130 minimax baseline used in Phase 4 analysis was
representative or an outlier. User pointed out there are many historical
draws in our DB; need to verify whether those came from hybrid_solver or
from a different strategy class entirely."""

import sqlite3, statistics
from pathlib import Path

conn = sqlite3.connect(str(Path.home() / ".tangled" / "game_stats.db"))
cur = conn.cursor()

print("=== All P1 (seat=1) vs alphaq games, grouped by strategy ===\n")
cur.execute("""
SELECT r.strategy, COUNT(g.id),
       SUM(CASE WHEN g.result='win'  THEN 1 ELSE 0 END),
       SUM(CASE WHEN g.result='draw' THEN 1 ELSE 0 END),
       SUM(CASE WHEN g.result='loss' THEN 1 ELSE 0 END),
       AVG(g.final_score), MIN(g.final_score), MAX(g.final_score)
FROM runs r JOIN games g ON g.run_id = r.id
WHERE r.opponent='alphaq' AND r.seat=1 AND g.final_score IS NOT NULL
GROUP BY r.strategy
ORDER BY COUNT(g.id) DESC
""")
print(f"  {'strategy':>20}  {'n':>5}  {'W':>3} {'D':>4} {'L':>4}  {'mean':>8}  {'range':>20}")
for r in cur.fetchall():
    strat, n, w, d, l, mean, lo, hi = r
    draw_pct = 100 * d / n if n else 0
    print(f"  {strat:>20}  {n:>5}  {w:>3} {d:>4} {l:>4}  {mean:+.4f}  "
          f"[{lo:+.3f}, {hi:+.3f}]  draws={draw_pct:.1f}%")

print("\n=== hybrid_solver-only P1 vs alphaq games, by run ===\n")
cur.execute("""
SELECT r.id, r.started, r.lut_variant, COUNT(g.id),
       SUM(CASE WHEN g.result='win'  THEN 1 ELSE 0 END),
       SUM(CASE WHEN g.result='draw' THEN 1 ELSE 0 END),
       SUM(CASE WHEN g.result='loss' THEN 1 ELSE 0 END),
       AVG(g.final_score), MIN(g.final_score), MAX(g.final_score)
FROM runs r JOIN games g ON g.run_id = r.id
WHERE r.opponent='alphaq' AND r.seat=1 AND r.strategy='hybrid_solver'
  AND g.final_score IS NOT NULL
GROUP BY r.id
ORDER BY r.started DESC
LIMIT 40
""")
print(f"  {'run':>4}  {'started':>20}  {'lut':>8}  {'n':>3}  "
      f"{'W':>2} {'D':>3} {'L':>3}  {'mean':>8}  {'min':>8}  {'max':>8}")
for r in cur.fetchall():
    rid, started, lut, n, w, d, l, mean, lo, hi = r
    print(f"  {rid:>4}  {started:>20}  {str(lut):>8}  {n:>3}  "
          f"{w:>2} {d:>3} {l:>3}  {mean:+.4f}  {lo:+.4f}  {hi:+.4f}")

print("\n=== Aggregate hybrid_solver P1 stats (excluding Phase 4 run 141) ===")
cur.execute("""
SELECT g.result, g.final_score
FROM runs r JOIN games g ON g.run_id = r.id
WHERE r.opponent='alphaq' AND r.seat=1 AND r.strategy='hybrid_solver'
  AND r.id != 141 AND g.final_score IS NOT NULL
""")
rows = cur.fetchall()
scores = [s for _, s in rows]
w = sum(1 for r, _ in rows if r == 'win')
d = sum(1 for r, _ in rows if r == 'draw')
l = sum(1 for r, _ in rows if r == 'loss')
print(f"  n={len(rows)}  W/D/L={w}/{d}/{l}  draw_rate={100*d/len(rows):.1f}%")
if scores:
    print(f"  score: mean={statistics.mean(scores):+.4f}  "
          f"median={statistics.median(scores):+.4f}  "
          f"stdev={statistics.stdev(scores):.4f}  "
          f"range=[{min(scores):+.4f}, {max(scores):+.4f}]")
    bins = [(-2.0, -0.7), (-0.7, -0.5), (-0.5, -0.2), (-0.2, 0.2),
            (0.2, 0.5), (0.5, 0.7), (0.7, 1.0)]
    print("  histogram:")
    for lo, hi in bins:
        cnt = sum(1 for s in scores if lo <= s < hi)
        bar = "#" * min(cnt, 60)
        print(f"    [{lo:+.1f}, {hi:+.1f}): {cnt:4d} {bar}")
