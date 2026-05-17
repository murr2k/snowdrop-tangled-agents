"""Identify which openings drive alphaq_explorer's 82.6% draw rate, and
contrast with what openings hybrid_solver direct picks.

Hypothesis: alphaq_explorer's only meaningful difference is that it
FORCES a specific opening (Thompson or round-robin), while hybrid_solver
direct lets MCTS flounder at grey=15 without an opening book (the
opening book is gated off for alphaq because no calibration_alphaq.mat).
If a small number of openings produce nearly all the draws, those
openings can be transferred to hybrid_solver via --oracle-override or
a simple opening_book change."""

import sqlite3
from pathlib import Path

conn = sqlite3.connect(str(Path.home() / ".tangled" / "game_stats.db"))
cur = conn.cursor()

print("=" * 70)
print("ALPHAQ_EXPLORER OPENING BREAKDOWN (P1 vs AlphaQ)")
print("=" * 70)
cur.execute("""
SELECT
  ('E' || CAST(g.opening_edge AS TEXT) || g.opening_color) AS opening,
  COUNT(*) as n,
  SUM(CASE WHEN g.result='win'  THEN 1 ELSE 0 END) as w,
  SUM(CASE WHEN g.result='draw' THEN 1 ELSE 0 END) as d,
  SUM(CASE WHEN g.result='loss' THEN 1 ELSE 0 END) as l,
  AVG(g.final_score) as mean_score,
  MIN(g.final_score), MAX(g.final_score)
FROM runs r JOIN games g ON g.run_id = r.id
WHERE r.opponent='alphaq' AND r.seat=1 AND r.strategy='alphaq_explorer'
  AND g.final_score IS NOT NULL AND g.opening_edge IS NOT NULL
GROUP BY opening
ORDER BY d DESC, w DESC
""")
print(f"\n  {'opening':>8}  {'n':>4}  {'W':>3} {'D':>4} {'L':>4}  "
      f"{'draw%':>6}  {'mean':>8}  {'min':>8}  {'max':>8}")
rows = cur.fetchall()
for r in rows:
    op, n, w, d, l, mean, lo, hi = r
    dpct = 100 * d / n if n else 0
    if n >= 5:  # only show openings tried at least 5 times
        print(f"  {op:>8}  {n:>4}  {w:>3} {d:>4} {l:>4}  {dpct:>5.1f}%  "
              f"{mean:+.4f}  {lo:+.4f}  {hi:+.4f}")
print(f"\n  (only openings with n >= 5 shown; {len([r for r in rows if r[1] >= 5])} of {len(rows)} qualified)")

print("\n" + "=" * 70)
print("HYBRID_SOLVER OPENING BREAKDOWN (P1 vs AlphaQ, excl run 141)")
print("=" * 70)
cur.execute("""
SELECT
  ('E' || CAST(g.opening_edge AS TEXT) || g.opening_color) AS opening,
  COUNT(*) as n,
  SUM(CASE WHEN g.result='win'  THEN 1 ELSE 0 END) as w,
  SUM(CASE WHEN g.result='draw' THEN 1 ELSE 0 END) as d,
  SUM(CASE WHEN g.result='loss' THEN 1 ELSE 0 END) as l,
  AVG(g.final_score) as mean_score
FROM runs r JOIN games g ON g.run_id = r.id
WHERE r.opponent='alphaq' AND r.seat=1 AND r.strategy='hybrid_solver'
  AND r.id != 141
  AND g.final_score IS NOT NULL AND g.opening_edge IS NOT NULL
GROUP BY opening
ORDER BY n DESC
""")
print(f"\n  {'opening':>8}  {'n':>4}  {'W':>3} {'D':>4} {'L':>4}  "
      f"{'draw%':>6}  {'mean':>8}")
for r in cur.fetchall():
    op, n, w, d, l, mean = r
    dpct = 100 * d / n if n else 0
    print(f"  {op:>8}  {n:>4}  {w:>3} {d:>4} {l:>4}  {dpct:>5.1f}%  {mean:+.4f}")

print("\n" + "=" * 70)
print("PHASE 4 (run 141) OPENING BREAKDOWN")
print("=" * 70)
cur.execute("""
SELECT
  ('E' || CAST(g.opening_edge AS TEXT) || g.opening_color) AS opening,
  COUNT(*) as n,
  SUM(CASE WHEN g.result='win'  THEN 1 ELSE 0 END) as w,
  SUM(CASE WHEN g.result='draw' THEN 1 ELSE 0 END) as d,
  SUM(CASE WHEN g.result='loss' THEN 1 ELSE 0 END) as l,
  AVG(g.final_score) as mean_score
FROM runs r JOIN games g ON g.run_id = r.id
WHERE r.id = 141 AND g.final_score IS NOT NULL AND g.opening_edge IS NOT NULL
GROUP BY opening
ORDER BY n DESC
""")
print(f"\n  {'opening':>8}  {'n':>4}  {'W':>3} {'D':>4} {'L':>4}  "
      f"{'draw%':>6}  {'mean':>8}")
for r in cur.fetchall():
    op, n, w, d, l, mean = r
    dpct = 100 * d / n if n else 0
    print(f"  {op:>8}  {n:>4}  {w:>3} {d:>4} {l:>4}  {dpct:>5.1f}%  {mean:+.4f}")

print("\n" + "=" * 70)
print("KEY QUESTION: which openings drive the alphaq_explorer draw rate?")
print("=" * 70)
cur.execute("""
SELECT
  ('E' || CAST(g.opening_edge AS TEXT) || g.opening_color) AS opening,
  COUNT(*) as n,
  SUM(CASE WHEN g.result='draw' THEN 1 ELSE 0 END) as d
FROM runs r JOIN games g ON g.run_id = r.id
WHERE r.opponent='alphaq' AND r.seat=1 AND r.strategy='alphaq_explorer'
  AND g.opening_edge IS NOT NULL
GROUP BY opening
HAVING n >= 10 AND d * 1.0 / n >= 0.80
ORDER BY d DESC
""")
print(f"\n  Openings tried >= 10 times with >= 80% draws:")
for r in cur.fetchall():
    op, n, d = r
    print(f"    {op}: n={n}, draws={d} ({100*d/n:.1f}%)")
