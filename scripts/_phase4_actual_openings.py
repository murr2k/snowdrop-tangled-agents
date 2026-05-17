"""What openings did Phase 4 (run 141) actually play? Compare to the
historical hybrid_solver openings too. The opening_edge column isn't
populated for those, so query the moves table at move_number=1."""

import sqlite3
from collections import Counter
from pathlib import Path

conn = sqlite3.connect(str(Path.home() / ".tangled" / "game_stats.db"))
cur = conn.cursor()

def opening_dist(label, where_clause, params=()):
    print(f"\n=== {label}")
    cur.execute(f"""
    SELECT m.edge, m.color, COUNT(*) as n,
           SUM(CASE WHEN g.result='win'  THEN 1 ELSE 0 END) as w,
           SUM(CASE WHEN g.result='draw' THEN 1 ELSE 0 END) as d,
           SUM(CASE WHEN g.result='loss' THEN 1 ELSE 0 END) as l,
           AVG(g.final_score) as mean_score
    FROM runs r
    JOIN games g ON g.run_id = r.id
    JOIN moves m ON m.game_id = g.id
    WHERE m.move_number = 1 AND m.player = 'us'
      AND {where_clause}
    GROUP BY m.edge, m.color
    ORDER BY n DESC
    """, params)
    rows = cur.fetchall()
    total = sum(r[2] for r in rows)
    print(f"  total games with first-move data: {total}")
    print(f"  {'opening':>8}  {'n':>4}  {'W':>3} {'D':>4} {'L':>4}  "
          f"{'draw%':>6}  {'mean':>8}")
    for edge, color, n, w, d, l, mean in rows:
        op = f"E{edge}{color}"
        dpct = 100 * d / n if n else 0
        print(f"  {op:>8}  {n:>4}  {w:>3} {d:>4} {l:>4}  {dpct:>5.1f}%  {mean:+.4f}")

opening_dist(
    "Phase 4 (run 141, expected mode)",
    "r.id = 141")

opening_dist(
    "hybrid_solver direct, P1 vs alphaq (excl run 141)",
    "r.opponent='alphaq' AND r.seat=1 AND r.strategy='hybrid_solver' AND r.id != 141")

# Cross-strategy: do all P1 vs alphaq E7G games (any strategy) draw heavily?
print("\n=== E7G across ALL strategies (P1 vs alphaq) ===")
cur.execute("""
SELECT r.strategy, COUNT(*) as n,
       SUM(CASE WHEN g.result='win'  THEN 1 ELSE 0 END),
       SUM(CASE WHEN g.result='draw' THEN 1 ELSE 0 END),
       SUM(CASE WHEN g.result='loss' THEN 1 ELSE 0 END),
       AVG(g.final_score)
FROM runs r
JOIN games g ON g.run_id = r.id
JOIN moves m ON m.game_id = g.id
WHERE m.move_number = 1 AND m.player = 'us'
  AND m.edge = 7 AND m.color = 'G'
  AND r.opponent = 'alphaq' AND r.seat = 1
GROUP BY r.strategy
ORDER BY n DESC
""")
for r in cur.fetchall():
    s, n, w, d, l, mean = r
    print(f"  strategy={s:>20}  n={n:>3}  W/D/L={w}/{d}/{l}  mean={mean:+.4f}")
