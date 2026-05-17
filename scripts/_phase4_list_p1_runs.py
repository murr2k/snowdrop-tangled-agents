import sqlite3, statistics
from pathlib import Path
conn = sqlite3.connect(str(Path.home() / ".tangled" / "game_stats.db"))
cur = conn.cursor()
cur.execute("""
SELECT id, started, planned_games, completed_games, strategy, opponent, seat, lut_variant
FROM runs WHERE opponent='alphaq' AND seat=1
ORDER BY started DESC LIMIT 30
""")
print("Recent P1-vs-alphaq runs (id, started, planned, done, strategy, opp, seat, lut):")
for r in cur.fetchall():
    print(" ", r)
print()
# For each run, summarise outcomes
cur.execute("""
SELECT r.id, r.lut_variant, COUNT(g.id),
       SUM(CASE WHEN g.result='win' THEN 1 ELSE 0 END),
       SUM(CASE WHEN g.result='draw' THEN 1 ELSE 0 END),
       SUM(CASE WHEN g.result='loss' THEN 1 ELSE 0 END),
       AVG(g.final_score), MIN(g.final_score), MAX(g.final_score)
FROM runs r JOIN games g ON g.run_id = r.id
WHERE r.opponent='alphaq' AND r.seat=1 AND g.final_score IS NOT NULL
GROUP BY r.id
ORDER BY r.started DESC
LIMIT 30
""")
print("Per-run outcome summary (id, lut, n, W, D, L, mean, min, max):")
for r in cur.fetchall():
    rid, lut, n, w, d, l, mean, lo, hi = r
    print(f"  run {rid:>4}: lut={lut!s:>8}  n={n:>3}  W/D/L={w}/{d}/{l}  "
          f"mean={mean:+.4f}  range=[{lo:+.4f}, {hi:+.4f}]")
