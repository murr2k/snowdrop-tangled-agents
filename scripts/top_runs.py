#!/usr/bin/env python3
"""List top 5 runs by game count with strategy and W/D/L stats."""

import sqlite3
from pathlib import Path

DB_PATH = Path.home() / ".tangled" / "game_stats.db"


def main():
    conn = sqlite3.connect(DB_PATH)

    cursor = conn.execute("""
        SELECT
            r.id,
            r.planned_games,
            r.completed_games,
            r.strategy,
            SUM(CASE WHEN g.result = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN g.result = 'draw' THEN 1 ELSE 0 END) as draws,
            SUM(CASE WHEN g.result = 'loss' THEN 1 ELSE 0 END) as losses
        FROM runs r
        LEFT JOIN games g ON g.run_id = r.id AND g.result IS NOT NULL
        GROUP BY r.id
        ORDER BY r.completed_games DESC
        LIMIT 5
    """)

    print(f"{'Run':<6} {'Games':<12} {'Strategy':<15} {'Wins':<8} {'Draws':<8} {'Losses':<8} {'Win%':<8}")
    print("-" * 73)

    for row in cursor:
        run_id, planned, completed, strategy, wins, draws, losses = row
        wins = wins or 0
        draws = draws or 0
        losses = losses or 0
        total = wins + draws + losses
        win_pct = f"{100*wins/total:.1f}%" if total > 0 else "N/A"

        print(f"{run_id:<6} {completed}/{planned:<10} {strategy or 'unknown':<15} {wins:<8} {draws:<8} {losses:<8} {win_pct:<8}")

    conn.close()


if __name__ == "__main__":
    main()
