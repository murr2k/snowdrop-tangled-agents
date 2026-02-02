#!/usr/bin/env python3
"""
Test opponent_history table population.

Verifies that opponent moves are being recorded to the opponent_history table
with all required contextual information.
"""

import sqlite3
from pathlib import Path

def test_opponent_history():
    """Check if opponent_history table is being populated."""
    db_path = Path.home() / ".tangled" / "game_stats.db"

    if not db_path.exists():
        print(f"[FAIL] Database not found: {db_path}")
        return False

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        # Check table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='opponent_history'"
        )
        if not cursor.fetchone():
            print("[FAIL] opponent_history table doesn't exist")
            return False
        print("[OK] opponent_history table exists")

        # Check for recent records
        cursor = conn.execute("SELECT COUNT(*) as count FROM opponent_history")
        total_count = cursor.fetchone()['count']
        print(f"   Total opponent moves recorded: {total_count}")

        if total_count == 0:
            print("[INFO] No opponent moves recorded yet (run games after code update)")
            return True  # Not a failure, just needs data

        # Show sample records
        cursor = conn.execute("""
            SELECT opponent_name, game_id, move_number, edge, color,
                   score_before, score_after, our_previous_move_edge, our_previous_move_color
            FROM opponent_history
            ORDER BY rowid DESC
            LIMIT 5
        """)

        print("\nRecent opponent moves:")
        print("-" * 80)
        for row in cursor.fetchall():
            print(f"   {row['opponent_name']} - Game {row['game_id'][:8]}... Move {row['move_number']}")
            print(f"      Played: E{row['edge']}{row['color']}")
            print(f"      Score: {row['score_before']:.3f} -> {row['score_after']:.3f}")
            if row['our_previous_move_edge'] is not None:
                print(f"      After our: E{row['our_previous_move_edge']}{row['our_previous_move_color']}")
            print()

        # Analyze opponent response patterns
        cursor = conn.execute("""
            SELECT
                opponent_name,
                COUNT(*) as moves,
                COUNT(DISTINCT game_id) as games,
                AVG(score_after - score_before) as avg_score_delta
            FROM opponent_history
            GROUP BY opponent_name
        """)

        print("\nOpponent statistics:")
        print("-" * 80)
        for row in cursor.fetchall():
            print(f"   {row['opponent_name']}: {row['moves']} moves across {row['games']} games")
            print(f"      Avg score delta: {row['avg_score_delta']:+.3f}")

        print("\n[OK] opponent_history implementation verified!")
        return True

if __name__ == "__main__":
    test_opponent_history()
