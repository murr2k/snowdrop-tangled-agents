#!/usr/bin/env python3
"""
Backfill policy_id for existing games.

Maps games to policy versions based on timestamps and known release points.
Run after migration v7 to populate historical data.

Usage:
    python -m snowdrop_tangled_agents.tools.backfill_policy_id
    python -m snowdrop_tangled_agents.tools.backfill_policy_id --dry-run
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

from snowdrop_tangled_agents.stats.collector import DEFAULT_DB_PATH

# Known policy release points (timestamp, policy_id)
# Games before this timestamp get this policy_id
POLICY_RELEASES = [
    # Format: (timestamp, policy_id, description)
    ("2026-01-22 23:34:00", "v0.6.0-bayesian-oracle", "Opponent modeling + calibrated feedback"),
    ("2026-01-21 00:00:00", "pre-opponent-modeling", "Before opponent modeling"),
    # Catch-all for very old games
    ("2000-01-01 00:00:00", "legacy", "Legacy games"),
]


def get_policy_for_timestamp(timestamp: str) -> str:
    """Determine policy_id based on game timestamp."""
    for release_time, policy_id, _ in POLICY_RELEASES:
        if timestamp >= release_time:
            return policy_id
    return "unknown"


def backfill_policy_id(db_path: Path = None, dry_run: bool = False) -> dict:
    """
    Backfill policy_id for games that don't have one.

    Args:
        db_path: Path to database
        dry_run: If True, show what would be done without making changes

    Returns:
        Dict with statistics about the backfill
    """
    db_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)

    # Find games without policy_id
    cursor = conn.execute("""
        SELECT id, timestamp FROM games
        WHERE policy_id IS NULL
        ORDER BY timestamp
    """)
    games_to_update = cursor.fetchall()

    stats = {
        'total_games': len(games_to_update),
        'by_policy': {},
        'dry_run': dry_run,
    }

    if not games_to_update:
        print("No games need backfilling.")
        return stats

    print(f"Found {len(games_to_update)} games without policy_id")
    print()

    # Group by policy for reporting
    updates = []
    for game_id, timestamp in games_to_update:
        policy_id = get_policy_for_timestamp(timestamp)
        updates.append((policy_id, game_id))
        stats['by_policy'][policy_id] = stats['by_policy'].get(policy_id, 0) + 1

    # Report what we'll do
    print("Policy assignments:")
    for policy_id, count in sorted(stats['by_policy'].items()):
        print(f"  {policy_id}: {count} games")
    print()

    if dry_run:
        print("DRY RUN - no changes made")
        return stats

    # Apply updates
    print("Applying updates...")
    conn.executemany(
        "UPDATE games SET policy_id = ? WHERE id = ?",
        updates
    )
    conn.commit()
    conn.close()

    print(f"Updated {len(updates)} games")
    return stats


def show_policy_distribution(db_path: Path = None):
    """Show current policy_id distribution."""
    db_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)

    cursor = conn.execute("""
        SELECT
            COALESCE(policy_id, '(null)') as policy,
            COUNT(*) as games,
            SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN result = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN result = 'draw' THEN 1 ELSE 0 END) as draws
        FROM games
        WHERE result IS NOT NULL
        GROUP BY policy_id
        ORDER BY MIN(timestamp)
    """)

    print("Current policy distribution:")
    print(f"{'Policy':<30} {'Games':>6} {'Wins':>6} {'Losses':>6} {'Draws':>6} {'Win%':>7}")
    print("-" * 70)

    for row in cursor.fetchall():
        policy, games, wins, losses, draws = row
        win_pct = wins / games * 100 if games > 0 else 0
        print(f"{policy:<30} {games:>6} {wins:>6} {losses:>6} {draws:>6} {win_pct:>6.1f}%")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Backfill policy_id for existing games")
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--show", "-s", action="store_true",
                       help="Show current policy distribution and exit")
    args = parser.parse_args()

    if args.show:
        show_policy_distribution()
        return

    backfill_policy_id(dry_run=args.dry_run)
    print()
    show_policy_distribution()


if __name__ == "__main__":
    main()
