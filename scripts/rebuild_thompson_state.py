#!/usr/bin/env python3
"""Rebuild AlphaQ Thompson sampling state from oracle-era game data only.

The state file (~/.tangled/alphaq_explorer_state.json) accumulates W/D/L counts
across all games ever played. Pre-oracle games dominate (~1467 games), diluting
the signal from oracle-era results (e.g. E13P looks near-uniform at 0.477 mean
despite being 0W/0D/2L in oracle era).

This script rebuilds the state from oracle-era games only (rowid >= MIN_ORACLE_ROWID),
preserving the same v2 format.

Usage:
    python scripts/rebuild_thompson_state.py [--min-rowid N] [--dry-run]
"""
import argparse
import json
import pathlib
import sqlite3
import sys
from datetime import datetime

DEFAULT_MIN_ORACLE_ROWID = 3199

STATE_PATH = pathlib.Path.home() / ".tangled" / "alphaq_explorer_state.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-rowid", type=int, default=DEFAULT_MIN_ORACLE_ROWID,
                        help=f"First rowid to include (default: {DEFAULT_MIN_ORACLE_ROWID})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be written without saving")
    parser.add_argument("--state-path", type=pathlib.Path, default=STATE_PATH,
                        help=f"Path to state file (default: {STATE_PATH})")
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from snowdrop_tangled_agents.stats.collector import DEFAULT_DB_PATH

    db_path = DEFAULT_DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Show what's in the DB
    total_alphaq = conn.execute(
        "SELECT COUNT(*) FROM games WHERE opponent LIKE '%lpha%'"
    ).fetchone()[0]
    oracle_era = conn.execute(
        "SELECT COUNT(*) FROM games WHERE opponent LIKE '%lpha%' AND rowid >= ?",
        (args.min_rowid,)
    ).fetchone()[0]

    print(f"\nAlphaQ games: {total_alphaq} total, {oracle_era} oracle-era (rowid >= {args.min_rowid})")

    # Build W/D/L counts from oracle-era games
    # Initialize all 30 openings to zero
    openings = {f"E{e}{c}": {'wins': 0, 'draws': 0, 'losses': 0}
                for e in range(15) for c in ['G', 'P']}

    games = conn.execute(
        "SELECT g.id, g.result, g.rowid "
        "FROM games g "
        "WHERE g.opponent LIKE '%lpha%' AND g.rowid >= ? "
        "ORDER BY g.rowid",
        (args.min_rowid,)
    ).fetchall()

    skipped = 0
    for game in games:
        game_id = game['id']
        result = game['result']

        if result not in ('win', 'draw', 'loss'):
            skipped += 1
            continue

        # First 'us' move determines the opening
        first_move = conn.execute(
            "SELECT edge, color FROM moves "
            "WHERE game_id = ? AND player = 'us' "
            "ORDER BY move_number ASC LIMIT 1",
            (game_id,)
        ).fetchone()

        if first_move is None:
            skipped += 1
            continue

        key = f"E{first_move['edge']}{first_move['color']}"
        if key not in openings:
            openings[key] = {'wins': 0, 'draws': 0, 'losses': 0}

        field = {'win': 'wins', 'draw': 'draws', 'loss': 'losses'}[result]
        openings[key][field] += 1

    conn.close()

    # Report
    print(f"\nOracle-era openings ({oracle_era - skipped} games, {skipped} skipped):")
    tested = [(k, v) for k, v in openings.items()
              if v['wins'] + v['draws'] + v['losses'] > 0]
    tested.sort(key=lambda x: x[1]['losses'] - x[1]['wins'])
    for k, v in tested:
        total = v['wins'] + v['draws'] + v['losses']
        alpha = 1 + v['wins'] + 0.5 * v['draws']
        beta = 1 + v['losses'] + 0.5 * v['draws']
        mean = alpha / (alpha + beta)
        print(f"  {k}: {v['wins']}W/{v['draws']}D/{v['losses']}L  mean={mean:.3f}")

    untested = [k for k, v in openings.items()
                if v['wins'] + v['draws'] + v['losses'] == 0]
    print(f"\n  {len(untested)} untested openings -> Beta(1,1), mean=0.5 each")

    # Load current state to show what rr_index to preserve
    rr_index = 0
    if args.state_path.exists():
        try:
            with open(args.state_path) as f:
                old_data = json.load(f)
            rr_index = old_data.get('rr_index', 0)
            old_played = old_data.get('games_played', 0)
            print(f"\nCurrent state: {old_played} games (will be replaced with {oracle_era - skipped})")
        except Exception as e:
            print(f"\nCould not read current state: {e}")

    games_played = sum(v['wins'] + v['draws'] + v['losses'] for v in openings.values())
    new_data = {
        'version': 2,
        'openings': openings,
        'games_played': games_played,
        'rr_index': rr_index,
        'last_updated': datetime.now().isoformat(),
    }

    if args.dry_run:
        print("\n[dry-run] Not saving.")
        return

    args.state_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = args.state_path.with_suffix('.json.bak')
    if args.state_path.exists():
        import shutil
        shutil.copy2(args.state_path, backup_path)
        print(f"\nBacked up old state to {backup_path}")

    with open(args.state_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    print(f"Saved oracle-era Thompson state to {args.state_path}")


if __name__ == "__main__":
    main()
