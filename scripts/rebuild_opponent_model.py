#!/usr/bin/env python3
"""Rebuild the AlphaQ opponent model from oracle-era game data only.

Oracle era: games played after the oracle + correct SA LUT were deployed.
We identify oracle-era games by rowid >= MIN_ORACLE_ROWID (3199, first game
played after commit b7422de "Validate oracle, fix bugs, remove wrong-LUT-era
biases").

Note: games.id is a hex TEXT primary key, NOT a sequential integer. Use rowid.

Usage:
    python scripts/rebuild_opponent_model.py [--min-rowid N] [--dry-run]
"""
import argparse
import sqlite3
import pathlib
import sys

# rowid of the first game played with oracle active (commit b7422de)
DEFAULT_MIN_ORACLE_ROWID = 3199

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-rowid", type=int, default=DEFAULT_MIN_ORACLE_ROWID,
                        help=f"First rowid to include (default: {DEFAULT_MIN_ORACLE_ROWID})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be built without saving")
    parser.add_argument("--include-all", action="store_true",
                        help="Include all games with policy decay instead of hard cutoff")
    args = parser.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from snowdrop_tangled_agents.stats.opponent_model import OpponentModel
    from snowdrop_tangled_agents.stats.collector import DEFAULT_DB_PATH

    db_path = DEFAULT_DB_PATH
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # Report what's in the DB
    total = conn.execute("SELECT COUNT(*) FROM games WHERE opponent LIKE '%lpha%'").fetchone()[0]
    oracle_era = conn.execute(
        "SELECT COUNT(*) FROM games WHERE opponent LIKE '%lpha%' AND rowid >= ?",
        (args.min_rowid,)
    ).fetchone()[0]

    policy_rows = conn.execute(
        "SELECT policy_id, COUNT(*) cnt, MIN(rowid) first_rowid, MAX(rowid) last_rowid, "
        "MIN(timestamp) first_ts, MAX(timestamp) last_ts "
        "FROM games WHERE opponent LIKE '%lpha%' GROUP BY policy_id ORDER BY first_rowid DESC LIMIT 15"
    ).fetchall()

    print(f"\nAlphaQ games in DB: {total} total, {oracle_era} oracle-era (rowid >= {args.min_rowid})")
    print("\nMost recent policies:")
    for r in policy_rows:
        marker = " <-- oracle era" if (r['last_rowid'] >= args.min_rowid) else ""
        print(f"  policy={str(r['policy_id'])[:12]:12s} games={r['cnt']:4d} "
              f"rowids={r['first_rowid']}-{r['last_rowid']} "
              f"({r['first_ts'][:10]}){marker}")

    conn.close()

    if args.include_all:
        # Use policy weights to decay old games
        print(f"\nBuilding model from ALL {total} games with policy-based decay...")
        model = OpponentModel("alphaq")
        model.load_from_database(db_path)
    else:
        # Hard cutoff: only oracle-era games, all weighted equally
        print(f"\nBuilding model from {oracle_era} oracle-era games only (rowid >= {args.min_rowid})...")
        model = _build_from_rowid_cutoff(db_path, args.min_rowid)

    model.print_summary()

    if args.dry_run:
        print("\n[dry-run] Not saving.")
        return

    out_path = pathlib.Path(__file__).parent.parent / \
               "snowdrop_tangled_agents" / "matlab" / "rl" / "data" / "opponent_model_alphaq.mat"
    model.save_mat(out_path)
    print(f"\nSaved to {out_path}")


def _build_from_rowid_cutoff(db_path, min_rowid):
    """Build OpponentModel using only games with rowid >= min_rowid."""
    import sqlite3
    from snowdrop_tangled_agents.stats.opponent_model import OpponentModel

    model = OpponentModel("alphaq")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        games = conn.execute(
            "SELECT id FROM games WHERE opponent LIKE '%lpha%' AND rowid >= ? ORDER BY rowid",
            (min_rowid,)
        ).fetchall()

        model.total_games = len(games)
        print(f"  Processing {model.total_games} games...")

        for game in games:
            game_id = game['id']
            moves = conn.execute(
                "SELECT move_number, player, edge, color, state_after "
                "FROM moves WHERE game_id = ? ORDER BY move_number, player",
                (game_id,)
            ).fetchall()

            our_last_move = None
            last_grey_count = 15

            for move in moves:
                player = move['player']
                edge = move['edge']
                color = move['color']
                state_after = move['state_after'] or ''

                if player == 'us':
                    our_last_move = (edge, color)
                    if state_after:
                        last_grey_count = state_after.count('-')
                elif player == 'opponent':
                    if our_last_move is not None:
                        model.update(our_last_move, (edge, color), last_grey_count, weight=1.0)
                    if state_after:
                        last_grey_count = state_after.count('-')

    finally:
        conn.close()

    return model


if __name__ == "__main__":
    main()
