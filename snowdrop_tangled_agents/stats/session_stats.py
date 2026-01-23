"""
Session-aware statistics report generator.

Generates live statistics for the current gaming session:
- If a game is in progress, includes all games from that session
- If no game is running, includes games from the most recent session

A "session" is defined as a contiguous group of games with timestamps
within SESSION_GAP_MINUTES of each other.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import statistics

from .collector import DEFAULT_DB_PATH


def utc_to_local(utc_str: str) -> datetime:
    """Convert UTC timestamp string from database to local datetime."""
    utc_dt = datetime.fromisoformat(utc_str.replace(' ', 'T')).replace(tzinfo=timezone.utc)
    return utc_dt.astimezone()


def format_local(utc_str: str) -> str:
    """Format UTC timestamp as local time string."""
    return utc_to_local(utc_str).strftime('%Y-%m-%d %H:%M')

# Gap in minutes that defines session boundary
SESSION_GAP_MINUTES = 30


@dataclass
class GameRecord:
    """Summary of a single game."""
    game_id: str
    timestamp: str
    opponent: str
    result: Optional[str]
    final_score: Optional[float]
    total_moves: Optional[int]
    strategy: str
    policy_id: Optional[str]
    model_entropy: Optional[float]
    model_top3_hit: Optional[float]
    prediction_accuracy: Optional[float]
    run_id: Optional[int] = None
    game_number: Optional[int] = None
    is_current: bool = False


@dataclass
class SessionStats:
    """Aggregated statistics for a session."""
    session_start: str
    session_end: str
    game_count: int
    completed_games: int
    in_progress: bool
    wins: int
    losses: int
    draws: int
    win_rate: float
    avg_score: Optional[float]
    median_score: Optional[float]
    min_score: Optional[float]
    max_score: Optional[float]
    score_std: Optional[float]
    avg_moves: Optional[float]
    avg_entropy: Optional[float]
    avg_top3_hit: Optional[float]
    avg_prediction_accuracy: Optional[float]
    games: list[GameRecord]


def get_session_boundary(db_path: Optional[Path] = None, gap_minutes: int = SESSION_GAP_MINUTES) -> tuple[Optional[str], bool]:
    """
    Determine the session boundary timestamp.

    Returns:
        Tuple of (session_start_timestamp, is_game_in_progress)
    """
    db_path = db_path or DEFAULT_DB_PATH

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT id, timestamp FROM games
            WHERE result IS NULL
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        in_progress = cursor.fetchone()

        if in_progress:
            game_id, current_ts = in_progress
            return _find_session_start(conn, current_ts, gap_minutes), True

        cursor = conn.execute("""
            SELECT timestamp FROM games
            WHERE result IS NOT NULL
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        most_recent = cursor.fetchone()

        if not most_recent:
            return None, False

        return _find_session_start(conn, most_recent[0], gap_minutes), False


def _find_session_start(conn, end_timestamp: str, gap_minutes: int = SESSION_GAP_MINUTES) -> str:
    """Walk backwards from end_timestamp to find session start."""
    cursor = conn.execute("""
        SELECT timestamp FROM games
        WHERE timestamp <= ?
        ORDER BY timestamp DESC
    """, (end_timestamp,))

    timestamps = [row[0] for row in cursor.fetchall()]

    if not timestamps:
        return end_timestamp

    session_start = timestamps[0]
    prev_time = datetime.fromisoformat(timestamps[0].replace(' ', 'T'))

    for ts in timestamps[1:]:
        current_time = datetime.fromisoformat(ts.replace(' ', 'T'))
        gap = prev_time - current_time

        if gap > timedelta(minutes=gap_minutes):
            break

        session_start = ts
        prev_time = current_time

    return session_start


def get_session_stats(db_path: Optional[Path] = None, gap_minutes: int = SESSION_GAP_MINUTES) -> Optional[SessionStats]:
    """Get statistics for the current or most recent session."""
    db_path = db_path or DEFAULT_DB_PATH

    session_start, in_progress = get_session_boundary(db_path, gap_minutes)

    if session_start is None:
        return None

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT
                id, timestamp, opponent, result, final_score, total_moves,
                strategy, policy_id, model_entropy, model_top3_hit,
                prediction_accuracy, run_id, game_number
            FROM games
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (session_start,))

        games = []
        for row in cursor.fetchall():
            games.append(GameRecord(
                game_id=row[0],
                timestamp=row[1],
                opponent=row[2],
                result=row[3],
                final_score=row[4],
                total_moves=row[5],
                strategy=row[6] or 'unknown',
                policy_id=row[7],
                model_entropy=row[8],
                model_top3_hit=row[9],
                prediction_accuracy=row[10],
                run_id=row[11],
                game_number=row[12],
                is_current=(row[3] is None)
            ))

        if not games:
            return None

        completed = [g for g in games if g.result is not None]
        scores = [g.final_score for g in completed if g.final_score is not None]
        moves = [g.total_moves for g in completed if g.total_moves is not None]
        entropies = [g.model_entropy for g in games if g.model_entropy is not None]
        top3_hits = [g.model_top3_hit for g in completed if g.model_top3_hit is not None]
        pred_accs = [g.prediction_accuracy for g in completed if g.prediction_accuracy is not None]

        wins = sum(1 for g in completed if g.result == 'win')
        losses = sum(1 for g in completed if g.result == 'loss')
        draws = sum(1 for g in completed if g.result == 'draw')

        return SessionStats(
            session_start=session_start,
            session_end=games[-1].timestamp,
            game_count=len(games),
            completed_games=len(completed),
            in_progress=in_progress,
            wins=wins,
            losses=losses,
            draws=draws,
            win_rate=wins / len(completed) if completed else 0.0,
            avg_score=statistics.mean(scores) if scores else None,
            median_score=statistics.median(scores) if scores else None,
            min_score=min(scores) if scores else None,
            max_score=max(scores) if scores else None,
            score_std=statistics.stdev(scores) if len(scores) > 1 else None,
            avg_moves=statistics.mean(moves) if moves else None,
            avg_entropy=statistics.mean(entropies) if entropies else None,
            avg_top3_hit=statistics.mean(top3_hits) if top3_hits else None,
            avg_prediction_accuracy=statistics.mean(pred_accs) if pred_accs else None,
            games=games
        )


def print_session_report(db_path: Optional[Path] = None, gap_minutes: int = SESSION_GAP_MINUTES, planned_games: Optional[int] = None):
    """Print a minimal session statistics report."""
    stats = get_session_stats(db_path, gap_minutes)

    if stats is None:
        print("No games found")
        return

    # Check for run info from the most recent game
    run_info = None
    if stats.games:
        latest_game = stats.games[-1]
        if latest_game.run_id:
            db_path = db_path or DEFAULT_DB_PATH
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT id, planned_games, completed_games FROM runs WHERE id = ?",
                    (latest_game.run_id,)
                )
                row = cursor.fetchone()
                if row:
                    run_info = {'id': row[0], 'planned_games': row[1], 'completed_games': row[2]}

    # Determine total games: run > --planned > session count
    if run_info:
        total_games = run_info['planned_games']
    elif planned_games:
        total_games = planned_games
    else:
        total_games = stats.game_count

    # Session info (convert UTC to local time for display)
    print(f"session_start = {format_local(stats.session_start)}")
    if stats.in_progress and stats.completed_games >= 2:
        # Estimate end time from play rate
        start_dt = utc_to_local(stats.session_start)
        last_dt = utc_to_local(stats.session_end)
        elapsed = (last_dt - start_dt).total_seconds()
        avg_per_game = elapsed / (stats.completed_games - 1) if stats.completed_games > 1 else 120
        games_remaining = total_games - stats.completed_games
        est_remaining = games_remaining * avg_per_game
        est_end = last_dt + timedelta(seconds=est_remaining)
        print(f"session_end = {est_end.strftime('%Y-%m-%d %H:%M')} (est)")
    elif stats.in_progress:
        print("session_end =")
    else:
        print(f"session_end = {format_local(stats.session_end)}")

    # Show run info if available
    if run_info:
        print(f"run = {run_info['id']}")
        print(f"games = {run_info['completed_games']}/{run_info['planned_games']}")
    else:
        print(f"games = {stats.completed_games}/{total_games}")

    # Results
    if stats.completed_games > 0:
        n = stats.completed_games
        print(f"wins = {stats.wins}, {stats.wins/n:.1%}")
        print(f"draws = {stats.draws}, {stats.draws/n:.1%}")
        print(f"losses = {stats.losses}, {stats.losses/n:.1%}")

    # Scores
    if stats.avg_score is not None:
        print(f"avg_score = {stats.avg_score:+.3f}")
        print(f"median_score = {stats.median_score:+.3f}")
        print(f"min_score = {stats.min_score:+.3f}")
        print(f"max_score = {stats.max_score:+.3f}")
        if stats.score_std is not None:
            print(f"score_std = {stats.score_std:.3f}")

    # Moves
    if stats.avg_moves is not None:
        print(f"avg_moves = {stats.avg_moves:.1f}")

    # Model metrics
    if stats.avg_entropy is not None:
        print(f"avg_entropy = {stats.avg_entropy:.3f}")
    if stats.avg_top3_hit is not None:
        print(f"avg_top3_hit = {stats.avg_top3_hit:.1%}")
    if stats.avg_prediction_accuracy is not None:
        print(f"avg_pred_accuracy = {stats.avg_prediction_accuracy:.3f}")

    # Trend analysis (need at least 4 completed games)
    if stats.completed_games >= 4:
        completed = [g for g in stats.games if g.result is not None and g.final_score is not None]
        if len(completed) >= 4:
            first_half = completed[:len(completed)//2]
            second_half = completed[len(completed)//2:]

            # Score trend
            first_avg = statistics.mean(g.final_score for g in first_half)
            second_avg = statistics.mean(g.final_score for g in second_half)
            print(f"score_trend = {second_avg - first_avg:+.3f}")

            # Win rate trend
            first_wins = sum(1 for g in first_half if g.result == 'win')
            second_wins = sum(1 for g in second_half if g.result == 'win')
            first_wr = first_wins / len(first_half) if first_half else 0
            second_wr = second_wins / len(second_half) if second_half else 0
            print(f"winrate_trend = {second_wr - first_wr:+.1%}")

            # Recent streak
            recent = completed[-5:] if len(completed) >= 5 else completed
            streak_str = ''.join('W' if g.result == 'win' else 'L' if g.result == 'loss' else 'D' for g in recent)
            print(f"recent_5 = {streak_str}")


def watch_session(db_path: Optional[Path] = None, interval: int = 60, gap_minutes: int = SESSION_GAP_MINUTES, planned_games: Optional[int] = None):
    """Watch session stats, refreshing at interval. Press q/Q/Esc to exit."""
    import os
    import sys
    import time

    # Platform-specific keyboard handling
    if os.name == 'nt':
        import msvcrt

        def check_exit_key():
            if msvcrt.kbhit():
                key = msvcrt.getch()
                # q, Q, or Esc
                if key in (b'q', b'Q', b'\x1b'):
                    return True
            return False
    else:
        import select
        import tty
        import termios

        old_settings = termios.tcgetattr(sys.stdin)

        def check_exit_key():
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key in ('q', 'Q', '\x1b'):
                    return True
            return False

        # Set terminal to raw mode for non-blocking input
        tty.setcbreak(sys.stdin.fileno())

    try:
        while True:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')

            # Fresh database read each time
            print_session_report(db_path, gap_minutes, planned_games)

            # Show refresh info
            print()
            print(f"[refreshes every {interval}s, press q to exit]")

            # Wait with periodic key checks
            for _ in range(interval * 10):
                if check_exit_key():
                    raise SystemExit
                time.sleep(0.1)

    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # Restore terminal on Unix
        if os.name != 'nt':
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\nExited.")


def cleanup_stale_games(db_path: Optional[Path] = None, dry_run: bool = True) -> int:
    """Mark stale in-progress games as abandoned."""
    db_path = db_path or DEFAULT_DB_PATH

    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("""
            SELECT id, timestamp FROM games WHERE result IS NULL
        """)
        stale = cursor.fetchall()

        if not stale:
            print("No stale games found")
            return 0

        print(f"Found {len(stale)} stale in-progress games:")
        for game_id, ts in stale:
            print(f"  {game_id} {ts}")

        if dry_run:
            print("\nDry run - no changes made. Use --cleanup --force to remove.")
            return len(stale)

        conn.execute("""
            UPDATE games SET result = 'abandoned', notes = 'cleaned up stale game'
            WHERE result IS NULL
        """)
        conn.commit()
        print(f"\nMarked {len(stale)} games as abandoned")
        return len(stale)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Session statistics report")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch mode, refresh every minute")
    parser.add_argument("--interval", "-i", type=int, default=60, help="Refresh interval in seconds (default: 60)")
    parser.add_argument("--gap", "-g", type=int, default=SESSION_GAP_MINUTES, help=f"Session gap in minutes (default: {SESSION_GAP_MINUTES})")
    parser.add_argument("--planned", "-p", type=int, help="Planned total games (for accurate end time estimate)")
    parser.add_argument("--cleanup", action="store_true", help="Clean up stale in-progress games")
    parser.add_argument("--force", action="store_true", help="Actually perform cleanup (not dry run)")
    parser.add_argument("--db", type=Path, help="Database path")
    args = parser.parse_args()

    if args.cleanup:
        cleanup_stale_games(args.db, dry_run=not args.force)
    elif args.watch:
        watch_session(args.db, args.interval, args.gap, args.planned)
    elif args.json:
        import json
        stats = get_session_stats(args.db, args.gap)
        if stats:
            data = {
                'session_start': stats.session_start,
                'session_end': stats.session_end,
                'game_count': stats.game_count,
                'completed_games': stats.completed_games,
                'in_progress': stats.in_progress,
                'wins': stats.wins,
                'losses': stats.losses,
                'draws': stats.draws,
                'win_rate': stats.win_rate,
                'avg_score': stats.avg_score,
                'median_score': stats.median_score,
                'min_score': stats.min_score,
                'max_score': stats.max_score,
                'score_std': stats.score_std,
                'avg_moves': stats.avg_moves,
                'avg_entropy': stats.avg_entropy,
                'avg_top3_hit': stats.avg_top3_hit,
                'avg_prediction_accuracy': stats.avg_prediction_accuracy,
                'games': [
                    {
                        'game_id': g.game_id,
                        'timestamp': g.timestamp,
                        'result': g.result,
                        'final_score': g.final_score,
                        'total_moves': g.total_moves,
                        'is_current': g.is_current
                    }
                    for g in stats.games
                ]
            }
            print(json.dumps(data, indent=2))
        else:
            print(json.dumps({'error': 'No games found'}))
    else:
        print_session_report(args.db, args.gap, args.planned)


if __name__ == "__main__":
    main()
