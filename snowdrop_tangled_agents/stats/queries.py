"""
Analysis queries for game statistics.

Provides functions to extract insights from collected game data,
including edge effectiveness, winning patterns, and opponent analysis.
"""

import sqlite3
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .collector import DEFAULT_DB_PATH, get_collector, connect_db


@dataclass
class EdgeStats:
    """Statistics for a specific edge/color combination."""
    edge: int
    color: str
    times_played: int
    avg_delta: float
    win_rate: float
    total_wins: int
    total_games: int


@dataclass
class MovePattern:
    """A move pattern with its outcome statistics."""
    move_number: int
    edge: int
    color: str
    state_before: str
    occurrences: int
    wins: int
    losses: int
    draws: int
    avg_final_score: float


def get_edge_effectiveness(
    db_path: Optional[Path] = None,
    opponent: Optional[str] = None,
    min_games: int = 3
) -> list[EdgeStats]:
    """
    Get effectiveness statistics for each edge/color combination.

    Args:
        db_path: Path to database
        opponent: Filter by opponent (None for all)
        min_games: Minimum games to include in results

    Returns:
        List of EdgeStats sorted by average score delta (best first)
    """
    db_path = db_path or DEFAULT_DB_PATH

    query = """
        SELECT
            m.edge,
            m.color,
            COUNT(*) as times_played,
            AVG(m.score_delta) as avg_delta,
            SUM(CASE WHEN g.result = 'win' THEN 1 ELSE 0 END) as wins,
            COUNT(DISTINCT g.id) as total_games
        FROM moves m
        JOIN games g ON m.game_id = g.id
        WHERE m.player = 'us'
          AND g.result IS NOT NULL
    """

    params = []
    if opponent:
        query += " AND g.opponent = ?"
        params.append(opponent)

    query += """
        GROUP BY m.edge, m.color
        HAVING COUNT(DISTINCT g.id) >= ?
        ORDER BY avg_delta DESC
    """
    params.append(min_games)

    with connect_db(db_path) as conn:
        cursor = conn.execute(query, params)
        results = []
        for row in cursor.fetchall():
            edge, color, times, avg_delta, wins, total = row
            results.append(EdgeStats(
                edge=edge,
                color=color,
                times_played=times,
                avg_delta=avg_delta or 0.0,
                win_rate=wins / total if total > 0 else 0.0,
                total_wins=wins,
                total_games=total
            ))
        return results


def get_winning_patterns(
    db_path: Optional[Path] = None,
    move_number: Optional[int] = None,
    min_occurrences: int = 2
) -> list[MovePattern]:
    """
    Find move patterns that lead to wins.

    Args:
        db_path: Path to database
        move_number: Filter by specific move number (None for all)
        min_occurrences: Minimum times pattern must occur

    Returns:
        List of MovePattern sorted by win rate
    """
    db_path = db_path or DEFAULT_DB_PATH

    query = """
        SELECT
            m.move_number,
            m.edge,
            m.color,
            m.state_after,
            COUNT(*) as occurrences,
            SUM(CASE WHEN g.result = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN g.result = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN g.result = 'draw' THEN 1 ELSE 0 END) as draws,
            AVG(g.final_score) as avg_final
        FROM moves m
        JOIN games g ON m.game_id = g.id
        WHERE m.player = 'us'
          AND g.result IS NOT NULL
    """

    params = []
    if move_number is not None:
        query += " AND m.move_number = ?"
        params.append(move_number)

    query += """
        GROUP BY m.move_number, m.edge, m.color
        HAVING COUNT(*) >= ?
        ORDER BY (wins * 1.0 / COUNT(*)) DESC, wins DESC
    """
    params.append(min_occurrences)

    with connect_db(db_path) as conn:
        cursor = conn.execute(query, params)
        results = []
        for row in cursor.fetchall():
            results.append(MovePattern(
                move_number=row[0],
                edge=row[1],
                color=row[2],
                state_before=row[3] or "",
                occurrences=row[4],
                wins=row[5],
                losses=row[6],
                draws=row[7],
                avg_final_score=row[8] or 0.0
            ))
        return results


def get_score_progression(
    db_path: Optional[Path] = None,
    result: Optional[str] = None
) -> dict[int, dict]:
    """
    Get average score by move number, grouped by game result.

    Args:
        db_path: Path to database
        result: Filter by result ('win', 'loss', 'draw', or None for all)

    Returns:
        Dict mapping move_number -> {'avg_score': float, 'count': int}
    """
    db_path = db_path or DEFAULT_DB_PATH

    query = """
        SELECT
            m.move_number,
            AVG(m.score_after) as avg_score,
            COUNT(*) as count
        FROM moves m
        JOIN games g ON m.game_id = g.id
        WHERE m.player = 'us'
          AND g.result IS NOT NULL
    """

    params = []
    if result:
        query += " AND g.result = ?"
        params.append(result)

    query += " GROUP BY m.move_number ORDER BY m.move_number"

    with connect_db(db_path) as conn:
        cursor = conn.execute(query, params)
        return {
            row[0]: {'avg_score': row[1], 'count': row[2]}
            for row in cursor.fetchall()
        }


def get_opening_sequences(
    db_path: Optional[Path] = None,
    num_moves: int = 4,
    min_occurrences: int = 2
) -> list[dict]:
    """
    Find common opening sequences and their outcomes.

    Args:
        db_path: Path to database
        num_moves: Number of opening moves to consider
        min_occurrences: Minimum times sequence must occur

    Returns:
        List of dicts with sequence info and win/loss/draw counts
    """
    db_path = db_path or DEFAULT_DB_PATH

    query = """
        WITH opening_moves AS (
            SELECT
                m.game_id,
                GROUP_CONCAT(
                    'E' || m.edge || m.color
                    ORDER BY m.move_number
                ) as sequence
            FROM moves m
            WHERE m.player = 'us'
              AND m.move_number <= ?
            GROUP BY m.game_id
        )
        SELECT
            o.sequence,
            COUNT(*) as occurrences,
            SUM(CASE WHEN g.result = 'win' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN g.result = 'loss' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN g.result = 'draw' THEN 1 ELSE 0 END) as draws,
            AVG(g.final_score) as avg_final
        FROM opening_moves o
        JOIN games g ON o.game_id = g.id
        WHERE g.result IS NOT NULL
        GROUP BY o.sequence
        HAVING COUNT(*) >= ?
        ORDER BY (wins * 1.0 / COUNT(*)) DESC, occurrences DESC
    """

    with connect_db(db_path) as conn:
        cursor = conn.execute(query, (num_moves, min_occurrences))
        return [
            {
                'sequence': row[0],
                'occurrences': row[1],
                'wins': row[2],
                'losses': row[3],
                'draws': row[4],
                'avg_final_score': row[5],
                'win_rate': row[2] / row[1] if row[1] > 0 else 0
            }
            for row in cursor.fetchall()
        ]


def get_critical_positions(
    db_path: Optional[Path] = None,
    score_swing_threshold: float = 0.5
) -> list[dict]:
    """
    Find positions where large score swings occurred.

    Args:
        db_path: Path to database
        score_swing_threshold: Minimum absolute score change to consider

    Returns:
        List of critical position details
    """
    db_path = db_path or DEFAULT_DB_PATH

    query = """
        SELECT
            m.game_id,
            m.move_number,
            m.edge,
            m.color,
            m.score_delta,
            m.state_after,
            g.result,
            g.final_score
        FROM moves m
        JOIN games g ON m.game_id = g.id
        WHERE m.player = 'us'
          AND ABS(m.score_delta) >= ?
          AND g.result IS NOT NULL
        ORDER BY ABS(m.score_delta) DESC
        LIMIT 50
    """

    with connect_db(db_path) as conn:
        cursor = conn.execute(query, (score_swing_threshold,))
        return [
            {
                'game_id': row[0],
                'move_number': row[1],
                'edge': row[2],
                'color': row[3],
                'score_delta': row[4],
                'state_after': row[5],
                'result': row[6],
                'final_score': row[7]
            }
            for row in cursor.fetchall()
        ]


def get_opponent_patterns(
    db_path: Optional[Path] = None,
    opponent: str = "melissa"
) -> dict:
    """
    Analyze opponent's move patterns.

    Args:
        db_path: Path to database
        opponent: Opponent name to analyze

    Returns:
        Dict with opponent analysis
    """
    db_path = db_path or DEFAULT_DB_PATH

    with connect_db(db_path) as conn:
        # First move preferences
        cursor = conn.execute("""
            SELECT m.edge, m.color, COUNT(*) as count
            FROM moves m
            JOIN games g ON m.game_id = g.id
            WHERE m.player = 'opponent'
              AND m.move_number = 1
              AND g.opponent = ?
            GROUP BY m.edge, m.color
            ORDER BY count DESC
            LIMIT 5
        """, (opponent,))
        first_moves = [
            {'edge': r[0], 'color': r[1], 'count': r[2]}
            for r in cursor.fetchall()
        ]

        # Response to our E9 Green opening
        cursor = conn.execute("""
            SELECT m.edge, m.color, COUNT(*) as count
            FROM moves m
            JOIN games g ON m.game_id = g.id
            WHERE m.player = 'opponent'
              AND m.move_number = 1
              AND g.opponent = ?
              AND EXISTS (
                  SELECT 1 FROM moves m2
                  WHERE m2.game_id = m.game_id
                    AND m2.player = 'us'
                    AND m2.move_number = 1
                    AND m2.edge = 9
                    AND m2.color = 'G'
              )
            GROUP BY m.edge, m.color
            ORDER BY count DESC
            LIMIT 5
        """, (opponent,))
        responses_to_e9 = [
            {'edge': r[0], 'color': r[1], 'count': r[2]}
            for r in cursor.fetchall()
        ]

        # Overall edge preferences
        cursor = conn.execute("""
            SELECT m.edge, m.color, COUNT(*) as count
            FROM moves m
            JOIN games g ON m.game_id = g.id
            WHERE m.player = 'opponent'
              AND g.opponent = ?
            GROUP BY m.edge, m.color
            ORDER BY count DESC
            LIMIT 10
        """, (opponent,))
        edge_preferences = [
            {'edge': r[0], 'color': r[1], 'count': r[2]}
            for r in cursor.fetchall()
        ]

    return {
        'opponent': opponent,
        'first_moves': first_moves,
        'responses_to_e9_green': responses_to_e9,
        'edge_preferences': edge_preferences
    }


def get_calibration_summary(db_path: Optional[Path] = None) -> dict:
    """
    Get summary statistics for adjudicator calibration.

    Returns:
        Dict with calibration metrics
    """
    db_path = db_path or DEFAULT_DB_PATH

    with connect_db(db_path) as conn:
        # Check if calibration table exists and has data
        cursor = conn.execute("""
            SELECT COUNT(*) FROM sqlite_master
            WHERE type='table' AND name='calibration'
        """)
        if cursor.fetchone()[0] == 0:
            return {'count': 0}

        cursor = conn.execute("""
            SELECT
                COUNT(*) as count,
                AVG(error) as mean_error,
                AVG(abs_error) as mean_abs_error,
                MAX(abs_error) as max_abs_error,
                MIN(error) as min_error,
                MAX(error) as max_error,
                AVG(website_score) as avg_website_score,
                AVG(predicted_score) as avg_predicted_score
            FROM calibration
        """)
        row = cursor.fetchone()

        if row[0] == 0:
            return {'count': 0}

        # Get error distribution
        cursor = conn.execute("""
            SELECT
                CASE
                    WHEN abs_error < 0.01 THEN 'exact'
                    WHEN abs_error < 0.1 THEN 'close'
                    WHEN abs_error < 0.5 THEN 'moderate'
                    ELSE 'large'
                END as error_category,
                COUNT(*) as count
            FROM calibration
            GROUP BY error_category
        """)
        error_dist = {r[0]: r[1] for r in cursor.fetchall()}

        return {
            'count': row[0],
            'mean_error': row[1] or 0,
            'mean_abs_error': row[2] or 0,
            'max_abs_error': row[3] or 0,
            'min_error': row[4] or 0,
            'max_error': row[5] or 0,
            'avg_website_score': row[6] or 0,
            'avg_predicted_score': row[7] or 0,
            'error_distribution': error_dist
        }


def get_calibration_details(
    db_path: Optional[Path] = None,
    limit: int = 20
) -> list[dict]:
    """
    Get detailed calibration records.

    Args:
        db_path: Path to database
        limit: Maximum records to return

    Returns:
        List of calibration records sorted by error magnitude
    """
    db_path = db_path or DEFAULT_DB_PATH

    with connect_db(db_path) as conn:
        # Check if calibration table exists
        cursor = conn.execute("""
            SELECT COUNT(*) FROM sqlite_master
            WHERE type='table' AND name='calibration'
        """)
        if cursor.fetchone()[0] == 0:
            return []

        cursor = conn.execute("""
            SELECT
                c.game_id,
                c.terminal_state,
                c.website_score,
                c.predicted_score,
                c.error,
                c.abs_error,
                g.result,
                c.timestamp
            FROM calibration c
            LEFT JOIN games g ON c.game_id = g.id
            ORDER BY c.abs_error DESC
            LIMIT ?
        """, (limit,))

        return [
            {
                'game_id': row[0],
                'terminal_state': row[1],
                'website_score': row[2],
                'predicted_score': row[3],
                'error': row[4],
                'abs_error': row[5],
                'result': row[6],
                'timestamp': row[7]
            }
            for row in cursor.fetchall()
        ]


def print_calibration_report(db_path: Optional[Path] = None):
    """Print a detailed calibration analysis report."""
    db_path = db_path or DEFAULT_DB_PATH

    summary = get_calibration_summary(db_path)

    print("\n" + "=" * 60)
    print("ADJUDICATOR CALIBRATION REPORT")
    print("=" * 60)

    if summary['count'] == 0:
        print("\nNo calibration data collected yet.")
        print("Play games to collect terminal state comparisons.")
        print("=" * 60)
        return

    print(f"\nTotal Calibrations: {summary['count']}")
    print(f"\nError Metrics (predicted - website):")
    print(f"  Mean Error:     {summary['mean_error']:+.4f}")
    print(f"  Mean Abs Error: {summary['mean_abs_error']:.4f}")
    print(f"  Max Abs Error:  {summary['max_abs_error']:.4f}")
    print(f"  Error Range:    [{summary['min_error']:+.4f}, {summary['max_error']:+.4f}]")

    print(f"\nScore Averages:")
    print(f"  Website Avg:    {summary['avg_website_score']:+.4f}")
    print(f"  Predicted Avg:  {summary['avg_predicted_score']:+.4f}")

    dist = summary.get('error_distribution', {})
    if dist:
        total = sum(dist.values())
        print(f"\nError Distribution:")
        for cat, count in sorted(dist.items()):
            pct = count / total * 100 if total > 0 else 0
            bar = "#" * int(pct / 2)
            label = {
                'exact': '< 0.01 (exact)',
                'close': '< 0.10 (close)',
                'moderate': '< 0.50 (moderate)',
                'large': '>= 0.50 (large)'
            }.get(cat, cat)
            print(f"  {label:20s} {count:3d} ({pct:5.1f}%) {bar}")

    # Show worst calibrations
    print("\n" + "-" * 60)
    print("LARGEST CALIBRATION ERRORS")
    print("-" * 60)

    details = get_calibration_details(db_path, limit=10)
    if details:
        print(f"{'State':<17} {'Website':>8} {'Predict':>8} {'Error':>8} {'Result':>6}")
        for d in details:
            state = d['terminal_state'][:15] if d['terminal_state'] else '?'*15
            result = (d['result'] or '?')[:6]
            print(f"{state:<17} {d['website_score']:+8.4f} {d['predicted_score']:+8.4f} {d['error']:+8.4f} {result:>6}")

    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)

    mean_abs = summary['mean_abs_error']
    if mean_abs < 0.05:
        print("Excellent calibration - predictions closely match website.")
    elif mean_abs < 0.2:
        print("Good calibration - minor systematic differences.")
    elif mean_abs < 0.5:
        print("Moderate calibration - noticeable prediction errors.")
        print("Consider investigating large-error cases.")
    else:
        print("POOR CALIBRATION - significant prediction errors!")
        print("Our terminal state evaluation may be fundamentally wrong.")
        print("This could explain poor game performance.")

    mean_err = summary['mean_error']
    if abs(mean_err) > 0.1:
        if mean_err > 0:
            print(f"\nSystematic OVER-prediction by {mean_err:+.3f}")
            print("We think positions are better than they are.")
        else:
            print(f"\nSystematic UNDER-prediction by {mean_err:+.3f}")
            print("We think positions are worse than they are.")

    print("\n" + "=" * 60)


def print_summary(db_path: Optional[Path] = None):
    """Print a summary of collected statistics."""
    db_path = db_path or DEFAULT_DB_PATH
    collector = get_collector(db_path)

    counts = collector.get_game_count()
    print("\n" + "=" * 60)
    print("GAME STATISTICS SUMMARY")
    print("=" * 60)

    print(f"\nTotal Games: {counts['total']}")
    print(f"  Wins:   {counts['wins']} ({counts['wins']/counts['total']*100:.1f}%)" if counts['total'] > 0 else "  Wins:   0")
    print(f"  Losses: {counts['losses']} ({counts['losses']/counts['total']*100:.1f}%)" if counts['total'] > 0 else "  Losses: 0")
    print(f"  Draws:  {counts['draws']} ({counts['draws']/counts['total']*100:.1f}%)" if counts['total'] > 0 else "  Draws:  0")

    if counts['total'] > 0:
        print("\n" + "-" * 60)
        print("TOP EDGE/COLOR COMBINATIONS (by avg score delta)")
        print("-" * 60)
        edges = get_edge_effectiveness(db_path, min_games=1)[:10]
        print(f"{'Edge':<8} {'Color':<8} {'Avg Delta':<12} {'Win Rate':<12} {'Games':<8}")
        for e in edges:
            print(f"E{e.edge:<7} {e.color:<8} {e.avg_delta:+.3f}       {e.win_rate:.1%}        {e.total_games}")

        print("\n" + "-" * 60)
        print("SCORE PROGRESSION (Wins vs Losses)")
        print("-" * 60)
        win_prog = get_score_progression(db_path, result='win')
        loss_prog = get_score_progression(db_path, result='loss')

        print(f"{'Move':<8} {'Win Avg':<12} {'Loss Avg':<12}")
        for move in range(1, 9):
            win_score = win_prog.get(move, {}).get('avg_score', 0)
            loss_score = loss_prog.get(move, {}).get('avg_score', 0)
            print(f"{move:<8} {win_score:+.3f}        {loss_score:+.3f}")

    print("\n" + "=" * 60)
