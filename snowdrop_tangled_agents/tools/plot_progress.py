#!/usr/bin/env python3
"""
Generate progress plots for Tangled game statistics.

Plots are saved to the plots/ directory with naming scheme:
    {plot_type}_{YYYYMMDD}_{HHMMSS}.png

Plot types:
    - progress: Win rate and score over time
    - edge: Edge effectiveness analysis
    - opening: Opening sequence analysis

Usage:
    python -m snowdrop_tangled_agents.tools.plot_progress
    python -m snowdrop_tangled_agents.tools.plot_progress --type progress
    python -m snowdrop_tangled_agents.tools.plot_progress --type edge
    python -m snowdrop_tangled_agents.tools.plot_progress --all
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from snowdrop_tangled_agents.stats.collector import DEFAULT_DB_PATH

PLOTS_DIR = Path(__file__).parent.parent.parent / "plots"


def get_output_path(plot_type: str) -> Path:
    """Generate output path with timestamp.

    Format: {plot_type}_{YYYYMMDD}_{HHMMSS}.png
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PLOTS_DIR / f"{plot_type}_{timestamp}.png"


def plot_progress(db_path: Path = None, window: int = 20, v06_start: int = 212) -> Path:
    """Plot win rate and score progression over time.

    Args:
        db_path: Path to database
        window: Rolling window size for averages
        v06_start: Game number where v0.6.0 started

    Returns:
        Path to saved plot
    """
    db_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)

    cur = conn.execute('''
        SELECT timestamp, result, final_score
        FROM games
        WHERE result IS NOT NULL
        ORDER BY timestamp
    ''')
    games = cur.fetchall()
    conn.close()

    if not games:
        print("No completed games found")
        return None

    # Calculate metrics
    results = [1 if g[1] == 'win' else 0 for g in games]
    scores = [g[2] or 0 for g in games]

    rolling_wr = []
    rolling_score = []
    for i in range(len(results)):
        start = max(0, i - window + 1)
        rolling_wr.append(sum(results[start:i+1]) / (i - start + 1) * 100)
        rolling_score.append(np.mean(scores[start:i+1]))

    cumulative_wr = [sum(results[:i+1]) / (i+1) * 100 for i in range(len(results))]

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Win rate
    ax1.plot(cumulative_wr, 'b-', alpha=0.3, label='Cumulative')
    ax1.plot(rolling_wr, 'b-', linewidth=2, label=f'Rolling {window}-game')
    ax1.axhline(y=15, color='g', linestyle='--', alpha=0.5, label='Target 15%')
    if v06_start < len(games):
        ax1.axvline(x=v06_start, color='r', linestyle=':', alpha=0.5, label='v0.6.0 start')
    ax1.set_ylabel('Win Rate (%)')
    ax1.set_title(f'Win Rate Over Time (n={len(games)} games)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(40, max(rolling_wr) + 5))

    # Score
    ax2.plot(rolling_score, 'purple', linewidth=2, label=f'Rolling {window}-game avg')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    if v06_start < len(games):
        ax2.axvline(x=v06_start, color='r', linestyle=':', alpha=0.5, label='v0.6.0 start')
    ax2.set_xlabel('Game Number')
    ax2.set_ylabel('Average Score')
    ax2.set_title('Score Over Time')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = get_output_path("progress")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")
    print(f"Total games: {len(games)}")
    print(f"Current rolling WR: {rolling_wr[-1]:.1f}%")
    print(f"Current rolling score: {rolling_score[-1]:.3f}")

    return output_path


def plot_edge_effectiveness(db_path: Path = None, min_games: int = 3) -> Path:
    """Plot edge/color effectiveness.

    Args:
        db_path: Path to database
        min_games: Minimum games for edge to be included

    Returns:
        Path to saved plot
    """
    db_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)

    cur = conn.execute('''
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
        GROUP BY m.edge, m.color
        HAVING COUNT(DISTINCT g.id) >= ?
        ORDER BY m.edge, m.color
    ''', (min_games,))

    data = cur.fetchall()
    conn.close()

    if not data:
        print("Not enough data for edge analysis")
        return None

    # Organize by edge
    edges = {}
    for edge, color, times, avg_delta, wins, total in data:
        if edge not in edges:
            edges[edge] = {}
        edges[edge][color] = {
            'delta': avg_delta or 0,
            'win_rate': wins / total if total > 0 else 0,
            'games': total
        }

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    edge_nums = sorted(edges.keys())
    x = np.arange(len(edge_nums))
    width = 0.35

    green_deltas = [edges[e].get('G', {}).get('delta', 0) for e in edge_nums]
    purple_deltas = [edges[e].get('P', {}).get('delta', 0) for e in edge_nums]

    ax1.bar(x - width/2, green_deltas, width, label='Green', color='green', alpha=0.7)
    ax1.bar(x + width/2, purple_deltas, width, label='Purple', color='purple', alpha=0.7)
    ax1.set_xlabel('Edge')
    ax1.set_ylabel('Avg Score Delta')
    ax1.set_title('Score Delta by Edge/Color')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'E{e}' for e in edge_nums])
    ax1.legend()
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3, axis='y')

    green_wr = [edges[e].get('G', {}).get('win_rate', 0) * 100 for e in edge_nums]
    purple_wr = [edges[e].get('P', {}).get('win_rate', 0) * 100 for e in edge_nums]

    ax2.bar(x - width/2, green_wr, width, label='Green', color='green', alpha=0.7)
    ax2.bar(x + width/2, purple_wr, width, label='Purple', color='purple', alpha=0.7)
    ax2.set_xlabel('Edge')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_title('Win Rate by Edge/Color')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'E{e}' for e in edge_nums])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = get_output_path("edge")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")

    return output_path


def plot_opening_analysis(db_path: Path = None, num_moves: int = 3) -> Path:
    """Plot opening sequence analysis.

    Args:
        db_path: Path to database
        num_moves: Number of opening moves to analyze

    Returns:
        Path to saved plot
    """
    db_path = db_path or DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)

    cur = conn.execute('''
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
        HAVING COUNT(*) >= 3
        ORDER BY occurrences DESC
        LIMIT 15
    ''', (num_moves,))

    data = cur.fetchall()
    conn.close()

    if not data:
        print("Not enough data for opening analysis")
        return None

    sequences = [d[0] for d in data]
    occurrences = [d[1] for d in data]
    win_rates = [d[2] / d[1] * 100 if d[1] > 0 else 0 for d in data]
    avg_scores = [d[5] or 0 for d in data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    y = np.arange(len(sequences))

    # Win rate by opening
    colors = ['green' if wr > 15 else 'red' if wr < 10 else 'gray' for wr in win_rates]
    ax1.barh(y, win_rates, color=colors, alpha=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(sequences, fontsize=9)
    ax1.set_xlabel('Win Rate (%)')
    ax1.set_title(f'Win Rate by Opening Sequence (first {num_moves} moves)')
    ax1.axvline(x=15, color='g', linestyle='--', alpha=0.5, label='Target 15%')
    ax1.grid(True, alpha=0.3, axis='x')

    # Add occurrence counts
    for i, (wr, occ) in enumerate(zip(win_rates, occurrences)):
        ax1.text(wr + 0.5, i, f'n={occ}', va='center', fontsize=8)

    # Average score by opening
    colors2 = ['green' if s > 0 else 'red' for s in avg_scores]
    ax2.barh(y, avg_scores, color=colors2, alpha=0.7)
    ax2.set_yticks(y)
    ax2.set_yticklabels(sequences, fontsize=9)
    ax2.set_xlabel('Average Final Score')
    ax2.set_title('Average Score by Opening Sequence')
    ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    output_path = get_output_path("opening")
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved: {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate Tangled game progress plots")
    parser.add_argument("--type", "-t", choices=["progress", "edge", "opening"],
                       default="progress", help="Type of plot to generate")
    parser.add_argument("--all", "-a", action="store_true", help="Generate all plot types")
    parser.add_argument("--window", "-w", type=int, default=20,
                       help="Rolling window size for progress plot")
    args = parser.parse_args()

    if args.all:
        plot_progress(window=args.window)
        plot_edge_effectiveness()
        plot_opening_analysis()
    elif args.type == "progress":
        plot_progress(window=args.window)
    elif args.type == "edge":
        plot_edge_effectiveness()
    elif args.type == "opening":
        plot_opening_analysis()


if __name__ == "__main__":
    main()
