"""
Statistics collection and analysis for Tangled games.

This module provides tools to collect game statistics and analyze
patterns to improve strategy.

Usage:
    from snowdrop_tangled_agents.stats import StatsCollector, get_collector

    # Use global collector
    collector = get_collector()
    game_id = collector.start_game(opponent="melissa")
    collector.record_move(game_id, 1, "us", 9, "G", 1.02)
    collector.end_game(game_id, "win", 2.04)

    # Analysis
    from snowdrop_tangled_agents.stats import queries
    queries.print_summary()
"""

from .collector import StatsCollector, get_collector, DEFAULT_DB_PATH
from . import queries

__all__ = [
    'StatsCollector',
    'get_collector',
    'DEFAULT_DB_PATH',
    'queries',
]
