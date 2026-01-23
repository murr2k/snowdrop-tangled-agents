"""
Statistics collection and analysis for Tangled games.

This module provides tools to collect game statistics and analyze
patterns to improve strategy.

Extended for MATLAB integration:
- Neural network model metadata storage
- Opponent profile tracking
- Training data versioning
- Database migrations

Usage:
    from snowdrop_tangled_agents.stats import StatsCollector, get_collector

    # Use global collector
    collector = get_collector()
    game_id = collector.start_game(opponent="melissa")
    collector.record_move(game_id, 1, "us", 9, "G", 1.02)
    collector.end_game(game_id, "win", 2.04)

    # Model management
    collector.save_model("value_net_v1", "value", 100, 0.05, "/path/to/model.mat")
    model = collector.get_active_model("value")

    # Opponent profiles
    collector.save_opponent("melissa", cluster_id=1, features=[...])
    opponent = collector.get_opponent("melissa")

    # Analysis
    from snowdrop_tangled_agents.stats import queries
    queries.print_summary()
"""

from .collector import StatsCollector, get_collector, DEFAULT_DB_PATH
from .migrations import run_migrations, get_migration_status
from .opponent_model import OpponentModel, get_opponent_model
from . import queries

__all__ = [
    'StatsCollector',
    'get_collector',
    'DEFAULT_DB_PATH',
    'run_migrations',
    'get_migration_status',
    'queries',
    'OpponentModel',
    'get_opponent_model',
]
