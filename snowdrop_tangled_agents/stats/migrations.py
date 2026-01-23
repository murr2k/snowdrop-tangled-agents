"""
Database schema migrations for MATLAB integration.

Adds support for:
- Neural network model metadata storage
- Opponent profile tracking
- Training data versioning

Migration versioning is tracked in a schema_version table.
"""

import sqlite3
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Migration definitions: (version, description, sql)
MIGRATIONS: List[Tuple[int, str, str]] = [
    # v1: Initial schema (handled by collector._init_database)
    # v2: Add models table for neural network metadata
    (2, "Add models table for neural network metadata", """
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            type TEXT NOT NULL,
            created DATETIME DEFAULT CURRENT_TIMESTAMP,
            training_games INTEGER,
            validation_loss REAL,
            file_path TEXT,
            hyperparameters TEXT,
            active BOOLEAN DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_models_type ON models(type);
        CREATE INDEX IF NOT EXISTS idx_models_active ON models(active);
    """),

    # v3: Add opponents table for opponent modeling
    (3, "Add opponents table for opponent modeling", """
        CREATE TABLE IF NOT EXISTS opponents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            cluster_id INTEGER,
            games_played INTEGER DEFAULT 0,
            win_rate REAL,
            features TEXT,
            last_updated DATETIME,
            notes TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_opponents_cluster ON opponents(cluster_id);
    """),

    # v4: Add training_data table for versioned training samples
    (4, "Add training_data table for versioned samples", """
        CREATE TABLE IF NOT EXISTS training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            version INTEGER NOT NULL,
            created DATETIME DEFAULT CURRENT_TIMESTAMP,
            source_game_id TEXT REFERENCES games(id),
            move_number INTEGER,
            features TEXT NOT NULL,
            target_value REAL,
            target_policy TEXT,
            quality_score REAL DEFAULT 1.0
        );

        CREATE INDEX IF NOT EXISTS idx_training_version ON training_data(version);
        CREATE INDEX IF NOT EXISTS idx_training_game ON training_data(source_game_id);
    """),

    # v5: Add opponent_history table for detailed move tracking
    (5, "Add opponent_history table for move pattern analysis", """
        CREATE TABLE IF NOT EXISTS opponent_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            opponent_name TEXT NOT NULL,
            game_id TEXT REFERENCES games(id),
            move_number INTEGER,
            board_state_before TEXT,
            edge INTEGER,
            color TEXT,
            score_before REAL,
            score_after REAL,
            our_previous_move_edge INTEGER,
            our_previous_move_color TEXT,
            UNIQUE(game_id, move_number)
        );

        CREATE INDEX IF NOT EXISTS idx_opponent_hist_name ON opponent_history(opponent_name);
        CREATE INDEX IF NOT EXISTS idx_opponent_hist_edge ON opponent_history(edge, color);
    """),

    # v6: Add comprehensive solver statistics to moves table
    (6, "Add solver statistics columns to moves table", """
        -- Strategy attribution
        ALTER TABLE moves ADD COLUMN strategy_used TEXT;

        -- Evaluation accuracy
        ALTER TABLE moves ADD COLUMN predicted_score REAL;
        ALTER TABLE moves ADD COLUMN prediction_error REAL;

        -- MCTS statistics (mcts_iterations already exists)
        ALTER TABLE moves ADD COLUMN mcts_tree_depth INTEGER;
        ALTER TABLE moves ADD COLUMN mcts_root_visits INTEGER;

        -- Minimax statistics
        ALTER TABLE moves ADD COLUMN minimax_nodes_searched INTEGER;
        ALTER TABLE moves ADD COLUMN minimax_prune_count INTEGER;
        ALTER TABLE moves ADD COLUMN minimax_depth INTEGER;

        -- Transposition table statistics
        ALTER TABLE moves ADD COLUMN trans_hits INTEGER;
        ALTER TABLE moves ADD COLUMN trans_misses INTEGER;

        -- Tabu search statistics
        ALTER TABLE moves ADD COLUMN tabu_restarts INTEGER;
        ALTER TABLE moves ADD COLUMN tabu_improved BOOLEAN;

        -- LUT statistics
        ALTER TABLE moves ADD COLUMN lut_used BOOLEAN;
        ALTER TABLE moves ADD COLUMN lut_grey_edges INTEGER;

        -- Move confidence (score gap to second-best)
        ALTER TABLE moves ADD COLUMN move_confidence REAL;
        ALTER TABLE moves ADD COLUMN second_best_edge INTEGER;
        ALTER TABLE moves ADD COLUMN second_best_score REAL;

        -- Timing (thinking_time already exists for our moves)
        ALTER TABLE moves ADD COLUMN opponent_think_time REAL;
        ALTER TABLE moves ADD COLUMN wall_clock_time REAL;

        -- Create indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_moves_strategy ON moves(strategy_used);
        CREATE INDEX IF NOT EXISTS idx_moves_lut_used ON moves(lut_used);
    """),

    # v7: Add policy tracking and opponent model metrics
    (7, "Add policy_id and opponent model learning metrics", """
        -- Policy version tracking
        ALTER TABLE games ADD COLUMN policy_id TEXT;
        CREATE INDEX IF NOT EXISTS idx_games_policy_id ON games(policy_id);

        -- Opponent model metrics at game start
        -- These track what the model knew before the game, enabling learning trajectory analysis
        ALTER TABLE games ADD COLUMN model_entropy REAL;
        ALTER TABLE games ADD COLUMN model_top3_hit REAL;
        ALTER TABLE games ADD COLUMN prediction_accuracy REAL;

        -- Additional model diagnostics
        ALTER TABLE games ADD COLUMN model_games_learned INTEGER;
        ALTER TABLE games ADD COLUMN model_moves_learned INTEGER;
    """),
]


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current schema version from database."""
    try:
        cursor = conn.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row[0] if row else 1  # v1 if no version table
    except sqlite3.OperationalError:
        # schema_version table doesn't exist - this is v1
        return 1


def create_version_table(conn: sqlite3.Connection):
    """Create schema_version table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            applied DATETIME DEFAULT CURRENT_TIMESTAMP,
            description TEXT
        )
    """)
    conn.commit()


def record_migration(conn: sqlite3.Connection, version: int, description: str):
    """Record a migration as applied."""
    conn.execute(
        "INSERT OR REPLACE INTO schema_version (version, description) VALUES (?, ?)",
        (version, description)
    )
    conn.commit()


def run_migrations(conn: sqlite3.Connection) -> int:
    """
    Run all pending migrations.

    Args:
        conn: SQLite database connection

    Returns:
        Number of migrations applied
    """
    create_version_table(conn)
    current_version = get_schema_version(conn)
    migrations_applied = 0

    for version, description, sql in MIGRATIONS:
        if version > current_version:
            logger.info(f"Applying migration v{version}: {description}")
            try:
                conn.executescript(sql)
                record_migration(conn, version, description)
                migrations_applied += 1
                logger.info(f"Migration v{version} applied successfully")
            except Exception as e:
                logger.error(f"Migration v{version} failed: {e}")
                raise

    if migrations_applied == 0:
        logger.debug(f"Database schema is up to date (v{current_version})")
    else:
        logger.info(f"Applied {migrations_applied} migrations (now at v{get_schema_version(conn)})")

    return migrations_applied


def get_migration_status(conn: sqlite3.Connection) -> dict:
    """
    Get migration status information.

    Returns:
        Dict with schema version, pending migrations, etc.
    """
    create_version_table(conn)
    current_version = get_schema_version(conn)

    # Get applied migrations
    cursor = conn.execute("SELECT version, applied, description FROM schema_version ORDER BY version")
    applied = [{"version": r[0], "applied": r[1], "description": r[2]} for r in cursor.fetchall()]

    # Get pending migrations
    pending = [(v, d) for v, d, _ in MIGRATIONS if v > current_version]

    return {
        "current_version": current_version,
        "latest_version": MIGRATIONS[-1][0] if MIGRATIONS else 1,
        "applied_migrations": applied,
        "pending_migrations": pending,
        "is_up_to_date": len(pending) == 0
    }
