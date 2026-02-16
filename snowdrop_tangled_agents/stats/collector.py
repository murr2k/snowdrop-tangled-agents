"""
SQLite-based statistics collector for Tangled game play.

Collects game outcomes, move sequences, and score progressions
for analysis and strategy improvement.

Extended for MATLAB integration:
- Neural network model metadata storage
- Opponent profile tracking
- Training data versioning
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

from .migrations import run_migrations, get_migration_status

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path.home() / ".tangled" / "game_stats.db"


def connect_db(db_path=None):
    """Create a SQLite connection with WAL mode and busy timeout for concurrent access."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


class StatsCollector:
    """
    Collects and stores game statistics in SQLite.

    Usage:
        collector = StatsCollector()

        # Start a new game
        game_id = collector.start_game(opponent="melissa")

        # Record each move
        collector.record_move(
            game_id=game_id,
            move_number=1,
            player="us",
            edge=9,
            color="G",
            score_after=1.02,
            state_after="--------G------"
        )

        # End the game
        collector.end_game(game_id, result="win", final_score=2.04)
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the stats collector.

        Args:
            db_path: Path to SQLite database file. Defaults to ~/.tangled/game_stats.db
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()
        logger.info(f"Stats collector initialized: {self.db_path}")

    def _init_database(self):
        """Create database tables if they don't exist and run migrations."""
        with connect_db(self.db_path) as conn:
            # Create base tables (v1 schema)
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS games (
                    id TEXT PRIMARY KEY,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    opponent TEXT NOT NULL,
                    graph TEXT DEFAULT 'petersen',
                    result TEXT,  -- 'win', 'loss', 'draw', NULL if in progress
                    final_score REAL,
                    total_moves INTEGER,
                    strategy TEXT,  -- 'hybrid', 'mcts', 'heuristic'
                    mcts_time REAL,  -- MCTS time limit used
                    notes TEXT
                );

                CREATE TABLE IF NOT EXISTS moves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL REFERENCES games(id),
                    move_number INTEGER NOT NULL,
                    player TEXT NOT NULL,  -- 'us', 'opponent'
                    edge INTEGER NOT NULL,  -- 0-14
                    color TEXT NOT NULL,    -- 'G', 'P'
                    score_after REAL,
                    score_delta REAL,
                    state_after TEXT,       -- 15-char board state
                    mcts_iterations INTEGER,
                    thinking_time REAL,     -- seconds
                    UNIQUE(game_id, move_number, player)
                );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_moves_edge_color
                    ON moves(edge, color);
                CREATE INDEX IF NOT EXISTS idx_moves_game
                    ON moves(game_id);
                CREATE INDEX IF NOT EXISTS idx_moves_player
                    ON moves(player);
                CREATE INDEX IF NOT EXISTS idx_games_result
                    ON games(result);
                CREATE INDEX IF NOT EXISTS idx_games_opponent
                    ON games(opponent);

                -- Calibration data for adjudicator comparison
                CREATE TABLE IF NOT EXISTS calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT REFERENCES games(id),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    terminal_state TEXT NOT NULL,    -- 15-char final board state
                    website_score REAL NOT NULL,     -- Score from tangled-game.com
                    predicted_score REAL NOT NULL,   -- Our evaluate_terminal_state() result
                    error REAL NOT NULL,             -- predicted - website
                    abs_error REAL NOT NULL          -- |error|
                );

                CREATE INDEX IF NOT EXISTS idx_calibration_game
                    ON calibration(game_id);
            """)
            conn.commit()

            # Run schema migrations for v2+ features
            run_migrations(conn)

    def start_game(
        self,
        opponent: str = "melissa",
        graph: str = "petersen",
        strategy: str = "hybrid",
        mcts_time: float = 8.0,
        policy_id: str = None,
        model_metrics: dict = None,
        run_id: int = None,
        game_number: int = None
    ) -> str:
        """
        Start tracking a new game.

        Args:
            opponent: Opponent name
            graph: Graph type
            strategy: Strategy being used
            mcts_time: MCTS time limit
            policy_id: Code version identifier (auto-detected if None)
            model_metrics: Opponent model metrics dict from GameMetricsTracker
            run_id: Optional run ID for batch tracking
            game_number: Optional game number within the run

        Returns:
            Unique game ID
        """
        game_id = str(uuid.uuid4())[:8]  # Short ID for readability

        # Auto-detect policy_id if not provided
        if policy_id is None:
            try:
                from snowdrop_tangled_agents.utils.version import get_policy_id
                policy_id = get_policy_id()
            except Exception:
                policy_id = 'unknown'

        # Extract model metrics
        model_entropy = None
        model_games_learned = None
        model_moves_learned = None
        if model_metrics:
            model_entropy = model_metrics.get('model_entropy')
            model_games_learned = model_metrics.get('model_games_learned')
            model_moves_learned = model_metrics.get('model_moves_learned')

        with connect_db(self.db_path) as conn:
            conn.execute("""
                INSERT INTO games (
                    id, opponent, graph, strategy, mcts_time,
                    policy_id, model_entropy, model_games_learned, model_moves_learned,
                    run_id, game_number
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, opponent, graph, strategy, mcts_time,
                policy_id, model_entropy, model_games_learned, model_moves_learned,
                run_id, game_number
            ))
            conn.commit()

        run_info = f" (run {run_id}, game {game_number})" if run_id else ""
        logger.debug(f"Started game {game_id} vs {opponent}{run_info} (policy={policy_id})")
        return game_id

    def record_move(
        self,
        game_id: str,
        move_number: int,
        player: str,
        edge: int,
        color: str,
        score_after: float,
        score_before: Optional[float] = None,
        state_after: Optional[str] = None,
        mcts_iterations: Optional[int] = None,
        thinking_time: Optional[float] = None,
        solver_stats: Optional[Dict[str, Any]] = None
    ):
        """
        Record a move in the current game.

        Args:
            game_id: Game ID from start_game()
            move_number: Move number (1-based)
            player: 'us' or 'opponent'
            edge: Edge index (0-14)
            color: 'G' (green) or 'P' (purple)
            score_after: Score after this move
            score_before: Score before this move (for delta calculation)
            state_after: Board state after move (15-char string)
            mcts_iterations: Number of MCTS iterations used
            thinking_time: Time spent thinking in seconds
            solver_stats: Optional dict with detailed solver statistics
        """
        score_delta = None
        if score_before is not None:
            score_delta = score_after - score_before

        # Extract solver stats if provided
        stats = solver_stats or {}

        with connect_db(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO moves
                (game_id, move_number, player, edge, color, score_after,
                 score_delta, state_after, mcts_iterations, thinking_time,
                 strategy_used, predicted_score, prediction_error,
                 mcts_tree_depth, mcts_root_visits, mcts_simulations,
                 minimax_nodes_searched, minimax_prune_count, minimax_depth,
                 trans_hits, trans_misses, tabu_restarts, tabu_improved,
                 lut_used, lut_grey_edges, move_confidence,
                 second_best_edge, second_best_score,
                 opponent_think_time, wall_clock_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                game_id, move_number, player, edge, color, score_after,
                score_delta, state_after,
                mcts_iterations or stats.get('mcts_iterations'),
                thinking_time,
                stats.get('strategy'),
                stats.get('predicted_score'),
                stats.get('prediction_error'),
                stats.get('mcts_tree_depth'),
                stats.get('mcts_root_visits'),
                stats.get('mcts_simulations'),
                stats.get('minimax_nodes_searched'),
                stats.get('minimax_prune_count'),
                stats.get('minimax_depth'),
                stats.get('trans_hits'),
                stats.get('trans_misses'),
                stats.get('tabu_restarts'),
                stats.get('tabu_improved'),
                stats.get('lut_used'),
                stats.get('lut_grey_edges'),
                stats.get('move_confidence'),
                stats.get('second_best_edge'),
                stats.get('second_best_score'),
                stats.get('opponent_think_time'),
                stats.get('wall_clock_time')
            ))
            conn.commit()

        strategy = stats.get('strategy', 'unknown')
        logger.debug(f"Game {game_id}: Move {move_number} - E{edge} {color} [{strategy}] -> {score_after:.3f}")

    def end_game(
        self,
        game_id: str,
        result: str,
        final_score: float,
        notes: Optional[str] = None,
        model_metrics: dict = None,
        opening_edge: Optional[int] = None,
        opening_color: Optional[str] = None,
        opening_mode: Optional[str] = None,
        mcts_iterations_setting: Optional[int] = None,
    ):
        """
        Mark a game as complete.

        Args:
            game_id: Game ID from start_game()
            result: 'win', 'loss', or 'draw'
            final_score: Final game score
            notes: Optional notes about the game
            model_metrics: Opponent model metrics dict from GameMetricsTracker
            opening_edge: First-move edge index (0-14)
            opening_color: First-move color ('G' or 'P')
            opening_mode: Opening selection mode ('forced', 'thompson', 'round_robin')
            mcts_iterations_setting: MCTS iterations setting used for this game
        """
        # Extract prediction accuracy metrics
        model_top3_hit = None
        prediction_accuracy = None
        if model_metrics:
            model_top3_hit = model_metrics.get('model_top3_hit')
            prediction_accuracy = model_metrics.get('prediction_accuracy')

        with connect_db(self.db_path) as conn:
            # Count total moves
            cursor = conn.execute(
                "SELECT COUNT(*) FROM moves WHERE game_id = ? AND player = 'us'",
                (game_id,)
            )
            total_moves = cursor.fetchone()[0]

            # Get run_id for this game
            cursor = conn.execute(
                "SELECT run_id FROM games WHERE id = ?",
                (game_id,)
            )
            row = cursor.fetchone()
            run_id = row[0] if row else None

            conn.execute("""
                UPDATE games
                SET result = ?, final_score = ?, total_moves = ?, notes = ?,
                    model_top3_hit = ?, prediction_accuracy = ?,
                    opening_edge = ?, opening_color = ?, opening_mode = ?,
                    mcts_iterations_setting = ?
                WHERE id = ?
            """, (result, final_score, total_moves, notes,
                  model_top3_hit, prediction_accuracy,
                  opening_edge, opening_color, opening_mode,
                  mcts_iterations_setting, game_id))
            conn.commit()

            # Update run completed count if this game belongs to a run
            if run_id:
                self.update_run_completed(run_id)

        logger.info(f"Game {game_id} ended: {result} ({final_score:+.3f})")

    def record_calibration(
        self,
        game_id: str,
        terminal_state: str,
        website_score: float,
        predicted_score: float
    ):
        """
        Record calibration data comparing our prediction to website score.

        Args:
            game_id: Game ID from start_game()
            terminal_state: Final 15-char board state (all G/P, no dashes)
            website_score: Score displayed on tangled-game.com
            predicted_score: Our evaluate_terminal_state() result
        """
        error = predicted_score - website_score
        abs_error = abs(error)

        with connect_db(self.db_path) as conn:
            conn.execute("""
                INSERT INTO calibration
                (game_id, terminal_state, website_score, predicted_score, error, abs_error)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (game_id, terminal_state, website_score, predicted_score, error, abs_error))
            conn.commit()

        logger.info(f"Calibration: website={website_score:+.4f}, predicted={predicted_score:+.4f}, error={error:+.4f}")

    def get_game_count(self) -> dict:
        """Get count of games by result."""
        with connect_db(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT result, COUNT(*) as count
                FROM games
                WHERE result IS NOT NULL
                GROUP BY result
            """)
            results = {row[0]: row[1] for row in cursor.fetchall()}

        return {
            'wins': results.get('win', 0),
            'losses': results.get('loss', 0),
            'draws': results.get('draw', 0),
            'total': sum(results.values())
        }

    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection for custom queries."""
        return connect_db(self.db_path)

    # ========== Model Management Methods ==========

    def save_model(
        self,
        name: str,
        model_type: str,
        training_games: int,
        validation_loss: float,
        file_path: str,
        hyperparameters: Optional[Dict[str, Any]] = None,
        set_active: bool = False
    ) -> int:
        """
        Save neural network model metadata.

        Args:
            name: Model name (e.g., 'value_net_v1')
            model_type: 'value', 'policy', or 'opponent'
            training_games: Number of games used for training
            validation_loss: Final validation loss
            file_path: Path to saved model file
            hyperparameters: Training hyperparameters dict
            set_active: Whether to mark this model as active

        Returns:
            Model ID
        """
        hyperparams_json = json.dumps(hyperparameters) if hyperparameters else None

        with connect_db(self.db_path) as conn:
            # If setting active, deactivate other models of same type
            if set_active:
                conn.execute(
                    "UPDATE models SET active = 0 WHERE type = ?",
                    (model_type,)
                )

            cursor = conn.execute("""
                INSERT OR REPLACE INTO models
                (name, type, training_games, validation_loss, file_path, hyperparameters, active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (name, model_type, training_games, validation_loss,
                  file_path, hyperparams_json, set_active))
            conn.commit()

            model_id = cursor.lastrowid
            logger.info(f"Saved model '{name}' (type={model_type}, loss={validation_loss:.4f})")
            return model_id

    def get_active_model(self, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Get the currently active model of a given type.

        Args:
            model_type: 'value', 'policy', or 'opponent'

        Returns:
            Model info dict or None if no active model
        """
        with connect_db(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM models WHERE type = ? AND active = 1
            """, (model_type,))
            row = cursor.fetchone()

            if row:
                model = dict(row)
                if model.get('hyperparameters'):
                    model['hyperparameters'] = json.loads(model['hyperparameters'])
                return model
            return None

    def get_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all models, optionally filtered by type.

        Args:
            model_type: Filter by type, or None for all

        Returns:
            List of model info dicts
        """
        with connect_db(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if model_type:
                cursor = conn.execute(
                    "SELECT * FROM models WHERE type = ? ORDER BY created DESC",
                    (model_type,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM models ORDER BY created DESC"
                )

            models = []
            for row in cursor.fetchall():
                model = dict(row)
                if model.get('hyperparameters'):
                    model['hyperparameters'] = json.loads(model['hyperparameters'])
                models.append(model)
            return models

    # ========== Opponent Management Methods ==========

    def save_opponent(
        self,
        name: str,
        cluster_id: Optional[int] = None,
        features: Optional[List[float]] = None,
        win_rate: Optional[float] = None,
        notes: Optional[str] = None
    ) -> int:
        """
        Save or update opponent profile.

        Args:
            name: Opponent name (e.g., 'melissa', 'amara')
            cluster_id: Cluster assignment from k-means
            features: Feature vector (20 elements)
            win_rate: Win rate against this opponent
            notes: Optional notes

        Returns:
            Opponent ID
        """
        features_json = json.dumps(features) if features else None

        with connect_db(self.db_path) as conn:
            # Check if opponent exists
            cursor = conn.execute(
                "SELECT id, games_played FROM opponents WHERE name = ?",
                (name,)
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing opponent
                cursor = conn.execute("""
                    UPDATE opponents
                    SET cluster_id = COALESCE(?, cluster_id),
                        features = COALESCE(?, features),
                        win_rate = COALESCE(?, win_rate),
                        notes = COALESCE(?, notes),
                        last_updated = CURRENT_TIMESTAMP
                    WHERE name = ?
                """, (cluster_id, features_json, win_rate, notes, name))
                conn.commit()
                return existing[0]
            else:
                # Insert new opponent
                cursor = conn.execute("""
                    INSERT INTO opponents (name, cluster_id, features, win_rate, notes, last_updated)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (name, cluster_id, features_json, win_rate, notes))
                conn.commit()
                return cursor.lastrowid

    def get_opponent(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get opponent profile by name.

        Args:
            name: Opponent name

        Returns:
            Opponent info dict or None
        """
        with connect_db(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM opponents WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()

            if row:
                opponent = dict(row)
                if opponent.get('features'):
                    opponent['features'] = json.loads(opponent['features'])
                return opponent
            return None

    def get_opponents(self, cluster_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all opponents, optionally filtered by cluster.

        Args:
            cluster_id: Filter by cluster, or None for all

        Returns:
            List of opponent info dicts
        """
        with connect_db(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if cluster_id is not None:
                cursor = conn.execute(
                    "SELECT * FROM opponents WHERE cluster_id = ? ORDER BY name",
                    (cluster_id,)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM opponents ORDER BY name"
                )

            opponents = []
            for row in cursor.fetchall():
                opponent = dict(row)
                if opponent.get('features'):
                    opponent['features'] = json.loads(opponent['features'])
                opponents.append(opponent)
            return opponents

    def increment_opponent_games(self, name: str, won: bool):
        """
        Increment opponent's games played and update win rate.

        Args:
            name: Opponent name
            won: Whether we won this game
        """
        with connect_db(self.db_path) as conn:
            # Get current stats
            cursor = conn.execute(
                "SELECT games_played, win_rate FROM opponents WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()

            if row:
                games_played = (row[0] or 0) + 1
                current_wins = (row[1] or 0) * (row[0] or 0)
                new_wins = current_wins + (1 if won else 0)
                new_win_rate = new_wins / games_played

                conn.execute("""
                    UPDATE opponents
                    SET games_played = ?, win_rate = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE name = ?
                """, (games_played, new_win_rate, name))
            else:
                # Create new opponent entry
                conn.execute("""
                    INSERT INTO opponents (name, games_played, win_rate, last_updated)
                    VALUES (?, 1, ?, CURRENT_TIMESTAMP)
                """, (name, 1.0 if won else 0.0))

            conn.commit()

    def record_opponent_move(
        self,
        opponent_name: str,
        game_id: str,
        move_number: int,
        board_state_before: str,
        edge: int,
        color: str,
        score_before: float,
        score_after: float,
        our_prev_edge: Optional[int] = None,
        our_prev_color: Optional[str] = None
    ):
        """
        Record an opponent move for pattern analysis.

        Args:
            opponent_name: Opponent name
            game_id: Game ID
            move_number: Move number in game
            board_state_before: Board state before opponent's move
            edge: Edge index played
            color: Color played ('G' or 'P')
            score_before: Score before move
            score_after: Score after move
            our_prev_edge: Our previous move's edge (if any)
            our_prev_color: Our previous move's color (if any)
        """
        with connect_db(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO opponent_history
                (opponent_name, game_id, move_number, board_state_before,
                 edge, color, score_before, score_after,
                 our_previous_move_edge, our_previous_move_color)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (opponent_name, game_id, move_number, board_state_before,
                  edge, color, score_before, score_after,
                  our_prev_edge, our_prev_color))
            conn.commit()

    # ========== Training Data Methods ==========

    def save_training_sample(
        self,
        version: int,
        features: List[float],
        target_value: Optional[float] = None,
        target_policy: Optional[List[float]] = None,
        source_game_id: Optional[str] = None,
        move_number: Optional[int] = None,
        quality_score: float = 1.0
    ) -> int:
        """
        Save a training sample for neural network training.

        Args:
            version: Training data version
            features: Feature vector (50 elements)
            target_value: Target value for value network
            target_policy: Target policy for policy network (30 elements)
            source_game_id: Source game ID
            move_number: Move number in source game
            quality_score: Sample quality (for filtering)

        Returns:
            Sample ID
        """
        features_json = json.dumps(features)
        policy_json = json.dumps(target_policy) if target_policy else None

        with connect_db(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO training_data
                (version, features, target_value, target_policy,
                 source_game_id, move_number, quality_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (version, features_json, target_value, policy_json,
                  source_game_id, move_number, quality_score))
            conn.commit()
            return cursor.lastrowid

    def get_training_data(
        self,
        version: Optional[int] = None,
        min_quality: float = 0.0,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get training data samples.

        Args:
            version: Filter by version, or None for all
            min_quality: Minimum quality score
            limit: Maximum samples to return

        Returns:
            List of training sample dicts
        """
        with connect_db(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = "SELECT * FROM training_data WHERE quality_score >= ?"
            params = [min_quality]

            if version is not None:
                query += " AND version = ?"
                params.append(version)

            query += " ORDER BY created DESC"

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor = conn.execute(query, params)

            samples = []
            for row in cursor.fetchall():
                sample = dict(row)
                sample['features'] = json.loads(sample['features'])
                if sample.get('target_policy'):
                    sample['target_policy'] = json.loads(sample['target_policy'])
                samples.append(sample)
            return samples

    def get_migration_status(self) -> Dict[str, Any]:
        """Get database migration status."""
        with connect_db(self.db_path) as conn:
            return get_migration_status(conn)

    # ========== Run Management Methods ==========

    def start_run(
        self,
        planned_games: int,
        description: Optional[str] = None,
        strategy: Optional[str] = None,
        opponent: Optional[str] = None,
        seat: int = 1,
        random_turns: Optional[str] = None,
        novel_branch: bool = False,
    ) -> int:
        """
        Start a new run (batch of planned games).

        Args:
            planned_games: Total number of games planned for this run
            description: Optional description of the run
            strategy: Strategy being used
            opponent: Opponent being played
            seat: Player seat (1 or 2)
            random_turns: Comma-separated random turn indices (e.g. "0,2,4")
            novel_branch: Whether novel branch forcing is enabled

        Returns:
            Run ID
        """
        with connect_db(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO runs (planned_games, description, strategy, opponent, pid,
                                  seat, random_turns, novel_branch)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (planned_games, description, strategy, opponent, os.getpid(),
                  seat, random_turns, 1 if novel_branch else 0))
            conn.commit()
            run_id = cursor.lastrowid
            logger.info(f"Started run {run_id}: {planned_games} games planned "
                        f"(seat={seat}, random_turns={random_turns}, novel_branch={novel_branch})")
            return run_id

    def get_active_run(self, pid: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """
        Get the most recent incomplete run, optionally filtered by PID.

        Args:
            pid: Process ID to filter by. If provided, only returns runs
                 started by this PID. If None, returns any incomplete run.

        Returns:
            Run info dict or None if no active run
        """
        with connect_db(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if pid is not None:
                cursor = conn.execute("""
                    SELECT * FROM runs
                    WHERE completed_games < planned_games AND pid = ?
                    ORDER BY started DESC
                    LIMIT 1
                """, (pid,))
            else:
                cursor = conn.execute("""
                    SELECT * FROM runs
                    WHERE completed_games < planned_games
                    ORDER BY started DESC
                    LIMIT 1
                """)
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_run(self, run_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a run by ID.

        Args:
            run_id: Run ID

        Returns:
            Run info dict or None
        """
        with connect_db(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM runs WHERE id = ?",
                (run_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_next_game_number(self, run_id: int) -> int:
        """
        Get the next game number for a run.

        Args:
            run_id: Run ID

        Returns:
            Next game number (1-based)
        """
        with connect_db(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COALESCE(MAX(game_number), 0) + 1 FROM games WHERE run_id = ?",
                (run_id,)
            )
            return cursor.fetchone()[0]

    def update_run_completed(self, run_id: int):
        """
        Update the completed_games count for a run.

        Args:
            run_id: Run ID
        """
        with connect_db(self.db_path) as conn:
            conn.execute("""
                UPDATE runs SET completed_games = (
                    SELECT COUNT(*) FROM games
                    WHERE run_id = ? AND result IS NOT NULL
                )
                WHERE id = ?
            """, (run_id, run_id))
            conn.commit()

    def get_or_create_run(
        self,
        planned_games: int,
        strategy: Optional[str] = None,
        opponent: Optional[str] = None,
        seat: int = 1,
        random_turns: Optional[str] = None,
        novel_branch: bool = False,
    ) -> tuple[int, int]:
        """
        Get active run matching config or create a new one.

        Multiple instances with the same config share the same run.
        Each unique (strategy, opponent, planned_games, seat, random_turns,
        novel_branch) configuration gets its own run.

        Args:
            planned_games: Total games planned
            strategy: Strategy being used
            opponent: Opponent being played
            seat: Player seat (1 or 2)
            random_turns: Comma-separated random turn indices
            novel_branch: Whether novel branch forcing is enabled

        Returns:
            Tuple of (run_id, next_game_number)
        """
        novel_int = 1 if novel_branch else 0

        # Find an incomplete run with exactly matching config
        with connect_db(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM runs
                WHERE completed_games < planned_games
                    AND strategy = ?
                    AND opponent = ?
                    AND planned_games = ?
                    AND COALESCE(seat, 1) = ?
                    AND COALESCE(random_turns, '') = ?
                    AND COALESCE(novel_branch, 0) = ?
                ORDER BY started DESC
                LIMIT 1
            """, (strategy, opponent, planned_games, seat,
                  random_turns or '', novel_int))
            active = cursor.fetchone()
            if active:
                active = dict(active)

        if active:
            run_id = active['id']
            game_number = self.get_next_game_number(run_id)
            logger.info(f"Joining run {run_id}: game {game_number}/{active['planned_games']} "
                        f"(seat={seat}, random_turns={random_turns}, novel_branch={novel_branch})")
            return run_id, game_number

        run_id = self.start_run(
            planned_games=planned_games,
            strategy=strategy,
            opponent=opponent,
            seat=seat,
            random_turns=random_turns,
            novel_branch=novel_branch,
        )
        return run_id, 1

    def abandon_game(self, game_id: str, reason: str = "stuck_opponent"):
        """
        Mark a game as abandoned (e.g., opponent stuck).

        Args:
            game_id: Game ID from start_game()
            reason: Reason for abandonment
        """
        with connect_db(self.db_path) as conn:
            conn.execute("""
                UPDATE games
                SET result = 'abandoned', notes = ?
                WHERE id = ? AND result IS NULL
            """, (reason, game_id))
            conn.commit()
        logger.info(f"Game {game_id} abandoned: {reason}")


# Global collector instance (lazy initialization)
_collector: Optional[StatsCollector] = None


def get_collector(db_path: Optional[Path] = None) -> StatsCollector:
    """Get or create the global stats collector."""
    global _collector
    if _collector is None:
        _collector = StatsCollector(db_path)
    return _collector
