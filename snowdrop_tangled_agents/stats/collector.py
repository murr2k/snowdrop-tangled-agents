"""
SQLite-based statistics collector for Tangled game play.

Collects game outcomes, move sequences, and score progressions
for analysis and strategy improvement.
"""

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Default database location
DEFAULT_DB_PATH = Path.home() / ".tangled" / "game_stats.db"


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
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
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

    def start_game(
        self,
        opponent: str = "melissa",
        graph: str = "petersen",
        strategy: str = "hybrid",
        mcts_time: float = 8.0
    ) -> str:
        """
        Start tracking a new game.

        Args:
            opponent: Opponent name
            graph: Graph type
            strategy: Strategy being used
            mcts_time: MCTS time limit

        Returns:
            Unique game ID
        """
        game_id = str(uuid.uuid4())[:8]  # Short ID for readability

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO games (id, opponent, graph, strategy, mcts_time)
                VALUES (?, ?, ?, ?, ?)
            """, (game_id, opponent, graph, strategy, mcts_time))
            conn.commit()

        logger.debug(f"Started game {game_id} vs {opponent}")
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
        thinking_time: Optional[float] = None
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
        """
        score_delta = None
        if score_before is not None:
            score_delta = score_after - score_before

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO moves
                (game_id, move_number, player, edge, color, score_after,
                 score_delta, state_after, mcts_iterations, thinking_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (game_id, move_number, player, edge, color, score_after,
                  score_delta, state_after, mcts_iterations, thinking_time))
            conn.commit()

        logger.debug(f"Game {game_id}: Move {move_number} - E{edge} {color} -> {score_after:.3f}")

    def end_game(
        self,
        game_id: str,
        result: str,
        final_score: float,
        notes: Optional[str] = None
    ):
        """
        Mark a game as complete.

        Args:
            game_id: Game ID from start_game()
            result: 'win', 'loss', or 'draw'
            final_score: Final game score
            notes: Optional notes about the game
        """
        with sqlite3.connect(self.db_path) as conn:
            # Count total moves
            cursor = conn.execute(
                "SELECT COUNT(*) FROM moves WHERE game_id = ? AND player = 'us'",
                (game_id,)
            )
            total_moves = cursor.fetchone()[0]

            conn.execute("""
                UPDATE games
                SET result = ?, final_score = ?, total_moves = ?, notes = ?
                WHERE id = ?
            """, (result, final_score, total_moves, notes, game_id))
            conn.commit()

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

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO calibration
                (game_id, terminal_state, website_score, predicted_score, error, abs_error)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (game_id, terminal_state, website_score, predicted_score, error, abs_error))
            conn.commit()

        logger.info(f"Calibration: website={website_score:+.4f}, predicted={predicted_score:+.4f}, error={error:+.4f}")

    def get_game_count(self) -> dict:
        """Get count of games by result."""
        with sqlite3.connect(self.db_path) as conn:
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
        return sqlite3.connect(self.db_path)


# Global collector instance (lazy initialization)
_collector: Optional[StatsCollector] = None


def get_collector(db_path: Optional[Path] = None) -> StatsCollector:
    """Get or create the global stats collector."""
    global _collector
    if _collector is None:
        _collector = StatsCollector(db_path)
    return _collector
