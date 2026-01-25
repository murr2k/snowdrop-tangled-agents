"""
WebSocket publisher for live stats dashboard.

Publishes game statistics to a remote dashboard server in real-time.
The dashboard URL and API key are configured via environment variables:
- TANGLED_DASHBOARD_URL: WebSocket URL (e.g., wss://tangled-stats.fly.dev/ws/publish)
- TANGLED_DASHBOARD_API_KEY: API key for authentication

If not configured, publishing is silently disabled.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import websocket library
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.debug("websocket-client not installed, live stats publishing disabled")


class StatsPublisher:
    """
    Publishes game statistics to a remote WebSocket dashboard.

    Thread-safe publisher that maintains a persistent connection
    and handles reconnection automatically.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        auto_connect: bool = True,
    ):
        """
        Initialize the stats publisher.

        Args:
            url: WebSocket URL. Defaults to TANGLED_DASHBOARD_URL env var.
            api_key: API key for auth. Defaults to TANGLED_DASHBOARD_API_KEY env var.
            auto_connect: Whether to connect automatically on first publish.
        """
        self.url = url or os.environ.get('TANGLED_DASHBOARD_URL')
        self.api_key = api_key or os.environ.get('TANGLED_DASHBOARD_API_KEY')

        self._ws: Optional[websocket.WebSocket] = None
        self._lock = threading.Lock()
        self._connected = False
        self._authenticated = False
        self._connect_attempts = 0
        self._max_connect_attempts = 3
        self._last_ping_time = 0
        self._ping_interval = 10

        # State for current session
        self._session_start: Optional[datetime] = None
        self._run_id: Optional[int] = None
        self._planned_games: Optional[int] = None

        # Turn time tracking for ETA
        self._total_turn_time: float = 0.0
        self._turn_count: int = 0
        self._current_game_turn_time: float = 0.0

        if auto_connect and self.is_configured():
            self._connect()

    def is_configured(self) -> bool:
        """Check if publishing is configured (URL and API key set)."""
        return bool(self.url and self.api_key and WEBSOCKET_AVAILABLE)

    def is_connected(self) -> bool:
        """Check if currently connected and authenticated."""
        return self._connected and self._authenticated

    def _connect(self) -> bool:
        """Establish WebSocket connection and authenticate."""
        if not self.is_configured():
            return False

        with self._lock:
            if self._connected:
                return True

            self._connect_attempts += 1
            if self._connect_attempts > self._max_connect_attempts:
                logger.warning(f"Max connection attempts ({self._max_connect_attempts}) exceeded")
                return False

            try:
                logger.info(f"Connecting to dashboard: {self.url}")
                self._ws = websocket.create_connection(
                    self.url,
                    timeout=60,
                    enable_multithread=True,
                    skip_utf8_validation=True,
                    header={"Sec-WebSocket-Extensions": ""},
                )
                import socket
                self._ws.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                if hasattr(socket, 'SIO_KEEPALIVE_VALS'):
                    self._ws.sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 30000, 10000))
                self._connected = True

                auth_msg = json.dumps({"api_key": self.api_key})
                self._ws.send(auth_msg)

                response = self._ws.recv()
                data = json.loads(response)

                if data.get("type") == "authenticated":
                    self._authenticated = True
                    self._connect_attempts = 0
                    logger.info("Dashboard connection authenticated")
                    return True
                else:
                    logger.error(f"Dashboard auth failed: {data}")
                    self._disconnect()
                    return False

            except Exception as e:
                logger.warning(f"Dashboard connection failed: {e}")
                self._disconnect()
                self._connect_attempts = self._max_connect_attempts
                return False

    def _disconnect(self):
        """Close WebSocket connection."""
        with self._lock:
            if self._ws:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None
            self._connected = False
            self._authenticated = False

    def _send(self, data: Dict[str, Any]) -> bool:
        """Send data to the dashboard with automatic reconnection."""
        max_retries = 2

        for attempt in range(max_retries + 1):
            if not self.is_connected():
                if not self._connect():
                    continue

            try:
                with self._lock:
                    if self._ws:
                        self._ws.send(json.dumps(data))
                        self._last_ping_time = time.time()
                        logger.debug("Stats published to dashboard")
                        return True
            except Exception as e:
                if attempt < max_retries:
                    logger.debug(f"Dashboard send failed (attempt {attempt + 1}), reconnecting: {e}")
                    self._disconnect()
                    time.sleep(0.2)
                else:
                    logger.warning(f"Dashboard send failed after {max_retries + 1} attempts: {e}")
                    self._disconnect()

        return False

    def set_session_info(
        self,
        run_id: Optional[int] = None,
        planned_games: Optional[int] = None,
    ):
        """Set session information for ETA calculation."""
        self._run_id = run_id
        self._planned_games = planned_games
        if self._session_start is None:
            self._session_start = datetime.now()
        self._total_turn_time = 0.0
        self._turn_count = 0
        self._current_game_turn_time = 0.0
        self._connect_attempts = 0
        if self.is_configured() and not self.is_connected():
            self._connect()

    def game_completed(self):
        """Call when a game completes to finalize turn time tracking."""
        self._total_turn_time += self._current_game_turn_time
        self._current_game_turn_time = 0.0

    def _calculate_eta(self, completed_games: int) -> Dict[str, Any]:
        """Calculate estimated time of completion based on turn times."""
        eta = {}
        planned = self._planned_games

        if not planned:
            return eta

        remaining = planned - completed_games
        MOVES_PER_GAME = 8

        if self._turn_count > 0:
            avg_turn_time = self._total_turn_time / self._turn_count
            current_game_turns = max(1, int(self._current_game_turn_time / avg_turn_time)) if avg_turn_time > 0 else 0
            completed_game_turns = self._turn_count - current_game_turns

            if completed_game_turns > 0 and completed_games > 0:
                avg_game_duration = self._total_turn_time / completed_games
            else:
                avg_game_duration = avg_turn_time * MOVES_PER_GAME
        elif completed_games >= 1 and self._session_start:
            elapsed = (datetime.now() - self._session_start).total_seconds()
            avg_game_duration = elapsed / completed_games
            avg_turn_time = avg_game_duration / MOVES_PER_GAME
        else:
            return eta

        if remaining > 0:
            est_remaining_seconds = remaining * avg_game_duration
            est_end = datetime.now() + timedelta(seconds=est_remaining_seconds)
            eta = {
                "estimated_end": est_end.isoformat(),
                "games_remaining": remaining,
                "avg_game_duration": avg_game_duration,
                "avg_turn_time": avg_turn_time,
            }

        return eta

    def publish_state(
        self,
        # Move info (optional - for during-game updates)
        move_edge: Optional[int] = None,
        move_color: Optional[str] = None,
        move_score: Optional[float] = None,
        move_thinking_time: Optional[float] = None,
        move_player: str = "us",
        # Board state
        board_state: Optional[str] = None,
        vertex_state: Optional[str] = None,
        # Session stats (fetched automatically if not provided)
        current_game: Optional[int] = None,
        completed_games: Optional[int] = None,
        wins: Optional[int] = None,
        draws: Optional[int] = None,
        losses: Optional[int] = None,
        # Score stats
        avg_score: Optional[float] = None,
        median_score: Optional[float] = None,
        min_score: Optional[float] = None,
        max_score: Optional[float] = None,
        std_score: Optional[float] = None,
        # Trends
        recent_5: Optional[str] = None,
        score_trend: Optional[float] = None,
        # Model metrics
        avg_entropy: Optional[float] = None,
        avg_top3_hit: Optional[float] = None,
        avg_pred_accuracy: Optional[float] = None,
    ) -> bool:
        """
        Publish the COMPLETE dashboard state. This is the ONE and ONLY message
        type sent to the dashboard. Every call sends ALL fields.

        Args:
            move_edge: Edge index (0-14) of current/last move
            move_color: 'G' for green/FM, 'P' for purple/AFM
            move_score: Score after this move
            move_thinking_time: Time spent calculating this move (seconds)
            move_player: 'us' or 'opponent'
            board_state: Current board state string (15 chars of G/P/-)
            vertex_state: Current vertex colors string (10 chars of R/B/-)
            current_game: Game number currently being played (1-indexed)
            completed_games: Number of completed games in this run
            wins/draws/losses: Result counts
            avg_score/median_score/min_score/max_score/std_score: Score statistics
            recent_5: String of last 5 results (e.g., "WDWLW")
            score_trend: Score trend slope
            avg_entropy/avg_top3_hit/avg_pred_accuracy: Opponent model metrics

        Returns:
            True if published successfully
        """
        if not self.is_configured():
            return False

        # Track turn time for ETA calculation
        if move_thinking_time is not None:
            self._current_game_turn_time += move_thinking_time
            self._turn_count += 1

        # Default board state
        board = board_state if board_state else "-" * 15
        vertices = vertex_state if vertex_state else "-----R-B--"
        edges_colored = sum(1 for c in board if c in 'GP')

        # Build move info
        move = None
        if move_edge is not None or move_color is not None:
            color_name = "Green" if move_color == 'G' else ("Purple" if move_color == 'P' else "")
            move = {
                "number": edges_colored,
                "edge": move_edge if move_edge is not None else 0,
                "color": move_color if move_color else "-",
                "color_name": color_name,
                "score": move_score if move_score is not None else 0.0,
                "thinking_time": move_thinking_time if move_thinking_time is not None else 0.0,
                "player": move_player,
            }

        # Calculate ETA
        games_completed = completed_games if completed_games is not None else 0
        eta = self._calculate_eta(games_completed)

        # Build the ONE message with ALL fields
        full_state = {
            "type": "full_state",
            "timestamp": datetime.now().isoformat(),
            # Move info
            "move": move,
            # Board state
            "board_state": board,
            "vertex_state": vertices,
            "edges_colored": edges_colored,
            # Session info
            "session": {
                "run_id": self._run_id,
                "current_game": current_game if current_game is not None else 0,
                "completed_games": completed_games if completed_games is not None else 0,
                "planned_games": self._planned_games,
            },
            # Results
            "results": {
                "wins": wins if wins is not None else 0,
                "draws": draws if draws is not None else 0,
                "losses": losses if losses is not None else 0,
            },
            # Score statistics
            "scores": {
                "avg": avg_score,
                "median": median_score,
                "min": min_score,
                "max": max_score,
                "std": std_score,
            },
            # Trends
            "trends": {
                "recent_5": recent_5 if recent_5 else "",
                "score_trend": score_trend,
            },
            # Opponent model metrics
            "model": {
                "avg_entropy": avg_entropy,
                "avg_top3_hit": avg_top3_hit,
                "avg_pred_accuracy": avg_pred_accuracy,
            },
            # ETA
            "eta": eta,
        }

        return self._send(full_state)

    def close(self):
        """Close the connection."""
        self._disconnect()


# Global publisher instance (lazy initialization)
_publisher: Optional[StatsPublisher] = None


def get_publisher() -> StatsPublisher:
    """Get the global stats publisher instance."""
    global _publisher
    if _publisher is None:
        _publisher = StatsPublisher(auto_connect=False)
    return _publisher
