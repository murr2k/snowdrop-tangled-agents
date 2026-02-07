"""
WebSocket publisher for live stats dashboard.

Publishes game statistics to a remote dashboard server in real-time.
The dashboard URL and API key are configured via environment variables:
- TANGLED_DASHBOARD_URL: WebSocket URL (e.g., wss://tangled-stats.fly.dev/ws/publish)
- TANGLED_DASHBOARD_API_KEY: API key for authentication

If not configured, publishing is silently disabled.

Connection is established in a background thread so it never blocks the game.
Retry with exponential backoff ensures transient failures are recovered from.
All connection attempts are timed and logged for diagnostics.
"""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

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
    and handles reconnection automatically in the background.
    """

    # Connection tuning
    CONNECT_TIMEOUT = 10          # seconds per connection attempt
    MAX_RETRIES = 5               # total attempts before giving up for this cycle
    INITIAL_BACKOFF = 1.0         # seconds before first retry
    MAX_BACKOFF = 30.0            # max seconds between retries
    BACKOFF_MULTIPLIER = 2.0      # exponential backoff factor
    RETRY_RESET_INTERVAL = 300.0  # seconds: reset retry state after this much idle time

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        auto_connect: bool = True,
    ):
        self.url = url or os.environ.get('TANGLED_DASHBOARD_URL')
        self.api_key = api_key or os.environ.get('TANGLED_DASHBOARD_API_KEY')

        self._ws: Optional['websocket.WebSocket'] = None
        self._lock = threading.Lock()
        self._connected = False
        self._authenticated = False
        self._last_ping_time = 0
        self._ping_interval = 10

        # Retry state
        self._retry_count = 0
        self._gave_up = False
        self._last_attempt_time = 0.0

        # Background connection thread
        self._connect_thread: Optional[threading.Thread] = None
        self._connecting = False  # True while background thread is running

        # Connection diagnostics (append-only log)
        self._diag_log: List[Dict[str, Any]] = []

        # State for current session
        self._session_start: Optional[datetime] = None
        self._run_id: Optional[int] = None
        self._planned_games: Optional[int] = None

        # Turn time tracking for ETA
        self._total_turn_time: float = 0.0
        self._turn_count: int = 0
        self._current_game_turn_time: float = 0.0

        if auto_connect and self.is_configured():
            self._start_connect_thread()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _record_diag(self, event: str, **kwargs):
        """Append a timestamped diagnostic entry."""
        entry = {
            "time": datetime.now().isoformat(),
            "elapsed_ms": round(kwargs.pop("elapsed_ms", 0), 1),
            "event": event,
            "attempt": self._retry_count,
            **kwargs,
        }
        self._diag_log.append(entry)
        # Keep only last 50 entries to bound memory
        if len(self._diag_log) > 50:
            self._diag_log = self._diag_log[-50:]

    def get_diagnostics(self) -> List[Dict[str, Any]]:
        """Return the connection diagnostics log for debugging."""
        return list(self._diag_log)

    def get_status(self) -> Dict[str, Any]:
        """Return a summary of the current connection status."""
        return {
            "configured": self.is_configured(),
            "connected": self._connected,
            "authenticated": self._authenticated,
            "connecting": self._connecting,
            "retry_count": self._retry_count,
            "gave_up": self._gave_up,
            "url": self.url,
            "diag_entries": len(self._diag_log),
        }

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Check if publishing is configured (URL and API key set)."""
        return bool(self.url and self.api_key and WEBSOCKET_AVAILABLE)

    def is_connected(self) -> bool:
        """Check if currently connected and authenticated."""
        return self._connected and self._authenticated

    def _start_connect_thread(self):
        """Launch background thread to establish connection with retries."""
        if self._connecting:
            return
        # Reset gave_up if enough time has passed since last attempt
        if self._gave_up and (time.monotonic() - self._last_attempt_time) > self.RETRY_RESET_INTERVAL:
            self._gave_up = False
            self._retry_count = 0
        if self._gave_up:
            return
        self._connecting = True
        t = threading.Thread(target=self._connect_with_retries, daemon=True, name="dashboard-connect")
        t.start()
        self._connect_thread = t

    def _connect_with_retries(self):
        """Background thread: retry connection with exponential backoff."""
        backoff = self.INITIAL_BACKOFF
        try:
            while self._retry_count < self.MAX_RETRIES:
                self._retry_count += 1
                self._last_attempt_time = time.monotonic()

                success = self._try_connect_once()
                if success:
                    return

                if self._retry_count < self.MAX_RETRIES:
                    self._record_diag("backoff", wait_s=round(backoff, 1))
                    logger.info(f"Dashboard: retry {self._retry_count}/{self.MAX_RETRIES} in {backoff:.1f}s")
                    time.sleep(backoff)
                    backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF)

            # Exhausted retries
            self._gave_up = True
            self._record_diag("gave_up", total_attempts=self._retry_count)
            logger.warning(f"Dashboard: gave up after {self._retry_count} attempts. "
                           f"Will retry after {self.RETRY_RESET_INTERVAL}s of inactivity.")
        finally:
            self._connecting = False

    def _try_connect_once(self) -> bool:
        """Single connection attempt. Returns True on success."""
        t0 = time.monotonic()
        try:
            logger.info(f"Dashboard: connecting to {self.url} (attempt {self._retry_count}/{self.MAX_RETRIES})")

            ws = websocket.create_connection(
                self.url,
                timeout=self.CONNECT_TIMEOUT,
                enable_multithread=True,
                skip_utf8_validation=True,
                header={"Sec-WebSocket-Extensions": ""},
            )
            elapsed_connect = (time.monotonic() - t0) * 1000

            # TCP keepalive
            import socket
            ws.sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, 'SIO_KEEPALIVE_VALS'):
                ws.sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 30000, 10000))

            # Authenticate
            t_auth = time.monotonic()
            ws.send(json.dumps({"api_key": self.api_key}))
            response = ws.recv()
            data = json.loads(response)
            elapsed_auth = (time.monotonic() - t_auth) * 1000
            elapsed_total = (time.monotonic() - t0) * 1000

            if data.get("type") == "authenticated":
                with self._lock:
                    self._ws = ws
                    self._connected = True
                    self._authenticated = True
                self._record_diag("connected",
                                  elapsed_ms=elapsed_total,
                                  connect_ms=round(elapsed_connect, 1),
                                  auth_ms=round(elapsed_auth, 1))
                logger.info(f"Dashboard: connected and authenticated in {elapsed_total:.0f}ms "
                            f"(tcp={elapsed_connect:.0f}ms, auth={elapsed_auth:.0f}ms)")
                # Reset retry count on success
                self._retry_count = 0
                self._gave_up = False
                return True
            else:
                ws.close()
                elapsed_total = (time.monotonic() - t0) * 1000
                self._record_diag("auth_rejected", elapsed_ms=elapsed_total, response=str(data)[:200])
                logger.error(f"Dashboard: auth rejected after {elapsed_total:.0f}ms: {data}")
                return False

        except Exception as e:
            elapsed_total = (time.monotonic() - t0) * 1000
            error_str = f"{type(e).__name__}: {e}"
            self._record_diag("connect_failed", elapsed_ms=elapsed_total, error=error_str[:200])
            logger.warning(f"Dashboard: connection failed after {elapsed_total:.0f}ms: {error_str}")
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

    def _ensure_connected(self) -> bool:
        """Ensure we're connected, starting background retry if not."""
        if self.is_connected():
            return True
        if not self.is_configured():
            return False
        self._start_connect_thread()
        return False

    def _send(self, data: Dict[str, Any]) -> bool:
        """Send data to the dashboard. Non-blocking if not connected."""
        if not self._ensure_connected():
            return False

        try:
            with self._lock:
                if self._ws:
                    self._ws.send(json.dumps(data))
                    self._last_ping_time = time.time()
                    return True
        except Exception as e:
            logger.debug(f"Dashboard: send failed: {e}")
            self._disconnect()
            # Trigger background reconnect for next send
            self._start_connect_thread()

        return False

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

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
        # Reset retry state for new session
        self._gave_up = False
        self._retry_count = 0
        if self.is_configured() and not self.is_connected():
            self._start_connect_thread()

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

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

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
        strategy: Optional[str] = None,
        opponent: Optional[str] = None,
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

        Non-blocking: if not yet connected, the message is silently dropped
        and a background reconnect is triggered.

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
                "strategy": strategy,
                "opponent": opponent,
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
