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
        self._last_stats: Optional[Dict[str, Any]] = None
        self._connect_attempts = 0
        self._max_connect_attempts = 3
        self._last_ping_time = 0
        self._ping_interval = 20  # Send ping every 20 seconds

        # State for current session
        self._session_start: Optional[datetime] = None
        self._run_id: Optional[int] = None
        self._planned_games: Optional[int] = None

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
                # Disable compression extensions to avoid RSV bit issues
                self._ws = websocket.create_connection(
                    self.url,
                    timeout=10,
                    enable_multithread=True,
                    skip_utf8_validation=True,
                    header={"Sec-WebSocket-Extensions": ""},
                )
                self._connected = True

                # Authenticate
                auth_msg = json.dumps({"api_key": self.api_key})
                self._ws.send(auth_msg)

                response = self._ws.recv()
                data = json.loads(response)

                if data.get("type") == "authenticated":
                    self._authenticated = True
                    self._connect_attempts = 0  # Reset on success
                    logger.info("Dashboard connection authenticated")
                    return True
                else:
                    logger.error(f"Dashboard auth failed: {data}")
                    self._disconnect()
                    return False

            except Exception as e:
                logger.warning(f"Dashboard connection failed: {e}")
                self._disconnect()
                # Don't retry immediately on connection errors
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

    def _send_ping(self) -> bool:
        """Send a keep-alive ping if needed."""
        now = time.time()
        if now - self._last_ping_time < self._ping_interval:
            return True

        try:
            with self._lock:
                if self._ws:
                    self._ws.ping()
                    self._last_ping_time = now
                    return True
        except Exception:
            return False
        return False

    def _send(self, data: Dict[str, Any]) -> bool:
        """Send data to the dashboard."""
        if not self.is_connected():
            if not self._connect():
                return False

        # Send keep-alive ping if needed
        if not self._send_ping():
            self._disconnect()
            if not self._connect():
                return False

        try:
            with self._lock:
                if self._ws:
                    self._ws.send(json.dumps(data))
                    self._last_ping_time = time.time()
                    logger.debug("Stats published to dashboard")
                    return True
        except Exception as e:
            logger.warning(f"Dashboard send failed: {e}")
            self._disconnect()
            # Try reconnect once
            if self._connect():
                try:
                    with self._lock:
                        if self._ws:
                            self._ws.send(json.dumps(data))
                            logger.debug("Stats published to dashboard (after reconnect)")
                            return True
                except Exception:
                    pass
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

    def publish_stats(
        self,
        session: Dict[str, Any],
        results: Dict[str, int],
        scores: Dict[str, Optional[float]],
        trends: Optional[Dict[str, Any]] = None,
        model: Optional[Dict[str, Any]] = None,
        current_game: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Publish stats update to the dashboard.

        Args:
            session: Session info (run_id, completed_games, planned_games)
            results: Win/draw/loss counts
            scores: Score statistics (avg, median, min, max, std)
            trends: Trend data (recent_5, score_trend, winrate_trend)
            model: Opponent model metrics (avg_entropy, avg_top3_hit, avg_pred_accuracy)
            current_game: Current game state if in progress

        Returns:
            True if published successfully
        """
        if not self.is_configured():
            return False

        # Calculate ETA
        eta = self._calculate_eta(session)

        stats_update = {
            "type": "stats_update",
            "timestamp": datetime.now().isoformat(),
            "session": session,
            "results": results,
            "scores": scores,
            "trends": trends or {},
            "model": model or {},
            "current_game": current_game,
            "eta": eta,
        }

        self._last_stats = stats_update
        return self._send(stats_update)

    def _calculate_eta(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate estimated time of completion."""
        eta = {}

        completed = session.get("completed_games", 0)
        planned = session.get("planned_games") or self._planned_games

        if not planned or completed < 2 or not self._session_start:
            return eta

        elapsed = (datetime.now() - self._session_start).total_seconds()
        avg_per_game = elapsed / completed
        remaining = planned - completed

        if remaining > 0:
            est_remaining_seconds = remaining * avg_per_game
            est_end = datetime.now() + timedelta(seconds=est_remaining_seconds)
            eta = {
                "estimated_end": est_end.isoformat(),
                "games_remaining": remaining,
                "avg_game_duration": avg_per_game,
            }

        return eta

    def publish_from_session_stats(self, session_stats, current_game: Optional[Dict[str, Any]] = None) -> bool:
        """
        Publish stats directly from a SessionStats object.

        Args:
            session_stats: SessionStats object from session_stats.py
            current_game: Optional current game state
        """
        if session_stats is None:
            return False

        # Extract run info from latest game if available
        run_id = None
        planned_games = None
        if session_stats.games:
            latest = session_stats.games[-1]
            run_id = latest.run_id
            # Try to get planned games from run
            if run_id:
                try:
                    from .collector import get_collector
                    collector = get_collector()
                    run_info = collector.get_run(run_id)
                    if run_info:
                        planned_games = run_info.get("planned_games")
                except Exception:
                    pass

        # Build recent_5 string (W/D/L for last 5 games)
        recent_5 = ""
        completed_games = [g for g in session_stats.games if g.result is not None]
        for game in completed_games[-5:]:
            if game.result == "win":
                recent_5 += "W"
            elif game.result == "draw":
                recent_5 += "D"
            elif game.result == "loss":
                recent_5 += "L"

        # Calculate score trend (last 10 games regression slope)
        score_trend = None
        if len(completed_games) >= 3:
            recent_scores = [g.final_score for g in completed_games[-10:] if g.final_score is not None]
            if len(recent_scores) >= 3:
                # Simple linear regression slope
                n = len(recent_scores)
                x_mean = (n - 1) / 2
                y_mean = sum(recent_scores) / n
                numerator = sum((i - x_mean) * (s - y_mean) for i, s in enumerate(recent_scores))
                denominator = sum((i - x_mean) ** 2 for i in range(n))
                if denominator > 0:
                    score_trend = numerator / denominator

        session = {
            "run_id": run_id,
            "completed_games": session_stats.completed_games,
            "planned_games": planned_games or session_stats.game_count,
        }

        results = {
            "wins": session_stats.wins,
            "draws": session_stats.draws,
            "losses": session_stats.losses,
        }

        scores = {
            "avg": session_stats.avg_score,
            "median": session_stats.median_score,
            "min": session_stats.min_score,
            "max": session_stats.max_score,
            "std": session_stats.score_std,
        }

        trends = {
            "recent_5": recent_5,
            "score_trend": score_trend,
        }

        model = {
            "avg_entropy": session_stats.avg_entropy,
            "avg_top3_hit": session_stats.avg_top3_hit,
            "avg_pred_accuracy": session_stats.avg_prediction_accuracy,
        }

        return self.publish_stats(
            session=session,
            results=results,
            scores=scores,
            trends=trends,
            model=model,
            current_game=current_game,
        )

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


def publish_session_stats(current_game: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convenience function to publish current session stats.

    Fetches current session stats and publishes to dashboard.
    """
    publisher = get_publisher()
    if not publisher.is_configured():
        return False

    try:
        from .session_stats import get_session_stats
        session_stats = get_session_stats()
        return publisher.publish_from_session_stats(session_stats, current_game)
    except Exception as e:
        logger.warning(f"Failed to publish session stats: {e}")
        return False
