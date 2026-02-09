"""Oracle Route Strategy - plays pre-computed winning routes against deterministic opponents.

Loads winning routes from the oracle-solver JSON output and plays moves verbatim.
When the opponent deviates from the predicted route, attempts to find an alternative
route that matches the current board state, or falls back to MCTS.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class OracleRouteStrategy:
    """Plays pre-computed oracle routes against a deterministic opponent.

    The strategy tracks the game state and follows a winning route move-by-move.
    If the opponent deviates from the oracle's prediction, it searches for an
    alternative route or falls back to a provided fallback strategy.
    """

    def __init__(
        self,
        routes_path: str = "oracle-solver/output/oracle_routes.json",
        fallback_strategy=None,
        route_index: int = 0,
        route_mode: str = "fixed",
    ):
        self.routes_path = Path(routes_path)
        self.fallback_strategy = fallback_strategy
        self.preferred_route_index = route_index
        self.route_mode = route_mode  # 'fixed' or 'cycle'
        self.routes = []
        self.active_route = None
        self.our_move_index = 0  # Which of our moves we're on (0-based)
        self.last_state = None
        self.deviated = False
        self.games_played = 0

        self._load_routes()

    def _load_routes(self):
        if not self.routes_path.exists():
            logger.error("Oracle routes file not found: %s", self.routes_path)
            return

        with open(self.routes_path) as f:
            data = json.load(f)

        self.routes = data.get("winning_routes", [])
        logger.info(
            "Loaded %d winning routes (oracle: %d states, %.1f%% deterministic)",
            len(self.routes),
            data.get("oracle_stats", {}).get("total_states_observed", 0),
            data.get("oracle_stats", {}).get("deterministic_pct", 0),
        )

        if self.routes:
            self._select_route(self.preferred_route_index)

    def _select_route(self, index: int):
        if index < len(self.routes):
            self.active_route = self.routes[index]
            self.our_move_index = 0
            self.deviated = False
            route = self.active_route
            logger.info(
                "Selected route %d: %s (score=%.4f, confidence=%.2f, gaps=%d)",
                index,
                route["terminal_state"],
                route["lut_score"],
                route["path_min_confidence"],
                route["oracle_gaps"],
            )
            # Log the full move sequence for this route
            our_moves = [m for m in route["moves"] if m["player"] == "us"]
            move_strs = [f"E{m['edge']}{m['color']}" for m in our_moves]
            logger.info("Route moves: %s", " -> ".join(move_strs))

    def _get_our_moves(self):
        """Extract our moves from the active route in order."""
        if not self.active_route:
            return []
        return [m for m in self.active_route["moves"] if m["player"] == "us"]

    def _get_opponent_moves(self):
        """Extract opponent moves from the active route in order."""
        if not self.active_route:
            return []
        return [m for m in self.active_route["moves"] if m["player"] == "opponent"]

    def _check_opponent_deviation(self, state: str, score_history: list) -> bool:
        """Check if the opponent has deviated from the oracle route.

        Compare the current board state against what we expect after
        both our last move and the opponent's predicted response.
        """
        if not self.active_route or self.our_move_index == 0:
            return False

        # Reconstruct expected state from the route up to current point
        expected_state = list("---------------")
        for move in self.active_route["moves"]:
            # Count how many of our moves have been played
            if move["player"] == "us" and move["turn"] > self.our_move_index * 2:
                break  # Haven't made this move yet
            if move["player"] == "opponent" and move["turn"] > self.our_move_index * 2:
                break
            expected_state[move["edge"]] = move["color"]

        expected = "".join(expected_state)

        # Compare with actual state
        if state != expected:
            # Find where deviation occurred
            for i in range(15):
                if state[i] != expected[i] and expected[i] != '-':
                    logger.warning(
                        "Opponent deviation detected at edge %d: expected %s, got %s",
                        i, expected[i], state[i],
                    )
            return True
        return False

    def _find_matching_route(self, state: str) -> Optional[dict]:
        """Find a route whose history matches the current board state."""
        move_count = sum(1 for c in state if c != '-')

        for i, route in enumerate(self.routes):
            # Build expected state at this point in the route
            expected = list("---------------")
            for move in route["moves"]:
                if move["turn"] > move_count:
                    break
                expected[move["edge"]] = move["color"]

            if "".join(expected) == state:
                logger.info("Found matching route %d for current state", i)
                return route

        return None

    def calculate_move(
        self,
        state: str,
        score: float,
        score_history: list,
    ) -> Optional[tuple[int, str]]:
        """Calculate the next move following the oracle route."""

        if not self.routes or not self.active_route:
            logger.warning("No oracle routes loaded, using fallback")
            return self._fallback(state, score, score_history)

        # Determine how many total moves have been made
        total_moves = sum(1 for c in state if c != '-')
        # Our moves are at positions 0, 2, 4, ... (we move first)
        # So our move index = total_moves // 2
        self.our_move_index = total_moves // 2

        our_moves = self._get_our_moves()

        # Check for opponent deviation (after the first exchange)
        if total_moves > 0 and not self.deviated:
            if self._check_opponent_deviation(state, score_history):
                self.deviated = True
                logger.warning("Opponent deviated from oracle route!")

                # Try to find an alternative route
                alt_route = self._find_matching_route(state)
                if alt_route:
                    self.active_route = alt_route
                    self.deviated = False
                    our_moves = self._get_our_moves()
                    logger.info("Switched to alternative matching route")
                else:
                    logger.warning("No matching route found, falling back")
                    return self._fallback(state, score, score_history)

        if self.deviated:
            return self._fallback(state, score, score_history)

        if self.our_move_index >= len(our_moves):
            logger.warning("Exhausted oracle moves (index=%d, have=%d)",
                           self.our_move_index, len(our_moves))
            return self._fallback(state, score, score_history)

        move = our_moves[self.our_move_index]
        edge = move["edge"]
        color = move["color"]

        # Validate move is legal
        if state[edge] != '-':
            logger.error(
                "Oracle move E%d%s is invalid (edge=%s), opponent may have deviated",
                edge, color, state[edge],
            )
            self.deviated = True
            alt_route = self._find_matching_route(state)
            if alt_route:
                self.active_route = alt_route
                self.deviated = False
                our_moves = self._get_our_moves()
                if self.our_move_index < len(our_moves):
                    move = our_moves[self.our_move_index]
                    edge, color = move["edge"], move["color"]
                    if state[edge] == '-':
                        logger.info("Recovered with alt route: E%d%s", edge, color)
                        return (edge, color)
            return self._fallback(state, score, score_history)

        logger.info(
            "Oracle move %d/%d: E%d%s (route target: %s, score=%.4f)",
            self.our_move_index + 1,
            len(our_moves),
            edge,
            color,
            self.active_route["terminal_state"],
            self.active_route["lut_score"],
        )

        return (edge, color)

    def _fallback(
        self,
        state: str,
        score: float,
        score_history: list,
    ) -> Optional[tuple[int, str]]:
        """Fall back to the backup strategy."""
        if self.fallback_strategy:
            logger.info("Using fallback strategy: %s", type(self.fallback_strategy).__name__)
            return self.fallback_strategy.calculate_move(state, score, score_history)

        # Last resort: pick first grey edge with heuristic color
        grey = [i for i, c in enumerate(state) if c == '-']
        if not grey:
            return None
        edge = grey[0]
        color = 'G' if edge in [9, 10, 11] else ('P' if edge in [5, 12, 13] else 'G')
        logger.warning("No fallback strategy, using heuristic: E%d%s", edge, color)
        return (edge, color)

    def end_game(self, result: str, final_score: float):
        """Called at end of game for logging."""
        if self.active_route:
            logger.info(
                "Oracle route result: %s (score=%.4f), target was %s (lut=%.4f)",
                result,
                final_score,
                self.active_route["terminal_state"],
                self.active_route["lut_score"],
            )
        self.games_played += 1

        # Reset for next game
        self.our_move_index = 0
        self.deviated = False
        self.last_state = None

        if self.route_mode == "cycle" and self.routes:
            next_index = self.games_played % len(self.routes)
            logger.info("Cycling to route %d/%d", next_index + 1, len(self.routes))
            self._select_route(next_index)
        else:
            self._select_route(self.preferred_route_index)
