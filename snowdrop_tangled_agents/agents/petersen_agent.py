"""
Petersen Graph Agent for Tangled Game.

SDK-compatible agent that uses the parameterized Petersen strategy,
with support for external score injection (for website play) and
learning from game outcomes.
"""

import logging
from typing import Optional

from snowdrop_tangled_game_engine import Game, GameAgentBase

from snowdrop_tangled_agents.strategy.petersen_strategy import PetersenStrategy


class PetersenAgent(GameAgentBase):
    """
    Tangled agent optimized for Petersen graph (graph #11).

    Uses a parameterized strategy with tunable edge priorities and
    adaptive color selection. Supports learning from game outcomes.

    For local SDK play:
        agent = PetersenAgent("player_1")
        move = agent.make_move(game)

    For website play with score injection:
        agent = PetersenAgent("player_1")
        agent.set_score(dom_score)  # Before each move
        move = agent.make_move(game)
        agent.record_move(edge, color, new_score)  # After move executes
    """

    def __init__(
        self,
        player_id: str = None,
        params_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Petersen agent.

        Args:
            player_id: Unique player identifier
            params_path: Path to load/save learned parameters
            **kwargs: Additional arguments (for compatibility)
        """
        super().__init__(player_id)

        self.strategy = PetersenStrategy(params_path=params_path)
        self.state = ['-'] * 15  # Track edge states: '-', 'G', 'P', '?'
        self.score = 0.0  # Current game score (injected for website play)
        self.score_history: list[tuple[int, str, float]] = []
        self.my_moves: list[tuple[int, str]] = []

    def make_move(self, game: Game) -> Optional[tuple[int, int, int]]:
        """
        Make a move in the game.

        Args:
            game: The Game instance

        Returns:
            Tuple of (move_type, move_index, move_state) or None if no legal moves.
        """
        legal_moves = game.get_legal_moves(self.id)

        if not legal_moves:
            logging.info("No legal moves available")
            return None

        # Filter out QUIT moves for analysis
        edge_moves = [m for m in legal_moves if m[0] == Game.MoveType.EDGE.value]

        if not edge_moves:
            logging.info("No edge moves available")
            return None

        # Get available edge indices
        available = {m[1] for m in edge_moves}

        # Update state: infer opponent moves (edges that were '-' but now unavailable)
        for i, s in enumerate(self.state):
            if s == '-' and i not in available:
                self.state[i] = '?'  # Opponent played (color unknown in local play)

        # Calculate move using strategy
        state_str = ''.join(self.state)
        result = self.strategy.calculate_move(state_str, self.score, self.score_history)

        if result is None:
            logging.info("Strategy returned no move")
            return None

        edge, color = result

        # Verify edge is actually available
        if edge not in available:
            logging.warning(f"Strategy chose unavailable edge {edge}, falling back")
            # Fall back to first available edge with appropriate color
            edge = next(iter(available))
            color = 'G' if edge in [9, 10, 11] else 'P' if edge in [5, 12, 13] else 'G'

        # Update our state tracking
        self.state[edge] = color
        self.my_moves.append((edge, color))

        # Convert color to Edge.State
        # FM (ferromagnetic) = Green, AFM (antiferromagnetic) = Purple
        from snowdrop_tangled_game_engine import Edge
        edge_state = Edge.State.FM.value if color == 'G' else Edge.State.AFM.value

        logging.debug(f"Playing edge {edge} with {'Green' if color == 'G' else 'Purple'}")

        return (Game.MoveType.EDGE.value, edge, edge_state)

    def set_score(self, score: float) -> None:
        """
        Inject current game score from external source (e.g., website DOM).

        Args:
            score: Current score (positive = we're ahead)
        """
        self.score = score

    def record_move(self, edge: int, color: str, score_after: float) -> None:
        """
        Record a move and its resulting score for learning.

        Call this after each move (yours or opponent's) when playing on website.

        Args:
            edge: Edge index that was played
            color: 'G' or 'P'
            score_after: Score after the move
        """
        self.score_history.append((edge, color, score_after))
        self.score = score_after

        # Update state if we have more info
        if self.state[edge] in ('-', '?'):
            self.state[edge] = color

    def end_game(self, result: str, final_score: float) -> None:
        """
        Signal game end and trigger learning update.

        Args:
            result: 'win', 'loss', or 'draw'
            final_score: Final game score
        """
        logging.info(f"Game ended: {result} with score {final_score}")

        # Update strategy parameters based on outcome
        self.strategy.update_from_trial(result, self.score_history, final_score)

        # Reset for next game
        self.reset()

    def reset(self) -> None:
        """Reset agent state for a new game."""
        self.state = ['-'] * 15
        self.score = 0.0
        self.score_history = []
        self.my_moves = []

    def get_strategy(self) -> PetersenStrategy:
        """Return the underlying strategy for direct access."""
        return self.strategy

    def get_state_string(self) -> str:
        """Return current state as 15-char string."""
        return ''.join(self.state)
