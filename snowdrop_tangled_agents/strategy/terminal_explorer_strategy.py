"""Terminal Explorer Strategy - maximizes terminal state diversity.

Systematically cycles through all 30 possible openings (15 edges x 2 colors)
to reach diverse terminal states. Uses MCTS for subsequent moves. Designed
for building empirical calibration data against a fixed opponent.
"""

import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)

# All 30 possible opening moves (15 edges x 2 colors)
ALL_OPENINGS = [(edge, color) for edge in range(15) for color in ('G', 'P')]


class TerminalExplorerStrategy:
    """Explores diverse terminal states through opening diversification.

    Cycles through all 30 possible openings in round-robin fashion.
    After the opening move, delegates to a fallback strategy (typically MCTS)
    for the remaining moves.
    """

    def __init__(
        self,
        fallback_strategy=None,
        randomize_midgame: bool = False,
    ):
        self.fallback_strategy = fallback_strategy
        self.randomize_midgame = randomize_midgame
        self.opening_index = 0
        self.games_played = 0
        self.current_opening = None

    def calculate_move(
        self,
        state: str,
        score: float,
        score_history: list,
    ) -> Optional[tuple[int, str]]:
        """Calculate the next move."""
        total_moves = sum(1 for c in state if c != '-')

        # Opening move: use the current opening from the cycle
        if total_moves == 0:
            edge, color = ALL_OPENINGS[self.opening_index]
            self.current_opening = (edge, color)
            logger.info(
                "Explorer opening %d/30: E%d%s (game %d)",
                self.opening_index + 1, edge, color, self.games_played + 1,
            )
            return (edge, color)

        # Mid-game: optionally randomize one move for extra diversity
        if self.randomize_midgame and total_moves == 2:
            grey = [i for i, c in enumerate(state) if c == '-']
            if grey:
                edge = random.choice(grey)
                color = random.choice(['G', 'P'])
                logger.info("Explorer random mid-game: E%d%s", edge, color)
                return (edge, color)

        # All other moves: delegate to fallback strategy
        if self.fallback_strategy:
            return self.fallback_strategy.calculate_move(state, score, score_history)

        # Last resort: random legal move
        grey = [i for i, c in enumerate(state) if c == '-']
        if not grey:
            return None
        edge = random.choice(grey)
        color = random.choice(['G', 'P'])
        return (edge, color)

    def end_game(self, result: str, final_score: float):
        """Called at end of game. Advances to next opening."""
        logger.info(
            "Explorer game %d result: %s (score=%.4f), opening was E%d%s",
            self.games_played + 1, result, final_score,
            self.current_opening[0] if self.current_opening else -1,
            self.current_opening[1] if self.current_opening else '?',
        )
        self.games_played += 1
        self.opening_index = self.games_played % len(ALL_OPENINGS)
        self.current_opening = None
