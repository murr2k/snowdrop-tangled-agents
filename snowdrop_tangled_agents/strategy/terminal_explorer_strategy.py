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
        random_move_turns: Optional[set] = None,
        novel_branch: bool = False,
        opening_index_start: int = 0,
    ):
        self.fallback_strategy = fallback_strategy
        self.randomize_midgame = randomize_midgame
        self.random_move_turns = random_move_turns if random_move_turns is not None else {0}
        self.novel_branch = novel_branch
        self.novel_cache = None
        self.opening_index = opening_index_start % len(ALL_OPENINGS)
        self.games_played = opening_index_start
        self.current_opening = None
        self.current_game_opening = None  # (edge, color) - read by play_tangled.py
        self.opening_mode = 'round_robin'  # read by play_tangled.py

    def calculate_move(
        self,
        state: str,
        score: float,
        score_history: list,
    ) -> Optional[tuple[int, str]]:
        """Calculate the next move."""
        total_moves = sum(1 for c in state if c != '-')
        # Our move index: how many of our moves have been played
        # Player 1 moves on turns 0, 2, 4, ... so our_move_count = total_moves // 2
        our_move_count = total_moves // 2

        grey = [i for i, c in enumerate(state) if c == '-']
        if not grey:
            return None

        # Check if this is a random turn
        if our_move_count in self.random_move_turns:
            # Try novel branch first if enabled
            if self.novel_cache:
                novel = self.novel_cache.get_novel_move(state)
                if novel:
                    edge, color = novel
                    if our_move_count == 0:
                        self.current_opening = (edge, color)
                        self.current_game_opening = (edge, color)
                    logger.info(
                        "Explorer novel move at turn %d: E%d%s (game %d)",
                        our_move_count, edge, color, self.games_played + 1,
                    )
                    return (edge, color)

            # Opening move: use round-robin cycle
            if our_move_count == 0:
                edge, color = ALL_OPENINGS[self.opening_index]
                self.current_opening = (edge, color)
                self.current_game_opening = (edge, color)
                logger.info(
                    "Explorer opening %d/30: E%d%s (game %d)",
                    self.opening_index + 1, edge, color, self.games_played + 1,
                )
                return (edge, color)

            # Non-opening random turn
            edge = random.choice(grey)
            color = random.choice(['G', 'P'])
            logger.info(
                "Explorer random turn %d: E%d%s (game %d)",
                our_move_count, edge, color, self.games_played + 1,
            )
            return (edge, color)

        # Non-random turn: try novel branch if enabled
        if self.novel_cache:
            novel = self.novel_cache.get_novel_move(state)
            if novel:
                edge, color = novel
                logger.info(
                    "Explorer novel move at turn %d: E%d%s (game %d)",
                    our_move_count, edge, color, self.games_played + 1,
                )
                return (edge, color)

        # All other moves: delegate to fallback strategy
        if self.fallback_strategy:
            return self.fallback_strategy.calculate_move(state, score, score_history)

        # Last resort: random legal move
        edge = random.choice(grey)
        color = random.choice(['G', 'P'])
        return (edge, color)

    def initialize(self, opponent: Optional[str] = None):
        """Initialize strategy, optionally loading novel branch cache."""
        if self.novel_branch and opponent:
            self.novel_cache = NovelBranchCache()
            self.novel_cache.load(opponent)
            logger.info(
                "Novel branch cache loaded: %d states with historical moves",
                len(self.novel_cache.seen),
            )

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
        self.current_game_opening = None


class NovelBranchCache:
    """Cache of historical moves to enable novel branch forcing.

    Loads all (board_state, edge, color) triples we've played against
    an opponent, then provides moves we haven't tried yet.
    """

    def __init__(self):
        self.seen = {}  # {state_before: set of (edge, color)}

    def load(self, opponent: str):
        """Load historical moves from the game stats database."""
        import sqlite3
        from pathlib import Path

        db_path = Path.home() / '.tangled' / 'game_stats.db'
        if not db_path.exists():
            logger.warning("DB not found at %s, novel branch disabled", db_path)
            return

        try:
            conn = sqlite3.connect(str(db_path))
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")

            # Get all our moves with the board state before each move
            # For move_number > 1, state_before is the previous move's state_after
            # For the first move (move_number=1), state_before is '---------------'
            rows = conn.execute("""
                SELECT m.edge, m.color,
                       COALESCE(prev.state_after, '---------------') as state_before
                FROM moves m
                JOIN games g ON m.game_id = g.id
                LEFT JOIN moves prev ON prev.game_id = m.game_id
                    AND prev.move_number = m.move_number - 1
                WHERE g.opponent = ? AND m.player = 'us'
            """, (opponent,)).fetchall()

            conn.close()

            for edge, color, state_before in rows:
                if state_before not in self.seen:
                    self.seen[state_before] = set()
                self.seen[state_before].add((edge, color))

            total_moves = sum(len(v) for v in self.seen.values())
            logger.info(
                "Novel branch: loaded %d unique (state, move) pairs across %d states for %s",
                total_moves, len(self.seen), opponent,
            )

        except Exception as e:
            logger.error("Failed to load novel branch cache: %s", e)

    def get_novel_move(self, state: str) -> Optional[tuple[int, str]]:
        """Return a random legal move we haven't played at this state before."""
        grey = [i for i, c in enumerate(state) if c == '-']
        if not grey:
            return None

        # All possible moves at this state
        all_moves = [(e, c) for e in grey for c in ('G', 'P')]
        seen_here = self.seen.get(state, set())

        # Filter to novel moves
        novel = [m for m in all_moves if m not in seen_here]

        if not novel:
            logger.debug("No novel moves at state %s (%d seen)", state, len(seen_here))
            return None

        return random.choice(novel)
