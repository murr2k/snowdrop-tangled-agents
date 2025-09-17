import random
import logging
from typing import Tuple

from snowdrop_tangled_game_engine import Game, GameAgentBase


class RandomRandyAgent(GameAgentBase):
    """
    This is an example of a simple RandomRandyAgent that makes random moves.

    Use this template to build your own agents. Here are the basic steps:

    1. Import the GameAgentBase class from snowdrop_tangled_game_engine.
        This will be the base class for your agent.
    2. Import the Game class from snowdrop_tangled_game_engine.
        This will be the state of the game that your agent will interact with. See the class for more details,
        but generally you'll have access to the full state of the game (the vertices and edges and their states)
        and some helpful methods for interacting with the game (get_legal_moves, etc.)
    3. Create a new class that inherits from GameAgentBase and implement the make_move method.
        The make_move method should take a Game object as an argument and return a tuple of the move type,
        move index, and move state.
        move_type is Game.MoveType IntEnum, and has values of NONE, EDGE, or QUIT.
        move_index is the index of the edge to change the state of, where the edges (i, j) i < j are in lexical order.
        move_state is the state to change the edge to.
                Edge.State.ZERO -- zero coupling / grey edge
                Edge.State.FM   -- FM coupling / green edge
                Edge.State.AFM  -- AFM coupling / purple edge
                (Edge.State.NONE is the initial state)

        The move should be returned as a tuple of these three values as integers.
        e.g. (Game.MoveType.EDGE.value, 3, Edge.State.FM.value) is a move that turns edge #3 green.
    """

    def __init__(self, player_id: str = None, **kwargs):
        super().__init__(player_id)

    def make_move(self, game: Game) -> Tuple[int, int, int] | None:
        """Make a move in the game.
        game: Game: The game instance

        Returns a tuple (move_type, move_index, move_state) or None if there are no legal moves.
        """

        legal_moves = game.get_legal_moves(self.id)

        if not legal_moves or (len(legal_moves) == 1 and legal_moves[0][0] == Game.MoveType.QUIT.value):
            logging.info("No legal moves available")
            return None

        while True:
            move = random.choice(legal_moves)
            if move[0] != Game.MoveType.QUIT.value:
                break

        return move
