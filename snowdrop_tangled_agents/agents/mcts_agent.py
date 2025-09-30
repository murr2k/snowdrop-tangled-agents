""" Geordie's online Monte Carlo Tree Search agent. Uses simulated annealing to online evaluate terminal states """
from snowdrop_tangled_game_engine import Game, GameAgentBase

from snowdrop_tangled_agents.agents.mcts_utils.infrastructure import get_action_from_two_states, hash_state_action
from snowdrop_tangled_agents.agents.mcts_utils.mcts_tangled import TangledMove
from snowdrop_tangled_agents.agents.mcts_utils.mcts_nodes import TwoPlayersGameMonteCarloTreeSearchNode
from snowdrop_tangled_agents.agents.mcts_utils.mcts_search import MonteCarloTreeSearch
from snowdrop_tangled_agents.agents.mcts_utils.converters import (convert_tangled_move_to_erik_move,
                                                                  convert_erik_game_instance_to_tangled_game_state_instance)


class MCTSAgent(GameAgentBase):

    def __init__(self, player_id: str = None, **kwargs):
        super().__init__(player_id)
        print('kwargs', kwargs)
        self.mcts_rollouts = kwargs['mcts_rollouts']
        self.rollout_adjudicator = kwargs['rollout_adjudicator']
        print('rollout adj:', self.rollout_adjudicator)

    def make_move(self, game: Game) -> tuple[int, int, int] | None:
        """Make a move in the game.
        game: Game: The game instance

        Returns a tuple of the move type, move index, and move state.
        """

        # this needs to pass in the rollout adjudicator
        initial_tangled_game_state = convert_erik_game_instance_to_tangled_game_state_instance(game, rollout_adjudicator=self.rollout_adjudicator)

        # My agents eat a TangledGameState instance and return a TangledMove instance.
        print('in mcts, using rollouts', self.mcts_rollouts)
        tangled_move = mcts_agent(initial_tangled_game_state, simulations_number=self.mcts_rollouts)

        legal_moves = game.get_legal_moves(self.id)

        if not legal_moves or (len(legal_moves) == 1 and legal_moves[0][0] == Game.MoveType.QUIT.value):
            print("No legal moves available")
            return None

        while True:
            erik_move = convert_tangled_move_to_erik_move(len(game.vertices), tangled_move)
            if erik_move not in legal_moves:
                print('something went wrong, your agent picked an illegal move!')
            if erik_move[0] != Game.MoveType.QUIT.value:
                break

        return erik_move


def mcts_agent(initial_state, simulations_number):
    # MCTS agent... eats initial_state (a TangledGameState instance) and returns a TangledMove instance

    root = TwoPlayersGameMonteCarloTreeSearchNode(state=initial_state)
    mcts = MonteCarloTreeSearch(root)

    best_node = mcts.best_action(simulations_number=simulations_number)  # best_action from current state

    next_state = best_node.state

    action = get_action_from_two_states(initial_state.board, next_state.board,
                                        vertex_count=next_state.number_of_vertices)

    return TangledMove(hash_state_action(initial_state.board, next_state.board, action), initial_state.next_to_move)
