import os
from typing import Any
import logging
import numpy as np

from snowdrop_tangled_game_engine import Game, GameAgentBase

from snowdrop_tangled_agents.agents.mcts_utils.converters import (convert_to_az_board, convert_from_az_board,
                                        convert_erik_game_instance_to_tangled_game_state_instance,
                                        convert_tangled_move_to_erik_move)
from snowdrop_tangled_agents.agents.mcts_utils.mcts_tangled import TangledMove
from snowdrop_tangled_agents.agents.mcts_utils.infrastructure import hash_state_action

from snowdrop_tangled_alphazero.game.TangledGame import TangledGame
from snowdrop_tangled_alphazero.monte_carlo.AZMCTS import AZ_MCTS
from snowdrop_tangled_alphazero.utilities.utilities import load_model_and_optimizer


class AlphaZeroAgent(GameAgentBase):

    def __init__(self, player_id: str = None, **kwargs):
        super().__init__(player_id)
        self.args1 = {'numMCTSSimsSelf': kwargs['mcts_rollouts'], 'cpuct': 1.0, 'fixed_token_game': True}

        graph_args: dict[str, Any] = {
            'graph_number': kwargs['graph_number'],  # 11 P_3, 2 K_3, 20 diamond, 19 barbell, 18 3-prism, 12 moser, 24 C_60
            'rollout_adjudicator': kwargs['rollout_adjudicator']}  # ['lookup_table', 'simulated_annealing', 'quantum_annealing']

        self.g = TangledGame(graph_args=graph_args,     # this is for MCTS
                             fixed_token_game=self.args1["fixed_token_game"])

        self.vertex_count = self.g.graph_properties["num_nodes"]

        # todo the reason why we can't load two different NNs is that NeuralNetWrapper calls TangledNeuralNet which has a NN architecture -- we'd need to have different files if we want two different NNs to fight
        self.nn = load_model_and_optimizer(board_size=self.g.getBoardSize(), action_size=self.g.getActionSize(),
                                           graph_number=graph_args["graph_number"],
                                         filepath=os.path.join(kwargs['neural_net_dir_path'], kwargs['neural_net_file_name']))

        self.mcts1 = AZ_MCTS(self.g, self.nn, self.args1)
        self.n1p = lambda x: np.argmax(self.mcts1.get_action_prob(x, temp=0, add_dirichlet_noise=False))

    def make_move(self, game: Game) -> tuple[int, int, int] | None:
        """Make a move in the game.
        game: Game: The game instance

        Returns a tuple of the move type, move index, and move state.
        """

        initial_tangled_game_state = convert_erik_game_instance_to_tangled_game_state_instance(game, self.g.adjudicator)

        tangled_move = self.alphazero_agent(initial_tangled_game_state)

        legal_moves = game.get_legal_moves(self.id)

        if not legal_moves or (len(legal_moves) == 1 and legal_moves[0][0] == Game.MoveType.QUIT.value):
            logging.info("No legal moves available")
            return None

        while True:
            erik_move = convert_tangled_move_to_erik_move(len(game.vertices), tangled_move)
            if erik_move not in legal_moves:
                print('something went wrong, your agent picked an illegal move!')
            if erik_move[0] != Game.MoveType.QUIT.value:
                break

        return erik_move

    def alphazero_agent(self, tangled_game_state):
        # tangled_game_state is a TangledGameState instance, returns a TangledMove state
        # tangled_game_state.board should be a list of length N+E, like [0,0,0,0,0,0] or [1,2,0,2,3,1]

        az_board = convert_to_az_board(tangled_game_state.board, self.vertex_count)

        # az_action is an integer from 0..11 for 3-vertex board
        az_action = self.n1p(self.g.getCanonicalForm(az_board, tangled_game_state.next_to_move))
        next_az_board, _ = self.g.getNextState(az_board, tangled_game_state.next_to_move, az_action)

        next_board = convert_from_az_board(next_az_board, self.vertex_count)

        return TangledMove(hash_state_action(tangled_game_state.board, next_board, az_action), tangled_game_state.next_to_move)
