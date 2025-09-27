from snowdrop_adjudicators.utils.utilities import convert_state_key_to_game_state
from snowdrop_tangled_agents.agents.mcts_utils.infrastructure import reverse_hashing_state_action, tangled_state_is_terminal, get_tangled_legal_actions
from snowdrop_tangled_agents.agents.mcts_utils.mcts_common import TwoPlayersAbstractGameState, AbstractGameAction

# The MCTS code requires writing specific classes for GameState and Move; here they are

class TangledMove(AbstractGameAction):
    def __init__(self, hash_string, player):
        # player = +1 (red) or -1 (blue) indicates which player is taking the action.
        # reverse_hashing_state_action(hash_string) returns the initial_state, next_state, int(action)
        # first two are lists, action is an integer 0 <= action <= N+3E-1

        self.hash_string = hash_string
        self.initial_state, self.next_state, self.action = reverse_hashing_state_action(self.hash_string)
        self.player = player

    def __repr__(self):
        # this thing is supposed to be a string that allows the instance to be recreated. If you print(instance)
        # this is what is returned. Not sure why this is here. I left out the player part.
        return self.hash_string


class TangledGameState(TwoPlayersAbstractGameState):

    # used to be x and o respectively
    red = 1
    blue = -1

    # when you first call the TangledGameState class (once per game type played) you need to pass it some
    # information about the game. We only want to do this once for the entire game tree, not for every game node!
    # so this is the solution.
    # "A classmethod() is a built-in function in Python that is used to define a method that is bound to the
    # class and not the instance of the class."
    @classmethod
    def load_game_info(cls, number_of_vertices, edge_list, rollout_adjudicator):
        cls.number_of_vertices = number_of_vertices
        cls.edge_list = edge_list
        # where adjudicator is set for MCTS -- setup must have already been run at this point
        cls.adjudicator = rollout_adjudicator

    def __init__(self, state, next_to_move=1):
        # state is list of length N+E, like [0,0,0,0,0,0] or [1,2,0,2,3,1]
        # next_to_move is whose turn it is to move (either 1 (red) or -1 (blue))
        self.board = state
        self.next_to_move = next_to_move
        self.vertex_state = state[:self.number_of_vertices]
        self.edge_state = state[self.number_of_vertices:]

    def game_result(self):
        # this function returns None if game not over, +1 if red won, -1 if blue won, and 0 if draw

        if tangled_state_is_terminal(self.board, self.number_of_vertices):

            game_state = convert_state_key_to_game_state(self.board,
                                                         self.number_of_vertices,
                                                         self.edge_list)

            results = self.adjudicator.adjudicate(game_state)   # use lookup table
            winner = results['winner']

            w = 666

            if winner == 'red':
                w = 1
            else:
                if winner == 'blue':
                    w = -1
                else:
                    if winner == 'draw':
                        w = 0
                    else:
                        print('Something went wrong!!!', winner)

            return w
        else:   # game not over - no result
            return None

    def is_game_over(self):
        return tangled_state_is_terminal(self.board, self.number_of_vertices)

    def is_move_legal(self, move):
        # move is an instance of the TangledMove class; this checks if correct player moves
        if move.player != self.next_to_move:
            print(move.player, self.next_to_move)
            return False
        else:
            return True

    def move(self, move):
        # move returns a TangledGameState instance from an input TangledMove instance

        if not self.is_move_legal(move):
            raise ValueError("move {0} on board {1} is not legal". format(move, self.board))

        if self.next_to_move == TangledGameState.red:
            next_to_move = TangledGameState.blue
        else:
            next_to_move = TangledGameState.red

        return TangledGameState(move.next_state, next_to_move)

    def get_legal_actions(self):
        # returns list of all TangledMove instances that are legal actions from current instance of TangledGameState

        returned_hashes = get_tangled_legal_actions(self.board, self.number_of_vertices)

        return [TangledMove(each_hash, self.next_to_move) for each_hash in returned_hashes]
