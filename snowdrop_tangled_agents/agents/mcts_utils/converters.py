import numpy as np

from snowdrop_tangled_agents.agents.mcts_utils.mcts_tangled import TangledGameState


# called by mcts_agent, optimal_agent
def convert_erik_game_instance_to_tangled_game_state_instance(erik_game_instance, rollout_adjudicator):
    # should be independent of which agent I use

    game_state = erik_game_instance.get_game_state()

    # this needs to pass in the rollout adjudicator
    TangledGameState.load_game_info(number_of_vertices=game_state['num_nodes'],
                                    edge_list=game_state['edges'],
                                    rollout_adjudicator=rollout_adjudicator)

    # for erik, 1 is red and 2 is blue; for geordie, +1 is red and -1 is blue; 1/2 ==> +1 / -1
    # at the start of the game
    if game_state['current_player_index'] == 1:
        next_to_move = 1
    else:
        next_to_move = -1

    # extract geordie state from erik state
    edge_state_list = [each[2] for each in game_state['edges']]

    vertex_state_list = [0] * game_state['num_nodes']
    if game_state['player1_node'] != -1:
        vertex_state_list[game_state['player1_node']] = 1
    if game_state['player2_node'] != -1:
        vertex_state_list[game_state['player2_node']] = 2

    initial_start_state = vertex_state_list + edge_state_list

    return TangledGameState(state=initial_start_state, next_to_move=next_to_move)


# called by mcts_agent, optimal_agent
# todo we have to track down getting this correct -- question is what tangled_move_instance looks like
def convert_tangled_move_to_erik_move(vertex_count, tangled_move_instance):
    # also independent of agent

    # erik_move is a list of three elements, like [0, 13, 2]  --> color edge 13 green
    #             list[tuple[int, int, int]]: A list of legal moves for the player.
    #                 Each move is a list of three integers: move type, move index, and move state.
    #                 Move types are from the MoveType enum.
    #     class MoveType(IntEnum):
    #         """
    #         The type of move that can be made in the game.
    #
    #         Values:
    #             NONE (int): No legal moves or it's not the player's turn.
    #             EDGE (int): Set a preferred edge state.
    #             QUIT (int): Quit the game.
    #         """
    #
    #         NONE = -1  # No legal moves or it's not the player's turn
    #         EDGE = 0
    #         QUIT = 1
    #                 Move indices are the edge index.
    #                 Move states are from the Edge.State enums.

    # first element is -1, 0, 1
    # second is index of edge
    # third is edge move type
    # class Edge:
    #     """
    #     An edge in the game graph.
    #
    #     Members:
    #         vertices (Tuple[int, int]): The two vertices that the edge connects.
    #         state (Edge.State): The state of the edge.
    #
    #     Enums:
    #         State: The state of an edge.
    #     """
    #
    #     class State(IntEnum):
    #         """
    #         The state of an edge.
    #         Use this to set or interpret the state of an edge.
    #
    #         Values:
    #             NONE (int): The edge has not been set.
    #             ZERO (int): The edge has been set to ZERO coupling.
    #             FM (int): The edge is ferromagnetic (FM).
    #             AFM (int): The edge is antiferromagnetic (AFM).
    #         """
    #
    #         NONE = 0
    #         ZERO = 1
    #         FM = 2
    #         AFM = 3

    # hash_string = '[0, 0, 0, 0, 0, 3]-[2, 0, 0, 0, 0, 3]|0'  (str)
    # action (int), initial_state (list), next_state (list)
    # player = -1

    # so we have to
    # (a) determine if the move was vertex, edge, or to terminal state
    # (b) which vertex/edge
    # (c) which player

    # tangled_move_instance.player returns +1/-1 based on which player made the move
    # instead of turn_count %2 which gives 0 if red chose it and 1 if blue chose it

    difference = [i - j for (i, j) in zip(tangled_move_instance.next_state, tangled_move_instance.initial_state)]
    idx = [i for i, e in enumerate(difference) if e != 0][0]

    # this first part can't happen with fixed tokens
    if idx < vertex_count:
        first_idx_to_return = 0  # vertex move
        second_idx_to_return = idx

        if tangled_move_instance.player == 1:
            third_idx_to_return = 1  # should be red making the move
        else:
            third_idx_to_return = 2  # should be blue making the move
    else:
        first_idx_to_return = 0  # edge move --> changed from 1 to 0 for new edge move
        second_idx_to_return = idx - vertex_count
        third_idx_to_return = tangled_move_instance.next_state[idx]

    return (first_idx_to_return, second_idx_to_return, third_idx_to_return)


def convert_to_az_board(my_board, vertex_count):
    # my_board should be a list of length N+E, like [0,0,0,0,0,0] or [1,2,0,2,3,1]

    vertex_state = my_board[:vertex_count]
    edge_state = my_board[vertex_count:]

    new_vertex_state = []
    for each in vertex_state:
        if each == 2:
            new_vertex_state.append(-1)
        else:
            new_vertex_state.append(each)

    new_edge_state = []
    for each in edge_state:
        if each == 0:
            new_edge_state += [0, 0, 0]
        if each == 1:
            new_edge_state += [1, 0, 0]
        if each == 2:
            new_edge_state += [0, 1, 0]
        if each == 3:
            new_edge_state += [0, 0, 1]

    # returns np.array of length N+3*E, like [0,0,0, 0,0,0, 0,0,0, 0,0,0]
    #     # or [1,-1,0, 0,1,0, 0,0,1, 1,0,0]
    return np.array(new_vertex_state + new_edge_state)


def convert_from_az_board(az_board, vertex_count):
    # az_board should be np.array of length N+3*E, like [0,0,0, 0,0,0, 0,0,0, 0,0,0]
    # or [1,-1,0, 0,1,0, 0,0,1, 1,0,0]

    az_board_list = list(az_board)

    vertex_state = az_board_list[:vertex_count]
    edge_state = az_board_list[vertex_count:]

    my_vertex_state = []
    for each in vertex_state:
        if each == -1:
            my_vertex_state.append(2)
        else:
            my_vertex_state.append(int(each))

    edge_groups = [edge_state[3 * k:3 * (k + 1)] for k in range(int(len(edge_state) / 3))]

    my_edge_state = []
    for each in edge_groups:  # each is a list of three bits like [0,1,0]
        if sum(each) == 0:
            edge_int = 0
        else:
            edge_int = 1 + each.index(1)

        my_edge_state.append(edge_int)

    return my_vertex_state + my_edge_state   # should be a list length N+E like [0,0,0,0,0,0] or [1,2,0,2,3,1]
