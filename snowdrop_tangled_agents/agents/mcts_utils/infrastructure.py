""" documents and implements a variety of infrastructural components in Tangled """
import ast
import math
import random
import gdown

# ------
# States
# ------

# There are N vertices and E edges in the game graph.
# Vertices are indexed starting at 0
# Edges are in dictionary order with tuples showing the vertices connected,
# ie ([0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3])
# Each vertex can be labeled 0 (unclaimed), 1 (claimed by red), or 2 (claimed by blue)
# Each edge can be labeled 0 (unselected), 1 (grey / J=0), 2 (green / FM / J=-1), or 3 (purple / AFM / J=+1)
# Each game state is stored as a list of these numbers, starting with N vertex states followed by E edge states
# So the length of the game state is N+E
# For example: for the fully connected four-vertex game board (N=4 and E=6)
# state = [0,0,0,0,0,0,0,0,0,0] would be the initial state of each game where nothing had been selected
# vertex_state = state[:N]    -- first N elements of the list are vertex states
# edge_state = state[N:]      -- everything else are edge states
# a possible terminal state could be
# terminal_state = [0,2,0,1,1,3,2,3,2,1]
# red plays on vertex 3 and blue plays on vertex 1
# edges [0, 1], [2, 3] are set to grey/J=0
# edges [0, 3], [1, 3] are set to green/J=-1
# edges [0, 2], [1, 2] are set to purple/J=+1

# -------
# Actions
# -------

# There are N+3E possible actions. The first N correspond to choosing a vertex, and the last 3E correspond to choosing
# a color for an edge (three color possibilities per edge = 3E).
# Each action is associated with an integer. If A<N the action corresponds to selecting a vertex. If A>=N the action
# corresponds to coloring an edge. The edges are in dictionary order, and for each edge the three integer actions are
# associated with +-1, J=0, J=+1 respectively.
# for example, for the 3-vertex 3-edge graph:
# action = 0, 1, 2 correspond to selecting vertex 0, 1, 2 respectively
# action = 3, 4, 5 correspond to selecting edge (0, 1) and setting J=-1, 0, +1 respectively
# action = 6, 7, 8 correspond to selecting edge (0, 2) and setting J=-1, 0, +1 respectively
# action = 9, 10, 11 correspond to selecting edge (1, 2) and setting J=-1, 0, +1 respectively

# ------------------------------
# Allowed Actions From Any State
# ------------------------------

# An important aspect of game play for any game is to know which actions are allowed from any given game state.
# In Tangled, the rules for legal actions are as follows:
# 1. Each player must select exactly one vertex; they can choose any unselected vertex during any of their turns
#       (but see 3)
# 2. Either player can color any unselected edge any color they choose
# 3. If there is exactly one unselected edge and the player has not yet selected a vertex, they must do so
# 4. Gameplay proceeds until all edges are colored and each player has selected a vertex, at which point the
#       game ends and the winner is determined

# -------------------------------------------
# Initial State => Action => Final State Hash
# -------------------------------------------

# In Tangled, actions are deterministic, so given an initial state and an action, the final state is known.
# This hashing function is used to store these triples.


# called by mcts_agent and optimal_agent
# This is the fundamental way we store information about gameplay
def hash_state_action(initial_state, final_state, action):
    # initial_state and final_state are lists of integers of length N+E
    # action is an integer 0 <= action <= N+3E-1
    # returns a string
    return str(initial_state) + "-" + str(final_state) + "|" + str(action)


# required by optimal_agent, called by mcts_tangled
# this reverses the hashing function
def reverse_hashing_state_action(hash_string):
    # hash_string is a string
    # returns two lists of integers of lengths N+E init_state_list, final_state_list
    # and an action integer 0 <= action <= N+3E-1
    state, action = hash_string.split("|")
    initial_state, final_state = [ast.literal_eval(e) for e in state.split("-")]
    return initial_state, final_state, int(action)


# called by mcts_agent
def get_action_from_two_states(initial_state, final_state, vertex_count):
    diff = [[k, final_state[k] - initial_state[k]] for k in range(len(initial_state))
            if (final_state[k] - initial_state[k] > 0)][0]

    if diff[0] < vertex_count:
        action = diff[0]
    else:
        action = vertex_count + (diff[0]-vertex_count)*3 + diff[1] - 1

    return action


# called by optimal_agent
def get_max_action(allowed_actions, q_table):
    # Given a Q-table where keys are initial_state-final_state|action hashes as strings and values are Q-values
    # returns best action, the next state that action leads to, and the max q value

    # allowed_actions is a list of initial_state-final_state|action hashes as strings
    max_action_list = []
    max_q_value = -float('inf')

    # first, find max_q_value
    for each in allowed_actions:
        # each[0] = action is an int between 0 and 11, each[1] is the state that action leads to
        _, final_state, action_taken = reverse_hashing_state_action(each)
        current_q_value = q_table[each]
        if current_q_value >= max_q_value:
            max_q_value = current_q_value

    # now find all actions with that same max_q_value
    for each in allowed_actions:
        _, final_state, action_taken = reverse_hashing_state_action(each)
        current_q_value = q_table[each]
        if current_q_value >= max_q_value:
            max_action_list.append([action_taken, final_state])

    # pick one of them at random
    max_action_to_return, final_state_to_return = random.choice(max_action_list)

    return max_action_to_return, final_state_to_return, max_q_value


# required by mcts_tangled
def add_vertices(internal_vertex_states, current_edge_state, current_player_has_not_chosen_a_vertex, current_player):
    # returns a list of hash_state_action strings corresponding to all the allowed actions for selecting a vertex

    returned_hashes = []
    initial_state = internal_vertex_states + current_edge_state

    for idx in range(len(internal_vertex_states)):
        # available, player hasn't chosen one yet
        if internal_vertex_states[idx] == 0 and current_player_has_not_chosen_a_vertex:
            internal_vertex_states[idx] = current_player
            final_state = internal_vertex_states + current_edge_state

            returned_hashes.append(hash_state_action(initial_state, final_state, action=idx))
            internal_vertex_states[idx] = 0    # reset just in case

    return returned_hashes


# required by mcts_tangled
def add_edges(current_vertex_state, internal_edge_states):
    # returns a list of hash_state_action strings corresponding to all the allowed actions for selecting an edge

    returned_hashes = []
    initial_state = current_vertex_state + internal_edge_states

    for idx in range(len(internal_edge_states)):
        if internal_edge_states[idx] == 0:  # unselected; three possibilities for changing it
            for k in [1, 2, 3]:
                internal_edge_states[idx] = k
                final_state = current_vertex_state + internal_edge_states

                returned_hashes.append(hash_state_action(initial_state, final_state,
                                                         action=len(current_vertex_state) + 3 * idx + k - 1))
            internal_edge_states[idx] = 0  # revert back to zero for the next value of idx

    return returned_hashes


# called by mcts_tangled
def get_tangled_legal_actions(initial_state, vertex_count):
    # this returns legal actions for either player from any initial state
    # if initial_state is a terminal state, this returns an empty list []

    # explicitly split vertex and edge states; each of these is a list
    vertex_states = initial_state[:vertex_count]
    edge_states = initial_state[vertex_count:]
    number_of_unselected_edges = edge_states.count(0)   # count the number of edges labeled zero

    # if the number of moves already made is even, it's red's turn
    if (initial_state.count(1) + initial_state.count(2) + initial_state.count(3)) % 2 == 0:
        current_player = 1  # red's turn
    else:
        current_player = 2

    # explicitly determine whether players have chosen a vertex
    has_chosen_a_vertex = [False, False]

    if 1 in vertex_states:
        has_chosen_a_vertex[0] = True
    if 2 in vertex_states:
        has_chosen_a_vertex[1] = True

    allowed_action_hashes = []

    if number_of_unselected_edges > 1:
        # if there is more than one unselected edge, players can either select a vertex if they haven't
        # already, or select an edge
        allowed_action_hashes += (
            add_vertices(vertex_states, edge_states, not has_chosen_a_vertex[current_player - 1], current_player))
        allowed_action_hashes += add_edges(vertex_states, edge_states)
    else:
        # in this case there's exactly one unselected edge. There's two possibilities corresponding to whether or
        # not the current player has already selected a vertex.
        if not has_chosen_a_vertex[current_player-1]:
            # answer is no, player has not chosen a vertex, so they have to!
            allowed_action_hashes += (
                add_vertices(vertex_states, edge_states, not has_chosen_a_vertex[current_player - 1], current_player))
        else:
            # answer is yes, player has chosen a vertex already, just add the edge candidates
            allowed_action_hashes += add_edges(vertex_states, edge_states)

    return allowed_action_hashes


# called by mcts_tangled
def tangled_state_is_terminal(state, vertex_count):
    # a state is terminal if both players have chosen vertices and all edges have been played

    vertex_states = state[:vertex_count]
    edge_states = state[vertex_count:]

    if edge_states.count(0) == 0 and 1 in vertex_states and 2 in vertex_states:
        return True
    else:
        return False


def get_q_table(graph_number, file_path):
    if graph_number == 2:
        q_table_url = 'https://drive.google.com/uc?id=1eOR-4lFgli2ZpLxD8iM6bvJg0YS2zLFW'
        gdown.download(q_table_url, file_path, quiet=False)

    if graph_number == 3:
        q_table_url = 'https://drive.google.com/uc?id=1LZhhPwY7T5i0_GFgBSvrSsBi27_DGyRQ'
        gdown.download(q_table_url, file_path, quiet=False)


def get_neural_net(graph_number, file_path):
    if graph_number == 2:
        nn_url = 'https://drive.google.com/uc?id=1GNY0GVXHv3i9s_FBJAnPgrmAtx1XYwwa'
        gdown.download(nn_url, file_path, quiet=False)

    if graph_number == 3:
        nn_url = 'https://drive.google.com/uc?id=1WTU9mg_hE4EPKFMSN6PdBwm2gdfg-jt7'
        gdown.download(nn_url, file_path, quiet=False)


def win_probability(rating1, rating2):
    return 1.0 * 1.0 / (1 + 1.0 * math.pow(10, 1.0 * (rating1 - rating2) / 400))


def update_elo_rating(player_1_current_elo, player_2_current_elo, d, k_factor=30):
    # d determines whether player 1 wins (d==1) or player 2 wins (d==2) or draw (d==3)
    p_b = win_probability(player_1_current_elo, player_2_current_elo)       # win probability of Player B
    p_a = win_probability(player_2_current_elo, player_1_current_elo)       # win probability of Player A

    # Updating the Elo Ratings
    if d == 1:
        player_1_current_elo += k_factor * (1 - p_a)
        player_2_current_elo += k_factor * (0 - p_b)
    else:
        if d == 2:
            player_1_current_elo += k_factor * (0 - p_a)
            player_2_current_elo += k_factor * (1 - p_b)
        else:
            player_1_current_elo += k_factor * (0.5 - p_a)
            player_2_current_elo += k_factor * (0.5 - p_b)

    return int(round(player_1_current_elo)), int(round(player_2_current_elo))
