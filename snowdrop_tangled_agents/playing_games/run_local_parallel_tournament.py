""" runs a round-robin tournament of Tangled agents, and returns WLD results and final Elo estimates """
import os
import time
import cProfile
import pstats
import math
import logging
import coloredlogs

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from snowdrop_tangled_game_engine import GraphProperties, GamePlayerBase, LocalGamePlayer, Game, GameAgentBase

from snowdrop_adjudicators import LookupTableAdjudicator, QuantumAnnealingAdjudicator, SimulatedAnnealingAdjudicator

from snowdrop_tangled_agents.utils.utilities import import_agent


def compute_equilibrium_elo(results):
    number_of_games = sum(results[(0, 1)])
    r = {}
    for k, v in results.items():
        if k[0] == 0:
            w = (v[1] + 0.5 * v[2])/ number_of_games
            r[k[1]] = 1000.0 + 400.0 * math.log10((0.0001+w) / (1 - w))
    return r


def play_game(player_1: GameAgentBase, player_2: GameAgentBase, terminal_state_adjudicator, args):
    """ instantiates a game between player_1 and player_2 adjudicated by terminal_state_adjudicator,
    and then plays it locally

    Returns:
        winner of the game, one of 'red', 'blue', 'draw'
    """
    game = Game()
    game.create_game(num_vertices=args['vertex_count'],
                     edges=args['edge_list'],
                     graph_id=args['graph_number'],
                     vertex_ownership=args['vertex_ownership'])
    GamePlayerBase.start_and_play_full_game(LocalGamePlayer, player_1=player_1, player_2=player_2, game=game)

    return terminal_state_adjudicator.adjudicate(game.get_game_state())['winner']    # one of 'red', 'blue', 'draw'


def tournament_worker(competitors, comp_1_idx, comp_2_idx, args, games_per_worker):

    # first, terminal state adjudicator
    terminal_state_adjudicator = None

    if args['terminal_state_adjudicator'] == 'simulated_annealing':
        terminal_state_adjudicator = SimulatedAnnealingAdjudicator()
    else:
        if args['terminal_state_adjudicator'] == 'quantum_annealing':
            terminal_state_adjudicator = QuantumAnnealingAdjudicator()
        else:
            if args['terminal_state_adjudicator'] == 'lookup_table':
                terminal_state_adjudicator = LookupTableAdjudicator()

    terminal_state_adjudicator.setup(**args['terminal_state_adjudicator_kwargs'])

    # next, player 1 and 2 rollout adjudicators
    rollout_adjudicators = {}
    for comp_idx in [comp_1_idx, comp_2_idx]:
        rollout_adjudicators[comp_idx] = None

        if competitors[comp_idx]['kwargs']['rollout_adjudicator'] == 'simulated_annealing':
            rollout_adjudicators[comp_idx] = SimulatedAnnealingAdjudicator()
        else:
            if competitors[comp_idx]['kwargs']['rollout_adjudicator'] == 'quantum_annealing':
                rollout_adjudicators[comp_idx] = QuantumAnnealingAdjudicator()
            else:
                if competitors[comp_idx]['kwargs']['rollout_adjudicator'] == 'lookup_table':
                    rollout_adjudicators[comp_idx] = LookupTableAdjudicator()

        if rollout_adjudicators[comp_idx] is not None:    # if random --> None
            rollout_adjudicators[comp_idx].setup(**competitors[comp_idx]['kwargs']['rollout_adjudicator_args'])

    # Create an agent from the class from a string
    comp_1_agent_class = import_agent(competitors[comp_1_idx]['agent_type'])
    comp_2_agent_class = import_agent(competitors[comp_2_idx]['agent_type'])

    comp_1_kwargs = {
        "rollout_adjudicator": rollout_adjudicators[comp_1_idx],
        "graph_number": args['graph_number'],
        # "lookup_args": competitors[comp_1_idx]['kwargs']['lookup_args'],
        "mcts_rollouts": competitors[comp_1_idx]['kwargs']['mcts_rollouts'],
        "neural_net_dir_path": competitors[comp_1_idx]['kwargs']['neural_net_dir_path'],
        "neural_net_file_name": competitors[comp_1_idx]['kwargs']['neural_net_file_name']
    }

    comp_2_kwargs = {
        "rollout_adjudicator": rollout_adjudicators[comp_2_idx],
        "graph_number": args['graph_number'],
        # "lookup_args": competitors[comp_1_idx]['kwargs']['lookup_args'],
        "mcts_rollouts": competitors[comp_2_idx]['kwargs']['mcts_rollouts'],
        "neural_net_dir_path": competitors[comp_2_idx]['kwargs']['neural_net_dir_path'],
        "neural_net_file_name": competitors[comp_2_idx]['kwargs']['neural_net_file_name']
    }

    # I think we've passed the rollout_adjudicators into the agents here via comp_1_kwargs and comp_2_kwargs
    # random should just work as it doesn't use kwargs

    # noinspection PyArgumentList
    player_1 = comp_1_agent_class(competitors[comp_1_idx]['name'], **comp_1_kwargs)
    # noinspection PyArgumentList
    player_2 = comp_2_agent_class(competitors[comp_2_idx]['name'], **comp_2_kwargs)

    worker_data_red = [play_game(player_1=player_1,
                                 player_2=player_2,
                                 terminal_state_adjudicator=terminal_state_adjudicator,
                                 args=args) for _ in range(games_per_worker)]
    worker_data_blue = [play_game(player_1=player_2,
                                 player_2=player_1,
                                 terminal_state_adjudicator=terminal_state_adjudicator,
                                 args=args) for _ in range(games_per_worker)]

    return [worker_data_red, worker_data_blue]


def parallel_competitive_tournament_play(competitors, args):

    print('beginning parallel competitive tournament play with', args['num_workers'], 'workers and',
          args['number_of_competitors'], 'competitors...')

    start = time.time()

    games_per_worker = args['number_of_games_per_matchup'] // args['num_workers']

    competition_data = {}

    # tournament will be round-robin; each competitor will play half their games as player 1 and half as player 2

    # here comp_2_idx < comp_1_idx
    for comp_1_idx in range(args['number_of_competitors']):
        for comp_2_idx in range(comp_1_idx):

            print('starting', competitors[comp_1_idx]['name'], 'vs', competitors[comp_2_idx]['name'], '...')
            start_here = time.time()

            futures = []

            # with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            with ProcessPoolExecutor(max_workers=args['num_workers']) as executor:
                for _ in range(args['num_workers']):  # eg 4
                    future = executor.submit(tournament_worker, competitors, comp_1_idx, comp_2_idx, args, games_per_worker)
                    futures.append(future)

                competition_data[(comp_1_idx, comp_2_idx)] = []

                for future in as_completed(futures):
                    competition_data[(comp_1_idx, comp_2_idx)].append(future.result())

            print('this round took', time.time() - start_here, 'seconds...')

    print('parallel round robin tournament took', time.time() - start, 'seconds.')

    return competition_data


# copy over neural nets from tangled-alphazero/results; make sure file_paths are correct
# copy over graph_#_adjudicated_unique_terminal_states_for_paper.pkl from tangled-adjudicate/data
# check args.graph_number, epsilon, anneal_time and num_reads are correct

def main():

    logging.getLogger(__name__)
    coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

    graph_properties = GraphProperties()

    for graph_number in [2, 11, 12, 18, 19, 20]:
        print('-------------------------------------------------------------')
        print('Tournament for graph number', graph_number, 'starting now ...')
        graph_index = graph_properties.allowed_graphs.index(graph_number)

        expected_random_wld = {2: [7./27, 7./27, 13./27],
                               11: [3./9, 3./9, 3./9],
                               12: [60514./177146, 60514./177146, 56118./177146],
                               18: [5038./19682, 5038./19682, 9606./19682],
                               19: [678./2187, 678./2187, 831./2187],
                               20: [53./243, 53./243, 137./243]}

        # parameters for the tournament
        args = {'graph_number': graph_number,
                'data_dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
                'terminal_state_adjudicator': 'lookup_table',
                'terminal_state_adjudicator_kwargs': {'epsilon': graph_properties.epsilon_values[graph_index],
                                                      'graph_number': graph_number},
                'number_of_games_per_matchup': 500,
                'num_workers': 10,
                'vertex_count': graph_properties.graph_database[graph_number]['num_nodes'],
                'edge_list': graph_properties.graph_database[graph_number]['edge_list'],
                'vertex_ownership': (graph_properties.graph_database[graph_number]['player1_node'],
                                     graph_properties.graph_database[graph_number]['player2_node'])}

        competitors = {
            0: {
                'name': 'Random_0',
                'agent_type': 'snowdrop_tangled_agents.RandomRandyAgent',
                'kwargs': {
                    'rollout_adjudicator': None,
                    'rollout_adjudicator_args': None,
                    'mcts_rollouts': None,
                    'Elo': 1000,
                    'WLD': [0, 0, 0],
                    'neural_net_dir_path': None,
                    'neural_net_file_name': None,
                }
            },
            # 1: {
            #     'name': 'Random_1',
            #     'agent_type': 'snowdrop_tangled_agents.RandomRandyAgent',
            #     'kwargs': {
            #         'rollout_adjudicator': None,
            #         'rollout_adjudicator_args': None,
            #         'mcts_rollouts': None,
            #         'Elo': 1000,
            #         'WLD': [0, 0, 0],
            #         'neural_net_dir_path': None,
            #         'neural_net_file_name': None,
            #     }
            # },
            1: {
                'name': 'MCTS_10',
                'agent_type': 'snowdrop_tangled_agents.MCTSAgent',
                'kwargs': {
                    'rollout_adjudicator': 'lookup_table',
                    'rollout_adjudicator_args': {'epsilon': graph_properties.epsilon_values[graph_index],
                                                 'graph_number': graph_number},
                    'mcts_rollouts': 10,
                    'Elo': 1000,
                    'WLD': [0, 0, 0],
                    'neural_net_dir_path': None,
                    'neural_net_file_name': None,
                }
            },
            2: {
                'name': 'AlphaZero_10',
                'agent_type': 'snowdrop_tangled_agents.AlphaZeroAgent',
                'kwargs': {
                    'rollout_adjudicator': 'lookup_table',
                    'rollout_adjudicator_args': {'epsilon': graph_properties.epsilon_values[graph_index],
                                                 'graph_number': graph_number},
                    'mcts_rollouts': 10,
                    'Elo': 1000,
                    'WLD': [0, 0, 0],
                    'neural_net_dir_path': os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
                    'neural_net_file_name': "alphazero_nn_graph_" + str(graph_number) + ".pth",
                }
            }
        }

        # # competitor 0 is baseline Elo of 1000 by definition
        # competitors = {
        #     0: {
        #         'name': 'MCTS[100, SA]',
        #         'agent_type': 'tangled_agent.MCTSAgent',
        #         'kwargs': {
        #             'rollout_adjudicator': 'simulated_annealing',
        #             'rollout_adjudicator_args': {'epsilon': graph_properties.epsilon_values[graph_number],
        #                                          'num_reads': 10000},
        #             'mcts_rollouts': 100,
        #             'Elo': 1000,
        #             'WLD': [0, 0, 0],
        #             'neural_net_dir_path': None,
        #             'neural_net_file_name': None,
        #         }
        #     },
        #     1: {
        #         'name': 'Random',
        #         'agent_type': 'tangled_agent.RandomRandyAgent',
        #         'kwargs': {
        #             'rollout_adjudicator': None,
        #             'rollout_adjudicator_args': None,
        #             'mcts_rollouts': None,
        #             'Elo': 1000,
        #             'WLD': [0, 0, 0],
        #             'neural_net_dir_path': None,
        #             'neural_net_file_name': None,
        #         }
        #     },
        #     2: {
        #         'name': 'MCTS[10, SA]',
        #         'agent_type': 'tangled_agent.MCTSAgent',
        #         'kwargs': {
        #             'rollout_adjudicator': 'simulated_annealing',
        #             'rollout_adjudicator_args': {'epsilon': graph_properties.epsilon_values[graph_number],
        #                                          'num_reads': 10000},
        #             'mcts_rollouts': 10,
        #             'Elo': 1000,
        #             'WLD': [0, 0, 0],
        #             'neural_net_dir_path': None,
        #             'neural_net_file_name': None,
        #         }
        #     },
        #     3: {
        #         'name': 'MCTS[10, QA]',
        #         'agent_type': 'tangled_agent.MCTSAgent',
        #         'kwargs': {
        #             'rollout_adjudicator': 'lookup_table',
        #             'rollout_adjudicator_args': {'epsilon': graph_properties.epsilon_values[graph_number],
        #                                          'graph_number': graph_number},
        #             'mcts_rollouts': 10,
        #             'Elo': 1000,
        #             'WLD': [0, 0, 0],
        #             'neural_net_dir_path': None,
        #             'neural_net_file_name': None,
        #         }
        #     },
        #     4: {
        #         'name': 'MCTS[100, QA]',
        #         'agent_type': 'tangled_agent.MCTSAgent',
        #         'kwargs': {
        #             'rollout_adjudicator': 'lookup_table',
        #             'rollout_adjudicator_args': {'epsilon': graph_properties.epsilon_values[graph_number],
        #                                          'graph_number': graph_number},
        #             'mcts_rollouts': 100,
        #             'Elo': 1000,
        #             'WLD': [0, 0, 0],
        #             'neural_net_dir_path': None,
        #             'neural_net_file_name': None,
        #         }
        #     },
        #     5: {
        #         'name': 'AlphaZero[10, SA]',
        #         'agent_type': 'tangled_agent.AlphaZeroAgent',
        #         'kwargs': {
        #             'rollout_adjudicator': 'simulated_annealing',
        #             'rollout_adjudicator_args': {'epsilon': graph_properties.epsilon_values[graph_number],
        #                                          'num_reads': 10000},
        #             'mcts_rollouts': 10,
        #             'Elo': 1000,
        #             'WLD': [0, 0, 0],
        #             'neural_net_dir_path': args['data_dir'],
        #             'neural_net_file_name': sa_nn_path,
        #         }
        #     },
        #     6: {
        #         'name': 'AlphaZero[10, QA]',
        #         'agent_type': 'tangled_agent.AlphaZeroAgent',
        #         'kwargs': {
        #             'rollout_adjudicator': 'lookup_table',
        #             'rollout_adjudicator_args': {'epsilon': graph_properties.epsilon_values[graph_number],
        #                                          'graph_number': graph_number},
        #             'mcts_rollouts': 10,
        #             'Elo': 1000,
        #             'WLD': [0, 0, 0],
        #             'neural_net_dir_path': args['data_dir'],
        #             'neural_net_file_name': qa_nn_path,
        #         }
        #     },
        #     7: {
        #         'name': 'AlphaZero[100, SA]',
        #         'agent_type': 'tangled_agent.AlphaZeroAgent',
        #         'kwargs': {
        #             'rollout_adjudicator': 'simulated_annealing',
        #             'rollout_adjudicator_args': {'epsilon': graph_properties.epsilon_values[graph_number],
        #                                          'num_reads': 10000},
        #             'mcts_rollouts': 100,
        #             'Elo': 1000,
        #             'WLD': [0, 0, 0],
        #             'neural_net_dir_path': args['data_dir'],
        #             'neural_net_file_name': sa_nn_path,
        #         }
        #     },
        #     8: {
        #         'name': 'AlphaZero[1000, QA]',
        #         'agent_type': 'tangled_agent.AlphaZeroAgent',
        #         'kwargs': {
        #             'rollout_adjudicator': 'lookup_table',
        #             'rollout_adjudicator_args': {'epsilon': graph_properties.epsilon_values[graph_number],
        #                                          'graph_number': graph_number},
        #             'mcts_rollouts': 1000,
        #             'Elo': 1000,
        #             'WLD': [0, 0, 0],
        #             'neural_net_dir_path': args['data_dir'],
        #             'neural_net_file_name': qa_nn_path,
        #         }
        #     }
        # }

        args['number_of_competitors'] = len(competitors)

        start = time.time()

        competition_data = parallel_competitive_tournament_play(competitors=competitors, args=args)

        print('tournament took', time.time() - start, 'seconds...')

        results = {}

        for k, v in competition_data.items():
            p1_wins_playing_red = 0
            p1_wins_playing_blue = 0
            p2_wins_playing_red = 0
            p2_wins_playing_blue = 0
            p1_draws_playing_red = 0
            p1_draws_playing_blue = 0

            for each in v:
                p1_wins_playing_red += each[0].count('red')
                p2_wins_playing_red += each[0].count('blue')
                p1_wins_playing_blue += each[1].count('blue')
                p2_wins_playing_blue += each[1].count('red')
                p1_draws_playing_red += each[0].count('draw')
                p1_draws_playing_blue += each[1].count('draw')

            print(competitors[k[0]]['name'], 'red and',
                  competitors[k[1]]['name'], 'blue: %d / %d / %d' % (p1_wins_playing_red, p2_wins_playing_red, p1_draws_playing_red))
            print(competitors[k[0]]['name'], 'blue and',
                  competitors[k[1]]['name'], 'red: %d / %d / %d' % (p1_wins_playing_blue, p2_wins_playing_blue, p1_draws_playing_blue))

            print(competitors[k[0]]['name'], 'vs', competitors[k[1]]['name'], '%d / %d / %d' %
                  (p1_wins_playing_red+p1_wins_playing_blue,
                   p2_wins_playing_red+p2_wins_playing_blue,
                   p1_draws_playing_red+p1_draws_playing_blue))

            results[(k[1], k[0])] = [p2_wins_playing_red+p2_wins_playing_blue,
                                     p1_wins_playing_red+p1_wins_playing_blue,
                                     p1_draws_playing_red+p1_draws_playing_blue]

            competitors[k[0]]['kwargs']['WLD'][0] += p1_wins_playing_red+p1_wins_playing_blue
            competitors[k[0]]['kwargs']['WLD'][1] += p2_wins_playing_red+p2_wins_playing_blue
            competitors[k[0]]['kwargs']['WLD'][2] += p1_draws_playing_red+p1_draws_playing_blue

            competitors[k[1]]['kwargs']['WLD'][0] += p2_wins_playing_red+p2_wins_playing_blue
            competitors[k[1]]['kwargs']['WLD'][1] += p1_wins_playing_red+p1_wins_playing_blue
            competitors[k[1]]['kwargs']['WLD'][2] += p1_draws_playing_red+p1_draws_playing_blue

        print()
        print('******************************************************')
        print('graph', args['graph_number'], 'tournament is done ... ')

        for idx in range(len(competitors)):
            print(competitors[idx]['name'], ': W/L/D of', competitors[idx]['kwargs']['WLD'])
            num_games = sum(competitors[idx]['kwargs']['WLD'])
            print(competitors[idx]['name'], ': W/L/D % tage of', [competitors[idx]['kwargs']['WLD'][k]/num_games for k in range(3)])
            print(competitors[idx]['name'], ': expected random W/L/D % tage of', [round(expected_random_wld[graph_number][k], 5) for k in range(3)])

        print()
        print('******************************************************')

        print('results here:', results)
        r = compute_equilibrium_elo(results)
        print('r is:', r)

        for k, v in r.items():
            competitors[k]['kwargs']['Elo'] = v

        # process_tournament_results_per_game(results=results, competitors=competitors)
        print('-------------------------------------------------------------')

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Ensures correct behavior in PyCharm

    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats('tottime').print_stats(4)   # show top 4 results
