""" runs a round-robin tournament of Tangled agents, and returns ranked W/L/D results """
import time
import cProfile
import pstats
import logging
import coloredlogs
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from snowdrop_tangled_agents.utils.utilities import import_agent

from snowdrop_tangled_game_engine import GraphProperties, GamePlayerBase, LocalGamePlayer, Game, GameAgentBase
from snowdrop_adjudicators import SimulatedAnnealingAdjudicator, SchrodingerEquationAdjudicator, Adjudicator


def play_game(player_1: GameAgentBase, player_2: GameAgentBase, terminal_state_adjudicator: Adjudicator, args):
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

    # create and setup Adjudicator instance; out of the box choices are simulated annealing or schrodinger equation
    # default is simulated annealing
    if args['terminal_state_adjudicator'] == 'simulated_annealing':
        terminal_state_adjudicator = SimulatedAnnealingAdjudicator()
    else:
        if args['terminal_state_adjudicator'] == 'schrodinger_equation':
            terminal_state_adjudicator = SchrodingerEquationAdjudicator()
        else:
            raise Exception("You can build your own adjudicator! Call it here!")

    terminal_state_adjudicator.setup(**args['terminal_state_adjudicator_kwargs'])

    # Create an agent from the class from a string
    comp_1_agent_class = import_agent(competitors[comp_1_idx]['agent_type'])
    comp_2_agent_class = import_agent(competitors[comp_2_idx]['agent_type'])

    # Random agents don't need any extra arguments, but this is how you add them if you do
    comp_1_kwargs = {"graph_number": args['graph_number']}
    comp_2_kwargs = {"graph_number": args['graph_number']}

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


def main():

    logging.getLogger(__name__)
    coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

    graph_properties = GraphProperties()

    for graph_number in [2, 11, 12, 18, 19, 20]:
        print('-------------------------------------------------------------')
        print('Tournament for graph number', graph_number, 'starting now ...')
        graph_index = graph_properties.allowed_graphs.index(graph_number)

        # as described in the X-Prize Phase 1 Submission document, simulated annealing mis-adjudicates some of the
        # terminal states for some of the graphs. Graphs 2 (P3), 11 (P3), 20 (diamond) all have the same adjudications
        # as ground truth (schrodinger, hardware quantum annealing, simulated annealing all agree on all terminal
        # states). Barbell (19), 3-Prism (18), and Moser Spindle (12) have increasing numbers of adjudication errors
        # for simulated annealing. If you run random vs random agents you should get unbiased sampling of terminal
        # states and therefore should see W/L/D numbers that match below. Note: simulated annealing probabilities
        # are stochastic and depend on num_reads; the numbers included here are for num_reads=10,000.

        # terminal state adjudications for all enumerated terminal states for these graphs
        expected_random_wld = {'schrodinger_equation': {2: [7. / 27, 7. / 27, 13. / 27],
                                                        11: [3. / 9, 3. / 9, 3. / 9],
                                                        12: [60514. / 177146, 60514. / 177146, 56118. / 177146],
                                                        18: [5038. / 19683, 5038. / 19683, 9607. / 19683],
                                                        19: [678. / 2187, 678. / 2187, 831. / 2187],
                                                        20: [53. / 243, 53. / 243, 137. / 243]},
        # contains systematic adjudication errors for graphs 12, 18, 19 -- this is why these are different
                               'simulated_annealing': {2: [7. / 27, 7. / 27, 13. / 27],
                                                       11: [3. / 9, 3. / 9, 3. / 9],
                                                       12: [74699. / 177146, 74715. / 177146, 27733. / 177146],
                                                       18: [5840. / 19683, 5841. / 19683, 8002. / 19683],
                                                       19: [724. / 2187, 724. / 2187, 739. / 2187],
                                                       20: [53. / 243, 53. / 243, 137. / 243]}}

        # parameters for the tournament
        args = {'graph_number': graph_number,
                'terminal_state_adjudicator': 'simulated_annealing',  # or 'schrodinger_equation' (takes forever tho)
                # anneal_time is required for Schrödinger adjudicator; num_reads for simulated annealing
                'terminal_state_adjudicator_kwargs': {'epsilon': graph_properties.epsilon_values[graph_index],
                                                      'anneal_time': graph_properties.anneal_times[graph_index],
                                                      'num_reads': 10000},
                'number_of_games_per_matchup': 50000,
                'num_workers': 10,
                'vertex_count': graph_properties.graph_database[graph_number]['num_nodes'],
                'edge_list': graph_properties.graph_database[graph_number]['edge_list'],
                'vertex_ownership': (graph_properties.graph_database[graph_number]['player1_node'],
                                     graph_properties.graph_database[graph_number]['player2_node'])}

        # add new agents here to add them to the round-robin tournament!
        competitors = {
            0: {'name': 'Random_0',
                'agent_type': 'snowdrop_tangled_agents.RandomRandyAgent',
                'kwargs': {'WLD': [0, 0, 0]}},
            1: {'name': 'Random_1',
                'agent_type': 'snowdrop_tangled_agents.RandomRandyAgent',
                'kwargs': {'WLD': [0, 0, 0]}}
        }

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

        print('graph', args['graph_number'], 'tournament is done ... ')

        for idx in range(len(competitors)):
            print(competitors[idx]['name'], ': W/L/D of', competitors[idx]['kwargs']['WLD'])
            num_games = sum(competitors[idx]['kwargs']['WLD'])
            print(competitors[idx]['name'], ': W/L/D % tage of',
                  [round(competitors[idx]['kwargs']['WLD'][k]/num_games, 2) for k in range(3)])
            print(competitors[idx]['name'], ': for', args['terminal_state_adjudicator'],
                  'terminal state adjudicator: expected random W/L/D % tage of',
                  [round(expected_random_wld[args['terminal_state_adjudicator']][graph_number][k], 2) for k in range(3)])


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Ensures correct behavior in PyCharm

    with cProfile.Profile() as pr:
        main()

    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats('tottime').print_stats(4)   # show top 4 results
