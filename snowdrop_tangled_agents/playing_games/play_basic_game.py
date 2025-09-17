import importlib
import os
import time
import logging
from typing import Any

from snowdrop_tangled_agents.agents.random_agent import RandomRandyAgent
from snowdrop_tangled_agents.utils.utilities import import_agent
from snowdrop_tangled_game_engine import GamePlayerBase, LocalGamePlayer, GameAgentBase, Game, GraphProperties
from snowdrop_adjudicators import LookupTableAdjudicator, GameState


# def play_remote_game(game_id: str, host: str, player: GameAgentBase, update_display: callable = None, force_new_credentials: bool = False, **kwargs):
#     """
#     Sets up and starts a remote game with a random agent.
#
#     Args:
#         game_id (str): The ID of the game to connect to (provided by create_game on the website).
#         host (str): The host URL for the remote game server (provided by create_game on the website).
#         player (GameAgentBase): The player agent to use.
#     """
#
#     # Start the game using the helper function for remote games (RemoteGamePlayer)
#     # This will connect to the server and join the game, generate your credentials (if needed),
#     # then start the game loop.
#     # This will repeatedly call the agent's make_move methods and pass it back to the
#     # server until the game is over
#     GamePlayerBase.start_game(RemoteGamePlayer, player, game_id, host, update_display=update_display, force_new_credentials=force_new_credentials, **kwargs)





# def setup_args() -> configargparse.ArgParser:
#     """
#     Set up handling of command-line arguments.
#     """
#
#     description = """
# Play a game of Tangled with one or two custom agents, either for local testing
#  or for online play."""
#
#     epilog = """
# Run this program with the --help flag to see the full list of options.
# """
#
#     usage = """
# Local game:
#     python -m tangled-agent --agent your_agent.YourAgentClass
# Remote game:
#     python -m tangled-agent --game-id <game-id> --agent your_agent.YourAgentClass [--player-index 1] [--new-credentials]
#
# Or you can use a configuration file (e.g. config.ini) with the following format:
#
# ```config.ini
# [DEFAULT]
# agent = your_agent.YourAgentClass
# agent_2 = your_agent.OtherAgentClass
# graph_number = 3
# game-id = <game-id>
# host = <game-host>
# new-credentials = False
# ```
#
# or a combination of both:
#
#     python -m tangled_agent --config config.ini --game_id <game-id> --host <game-host> --new-credentials
#
# If a game-id is provided, the script will connect to the remote game server and join the game as either player 1 or 2 using the specified GameAgent subclass.
# """
#
#     parser = configargparse.ArgParser(description=description,
#                                       epilog=epilog,
#                                       usage=usage,
#                                       default_config_files=['config.ini'])
#
#     parser.add('--config', is_config_file=True, help='Path to the configuration file.')
#     parser.add_argument('--game-id', type=str, default=None, help='The ID of the game to connect to for a remote game.')
#     parser.add_argument('--host', type=str, default='https://game-service-fastapi-blue-pond-7261.fly.dev',
#                         help='The host URL for the remote game server.')
#     parser.add_argument('--new-credentials', action='store_true', help='Force new credentials.')
#
#     parser.add_argument('--agent', type=str, default="tangled_agent.RandomRandyAgent",
#                         help='The qualified name of the game agent class (module.ClassName) to use.')
#     # parser.add_argument('--agent', type=str, default="tangled_agent.MCTSAgent", help='The qualified name of the game agent class (module.ClassName) to use for player 2.')
#     # parser.add_argument('--agent', type=str, default="tangled_agent.OptimalAgent", help='The qualified name of the game agent class (module.ClassName) to use for player 2.')
#     # parser.add_argument('--agent', type=str, default="tangled_agent.AlphaZeroAgent", help='The qualified name of the game agent class (module.ClassName) to use for player 2.')
#
#     # parser.add_argument('--agent-2', type=str, default="tangled_agent.RandomRandyAgent", help='The qualified name of the game agent class (module.ClassName) to use for player 2.')
#     # parser.add_argument('--agent-2', type=str, default="tangled_agent.MCTSAgent", help='The qualified name of the game agent class (module.ClassName) to use for player 2.')
#     # parser.add_argument('--agent-2', type=str, default="tangled_agent.OptimalAgent", help='The qualified name of the game agent class (module.ClassName) to use for player 2.')
#     parser.add_argument('--agent-2', type=str, default="tangled_agent.AlphaZeroAgent",
#                         help='The qualified name of the game agent class (module.ClassName) to use for player 2.')
#
#     parser.add_argument('--agent-name', type=str, default="default",
#                         help='The name of the player agent to use for record keeping (online games).')
#     parser.add_argument('--player-index', type=int, default=0,
#                         help='The index of the player to be in the game (0=no preference, 1 or 2).')
#     parser.add_argument('--graph-display-delay', type=float, default=1.0,
#                         help='When using GRAPH display, the GUI update delay after making a move.')
#     parser.add_argument('--graph-number', type=int, default=4,
#                         help='Standard integer graph number; default is graph_number=3, the complete graph on 4 vertices.')
#     valid_log_levels = ['PRETTY', 'GRAPH', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
#     parser.add_argument('--log-level', type=str, default='PRETTY', choices=valid_log_levels,
#                         help=f'The logging level to use ({", ".join(valid_log_levels)}).')
#
#     return parser


# def main():
#     """
#     Main entry point for the script.
#     Get the command-line arguments and start the game.
#     """
#
#     graph_properties = GraphProperties()
#
#     args: dict[str, Any] = {'graph_number': 11,
#                             'agent_1': 'RandomRandyAgent',
#                             'agent_2': 'RandomRandyAgent'}
#
#                             # 'game_id': None}
#
#     # script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the current script
#     # data_dir_path = os.path.join(script_dir, 'geordie_agents')
#
#     # # if using OptimalAgent, need to download / load in lookup table; only provided here for graph_number 2 or 3
#     # q_table = None
#     # if 'Optimal' in args.agent or 'Optimal' in args.agent_2:
#     #     if args.graph_number in [2, 3]:
#     #         file_path = os.path.join(data_dir_path, 'graph_' + str(args.graph_number) + '_q_table.pkl')
#     #         if not os.path.exists(file_path):
#     #             get_q_table(args.graph_number, file_path)
#     #         with open(file_path, "rb") as fp:
#     #             q_table = pickle.load(fp)
#     #     else:
#     #         sys.exit(print('Optimal agent only available for graph_number 2 or 3...'))
#     #
#     # # if using AlphaZeroAgent, need to download neural net; only provided here for graph_number 2 or 3
#     # if 'Alpha' in args.agent or 'Alpha' in args.agent_2:
#     #     if args.graph_number in [2, 3]:
#     #         file_path = os.path.join(data_dir_path, 'graph_' + str(args.graph_number) + '_neural_net.weights.h5')
#     #         if not os.path.exists(file_path):
#     #             get_neural_net(args.graph_number, file_path)
#     #     else:
#     #         sys.exit(print('AlphaZero neural net only available for graph_number 2 or 3...'))
#
#     # these are keyword arguments that cover the needs of AlphaZeroAgent, OptimalAgent and MCTSAgent. If you are using
#     # RandomRandyAgent these aren't used.
#
#     player1_kwargs = {
#         "graph_number": args['graph_number']
#         # "mcts_rollouts": 50,
#         # "q_table": q_table,
#         # "neural_net_dir_path": data_dir_path,
#         # "neural_net_file_name": 'graph_' + str(args['graph_number']) + '_neural_net.weights.h5',
#     }
#
#     player2_kwargs = {
#         "graph_number": args['graph_number']
#         # "mcts_rollouts": 50,
#         # "q_table": q_table,
#         # "neural_net_dir_path": data_dir_path,
#         # "neural_net_file_name": 'graph_' + str(args['graph_number']) + '_neural_net.weights.h5',
#     }
#
#     # Create agents from the class from a string
#     agent_class_1 = import_agent(args['agent_1'])
#     agent_class_2 = import_agent(args['agent_2'])
#
#     # Create two agents with names (e.g. RandomRandyAgent("player1"))
#     player1 = agent_class_1(f'{agent_class_1.__name__}', **player1_kwargs)
#     player2 = agent_class_2(f'{agent_class_2.__name__}', **player2_kwargs)
#
#     logging.info(f"Game Specification: {graph_properties}")
#
#     game = Game()
#
#     game.create_game(graph_properties.graph_database[args['graph_number']]['num_nodes'],
#                      graph_properties.graph_database[args['graph_number']]['edge_list'],)
#
#     play_local_game(player1, player2, game)  # Play a local game with the two agents
#
#     time.sleep(2)
#
# if __name__ == "__main__":
#     main()

