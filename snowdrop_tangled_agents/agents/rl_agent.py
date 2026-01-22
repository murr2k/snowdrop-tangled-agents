"""
Reinforcement Learning Agent for Tangled Game.

This agent uses a trained PPO model from MATLAB to make move decisions.
It requires MATLAB Engine for Python to be installed.
"""

import logging
from pathlib import Path
from typing import Optional

from snowdrop_tangled_game_engine import Game, GameAgentBase, Edge

logger = logging.getLogger(__name__)

# Try to import MATLAB engine
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    logger.warning("MATLAB Engine not available. RLAgent will use fallback strategy.")


class RLAgent(GameAgentBase):
    """
    RL-based agent using trained PPO model from MATLAB.

    Falls back to petersen strategy if MATLAB is unavailable.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        fallback_to_petersen: bool = True,
        **kwargs
    ):
        """
        Initialize the RL agent.

        Args:
            model_path: Path to trained .mat file. If None, uses default.
            fallback_to_petersen: Use petersen strategy if MATLAB unavailable.
        """
        super().__init__(**kwargs)

        self.fallback_to_petersen = fallback_to_petersen
        self.matlab_engine = None
        self.agent_loaded = False

        # Default model path
        if model_path is None:
            default_path = Path(__file__).parent.parent / "matlab" / "models" / "agent_selfplay_v1.mat"
            model_path = str(default_path)

        self.model_path = model_path

        # Try to initialize MATLAB
        if MATLAB_AVAILABLE:
            try:
                self._init_matlab()
            except Exception as e:
                logger.warning(f"Failed to initialize MATLAB: {e}")

    def _init_matlab(self):
        """Initialize MATLAB engine and load agent."""
        logger.info("Starting MATLAB engine...")
        self.matlab_engine = matlab.engine.start_matlab()

        # Add RL toolbox path
        rl_path = Path(__file__).parent.parent / "matlab" / "rl"
        self.matlab_engine.addpath(str(rl_path), nargout=0)

        # Load agent
        logger.info(f"Loading agent from {self.model_path}")
        self.matlab_engine.eval(f"load('{self.model_path}', 'agent')", nargout=0)
        self.agent_loaded = True
        logger.info("RL agent loaded successfully")

    def make_move(self, game: Game) -> tuple[int, int, int]:
        """
        Make a move using the trained RL policy.

        Args:
            game: Current game state

        Returns:
            (move_type, edge_index, edge_state) tuple
        """
        # Get board state
        state = self._get_state_string(game)

        # Try RL policy first
        if self.agent_loaded and self.matlab_engine is not None:
            try:
                action = self._get_rl_action(state, game)
                if action is not None:
                    return action
            except Exception as e:
                logger.warning(f"RL action failed: {e}, using fallback")

        # Fallback to petersen strategy
        if self.fallback_to_petersen:
            return self._petersen_action(game)

        # Last resort: random valid move
        return self._random_action(game)

    def _get_state_string(self, game: Game) -> str:
        """Convert game state to 15-char string."""
        state = ['-'] * 15

        for idx, edge in enumerate(game.graph.edges):
            if edge.state == Edge.State.FM:
                state[idx] = 'G'
            elif edge.state == Edge.State.AFM:
                state[idx] = 'P'
            else:
                state[idx] = '-'

        return ''.join(state)

    def _get_rl_action(self, state: str, game: Game) -> Optional[tuple[int, int, int]]:
        """Get action from RL policy via MATLAB."""
        # Build observation vector
        obs = self._state_to_observation(state)

        # Convert to MATLAB format
        obs_matlab = matlab.double(obs)

        # Get action from agent
        action_cell = self.matlab_engine.getAction(
            self.matlab_engine.eval("agent"),
            obs_matlab
        )

        # Extract action (1-30)
        if isinstance(action_cell, list):
            action = int(action_cell[0])
        else:
            action = int(action_cell)

        # Decode action
        if action <= 15:
            edge_idx = action - 1
            color = Edge.State.FM  # Green
        else:
            edge_idx = action - 16
            color = Edge.State.AFM  # Purple

        # Validate action
        if game.graph.edges[edge_idx].state == Edge.State.ZERO:
            return (Game.MoveType.EDGE, edge_idx, color)

        # Invalid action - try to find valid one
        logger.debug(f"RL selected invalid edge {edge_idx}, finding alternative")
        return None

    def _state_to_observation(self, state: str) -> list:
        """Convert state string to 50-dim observation vector."""
        obs = [0.0] * 50

        # [0:15] Board state
        for i in range(15):
            if state[i] == 'G':
                obs[i] = 1.0
            elif state[i] == 'P':
                obs[i] = -1.0
            else:
                obs[i] = 0.0

        # [15] Turn indicator (our turn = +1)
        obs[15] = 1.0

        # [16:31] Edge categories
        my_edges = [9, 10, 11]
        opp_edges = [5, 12, 13]
        hub_edges = [2, 10, 12]

        for i in range(15):
            if i in my_edges:
                obs[16 + i] = 0.5
            elif i in opp_edges:
                obs[16 + i] = -0.5
            elif i in hub_edges:
                obs[16 + i] = 0.25
            else:
                obs[16 + i] = 0.0

        # [31] Grey count
        grey_count = state.count('-')
        obs[31] = grey_count / 15.0

        # [32:35] Score momentum (placeholder)
        obs[32:35] = [0.0, 0.0, 0.0]

        # [35:50] Game phase one-hot
        if grey_count > 10:
            obs[35:40] = [1.0] * 5
        elif grey_count >= 5:
            obs[40:45] = [1.0] * 5
        else:
            obs[45:50] = [1.0] * 5

        return obs

    def _petersen_action(self, game: Game) -> tuple[int, int, int]:
        """Fallback: Use petersen strategy."""
        from snowdrop_tangled_agents.strategy.petersen_strategy import PetersenStrategy

        state = self._get_state_string(game)
        strategy = PetersenStrategy()

        move = strategy.calculate_move(state, 0.0, [])
        if move:
            edge_idx, color_str = move
            color = Edge.State.FM if color_str == 'G' else Edge.State.AFM
            return (Game.MoveType.EDGE, edge_idx, color)

        return self._random_action(game)

    def _random_action(self, game: Game) -> tuple[int, int, int]:
        """Last resort: Random valid move."""
        import random

        grey_edges = [
            i for i, e in enumerate(game.graph.edges)
            if e.state == Edge.State.ZERO
        ]

        if grey_edges:
            edge_idx = random.choice(grey_edges)
            color = random.choice([Edge.State.FM, Edge.State.AFM])
            return (Game.MoveType.EDGE, edge_idx, color)

        return (Game.MoveType.QUIT, 0, 0)

    def cleanup(self):
        """Clean up MATLAB engine."""
        if self.matlab_engine is not None:
            try:
                self.matlab_engine.quit()
            except:
                pass
            self.matlab_engine = None
