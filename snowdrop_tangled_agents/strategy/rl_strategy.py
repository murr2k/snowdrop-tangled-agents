"""
RL Strategy for web play using trained MATLAB model.

This wraps the RLAgent logic for use with the WebPlayer strategy interface.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import MATLAB engine
MATLAB_AVAILABLE = False
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    logger.info("MATLAB Engine not available. RLStrategy will use fallback.")


class RLStrategy:
    """
    RL-based strategy using trained PPO model from MATLAB.

    Falls back to petersen strategy if MATLAB is unavailable.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        fallback_to_petersen: bool = True,
    ):
        self.fallback_to_petersen = fallback_to_petersen
        self.matlab_engine = None
        self.agent_loaded = False
        self.move_history = []

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
        logger.info("Starting MATLAB engine for RL strategy...")
        self.matlab_engine = matlab.engine.start_matlab()

        # Add RL toolbox path
        rl_path = Path(__file__).parent.parent / "matlab" / "rl"
        self.matlab_engine.addpath(str(rl_path), nargout=0)

        # Load agent
        if Path(self.model_path).exists():
            logger.info(f"Loading agent from {self.model_path}")
            self.matlab_engine.eval(f"load('{self.model_path}', 'agent')", nargout=0)
            self.agent_loaded = True
            logger.info("RL agent loaded successfully")
        else:
            logger.warning(f"Model not found: {self.model_path}")

    def calculate_move(self, state: str, score: float, history: list) -> Optional[tuple[int, str]]:
        """
        Calculate the next move using the trained RL policy.

        Args:
            state: 15-char board state string ('-'=grey, 'G'=green, 'P'=purple)
            score: Current score
            history: List of (edge, color, score) tuples

        Returns:
            (edge_index, color) tuple or None
        """
        # Try RL policy first
        if self.agent_loaded and self.matlab_engine is not None:
            try:
                action = self._get_rl_action(state)
                if action is not None:
                    return action
            except Exception as e:
                logger.warning(f"RL action failed: {e}, using fallback")

        # Fallback to petersen strategy
        if self.fallback_to_petersen:
            return self._petersen_action(state, score, history)

        # Last resort: random valid move
        return self._random_action(state)

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

    def _get_rl_action(self, state: str) -> Optional[tuple[int, str]]:
        """Get action from RL policy via MATLAB."""
        # Build observation vector
        obs = self._state_to_observation(state)

        # Convert to MATLAB format
        obs_matlab = matlab.double([obs])  # Row vector for MATLAB

        # Get action from agent using getAction
        try:
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
                color = 'G'  # Green
            else:
                edge_idx = action - 16
                color = 'P'  # Purple

            # Validate action
            if 0 <= edge_idx < 15 and state[edge_idx] == '-':
                logger.info(f"RL selected: E{edge_idx} {color}")
                return (edge_idx, color)

            # Invalid action - try to find valid one
            logger.debug(f"RL selected invalid edge {edge_idx}, finding alternative")
            return None

        except Exception as e:
            logger.warning(f"MATLAB getAction error: {e}")
            return None

    def _petersen_action(self, state: str, score: float, history: list) -> Optional[tuple[int, str]]:
        """Fallback: Use petersen strategy."""
        from snowdrop_tangled_agents.strategy.petersen_strategy import PetersenStrategy

        strategy = PetersenStrategy()
        return strategy.calculate_move(state, score, history)

    def _random_action(self, state: str) -> Optional[tuple[int, str]]:
        """Last resort: Random valid move."""
        import random

        grey_edges = [i for i, c in enumerate(state) if c == '-']

        if grey_edges:
            edge_idx = random.choice(grey_edges)
            color = random.choice(['G', 'P'])
            return (edge_idx, color)

        return None

    def record_move(self, edge: int, color: str, score: float):
        """Record a move for potential learning."""
        self.move_history.append((edge, color, score))

    def end_game(self, result: str, final_score: float):
        """Called at end of game."""
        logger.info(f"RL game ended: {result}, score={final_score:.4f}")
        self.move_history = []

    def cleanup(self):
        """Clean up MATLAB engine."""
        if self.matlab_engine is not None:
            try:
                self.matlab_engine.quit()
            except:
                pass
            self.matlab_engine = None
