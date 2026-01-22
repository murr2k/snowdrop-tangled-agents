"""
RL Strategy for web play using trained MATLAB model.

This wraps the RLAgent logic for use with the WebPlayer strategy interface.
Supports both pure RL inference and MC-guided ensemble inference.
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

    Supports two modes:
    1. Pure RL: Fast inference using just the neural network
    2. Ensemble: RL priors + MC rollouts for improved accuracy

    Falls back to petersen strategy if MATLAB is unavailable.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        fallback_to_petersen: bool = True,
        use_ensemble: bool = False,
        num_workers: int = 22,
        rollouts_per_action: int = 50,
        top_k: int = 5,
    ):
        """
        Initialize the RL strategy.

        Args:
            model_path: Path to trained .mat file
            fallback_to_petersen: Use petersen strategy if MATLAB unavailable
            use_ensemble: Use MC-guided ensemble (slower but more accurate)
            num_workers: Parallel workers for MC rollouts
            rollouts_per_action: Number of MC rollouts per candidate action
            top_k: Number of top candidates to evaluate with MC
        """
        self.fallback_to_petersen = fallback_to_petersen
        self.use_ensemble = use_ensemble
        self.num_workers = num_workers
        self.rollouts_per_action = rollouts_per_action
        self.top_k = top_k

        self.matlab_engine = None
        self.agent_loaded = False
        self.ensemble_ready = False
        self.move_history = []

        # Default model path
        if model_path is None:
            # Try ensemble model first, then fallback to selfplay model
            ensemble_path = Path(__file__).parent.parent / "matlab" / "models" / "agent_ensemble_v1.mat"
            selfplay_path = Path(__file__).parent.parent / "matlab" / "models" / "agent_selfplay_v1.mat"

            if ensemble_path.exists():
                model_path = str(ensemble_path)
            elif selfplay_path.exists():
                model_path = str(selfplay_path)
            else:
                model_path = str(ensemble_path)  # Will fail gracefully

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

            # Initialize ensemble if requested
            if self.use_ensemble:
                self._init_ensemble()
        else:
            logger.warning(f"Model not found: {self.model_path}")

    def _init_ensemble(self):
        """Initialize the ensemble policy for MC-guided inference."""
        try:
            logger.info(f"Initializing ensemble (workers={self.num_workers}, rollouts={self.rollouts_per_action})...")

            # Create ensemble in MATLAB
            self.matlab_engine.eval(f'''
                ensemble = EnsemblePolicy(agent, ...
                    'TopK', {self.top_k}, ...
                    'RolloutsPerAction', {self.rollouts_per_action}, ...
                    'NumWorkers', {self.num_workers});
            ''', nargout=0)

            self.ensemble_ready = True
            logger.info("Ensemble policy ready")

        except Exception as e:
            logger.warning(f"Failed to initialize ensemble: {e}")
            self.use_ensemble = False

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
        # Try ensemble first if available
        if self.use_ensemble and self.ensemble_ready:
            try:
                action = self._get_ensemble_action(state)
                if action is not None:
                    return action
            except Exception as e:
                logger.warning(f"Ensemble action failed: {e}, falling back to RL")

        # Try pure RL policy
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

    def _get_ensemble_action(self, state: str) -> Optional[tuple[int, str]]:
        """Get action from MC-guided ensemble via MATLAB."""
        try:
            # Call ensemble selectAction
            self.matlab_engine.workspace['state_str'] = state
            self.matlab_engine.eval('''
                [action_out, info_out] = ensemble.selectActionDetailed(state_str);
            ''', nargout=0)

            action = int(self.matlab_engine.workspace['action_out'])

            # Log ensemble decision info
            try:
                info = self.matlab_engine.workspace['info_out']
                candidates = info['candidates']
                scores = info['combinedScores']
                logger.info(f"Ensemble: {len(candidates)} candidates evaluated with MC")
            except:
                pass

            # Decode action
            if action <= 15:
                edge_idx = action - 1
                color = 'G'
            else:
                edge_idx = action - 16
                color = 'P'

            # Validate
            if 0 <= edge_idx < 15 and state[edge_idx] == '-':
                logger.info(f"Ensemble selected: E{edge_idx} {color}")
                return (edge_idx, color)

            return None

        except Exception as e:
            logger.warning(f"MATLAB ensemble error: {e}")
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

    def _get_rl_action(self, state: str) -> Optional[tuple[int, str]]:
        """Get action from RL policy via MATLAB with action masking."""
        # Build observation vector
        obs = self._state_to_observation(state)

        try:
            # Get actor network probabilities directly (not getAction which ignores mask)
            self.matlab_engine.workspace['obs_py'] = matlab.double([obs])
            self.matlab_engine.eval('''
                actor = getActor(agent);
                actorNet = getModel(actor);
                dlObs = dlarray(obs_py', "CB");
                probs = forward(actorNet, dlObs);
                probs = extractdata(probs);
            ''', nargout=0)

            probs = self.matlab_engine.workspace['probs']

            # Build action mask (only grey edges are valid)
            masked_probs = []
            for i in range(30):
                edge = i if i < 15 else i - 15
                if state[edge] == '-':  # Grey edge - valid
                    masked_probs.append((i, probs[i][0]))
                else:
                    masked_probs.append((i, 0.0))

            # Find best valid action
            if not any(p > 0 for _, p in masked_probs):
                logger.warning("No valid actions available")
                return None

            # Get argmax of masked probabilities
            best_action, best_prob = max(masked_probs, key=lambda x: x[1])

            # Decode action (1-indexed in MATLAB, 0-indexed here)
            if best_action < 15:
                edge_idx = best_action
                color = 'G'  # Green
            else:
                edge_idx = best_action - 15
                color = 'P'  # Purple

            # Log top 3 choices for debugging
            sorted_probs = sorted(masked_probs, key=lambda x: x[1], reverse=True)[:3]
            choices = []
            for idx, prob in sorted_probs:
                if prob > 0:
                    e = idx if idx < 15 else idx - 15
                    c = 'G' if idx < 15 else 'P'
                    choices.append(f"E{e}{c}:{prob:.2f}")
            logger.info(f"RL selected: E{edge_idx} {color} (top choices: {', '.join(choices)})")

            return (edge_idx, color)

        except Exception as e:
            logger.warning(f"MATLAB policy error: {e}")
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


class EnsembleStrategy(RLStrategy):
    """
    Convenience class for ensemble-mode RL strategy.

    This is equivalent to RLStrategy with use_ensemble=True.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        num_workers: int = 22,
        rollouts_per_action: int = 50,
        top_k: int = 5,
    ):
        super().__init__(
            model_path=model_path,
            fallback_to_petersen=True,
            use_ensemble=True,
            num_workers=num_workers,
            rollouts_per_action=rollouts_per_action,
            top_k=top_k,
        )
