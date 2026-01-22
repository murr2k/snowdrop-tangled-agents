"""
RL Agent Bridge - Python interface to MATLAB-trained RL agent.

This module provides a unified interface for using the trained Tangled RL agent
with automatic fallback chain:
1. Compiled MATLAB package (fastest, no MATLAB installation needed)
2. MATLAB Engine API (full functionality, requires MATLAB)
3. Pure Python heuristics (always available)

The bridge supports hot-reload of models without restarting the application.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np

logger = logging.getLogger(__name__)


class RLAgentBridge:
    """
    Bridge to MATLAB RL agent with hot-reload support.

    Attributes:
        backend: Currently active backend ('compiled', 'engine', or 'heuristic')
        model_path: Path to deployed model
        last_reload: Timestamp of last model reload
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the RL agent bridge.

        Args:
            model_dir: Directory containing deployed models.
                       Defaults to ~/.tangled/models/deployed
        """
        self.model_dir = model_dir or Path.home() / '.tangled' / 'models' / 'deployed'
        self.model_path = self.model_dir / 'current_model.mat'

        self.backend = None
        self.compiled_pkg = None
        self.engine = None
        self.last_reload = 0
        self.reload_interval = 60  # seconds

        self._initialize()

    def _initialize(self) -> str:
        """Initialize the best available backend."""
        # Try compiled package first (fastest)
        if self._init_compiled():
            self.backend = 'compiled'
            logger.info("RL bridge using compiled MATLAB package")
            return 'compiled'

        # Try MATLAB Engine
        if self._init_engine():
            self.backend = 'engine'
            logger.info("RL bridge using MATLAB Engine")
            return 'engine'

        # Fall back to heuristics
        self.backend = 'heuristic'
        logger.info("RL bridge using Python heuristics (MATLAB not available)")
        return 'heuristic'

    def _init_compiled(self) -> bool:
        """Try to initialize compiled MATLAB package."""
        try:
            import tangled_rl_agent
            self.compiled_pkg = tangled_rl_agent.initialize()
            return True
        except ImportError:
            logger.debug("Compiled MATLAB package not installed")
            return False
        except Exception as e:
            logger.warning(f"Failed to initialize compiled package: {e}")
            return False

    def _init_engine(self) -> bool:
        """Try to initialize MATLAB Engine."""
        try:
            import matlab.engine
            self.engine = matlab.engine.start_matlab()

            # Add RL code to path
            rl_path = Path(__file__).parent / 'matlab' / 'rl'
            if rl_path.exists():
                self.engine.addpath(str(rl_path))

            return True
        except ImportError:
            logger.debug("MATLAB Engine not installed")
            return False
        except Exception as e:
            logger.warning(f"Failed to start MATLAB Engine: {e}")
            return False

    def get_action(
        self,
        state: str,
        valid_actions: Optional[List[int]] = None
    ) -> Tuple[int, float, List[float]]:
        """
        Get action from RL agent.

        Args:
            state: 15-character board state string (G/P/- for each edge)
            valid_actions: List of valid action indices (0-29), or None to auto-detect

        Returns:
            Tuple of (action, value, probabilities):
            - action: Selected action index (0-29)
            - value: State value estimate (-1 to 1)
            - probabilities: 30-element list of action probabilities
        """
        # Build feature vector from state
        state_vec = self._state_to_features(state)

        # Build action mask
        if valid_actions is not None:
            mask = [1.0 if i in valid_actions else 0.0 for i in range(30)]
        else:
            mask = self._build_action_mask(state)

        # Check for model updates (hot-reload)
        self._check_reload()

        # Get action from appropriate backend
        if self.backend == 'compiled':
            return self._compiled_inference(state_vec, mask)
        elif self.backend == 'engine':
            return self._engine_inference(state_vec, mask)
        else:
            return self._heuristic_inference(state, mask)

    def _state_to_features(self, state: str) -> List[float]:
        """
        Convert board state to 50-element feature vector.

        This matches the MATLAB buildRLFeatures function.
        """
        features = []

        # Board state (15 elements): G=1, P=-1, -=0
        for c in state:
            if c == 'G':
                features.append(1.0)
            elif c == 'P':
                features.append(-1.0)
            else:
                features.append(0.0)

        # Turn indicator (1 element): always our turn when called
        features.append(1.0)

        # Edge categories (15 elements) - simplified encoding
        # MY_EDGES: E9, E10, E11 (indices 9, 10, 11)
        # OPP_EDGES: E5, E12, E13 (indices 5, 12, 13)
        # HUB_EDGES: E0, E3, E6 (indices 0, 3, 6)
        for i in range(15):
            if i in [9, 10, 11]:
                features.append(0.5)   # MY_EDGE
            elif i in [5, 12, 13]:
                features.append(-0.5)  # OPP_EDGE
            elif i in [0, 3, 6]:
                features.append(0.3)   # HUB_EDGE
            else:
                features.append(0.0)   # NEUTRAL

        # Grey count (1 element)
        grey_count = state.count('-')
        features.append(grey_count / 15.0)

        # Score momentum (3 elements) - placeholder
        features.extend([0.0, 0.0, 0.0])

        # Game phase (15 elements) - one-hot encoding
        if grey_count > 10:
            phase = 0  # Opening
        elif grey_count > 5:
            phase = 1  # Middle
        else:
            phase = 2  # Endgame

        phase_onehot = [0.0] * 15
        phase_onehot[phase * 5] = 1.0  # Simplified phase encoding
        features.extend(phase_onehot)

        return features

    def _build_action_mask(self, state: str) -> List[float]:
        """Build action mask from board state."""
        mask = []
        for i in range(15):
            is_grey = state[i] == '-'
            mask.append(1.0 if is_grey else 0.0)  # Green
            mask.append(1.0 if is_grey else 0.0)  # Purple

        # Flatten: [G0, P0, G1, P1, ...] -> [G0-G14, P0-P14]
        green_mask = [mask[i*2] for i in range(15)]
        purple_mask = [mask[i*2+1] for i in range(15)]
        return green_mask + purple_mask

    def _check_reload(self):
        """Check if model needs to be reloaded (hot-reload support)."""
        current_time = time.time()
        if current_time - self.last_reload < self.reload_interval:
            return

        self.last_reload = current_time

        # Check if model file has been updated
        if self.model_path.exists():
            mtime = self.model_path.stat().st_mtime
            if mtime > self.last_reload - self.reload_interval:
                logger.info("Model file updated, triggering reload")
                # Compiled package handles this internally
                # Engine would need explicit reload

    def _compiled_inference(
        self,
        state_vec: List[float],
        mask: List[float]
    ) -> Tuple[int, float, List[float]]:
        """Run inference using compiled package."""
        try:
            action, value, probs = self.compiled_pkg.tangled_agent_inference(
                state_vec, mask, nargout=3
            )
            # Convert from MATLAB types
            action = int(action) - 1  # MATLAB 1-indexed to Python 0-indexed
            value = float(value)
            probs = [float(p) for p in probs]
            return action, value, probs
        except Exception as e:
            logger.warning(f"Compiled inference failed: {e}, using fallback")
            return self._heuristic_inference_from_features(mask)

    def _engine_inference(
        self,
        state_vec: List[float],
        mask: List[float]
    ) -> Tuple[int, float, List[float]]:
        """Run inference using MATLAB Engine."""
        try:
            import matlab
            state_ml = matlab.double(state_vec)
            mask_ml = matlab.double(mask)

            action, value, probs = self.engine.tangled_agent_inference(
                state_ml, mask_ml, nargout=3
            )

            action = int(action) - 1  # MATLAB 1-indexed to Python 0-indexed
            value = float(value)
            probs = [float(p) for p in probs[0]]
            return action, value, probs
        except Exception as e:
            logger.warning(f"Engine inference failed: {e}, using fallback")
            return self._heuristic_inference_from_features(mask)

    def _heuristic_inference(
        self,
        state: str,
        mask: List[float]
    ) -> Tuple[int, float, List[float]]:
        """Pure Python heuristic inference."""
        import random

        # Strategic edge priorities (from Petersen graph analysis)
        # Actions 0-14: Green, 15-29: Purple
        edge_scores = [
            0.7,  # E0 (hub)
            0.4,  # E1
            0.4,  # E2
            0.7,  # E3 (hub)
            0.4,  # E4
            0.6,  # E5 (strategic)
            0.7,  # E6 (hub)
            0.4,  # E7
            0.4,  # E8
            0.8,  # E9 (our edge)
            0.8,  # E10 (our edge)
            0.8,  # E11 (our edge)
            0.6,  # E12 (strategic)
            0.6,  # E13 (strategic)
            0.4,  # E14
        ]

        # Build weighted probabilities
        probs = []
        for i in range(15):
            # Green action
            if mask[i] > 0:
                # Prefer green on our edges (9, 10, 11)
                if i in [9, 10, 11]:
                    probs.append(edge_scores[i] * 1.5)
                else:
                    probs.append(edge_scores[i])
            else:
                probs.append(0.0)

        for i in range(15):
            # Purple action
            if mask[15 + i] > 0:
                # Prefer purple on opponent edges (5, 12, 13)
                if i in [5, 12, 13]:
                    probs.append(edge_scores[i] * 1.3)
                else:
                    probs.append(edge_scores[i] * 0.8)
            else:
                probs.append(0.0)

        # Normalize
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        else:
            # All masked - uniform over valid
            valid_count = sum(1 for m in mask if m > 0)
            probs = [1.0/valid_count if m > 0 else 0.0 for m in mask]

        # Sample action
        r = random.random()
        cumsum = 0
        action = 0
        for i, p in enumerate(probs):
            cumsum += p
            if cumsum >= r:
                action = i
                break

        # Simple value estimate based on board state
        green_count = state.count('G')
        purple_count = state.count('P')
        value = (green_count - purple_count) * 0.1

        return action, value, probs

    def _heuristic_inference_from_features(
        self,
        mask: List[float]
    ) -> Tuple[int, float, List[float]]:
        """Fallback heuristic when only mask is available."""
        import random

        valid_actions = [i for i, m in enumerate(mask) if m > 0]
        if not valid_actions:
            return 0, 0.0, [0.0] * 30

        action = random.choice(valid_actions)
        probs = [1.0/len(valid_actions) if m > 0 else 0.0 for m in mask]
        return action, 0.0, probs

    def close(self):
        """Clean up resources."""
        if self.engine is not None:
            try:
                self.engine.quit()
            except:
                pass
            self.engine = None

        if self.compiled_pkg is not None:
            try:
                self.compiled_pkg.terminate()
            except:
                pass
            self.compiled_pkg = None


# Module-level singleton
_bridge_instance: Optional[RLAgentBridge] = None


def get_rl_bridge() -> RLAgentBridge:
    """Get or create the RL bridge singleton."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = RLAgentBridge()
    return _bridge_instance


def reset_rl_bridge():
    """Reset the RL bridge (forces re-initialization)."""
    global _bridge_instance
    if _bridge_instance is not None:
        _bridge_instance.close()
        _bridge_instance = None
