"""
Bridge to compiled MATLAB packages (no MATLAB Engine required).

Uses MATLAB Compiler SDK packages for neural network inference and
opponent modeling. Falls back gracefully if packages are not installed.

Packages:
- tangled_value_network: Position evaluation with neural networks
- tangled_opponent_model: Opponent style classification and adaptation
- tangled_training: Model training (requires more toolbox runtimes)

Prerequisites:
- MATLAB Runtime R2026a (free download from MathWorks)
- Compiled packages installed via pip/setup.py

IMPORTANT: The Runtime version must match the MATLAB version used to compile
the packages. Packages compiled with R2026a require Runtime R2026a.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# Import matlab module for array conversion
try:
    import matlab
    MATLAB_MODULE_AVAILABLE = True
except ImportError:
    MATLAB_MODULE_AVAILABLE = False
    logger.debug("matlab module not available for array conversion")

# Try to import compiled packages
VALUE_NETWORK_AVAILABLE = False
OPPONENT_MODEL_AVAILABLE = False
TRAINING_AVAILABLE = False

try:
    import tangled_value_network
    VALUE_NETWORK_AVAILABLE = True
    logger.info("tangled_value_network package available")
except ImportError:
    logger.debug("tangled_value_network package not installed")

try:
    import tangled_opponent_model
    OPPONENT_MODEL_AVAILABLE = True
    logger.info("tangled_opponent_model package available")
except ImportError:
    logger.debug("tangled_opponent_model package not installed")

try:
    import tangled_training
    TRAINING_AVAILABLE = True
    logger.info("tangled_training package available")
except ImportError:
    logger.debug("tangled_training package not installed")


def packages_available() -> Dict[str, bool]:
    """Check which compiled packages are available."""
    return {
        'value_network': VALUE_NETWORK_AVAILABLE,
        'opponent_model': OPPONENT_MODEL_AVAILABLE,
        'training': TRAINING_AVAILABLE,
    }


class CompiledMatlabBridge:
    """
    Bridge to compiled MATLAB packages.

    Provides fast neural network inference and opponent modeling
    without requiring a full MATLAB installation (only MATLAB Runtime).
    """

    def __init__(self):
        self.value_pkg = None
        self.opponent_pkg = None
        self.training_pkg = None
        self.initialized = False
        self._model_dir: Optional[Path] = None

    def initialize(self, model_dir: Optional[Path] = None) -> bool:
        """
        Initialize compiled packages.

        Args:
            model_dir: Directory containing .mat model files

        Returns:
            True if at least one package initialized successfully

        Note:
            Compiled packages and MATLAB Engine cannot be used in the same
            Python process due to DLL conflicts. If MATLAB Engine is already
            running, this will return False.
        """
        if model_dir:
            self._model_dir = Path(model_dir)
        else:
            self._model_dir = Path.home() / ".tangled" / "models"

        # Check if MATLAB Engine is already running (DLL conflict)
        try:
            import sys
            if 'matlab.engine' in sys.modules:
                # Check if an engine instance exists
                import matlab.engine
                sessions = matlab.engine.find_matlab()
                if sessions:
                    logger.info("MATLAB Engine detected - skipping compiled packages to avoid DLL conflict")
                    return False
        except Exception:
            pass  # matlab.engine not loaded, safe to proceed

        success = False

        # Initialize value network package
        if VALUE_NETWORK_AVAILABLE:
            try:
                self.value_pkg = tangled_value_network.initialize()
                logger.info("Value network package initialized")
                success = True
            except Exception as e:
                logger.warning(f"Failed to initialize value network: {e}")

        # Initialize opponent model package
        if OPPONENT_MODEL_AVAILABLE:
            try:
                self.opponent_pkg = tangled_opponent_model.initialize()
                logger.info("Opponent model package initialized")
                success = True
            except Exception as e:
                logger.warning(f"Failed to initialize opponent model: {e}")

        # Initialize training package (optional)
        if TRAINING_AVAILABLE:
            try:
                self.training_pkg = tangled_training.initialize()
                logger.info("Training package initialized")
            except Exception as e:
                logger.debug(f"Training package not initialized: {e}")

        self.initialized = success
        return success

    def is_available(self) -> bool:
        """Check if bridge is available for use."""
        return self.initialized

    def evaluate_position(
        self,
        state: str,
        is_our_turn: bool
    ) -> Tuple[float, Dict[Tuple[int, str], float]]:
        """
        Evaluate position using compiled value network.

        Args:
            state: 15-char board state ('G', 'P', '-')
            is_our_turn: True if it's our turn

        Returns:
            (value, policy_dict) where:
            - value: Expected outcome in [-1, 1]
            - policy_dict: {(edge, color): probability} for available actions
        """
        if not self.initialized or self.value_pkg is None:
            return 0.0, {}

        try:
            # Convert state string to numeric vector
            state_list = [
                1.0 if c == 'G' else (-1.0 if c == 'P' else 0.0)
                for c in state
            ]
            turn_val = 1.0 if is_our_turn else -1.0

            # Convert to MATLAB arrays (required for compiled packages)
            if MATLAB_MODULE_AVAILABLE:
                state_vec = matlab.double(state_list)
                turn = matlab.double([turn_val])
            else:
                state_vec = state_list
                turn = turn_val

            # Call compiled function
            result = self.value_pkg.evaluate_position_nn(state_vec, turn, nargout=2)

            # Parse results
            value = float(result[0])

            # Extract policy from matlab.double array
            policy_result = result[1]
            if hasattr(policy_result, '_data'):
                # matlab.double object - get underlying data
                policy_raw = list(policy_result._data)
            elif hasattr(policy_result, '__iter__'):
                policy_raw = [float(x) for x in policy_result]
            else:
                policy_raw = []

            # Convert policy array to dict (only for grey/available edges)
            policy = {}
            for i in range(15):
                if state[i] == '-' and len(policy_raw) >= 30:
                    policy[(i, 'G')] = float(policy_raw[i * 2])
                    policy[(i, 'P')] = float(policy_raw[i * 2 + 1])

            return value, policy

        except Exception as e:
            logger.warning(f"Compiled position evaluation failed: {e}")
            return 0.0, {}

    def classify_opponent(
        self,
        features: List[float]
    ) -> Tuple[int, float]:
        """
        Classify opponent play style.

        Args:
            features: 20-element opponent feature vector

        Returns:
            (style, confidence) where:
            - style: Cluster ID (1=aggressive, 2=defensive, 3=balanced)
            - confidence: Classification confidence [0, 1]
        """
        if not self.initialized or self.opponent_pkg is None:
            return 0, 0.0

        try:
            # Convert to MATLAB array
            if MATLAB_MODULE_AVAILABLE:
                features_arr = matlab.double(features)
            else:
                features_arr = features

            result = self.opponent_pkg.classify_opponent(
                features_arr,
                nargout=2
            )
            style = int(result[0])
            confidence = float(result[1])
            return style, confidence

        except Exception as e:
            logger.warning(f"Opponent classification failed: {e}")
            return 0, 0.0

    def adapt_priors(
        self,
        state: str,
        opponent_features: List[float],
        base_priors: List[float],
        style: Optional[int] = None
    ) -> List[float]:
        """
        Adapt action priors based on opponent model.

        Args:
            state: 15-char board state
            opponent_features: 20-element opponent feature vector
            base_priors: 30-element base action probabilities
            style: Optional opponent style override

        Returns:
            30-element adapted action probabilities
        """
        if not self.initialized or self.opponent_pkg is None:
            return base_priors

        try:
            state_list = [
                1.0 if c == 'G' else (-1.0 if c == 'P' else 0.0)
                for c in state
            ]

            # Convert to MATLAB arrays
            if MATLAB_MODULE_AVAILABLE:
                state_vec = matlab.double(state_list)
                opp_features = matlab.double(opponent_features)
                priors = matlab.double(base_priors)
            else:
                state_vec = state_list
                opp_features = opponent_features
                priors = base_priors

            if style is not None:
                result = self.opponent_pkg.adapt_to_opponent(
                    state_vec, opp_features, priors, float(style),
                    nargout=1
                )
            else:
                result = self.opponent_pkg.adapt_to_opponent(
                    state_vec, opp_features, priors,
                    nargout=1
                )

            # Extract result from matlab.double array
            if hasattr(result, '_data'):
                return list(result._data)
            elif hasattr(result, '__iter__'):
                return [float(x) for x in result]
            else:
                return base_priors

        except Exception as e:
            logger.warning(f"Prior adaptation failed: {e}")
            return base_priors

    def train_value_network(
        self,
        db_path: str = "",
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Train value network from game history.

        Args:
            db_path: Path to SQLite database (empty for default)
            options: Training options dict

        Returns:
            Metrics dict with training_samples, validation_loss, model_path
        """
        if not TRAINING_AVAILABLE or self.training_pkg is None:
            raise RuntimeError("Training package not available")

        try:
            if options is None:
                options = {}

            result = self.training_pkg.train_value_network(
                db_path, options,
                nargout=2
            )

            # Parse metrics
            metrics = {
                'training_samples': int(result[1].get('training_samples', 0)),
                'validation_loss': float(result[1].get('final_val_loss', 0)),
                'model_path': str(result[1].get('model_path', '')),
            }
            return metrics

        except Exception as e:
            logger.error(f"Value network training failed: {e}")
            raise

    def cluster_opponents(
        self,
        db_path: str = "",
        k: int = 3
    ) -> Tuple[List[int], List[List[float]]]:
        """
        Cluster opponents by play style.

        Args:
            db_path: Path to SQLite database
            k: Number of clusters

        Returns:
            (labels, centroids) where:
            - labels: Cluster assignment per opponent
            - centroids: Cluster center feature vectors
        """
        if not TRAINING_AVAILABLE or self.training_pkg is None:
            raise RuntimeError("Training package not available")

        try:
            result = self.training_pkg.cluster_opponents(
                db_path, float(k),
                nargout=2
            )

            labels = [int(x) for x in result[0]]
            centroids = [list(row) for row in result[1]]
            return labels, centroids

        except Exception as e:
            logger.error(f"Opponent clustering failed: {e}")
            raise


# Module-level singleton
_compiled_bridge: Optional[CompiledMatlabBridge] = None


def get_compiled_bridge() -> Optional[CompiledMatlabBridge]:
    """
    Get compiled bridge instance if any packages are available.

    Returns:
        CompiledMatlabBridge instance or None if no packages available
    """
    global _compiled_bridge

    # Check if any packages are available
    if not any(packages_available().values()):
        return None

    if _compiled_bridge is None:
        _compiled_bridge = CompiledMatlabBridge()

    return _compiled_bridge
