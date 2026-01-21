"""
Training orchestration for MATLAB neural networks.

Provides Python interface to MATLAB training functions:
- Value network training
- Policy network training
- Opponent clustering

Supports both MATLAB Engine API and compiled packages.
Falls back to informative error messages when MATLAB unavailable.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from ..stats import get_collector, DEFAULT_DB_PATH

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates MATLAB-based neural network training.

    Attempts to use:
    1. Compiled training package (tangled_training)
    2. MATLAB Engine API
    3. Reports unavailability with helpful messages
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize training orchestrator.

        Args:
            db_path: Path to SQLite database (None for default)
        """
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self._matlab_engine = None
        self._compiled_available = False
        self._engine_available = False

        self._check_availability()

    def _check_availability(self):
        """Check what training backends are available."""
        # Check compiled packages
        try:
            import tangled_training
            self._compiled_available = True
            logger.info("Compiled training package available")
        except ImportError:
            logger.debug("Compiled training package not installed")

        # Check MATLAB Engine
        try:
            from .bridge import get_bridge
            bridge = get_bridge()
            if bridge.connect():
                self._matlab_engine = bridge
                self._engine_available = True
                logger.info("MATLAB Engine available for training")
        except Exception as e:
            logger.debug(f"MATLAB Engine not available: {e}")

    def is_available(self) -> bool:
        """Check if any training backend is available."""
        return self._compiled_available or self._engine_available

    def get_status(self) -> Dict[str, Any]:
        """Get training system status."""
        # Check database
        collector = get_collector(self.db_path)
        game_counts = collector.get_game_count()
        migration_status = collector.get_migration_status()

        return {
            'database_path': str(self.db_path),
            'total_games': game_counts.get('total', 0),
            'wins': game_counts.get('wins', 0),
            'losses': game_counts.get('losses', 0),
            'schema_version': migration_status.get('current_version', 1),
            'compiled_training': self._compiled_available,
            'matlab_engine': self._engine_available,
            'ready_for_training': (
                self.is_available() and game_counts.get('total', 0) >= 50
            ),
        }

    def train_value_network(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        model_name: str = 'value_net_v1',
        output_dir: Optional[Path] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train value network from game history.

        Args:
            epochs: Maximum training epochs
            batch_size: Mini-batch size
            learning_rate: Initial learning rate
            model_name: Name for saved model
            output_dir: Directory to save model (None for default)
            verbose: Show training progress

        Returns:
            Dict with training_samples, validation_loss, model_path

        Raises:
            RuntimeError: If no training backend available
        """
        if output_dir is None:
            output_dir = Path.home() / ".tangled" / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        options = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'model_name': model_name,
            'output_dir': str(output_dir),
            'verbose': verbose,
        }

        # Try compiled package first
        if self._compiled_available:
            try:
                import tangled_training
                pkg = tangled_training.initialize()
                result = pkg.train_value_network(str(self.db_path), options, nargout=2)

                # Parse result
                metrics = {
                    'training_samples': int(result[1].get('training_samples', 0)),
                    'validation_loss': float(result[1].get('final_val_loss', 0)),
                    'validation_mae': float(result[1].get('final_val_mae', 0)),
                    'model_path': str(result[1].get('model_path', '')),
                    'backend': 'compiled',
                }
                return metrics
            except Exception as e:
                logger.warning(f"Compiled training failed: {e}")

        # Try MATLAB Engine
        if self._engine_available and self._matlab_engine:
            try:
                import matlab

                # Convert options to MATLAB struct
                opts_struct = matlab.double([])  # Placeholder
                result = self._matlab_engine.call_function(
                    'train_value_network',
                    str(self.db_path),
                    nargout=2
                )

                metrics = {
                    'training_samples': int(result[1].get('training_samples', 0)),
                    'validation_loss': float(result[1].get('final_val_loss', 0)),
                    'model_path': str(result[1].get('model_path', '')),
                    'backend': 'engine',
                }
                return metrics
            except Exception as e:
                logger.warning(f"MATLAB Engine training failed: {e}")

        # No backend available
        raise RuntimeError(
            "No training backend available. Install either:\n"
            "  1. Compiled packages: pip install tangled_training\n"
            "  2. MATLAB with Deep Learning Toolbox\n"
            f"Database path: {self.db_path}"
        )

    def train_policy_network(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        model_name: str = 'policy_net_v1',
        output_dir: Optional[Path] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train policy network from game history.

        Args:
            epochs: Maximum training epochs
            batch_size: Mini-batch size
            learning_rate: Initial learning rate
            model_name: Name for saved model
            output_dir: Directory to save model
            verbose: Show training progress

        Returns:
            Dict with training_samples, validation_accuracy, model_path
        """
        if output_dir is None:
            output_dir = Path.home() / ".tangled" / "models"
        output_dir.mkdir(parents=True, exist_ok=True)

        options = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'model_name': model_name,
            'output_dir': str(output_dir),
            'verbose': verbose,
        }

        # Try compiled package
        if self._compiled_available:
            try:
                import tangled_training
                pkg = tangled_training.initialize()
                result = pkg.train_policy_network(str(self.db_path), options, nargout=2)

                metrics = {
                    'training_samples': int(result[1].get('training_samples', 0)),
                    'validation_accuracy': float(result[1].get('final_val_accuracy', 0)),
                    'model_path': str(result[1].get('model_path', '')),
                    'backend': 'compiled',
                }
                return metrics
            except Exception as e:
                logger.warning(f"Compiled training failed: {e}")

        # Try MATLAB Engine
        if self._engine_available and self._matlab_engine:
            try:
                result = self._matlab_engine.call_function(
                    'train_policy_network',
                    str(self.db_path),
                    nargout=2
                )

                metrics = {
                    'training_samples': int(result[1].get('training_samples', 0)),
                    'validation_accuracy': float(result[1].get('final_val_accuracy', 0)),
                    'model_path': str(result[1].get('model_path', '')),
                    'backend': 'engine',
                }
                return metrics
            except Exception as e:
                logger.warning(f"MATLAB Engine training failed: {e}")

        raise RuntimeError(
            "No training backend available for policy network.\n"
            "Install MATLAB with Deep Learning Toolbox or compiled packages."
        )

    def cluster_opponents(
        self,
        k: int = 3,
        min_games: int = 5,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Cluster opponents by play style.

        Args:
            k: Number of clusters
            min_games: Minimum games per opponent to include
            verbose: Show progress

        Returns:
            Dict with labels, centroids, num_opponents
        """
        options = {
            'min_games': min_games,
            'verbose': verbose,
        }

        # Try compiled package
        if self._compiled_available:
            try:
                import tangled_training
                pkg = tangled_training.initialize()
                result = pkg.cluster_opponents(str(self.db_path), float(k), nargout=3)

                return {
                    'labels': [int(x) for x in result[0]],
                    'centroids': [list(row) for row in result[1]],
                    'num_opponents': len(result[0]),
                    'backend': 'compiled',
                }
            except Exception as e:
                logger.warning(f"Compiled clustering failed: {e}")

        # Try MATLAB Engine
        if self._engine_available and self._matlab_engine:
            try:
                result = self._matlab_engine.call_function(
                    'cluster_opponents',
                    str(self.db_path),
                    float(k),
                    nargout=3
                )

                return {
                    'labels': [int(x) for x in result[0]],
                    'centroids': [list(row) for row in result[1]],
                    'num_opponents': len(result[0]),
                    'backend': 'engine',
                }
            except Exception as e:
                logger.warning(f"MATLAB clustering failed: {e}")

        raise RuntimeError(
            "No training backend available for opponent clustering.\n"
            "Install MATLAB with Statistics and ML Toolbox or compiled packages."
        )

    def extract_opponent_features(
        self,
        opponent_name: str
    ) -> List[float]:
        """
        Extract feature vector for an opponent.

        Args:
            opponent_name: Name of opponent

        Returns:
            20-element feature vector
        """
        # Try compiled package
        if self._compiled_available:
            try:
                import tangled_training
                pkg = tangled_training.initialize()
                result = pkg.extract_opponent_features(str(self.db_path), opponent_name)
                return list(result)
            except Exception as e:
                logger.warning(f"Compiled feature extraction failed: {e}")

        # Try MATLAB Engine
        if self._engine_available and self._matlab_engine:
            try:
                result = self._matlab_engine.call_function(
                    'db_utils',
                    'get_opponent_features',
                    str(self.db_path),
                    opponent_name,
                    nargout=1
                )
                return list(result)
            except Exception as e:
                logger.warning(f"MATLAB feature extraction failed: {e}")

        # Fallback: return zeros (no features available)
        logger.warning(f"Could not extract features for {opponent_name}")
        return [0.0] * 20


def get_training_orchestrator(
    db_path: Optional[Path] = None
) -> TrainingOrchestrator:
    """Get training orchestrator instance."""
    return TrainingOrchestrator(db_path)


def print_training_status():
    """Print training system status to console."""
    orchestrator = get_training_orchestrator()
    status = orchestrator.get_status()

    print("\n=== MATLAB Training System Status ===")
    print(f"Database: {status['database_path']}")
    print(f"Games in database: {status['total_games']} "
          f"({status['wins']}W / {status['losses']}L)")
    print(f"Schema version: {status['schema_version']}")
    print()
    print("Backend availability:")
    print(f"  Compiled packages: {'Yes' if status['compiled_training'] else 'No'}")
    print(f"  MATLAB Engine:     {'Yes' if status['matlab_engine'] else 'No'}")
    print()

    if status['ready_for_training']:
        print("Status: READY for training")
    elif not orchestrator.is_available():
        print("Status: No training backend available")
        print("  Install MATLAB or compiled packages")
    elif status['total_games'] < 50:
        print(f"Status: Need more games ({50 - status['total_games']} more)")
        print("  Play more games to collect training data")
    print("=" * 40)
