"""
MATLAB-Enhanced Strategy for Tangled Game.

Combines MCTS with MATLAB toolbox evaluations:
- Neural network value/policy networks for position evaluation
- Opponent modeling for adaptive play style
- Simulated annealing for deep analysis of critical positions

Supports three backends via unified bridge:
1. Compiled MATLAB packages (fastest, no license)
2. MATLAB Engine API (full functionality)
3. Pure Python heuristics (always available)
"""

import logging
from typing import Optional, Dict, List, Tuple

from ..strategy.mcts_strategy import (
    MCTSStrategy,
    HybridStrategy,
    evaluate_terminal_state,
    compute_action_prior,
    MY_EDGES,
    OPP_EDGES,
    HUB_EDGES,
    NUM_EDGES,
)
from ..stats import get_collector
from .bridge import get_bridge, MatlabBridge
from .unified_bridge import get_unified_bridge, UnifiedMatlabBridge

logger = logging.getLogger(__name__)


class MatlabEnhancedStrategy:
    """
    Hybrid strategy combining MCTS with MATLAB toolbox analysis.

    Features:
    - Neural network value/policy networks for position evaluation
    - Opponent style classification and adaptive priors
    - SA validation of critical positions

    Uses unified bridge with fallback chain:
    1. Compiled MATLAB packages (fastest)
    2. MATLAB Engine API
    3. Pure Python heuristics
    """

    def __init__(
        self,
        mcts_time_limit: float = 2.0,
        mcts_iterations: int = 5000,
        use_nn_priors: bool = True,
        use_opponent_adaptation: bool = True,
        use_sa_validation: bool = True,
        sa_threshold: int = 4,
        opening_moves: int = 3,
    ):
        """
        Initialize MATLAB-enhanced strategy.

        Args:
            mcts_time_limit: Time limit per MCTS move
            mcts_iterations: Max MCTS iterations
            use_nn_priors: Use neural network for action priors
            use_opponent_adaptation: Adapt to opponent play style
            use_sa_validation: Use SA to validate critical moves
            sa_threshold: Edge count threshold for SA activation
            opening_moves: Number of opening book moves
        """
        # Use unified bridge (with fallback chain)
        self.unified_bridge = get_unified_bridge()
        self.bridge = get_bridge()  # Keep for backwards compatibility

        self.use_nn_priors = use_nn_priors
        self.use_opponent_adaptation = use_opponent_adaptation
        self.use_sa_validation = use_sa_validation
        self.sa_threshold = sa_threshold
        self.opening_moves = opening_moves

        # Core MCTS strategy
        self.mcts = MCTSStrategy(
            time_limit=mcts_time_limit,
            max_iterations=mcts_iterations,
            prior_weight=3.0,
        )
        self.base_time_limit = mcts_time_limit

        # Opening book - secure our edges first
        self.opening_sequence = [
            (9, 'G'),   # E9: Our spoke
            (10, 'G'),  # E10: Our hub connection
            (11, 'G'),  # E11: Our outer edge
            (5, 'P'),   # E5: Attack opponent spoke
            (12, 'P'),  # E12: Attack opponent hub
        ]

        # Game history for opponent modeling
        self.game_history: List[Tuple[int, str, float]] = []
        self.opponent_model: Dict = {}
        self.opponent_features: Optional[List[float]] = None
        self.opponent_style: Optional[int] = None
        self.current_opponent: Optional[str] = None

        # Statistics
        self.nn_calls = 0
        self.sa_calls = 0
        self.adapt_calls = 0
        self.backend: Optional[str] = None

        # Backwards compatibility
        self.rl_calls = 0
        self.matlab_available = False

    def initialize(self, opponent: Optional[str] = None) -> bool:
        """
        Initialize MATLAB connection and opponent model.

        Args:
            opponent: Opponent name for loading opponent model

        Returns:
            True if any MATLAB backend available, False otherwise.
            Falls back to pure MCTS/heuristics if MATLAB unavailable.
        """
        # Connect to best available backend
        self.backend = self.unified_bridge.connect()
        self.matlab_available = self.backend in ('compiled', 'engine')

        if self.matlab_available:
            logger.info(f"MATLAB-enhanced strategy initialized (backend: {self.backend})")
        else:
            logger.info("Using heuristic fallback (MATLAB unavailable)")

        # Load opponent model if available
        if opponent:
            self.current_opponent = opponent
            self._load_opponent_model(opponent)

        return self.matlab_available

    def _load_opponent_model(self, opponent: str):
        """Load opponent features and classification from database."""
        if not self.use_opponent_adaptation:
            return

        try:
            collector = get_collector()
            opponent_data = collector.get_opponent(opponent)

            if opponent_data:
                self.opponent_features = opponent_data.get('features')
                self.opponent_style = opponent_data.get('cluster_id')
                logger.debug(f"Loaded opponent model: style={self.opponent_style}")
            else:
                # Try to classify from history
                if self.unified_bridge.is_available():
                    style, conf = self.unified_bridge.classify_opponent(
                        opponent_name=opponent
                    )
                    if style > 0:
                        self.opponent_style = style
                        logger.debug(f"Classified opponent: style={style}, conf={conf:.2f}")
        except Exception as e:
            logger.debug(f"Could not load opponent model: {e}")

    def calculate_move(
        self,
        state: str,
        score: float = 0.0,
        score_history: list = None
    ) -> Optional[Tuple[int, str]]:
        """
        Calculate the best move using MATLAB-enhanced strategy.

        Strategy:
        1. Opening phase: Use heuristic sequence
        2. Midgame: MCTS with neural network priors + opponent adaptation
        3. Endgame: SA validation of candidate moves

        Args:
            state: 15-char board state
            score: Current game score
            score_history: History of (edge, color, score) tuples

        Returns:
            (edge_index, color) or None if no moves available
        """
        grey_count = state.count('-')
        if grey_count == 0:
            return None

        total_moves = NUM_EDGES - grey_count
        our_move_count = (total_moves + 1) // 2

        # Opening phase: use heuristic sequence
        if our_move_count < self.opening_moves:
            for edge, color in self.opening_sequence:
                if state[edge] == '-':
                    logger.debug(f"Opening move: E{edge} {color}")
                    return (edge, color)

        # Get neural network priors
        nn_priors: Dict[Tuple[int, str], float] = {}
        if self.use_nn_priors and self.unified_bridge.is_available():
            try:
                value, nn_priors = self.unified_bridge.evaluate_position(
                    state, is_our_turn=True
                )
                self.nn_calls += 1
                self.rl_calls += 1  # Backwards compatibility

                logger.debug(f"NN value: {value:.3f}, priors: {len(nn_priors)} actions")
            except Exception as e:
                logger.debug(f"NN evaluation skipped: {e}")

        # Apply opponent adaptation to priors
        if (self.use_opponent_adaptation and
            self.opponent_features is not None and
            (nn_priors or grey_count > 0)):

            try:
                adapted_priors = self.unified_bridge.adapt_priors(
                    state,
                    self.opponent_features,
                    nn_priors if nn_priors else None,
                    self.opponent_style
                )
                if adapted_priors:
                    nn_priors = adapted_priors
                    self.adapt_calls += 1
                    logger.debug(f"Applied opponent adaptation (style={self.opponent_style})")
            except Exception as e:
                logger.debug(f"Opponent adaptation skipped: {e}")

        # Calculate MCTS move with enhanced priors
        if nn_priors:
            move = self._mcts_with_nn_priors(state, score, score_history, nn_priors)
        else:
            move = self.mcts.calculate_move(state, score, score_history)

        if move is None:
            return None

        # SA validation for critical positions (endgame)
        if (self.backend in ('compiled', 'engine') and
            self.use_sa_validation and
            grey_count <= self.sa_threshold):

            move = self._sa_validate_move(state, move, score)

        return move

    def _mcts_with_nn_priors(
        self,
        state: str,
        score: float,
        score_history: list,
        nn_priors: Dict[Tuple[int, str], float]
    ) -> Optional[Tuple[int, str]]:
        """
        Run MCTS with neural network enhanced action priors.

        Blends heuristic priors with neural network output.
        """
        # Store original prior function
        original_compute = compute_action_prior

        def enhanced_prior(edge: int, color: str, is_our_turn: bool) -> float:
            """Blend heuristic and neural network priors."""
            heuristic = original_compute(edge, color, is_our_turn)
            nn = nn_priors.get((edge, color), 0.5)

            # Weighted blend: 60% NN, 40% heuristic
            return 0.6 * nn + 0.4 * heuristic

        # Temporarily replace prior computation
        try:
            import snowdrop_tangled_agents.strategy.mcts_strategy as mcts_module
            mcts_module.compute_action_prior = enhanced_prior

            result = self.mcts.calculate_move(state, score, score_history)
            return result

        finally:
            mcts_module.compute_action_prior = original_compute

    # Backwards compatibility alias
    _mcts_with_rl_priors = _mcts_with_nn_priors

    def _sa_validate_move(
        self,
        state: str,
        candidate_move: Tuple[int, str],
        score: float
    ) -> Tuple[int, str]:
        """
        Use simulated annealing to validate and potentially improve move.

        For critical late-game positions, SA provides deeper analysis
        than MCTS rollouts.
        """
        # SA only available via MATLAB Engine (not compiled packages)
        if not self.bridge.is_available():
            return candidate_move

        try:
            mean_val, confidence, sa_moves = self.bridge.run_simulated_annealing(
                state, num_samples=50
            )
            self.sa_calls += 1

            # If SA is confident and suggests different move, consider it
            if confidence < 0.3 and sa_moves:
                sa_best = sa_moves[0]
                if sa_best != candidate_move:
                    logger.info(
                        f"SA override: {candidate_move} -> {sa_best} "
                        f"(confidence={confidence:.2f})"
                    )
                    return sa_best

        except Exception as e:
            logger.debug(f"SA validation skipped: {e}")

        return candidate_move

    def record_move(self, edge: int, color: str, score_after: float):
        """Record a move for opponent modeling."""
        self.game_history.append((edge, color, score_after))

    def end_game(self, result: str, final_score: float):
        """
        Process end of game.

        Updates opponent model if enough data collected.
        """
        if self.matlab_available and len(self.game_history) >= 5:
            try:
                # Update opponent model periodically
                self.opponent_model = self.bridge.identify_opponent_strategy(
                    [self.game_history]
                )
                logger.debug(f"Opponent model updated: {self.opponent_model}")
            except Exception as e:
                logger.debug(f"Opponent modeling skipped: {e}")

        self.game_history = []

    def get_stats(self) -> dict:
        """Return strategy statistics."""
        mcts_stats = self.mcts.get_stats()
        return {
            **mcts_stats,
            'matlab_available': self.matlab_available,
            'backend': self.backend,
            'nn_calls': self.nn_calls,
            'rl_calls': self.rl_calls,  # Backwards compatibility
            'sa_calls': self.sa_calls,
            'adapt_calls': self.adapt_calls,
            'opponent_style': self.opponent_style,
            'opponent_model': self.opponent_model,
        }


class MatlabMCTSStrategy(MCTSStrategy):
    """
    MCTS Strategy with optional MATLAB RL value network for evaluation.

    Extends base MCTS to use MATLAB for terminal state evaluation
    when available, providing potentially more accurate values.
    """

    def __init__(self, *args, use_matlab_eval: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge = get_bridge()
        self.use_matlab_eval = use_matlab_eval
        self._matlab_checked = False
        self._matlab_available = False

    def _simulate(self, state: str, is_our_turn: bool) -> float:
        """
        Simulate with optional MATLAB evaluation.

        Uses MATLAB RL network for faster approximate evaluation
        when available, falls back to SA adjudicator otherwise.
        """
        current_state = list(state)
        current_turn = is_our_turn

        # Get available edges
        available = [i for i, c in enumerate(current_state) if c == '-']

        while available:
            # Select action using heuristic
            if self.use_heuristic_rollout:
                action = self._heuristic_action(current_state, available, current_turn)
            else:
                import random
                edge = random.choice(available)
                color = random.choice(['G', 'P'])
                action = (edge, color)

            edge, color = action
            current_state[edge] = color
            available.remove(edge)
            current_turn = not current_turn

        # Evaluate terminal state
        terminal_state = ''.join(current_state)

        # Try MATLAB evaluation for speed
        if self.use_matlab_eval and self._check_matlab():
            try:
                value, _ = self.bridge.evaluate_position_rl(terminal_state, True)
                if value != 0.0:  # Valid result
                    return value
            except Exception:
                pass

        # Fall back to standard evaluation
        return evaluate_terminal_state(terminal_state)

    def _check_matlab(self) -> bool:
        """Check MATLAB availability once."""
        if not self._matlab_checked:
            self._matlab_available = self.bridge.connect()
            self._matlab_checked = True
        return self._matlab_available


class HybridSolverStrategy:
    """
    Strategy using the D-Wave inspired HybridTangledSolver.

    Combines:
    - Alpha-beta minimax at shallow depths (exact)
    - MCTS with tabu-guided rollouts (deep exploration)
    - Expanded LUT with ~19M exact minimax values (0-3 grey edges)

    Requires MATLAB Engine connection to a shared session.
    """

    def __init__(
        self,
        time_limit: float = 10.0,
        minimax_depth: int = 4,
        mcts_iterations: int = 5000,
        player: int = 1,
    ):
        """
        Initialize HybridSolverStrategy.

        Args:
            time_limit: Total time budget per move (seconds)
            minimax_depth: Depth for exact alpha-beta search
            mcts_iterations: MCTS iterations for deep exploration
            player: Player perspective (1 or 2)
        """
        self.time_limit = time_limit
        self.minimax_depth = minimax_depth
        self.mcts_iterations = mcts_iterations
        self.player = player

        self.bridge = get_bridge()
        self.engine = None
        self.solver_initialized = False

        # Statistics
        self.moves_calculated = 0
        self.total_time = 0.0
        self.last_strategy = ''
        self.last_score = 0.0

    def initialize(self, opponent: Optional[str] = None) -> bool:
        """
        Initialize MATLAB connection and create solver instance.

        Args:
            opponent: Opponent name (unused, for API compatibility)

        Returns:
            True if MATLAB connection successful.
        """
        if not self.bridge.connect():
            logger.error("Failed to connect to MATLAB Engine")
            return False

        self.engine = self.bridge.engine

        # Add RL path to MATLAB
        try:
            import os
            rl_path = os.path.join(
                os.path.dirname(__file__), 'rl'
            )
            self.engine.addpath(rl_path, nargout=0)

            # Create solver instance in MATLAB
            self.engine.eval(
                f"hybridSolver = HybridTangledSolver("
                f"'TimeLimit', {self.time_limit}, "
                f"'MinimaxDepth', {self.minimax_depth}, "
                f"'MCTSIterations', {self.mcts_iterations}, "
                f"'Player', {self.player});",
                nargout=0
            )

            self.solver_initialized = True
            logger.info(
                f"HybridSolverStrategy initialized: "
                f"time_limit={self.time_limit}, depth={self.minimax_depth}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize HybridTangledSolver: {e}")
            return False

    def calculate_move(
        self,
        state: str,
        score: float = 0.0,
        score_history: list = None
    ) -> Optional[Tuple[int, str]]:
        """
        Calculate best move using HybridTangledSolver.

        Args:
            state: 15-char board state ('G', 'P', or '-')
            score: Current game score (unused)
            score_history: Move history (unused)

        Returns:
            (edge_index, color) or None if no moves available.
        """
        if not self.solver_initialized:
            if not self.initialize():
                logger.error("Solver not initialized")
                return None

        grey_count = state.count('-')
        if grey_count == 0:
            return None

        try:
            import time
            start = time.time()

            # Call MATLAB solver
            self.engine.eval(
                f"[solverEdge, solverColor, solverInfo] = hybridSolver.solve('{state}');",
                nargout=0
            )

            # Get results
            edge = int(self.engine.eval("solverEdge"))
            color = str(self.engine.eval("solverColor"))

            # Get info for statistics
            try:
                self.last_strategy = str(self.engine.eval("solverInfo.strategy"))
                self.last_score = float(self.engine.eval("solverInfo.score"))
            except Exception:
                pass

            elapsed = time.time() - start
            self.total_time += elapsed
            self.moves_calculated += 1

            logger.debug(
                f"HybridSolver: E{edge} {color} "
                f"(strategy={self.last_strategy}, score={self.last_score:.3f}, "
                f"time={elapsed:.2f}s)"
            )

            return (edge, color)

        except Exception as e:
            logger.error(f"HybridSolver error: {e}")
            return None

    def set_player(self, player: int):
        """Set player perspective (1 or 2)."""
        self.player = player
        if self.solver_initialized:
            try:
                self.engine.eval(f"hybridSolver.setPlayer({player});", nargout=0)
            except Exception as e:
                logger.warning(f"Failed to set player: {e}")

    def get_stats(self) -> dict:
        """Return strategy statistics."""
        stats = {
            'moves_calculated': self.moves_calculated,
            'total_time': self.total_time,
            'avg_time_per_move': self.total_time / max(1, self.moves_calculated),
            'last_strategy': self.last_strategy,
            'last_score': self.last_score,
            'solver_initialized': self.solver_initialized,
        }

        # Get solver stats from MATLAB if available
        if self.solver_initialized:
            try:
                self.engine.eval("solverStats = hybridSolver.getStats();", nargout=0)
                stats['lut_loaded'] = bool(self.engine.eval("solverStats.lutLoaded"))
                stats['lut_entries'] = int(self.engine.eval("solverStats.lutEntries"))
            except Exception:
                pass

        return stats

    def end_game(self, result: str, final_score: float):
        """Process end of game (API compatibility)."""
        pass
