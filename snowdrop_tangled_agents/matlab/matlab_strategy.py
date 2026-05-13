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
from pathlib import Path
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
    - REINFORCE-style learning from game outcomes

    Requires MATLAB Engine connection to a shared session.
    """

    NUM_EDGES = 15

    def __init__(
        self,
        time_limit: float = 10.0,
        minimax_depth: int = 4,
        mcts_iterations: int = 5000,
        player: int = 1,
        learning_rate: float = 0.03,
        adjustments_path: Optional[Path] = None,
    ):
        """
        Initialize HybridSolverStrategy.

        Args:
            time_limit: Total time budget per move (seconds)
            minimax_depth: Depth for exact alpha-beta search
            mcts_iterations: MCTS iterations for deep exploration
            player: Player perspective (1 or 2)
            learning_rate: Rate for edge adjustment learning (default 0.03)
            adjustments_path: Path to persist learned adjustments (default ~/.tangled/hybrid_solver_adjustments.json)
        """
        self.time_limit = time_limit
        self.minimax_depth = minimax_depth
        self.mcts_iterations = mcts_iterations
        self.player = player
        self.lut_file = 'terminal_scores.mat'  # override per-strategy after construction

        self.bridge = get_bridge()
        self.engine = None
        self.solver_initialized = False

        # Statistics
        self.moves_calculated = 0
        self.total_time = 0.0
        self.last_strategy = ''
        self.last_score = 0.0

        # Learning: track move history and outcomes
        self.move_history: List[Tuple[int, str, float]] = []  # [(edge, color, score_after), ...]
        self.game_results: List[Tuple[str, float, List]] = []  # [(result, final_score, moves), ...]

        # Learned edge value adjustments (start at 0, adjusted by learning)
        self.edge_adjustments = [0.0] * self.NUM_EDGES

        # Learning rate - reduced to prevent overreaction to losses
        self.learning_rate = learning_rate

        # Persistence path for learned adjustments
        self.adjustments_path = adjustments_path or Path.home() / ".tangled" / "hybrid_solver_adjustments.json"
        self._load_adjustments()

    def initialize(self, opponent: Optional[str] = None) -> bool:
        """
        Initialize MATLAB connection and create solver instance.

        Args:
            opponent: Opponent name, forwarded to MATLAB for opponent-conditional calibration.

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

            # Force reload of all classdef files from disk.  A shared
            # MATLAB session may have cached an older version of
            # TangledMCTS (or any other class) from before a fix was
            # committed.  `clear classes` discards those cached
            # definitions so the next construction picks up the current
            # .m files on the path.
            self.engine.eval("clear classes", nargout=0)

            # Clean up any stale parallel pool from previous runs
            # This prevents worker exhaustion if a previous run crashed
            logger.debug("Cleaning up any stale parallel pools...")
            try:
                self.engine.eval(
                    "pool = gcp('nocreate'); if ~isempty(pool), delete(pool); end",
                    nargout=0
                )
                logger.debug("Stale pool cleanup complete")
            except Exception as e:
                logger.debug(f"Pool cleanup failed (non-critical): {e}")

            # Sanitize string args for safe MATLAB string embedding
            opponent_name = (opponent or '').replace("'", "''")
            lut_file = self.lut_file.replace("'", "''")
            self.engine.eval(
                f"hybridSolver = HybridTangledSolver("
                f"'TimeLimit', {self.time_limit}, "
                f"'MinimaxDepth', {self.minimax_depth}, "
                f"'MCTSIterations', {self.mcts_iterations}, "
                f"'Player', {self.player}, "
                f"'Opponent', '{opponent_name}', "
                f"'MCTSLUTFile', '{lut_file}');",
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
    ) -> Optional[Tuple[int, str, dict]]:
        """
        Calculate best move using HybridTangledSolver.

        Args:
            state: 15-char board state ('G', 'P', or '-')
            score: Current game score (unused)
            score_history: Move history (unused)

        Returns:
            (edge_index, color, solver_stats) or None if no moves available.
            solver_stats contains detailed statistics for database recording.
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

            # Call MATLAB solver with timeout protection
            # Use async mode to enable timeout
            # With parallel rollouts (6 workers × 100 rollouts/worker = 600/iter),
            # each MCTS iteration takes ~65ms (dominated by parfor overhead).
            # Hybrid path evaluates up to 5 candidate moves, each with full
            # iteration count: 5 × iterations × 0.065s + pool/minimax overhead.
            # Examples: 5K iters ≈ 28 min worst case, 20K ≈ 108 min, 100K ≈ 9 hr.
            timeout_seconds = max(3600, self.mcts_iterations // 2)  # 1 hr minimum, scales with iterations
            try:
                future = self.engine.eval(
                    f"[solverEdge, solverColor, solverInfo] = hybridSolver.solve('{state}');",
                    nargout=0,
                    background=True
                )
                # Wait for completion with timeout
                future.result(timeout=timeout_seconds)
            except TimeoutError:
                logger.error(f"MATLAB timeout after {timeout_seconds}s - MCTS may be hanging or too slow")
                logger.error(f"Current MCTS settings: iterations={self.mcts_iterations}, time_limit={self.time_limit}s")
                logger.error("Try reducing --mcts-iterations or check MATLAB console for diagnostic output")
                return None
            except Exception as matlab_error:
                # Check for specific MATLAB errors
                error_msg = str(matlab_error)
                if 'ParallelRequired' in error_msg or 'PoolNotReady' in error_msg \
                        or 'PoolVerificationFailed' in error_msg or 'InsufficientWorkers' in error_msg:
                    logger.error(f"MATLAB parallel pool failed to initialize: {error_msg}")
                    logger.error("MCTS requires a working parallel pool. Check Parallel Computing Toolbox.")
                    raise RuntimeError(
                        "MCTS parallel pool initialization failed. "
                        "Ensure MATLAB Parallel Computing Toolbox is installed and workers are available."
                    ) from matlab_error
                elif 'OutOfMemory' in error_msg or 'out of memory' in error_msg:
                    logger.error(f"MATLAB out of memory: {error_msg}")
                    logger.error(f"Try reducing --mcts-iterations (current: {self.mcts_iterations})")
                    return None
                elif 'SearchFailed' in error_msg:
                    logger.error(f"MATLAB search failed: {error_msg}")
                    return None
                else:
                    # Unknown MATLAB error - re-raise
                    raise

            # Get results
            edge = int(self.engine.eval("solverEdge"))
            color = str(self.engine.eval("solverColor"))

            elapsed = time.time() - start
            self.total_time += elapsed
            self.moves_calculated += 1

            # Extract detailed statistics from MATLAB
            solver_stats = self._extract_solver_stats(elapsed, grey_count)

            # Update local tracking
            self.last_strategy = solver_stats.get('strategy', 'unknown')
            self.last_score = solver_stats.get('predicted_score', 0.0)

            logger.debug(
                f"HybridSolver: E{edge} {color} "
                f"(strategy={self.last_strategy}, score={self.last_score:.3f}, "
                f"time={elapsed:.2f}s)"
            )

            return (edge, color, solver_stats)

        except Exception as e:
            logger.error(f"HybridSolver error: {e}")
            logger.error("If MATLAB is unresponsive, try restarting MATLAB and re-running")
            return None

    def _extract_solver_stats(self, elapsed: float, grey_count: int) -> dict:
        """Extract detailed statistics from MATLAB solverInfo struct."""
        stats = {
            'wall_clock_time': elapsed,
            'lut_grey_edges': grey_count,
        }

        # Strategy and score (always available)
        try:
            stats['strategy'] = str(self.engine.eval("solverInfo.strategy"))
            stats['predicted_score'] = float(self.engine.eval("solverInfo.score"))
        except Exception:
            stats['strategy'] = 'unknown'
            stats['predicted_score'] = 0.0

        # Time from MATLAB's perspective
        try:
            stats['thinking_time'] = float(self.engine.eval("solverInfo.time"))
        except Exception:
            pass

        # Strategy-specific statistics
        strategy = stats.get('strategy', '')

        if strategy == 'minimax':
            try:
                stats['minimax_depth'] = int(self.engine.eval("solverInfo.depth"))
                stats['minimax_nodes_searched'] = int(self.engine.eval("solverInfo.nodesSearched"))
                stats['minimax_prune_count'] = int(self.engine.eval("solverInfo.pruneCount"))
            except Exception:
                pass

        elif strategy == 'hybrid':
            try:
                stats['minimax_depth'] = int(self.engine.eval(
                    "hybridSolver.MinimaxDepth"
                ))
                # MCTS stats from most recent search
                stats['mcts_iterations'] = int(self.engine.eval(
                    "hybridSolver.LastMCTSIterations"
                ))
                # Tabu improved flag
                stats['tabu_improved'] = bool(self.engine.eval("solverInfo.tabuImproved"))
                if stats['tabu_improved']:
                    stats['tabu_restarts'] = int(self.engine.eval(
                        "hybridSolver.LastTabuRestarts"
                    ))
            except Exception:
                pass

        elif strategy == 'mcts':
            try:
                stats['mcts_iterations'] = int(self.engine.eval("solverInfo.mctsIterations"))
                stats['mcts_root_visits'] = int(self.engine.eval("solverInfo.mctsRootVisits"))
                stats['mcts_simulations'] = int(self.engine.eval("solverInfo.mctsSimulations"))
                stats['tabu_improved'] = bool(self.engine.eval("solverInfo.tabuImproved"))
                if stats['tabu_improved']:
                    stats['tabu_restarts'] = int(self.engine.eval(
                        "hybridSolver.LastTabuRestarts"
                    ))
            except Exception:
                pass

        elif strategy == 'opening':
            # Opening book moves are instant, high confidence
            stats['move_confidence'] = 0.9

        # LUT usage (from solver stats)
        try:
            stats['lut_used'] = bool(self.engine.eval("hybridSolver.LUTLoaded"))
        except Exception:
            stats['lut_used'] = False

        return stats

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

        # Add learning stats
        stats['game_stats'] = self.get_game_stats()
        stats['edge_adjustments'] = self.edge_adjustments

        return stats

    def record_move(self, edge: int, color: str, score_after: float):
        """Record a move and its resulting score for learning."""
        self.move_history.append((edge, color, score_after))

    def end_game(self, result: str, final_score: float):
        """
        Called at end of game to trigger learning update.

        Args:
            result: 'win', 'loss', or 'draw'
            final_score: Final game score
        """
        # Cleanup parallel pool to free workers for next game
        if self.solver_initialized and self.engine:
            try:
                self.engine.eval(
                    "if exist('hybridSolver','var') && isvalid(hybridSolver), "
                    "hybridSolver.MCTS.cleanupPool(); end",
                    nargout=0
                )
                logger.debug("Cleaned up MATLAB parallel pool")
            except Exception as e:
                logger.debug(f"Parallel pool cleanup failed (non-critical): {e}")

        if not self.move_history:
            return

        # Store game result
        self.game_results.append((result, final_score, list(self.move_history)))

        # Learn from this game
        self._learn_from_game(result, final_score)

        # Persist learned adjustments
        self._save_adjustments()

        # Clear history for next game
        self.move_history = []

    def _learn_from_game(self, result: str, final_score: float):
        """
        Update edge adjustments based on game outcome.

        Uses REINFORCE-style policy gradient:
        - Winning moves get positive reinforcement
        - Losing moves get negative reinforcement
        - Magnitude depends on score margin
        - Later moves get more credit/blame (temporal credit assignment)
        """
        if not self.move_history:
            return

        # Continuous score-weighted reward.  Win and loss branches are
        # structurally unchanged.  Draw branch replaces the flat ±0.1 with
        # a score-proportional signal so that near-miss draws (e.g. +0.78)
        # contribute meaningful positive gradient even when wins are absent.
        clamped = max(-2.0, min(2.0, final_score))
        if result == 'win':
            reward = 1.0 + clamped / 2.0                    # 1.0 to 2.0
        elif result == 'draw':
            reward = clamped * 0.65                          # ~[-1.3, +1.3]
        else:  # loss
            reward = -1.0 + clamped / 2.0                   # -2.0 to -1.0

        # Discount factor for temporal credit assignment
        gamma = 0.9
        n_moves = len(self.move_history)

        # Update edge adjustments with discounted rewards
        for i, (edge, color, score) in enumerate(self.move_history):
            # Later moves get more credit/blame (less discounting)
            discount = gamma ** (n_moves - i - 1)
            update = self.learning_rate * reward * discount

            # Apply update (same for both colors - we're learning edge value, not color preference)
            self.edge_adjustments[edge] += update * 0.5

            # Clamp adjustments to reasonable range
            self.edge_adjustments[edge] = max(-1.0, min(1.0, self.edge_adjustments[edge]))

        # Log learning stats
        logger.info(f"Learning: {result} (score {final_score:.2f}), reward={reward:.2f}")
        logger.debug(f"Edge adjustments: {[f'{a:.2f}' for a in self.edge_adjustments]}")

    def _load_adjustments(self):
        """Load learned edge adjustments from disk."""
        try:
            if self.adjustments_path.exists():
                import json
                with open(self.adjustments_path) as f:
                    data = json.load(f)
                self.edge_adjustments = data.get('edge_adjustments', [0.0] * self.NUM_EDGES)
                games_learned = data.get('games_learned', 0)
                logger.info(f"Loaded edge adjustments from {self.adjustments_path} ({games_learned} games learned)")
        except Exception as e:
            logger.warning(f"Failed to load adjustments: {e}")
            self.edge_adjustments = [0.0] * self.NUM_EDGES

    def _save_adjustments(self):
        """Save learned edge adjustments to disk."""
        try:
            import json
            self.adjustments_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'edge_adjustments': self.edge_adjustments,
                'games_learned': len(self.game_results),
                'last_updated': __import__('datetime').datetime.now().isoformat(),
            }
            with open(self.adjustments_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved edge adjustments to {self.adjustments_path}")
        except Exception as e:
            logger.warning(f"Failed to save adjustments: {e}")

    def get_learned_adjustments(self) -> List[float]:
        """Return current learned edge adjustments."""
        return self.edge_adjustments.copy()

    def get_game_stats(self) -> dict:
        """Return statistics from all games played this session."""
        if not self.game_results:
            return {'games': 0, 'wins': 0, 'losses': 0, 'draws': 0}

        wins = sum(1 for r, _, _ in self.game_results if r == 'win')
        losses = sum(1 for r, _, _ in self.game_results if r == 'loss')
        draws = sum(1 for r, _, _ in self.game_results if r == 'draw')

        return {
            'games': len(self.game_results),
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / len(self.game_results) if self.game_results else 0,
            'edge_adjustments': self.edge_adjustments
        }


class AmaraExplorerStrategy:
    """
    Strategy for systematically exploring openings against Amara.

    Wraps HybridSolverStrategy but forces different opening moves to
    explore branches of the game tree that Amara may not have seen
    during training.

    Cycles through all 30 possible first moves (15 edges × 2 colors)
    to find potential weaknesses in Amara's play.

    Based on advice from Geordie Rose: "make moves that are not the
    best ones" in the early game to reach positions unlikely to have
    been seen during training.
    """

    NUM_EDGES = 15
    COLORS = ['G', 'P']  # Green, Purple

    # All 30 possible opening moves
    ALL_OPENINGS = [(edge, color) for edge in range(15) for color in ['G', 'P']]

    def __init__(
        self,
        time_limit: float = 10.0,
        minimax_depth: int = 4,
        mcts_iterations: int = 5000,
        player: int = 1,
        state_path: Optional[Path] = None,
    ):
        """
        Initialize AmaraExplorerStrategy.

        Args:
            time_limit: Time budget per move for underlying solver
            minimax_depth: Depth for minimax in underlying solver
            mcts_iterations: MCTS iterations in underlying solver
            player: Player perspective (1 or 2)
            state_path: Path to persist exploration state
        """
        # Create underlying hybrid solver (no learning - we want controlled experiments)
        self.solver = HybridSolverStrategy(
            time_limit=time_limit,
            minimax_depth=minimax_depth,
            mcts_iterations=mcts_iterations,
            player=player,
            learning_rate=0.0,  # Disable learning during exploration
        )

        # Exploration state
        self.state_path = state_path or Path.home() / ".tangled" / "amara_explorer_state.json"
        self.current_opening_index = 0
        self.exploration_results = {}  # {opening: [(score, result), ...]}
        self._load_state()

        # Track current game
        self.current_game_opening = None
        self.move_count = 0

    def _load_state(self):
        """Load exploration state from disk."""
        try:
            if self.state_path.exists():
                import json
                with open(self.state_path) as f:
                    data = json.load(f)
                self.current_opening_index = data.get('current_opening_index', 0)
                self.exploration_results = data.get('exploration_results', {})
                logger.info(
                    f"Loaded explorer state: opening {self.current_opening_index}/30, "
                    f"{sum(len(v) for v in self.exploration_results.values())} games recorded"
                )
        except Exception as e:
            logger.warning(f"Failed to load explorer state: {e}")

    def _save_state(self):
        """Save exploration state to disk."""
        try:
            import json
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'current_opening_index': self.current_opening_index,
                'exploration_results': self.exploration_results,
                'last_updated': __import__('datetime').datetime.now().isoformat(),
            }
            with open(self.state_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save explorer state: {e}")

    def initialize(self, opponent: Optional[str] = None) -> bool:
        """Initialize the underlying solver."""
        return self.solver.initialize(opponent)

    def calculate_move(
        self,
        state: str,
        score: float = 0.0,
        score_history: list = None
    ) -> Optional[Tuple[int, str, dict]]:
        """
        Calculate move, forcing specific opening on first move.

        Args:
            state: 15-char board state
            score: Current score
            score_history: Score history

        Returns:
            (edge, color, stats) tuple
        """
        grey_count = state.count('-')

        # First move of game (all edges grey)?
        if grey_count == 15:
            self.move_count = 1

            # Get the opening for this game
            opening = self.ALL_OPENINGS[self.current_opening_index]
            self.current_game_opening = opening
            edge, color = opening

            logger.info(
                f"AmaraExplorer: Forcing opening {self.current_opening_index + 1}/30: "
                f"E{edge} {color}"
            )

            stats = {
                'strategy': 'amara_explorer_opening',
                'predicted_score': 0.0,
                'opening_index': self.current_opening_index,
                'forced_opening': f"E{edge}{color}",
            }

            return (edge, color, stats)

        # After first move, use underlying solver
        self.move_count += 1
        return self.solver.calculate_move(state, score, score_history)

    def record_move(self, edge: int, color: str, score_after: float):
        """Record move (delegate to solver for non-opening moves)."""
        if self.move_count > 1:
            self.solver.record_move(edge, color, score_after)

    def end_game(self, result: str, final_score: float):
        """
        Record game result and advance to next opening.

        Args:
            result: 'win', 'loss', or 'draw'
            final_score: Final game score
        """
        # Record result for this opening
        if self.current_game_opening:
            opening_key = f"E{self.current_game_opening[0]}{self.current_game_opening[1]}"
            if opening_key not in self.exploration_results:
                self.exploration_results[opening_key] = []
            self.exploration_results[opening_key].append({
                'score': final_score,
                'result': result,
            })

            logger.info(
                f"AmaraExplorer: Opening {opening_key} -> {result} (score: {final_score:+.4f})"
            )

        # Advance to next opening
        self.current_opening_index = (self.current_opening_index + 1) % len(self.ALL_OPENINGS)

        # Save state
        self._save_state()

        # Reset for next game
        self.current_game_opening = None
        self.move_count = 0

    def get_exploration_summary(self) -> dict:
        """Get summary of exploration results."""
        summary = {
            'total_openings': len(self.ALL_OPENINGS),
            'openings_tested': len(self.exploration_results),
            'current_index': self.current_opening_index,
            'next_opening': f"E{self.ALL_OPENINGS[self.current_opening_index][0]}"
                           f"{self.ALL_OPENINGS[self.current_opening_index][1]}",
            'results_by_opening': {},
        }

        for opening, results in self.exploration_results.items():
            wins = sum(1 for r in results if r['result'] == 'win')
            losses = sum(1 for r in results if r['result'] == 'loss')
            draws = sum(1 for r in results if r['result'] == 'draw')
            avg_score = sum(r['score'] for r in results) / len(results) if results else 0

            summary['results_by_opening'][opening] = {
                'games': len(results),
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'avg_score': avg_score,
            }

        # Find best opening so far
        if summary['results_by_opening']:
            best = max(
                summary['results_by_opening'].items(),
                key=lambda x: (x[1]['wins'], x[1]['avg_score'])
            )
            summary['best_opening'] = best[0]
            summary['best_opening_stats'] = best[1]

        return summary

    def get_stats(self) -> dict:
        """Get strategy statistics."""
        stats = self.solver.get_stats() if hasattr(self.solver, 'get_stats') else {}
        stats['exploration'] = self.get_exploration_summary()
        return stats


class AmaraKillerStrategy:
    """
    Optimized strategy for defeating Amara based on exploration results.

    Uses proven winning openings discovered through systematic exploration
    of all 30 possible first moves. Prioritizes openings that found gaps
    in Amara's D-Wave training data.

    Winning openings (Run 32, January 25, 2026):
    - E14P: +1.799 (highest score)
    - E1G:  +1.382
    - E4G:  +0.864
    - E4P:  -1.547 (win)
    - E12P: -1.474 (win)
    """

    # Proven winning openings, ordered by score
    WINNING_OPENINGS = [
        (14, 'P'),  # E14P: +1.799 - Best opening
        (1, 'G'),   # E1G:  +1.382 - Inner cross
        (4, 'G'),   # E4G:  +0.864 - Inner cross
        (4, 'P'),   # E4P:  win    - Same edge weakness
        (12, 'P'),  # E12P: win    - Outer ring
    ]

    # Openings to avoid (losses)
    LOSING_OPENINGS = [
        (10, 'P'),  # E10P: -0.138
        (12, 'G'),  # E12G: -0.131
        (0, 'P'),   # E0P:  -0.060
    ]

    def __init__(
        self,
        time_limit: float = 10.0,
        minimax_depth: int = 4,
        mcts_iterations: int = 5000,
        player: int = 1,
        opening_mode: str = 'best',
        state_path: Optional[Path] = None,
    ):
        """
        Initialize AmaraKillerStrategy.

        Args:
            time_limit: Time budget per move for underlying solver
            minimax_depth: Depth for minimax in underlying solver
            mcts_iterations: MCTS iterations in underlying solver
            player: Player perspective (1 or 2)
            opening_mode: 'best' (always E14P), 'cycle' (rotate through winners),
                         'random' (random winning opening)
            state_path: Path to persist state for cycle mode
        """
        self.solver = HybridSolverStrategy(
            time_limit=time_limit,
            minimax_depth=minimax_depth,
            mcts_iterations=mcts_iterations,
            player=player,
            learning_rate=0.03,  # Enable learning
        )

        self.opening_mode = opening_mode
        self.state_path = state_path or Path.home() / ".tangled" / "amara_killer_state.json"

        # Track wins/games per opening
        self.opening_stats = {f"E{e}{c}": {'wins': 0, 'games': 0}
                             for e, c in self.WINNING_OPENINGS}
        self.current_opening_index = 0
        self._load_state()

        # Current game state
        self.current_game_opening = None
        self.move_count = 0

    def _load_state(self):
        """Load state from disk."""
        try:
            if self.state_path.exists():
                import json
                with open(self.state_path) as f:
                    data = json.load(f)
                self.current_opening_index = data.get('current_opening_index', 0)
                self.opening_stats = data.get('opening_stats', self.opening_stats)
                total_games = sum(s['games'] for s in self.opening_stats.values())
                total_wins = sum(s['wins'] for s in self.opening_stats.values())
                logger.info(
                    f"Loaded amara_killer state: {total_wins}/{total_games} wins, "
                    f"mode={self.opening_mode}"
                )
        except Exception as e:
            logger.warning(f"Failed to load amara_killer state: {e}")

    def _save_state(self):
        """Save state to disk."""
        try:
            import json
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'current_opening_index': self.current_opening_index,
                'opening_stats': self.opening_stats,
                'last_updated': __import__('datetime').datetime.now().isoformat(),
            }
            with open(self.state_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save amara_killer state: {e}")

    def _select_opening(self) -> Tuple[int, str]:
        """Select opening based on mode."""
        if self.opening_mode == 'best':
            # Always use the highest-scoring opening
            return self.WINNING_OPENINGS[0]

        elif self.opening_mode == 'cycle':
            # Cycle through winning openings
            opening = self.WINNING_OPENINGS[self.current_opening_index]
            return opening

        elif self.opening_mode == 'random':
            # Random winning opening
            import random
            return random.choice(self.WINNING_OPENINGS)

        else:
            # Default to best
            return self.WINNING_OPENINGS[0]

    def initialize(self, opponent: Optional[str] = None) -> bool:
        """Initialize the underlying solver."""
        return self.solver.initialize(opponent)

    def calculate_move(
        self,
        state: str,
        score: float = 0.0,
        score_history: list = None
    ) -> Optional[Tuple[int, str, dict]]:
        """
        Calculate move, using winning openings on first move.

        Args:
            state: 15-char board state
            score: Current score
            score_history: Score history

        Returns:
            (edge, color, stats) tuple
        """
        grey_count = state.count('-')

        # First move of game (all edges grey)?
        if grey_count == 15:
            self.move_count = 1

            # Select winning opening
            edge, color = self._select_opening()
            self.current_game_opening = (edge, color)

            opening_key = f"E{edge}{color}"
            logger.info(f"AmaraKiller: Using opening {opening_key}")

            stats = {
                'strategy': 'amara_killer_opening',
                'predicted_score': 1.0,  # Optimistic - these are winning openings
                'opening': opening_key,
                'opening_mode': self.opening_mode,
            }

            return (edge, color, stats)

        # After first move, use underlying solver
        self.move_count += 1
        return self.solver.calculate_move(state, score, score_history)

    def record_move(self, edge: int, color: str, score_after: float):
        """Record move for learning."""
        if self.move_count > 1:
            self.solver.record_move(edge, color, score_after)

    def end_game(self, result: str, final_score: float):
        """
        Record game result and update statistics.

        Args:
            result: 'win', 'loss', or 'draw'
            final_score: Final game score
        """
        # Update opening stats
        if self.current_game_opening:
            opening_key = f"E{self.current_game_opening[0]}{self.current_game_opening[1]}"
            if opening_key in self.opening_stats:
                self.opening_stats[opening_key]['games'] += 1
                if result == 'win':
                    self.opening_stats[opening_key]['wins'] += 1

            logger.info(
                f"AmaraKiller: {opening_key} -> {result} (score: {final_score:+.4f})"
            )

        # Advance cycle index
        if self.opening_mode == 'cycle':
            self.current_opening_index = (
                (self.current_opening_index + 1) % len(self.WINNING_OPENINGS)
            )

        # Delegate to solver for learning
        self.solver.end_game(result, final_score)

        # Save state
        self._save_state()

        # Reset for next game
        self.current_game_opening = None
        self.move_count = 0

    def get_stats(self) -> dict:
        """Get strategy statistics."""
        stats = self.solver.get_stats() if hasattr(self.solver, 'get_stats') else {}

        total_games = sum(s['games'] for s in self.opening_stats.values())
        total_wins = sum(s['wins'] for s in self.opening_stats.values())

        stats['amara_killer'] = {
            'opening_mode': self.opening_mode,
            'total_games': total_games,
            'total_wins': total_wins,
            'win_rate': total_wins / total_games if total_games > 0 else 0,
            'opening_stats': self.opening_stats,
        }

        return stats


class MelissaKillerStrategy:
    """
    Optimized strategy for defeating Melissa based on exploration results.

    Uses proven winning openings discovered through systematic exploration.
    Unlike Amara, Melissa has more variance, so we use openings that
    consistently won across multiple runs.

    Run 37 analysis (30 games, January 25, 2026):
    - E12P: 60% win rate (6W/2D/2L) - Best opening
    - E13P: 50% win rate (5W/2D/3L) - Second best
    - E9G:  10% win rate (1W/5D/4L) - Dropped from rotation
    """

    # Proven winning openings against Melissa
    WINNING_OPENINGS = [
        (12, 'P'),  # E12P: 60% win rate - primary opening
        (13, 'P'),  # E13P: 50% win rate - secondary opening
    ]

    # Safe openings (never lost in both runs)
    SAFE_OPENINGS = [
        (4, 'P'),   # E4P:  draw/draw
        (5, 'P'),   # E5P:  draw/draw
        (13, 'P'),  # E13P: draw/win
        (14, 'P'),  # E14P: draw/draw
    ]

    def __init__(
        self,
        time_limit: float = 10.0,
        minimax_depth: int = 4,
        mcts_iterations: int = 5000,
        player: int = 1,
        opening_mode: str = 'best',
        state_path: Optional[Path] = None,
    ):
        """
        Initialize MelissaKillerStrategy.

        Args:
            time_limit: Time budget per move for underlying solver
            minimax_depth: Depth for minimax in underlying solver
            mcts_iterations: MCTS iterations in underlying solver
            player: Player perspective (1 or 2)
            opening_mode: 'best' (always E9G), 'cycle' (rotate through winners),
                         'random' (random winning opening)
            state_path: Path to persist state for cycle mode
        """
        self.solver = HybridSolverStrategy(
            time_limit=time_limit,
            minimax_depth=minimax_depth,
            mcts_iterations=mcts_iterations,
            player=player,
            learning_rate=0.03,
        )

        self.opening_mode = opening_mode
        self.state_path = state_path or Path.home() / ".tangled" / "melissa_killer_state.json"

        self.opening_stats = {f"E{e}{c}": {'wins': 0, 'games': 0}
                             for e, c in self.WINNING_OPENINGS}
        self.current_opening_index = 0
        self._load_state()

        self.current_game_opening = None
        self.move_count = 0

    def _load_state(self):
        """Load state from disk."""
        try:
            if self.state_path.exists():
                import json
                with open(self.state_path) as f:
                    data = json.load(f)
                self.current_opening_index = data.get('current_opening_index', 0)
                self.opening_stats = data.get('opening_stats', self.opening_stats)
                total_games = sum(s['games'] for s in self.opening_stats.values())
                total_wins = sum(s['wins'] for s in self.opening_stats.values())
                logger.info(
                    f"Loaded melissa_killer state: {total_wins}/{total_games} wins"
                )
        except Exception as e:
            logger.warning(f"Failed to load melissa_killer state: {e}")

    def _save_state(self):
        """Save state to disk."""
        try:
            import json
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'current_opening_index': self.current_opening_index,
                'opening_stats': self.opening_stats,
                'last_updated': __import__('datetime').datetime.now().isoformat(),
            }
            with open(self.state_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save melissa_killer state: {e}")

    def _select_opening(self) -> Tuple[int, str]:
        """Select opening based on mode."""
        if self.opening_mode == 'best':
            return self.WINNING_OPENINGS[0]
        elif self.opening_mode == 'cycle':
            opening = self.WINNING_OPENINGS[self.current_opening_index]
            return opening
        elif self.opening_mode == 'random':
            import random
            return random.choice(self.WINNING_OPENINGS)
        else:
            return self.WINNING_OPENINGS[0]

    def initialize(self, opponent: Optional[str] = None) -> bool:
        """Initialize the underlying solver."""
        return self.solver.initialize(opponent)

    def calculate_move(
        self,
        state: str,
        score: float = 0.0,
        score_history: list = None
    ) -> Optional[Tuple[int, str, dict]]:
        """Calculate move, using winning openings on first move."""
        grey_count = state.count('-')

        if grey_count == 15:
            self.move_count = 1
            edge, color = self._select_opening()
            self.current_game_opening = (edge, color)

            opening_key = f"E{edge}{color}"
            logger.info(f"MelissaKiller: Using opening {opening_key}")

            stats = {
                'strategy': 'melissa_killer_opening',
                'predicted_score': 1.0,
                'opening': opening_key,
                'opening_mode': self.opening_mode,
            }

            return (edge, color, stats)

        self.move_count += 1
        return self.solver.calculate_move(state, score, score_history)

    def record_move(self, edge: int, color: str, score_after: float):
        """Record move for learning."""
        if self.move_count > 1:
            self.solver.record_move(edge, color, score_after)

    def end_game(self, result: str, final_score: float):
        """Record game result and update statistics."""
        if self.current_game_opening:
            opening_key = f"E{self.current_game_opening[0]}{self.current_game_opening[1]}"
            if opening_key in self.opening_stats:
                self.opening_stats[opening_key]['games'] += 1
                if result == 'win':
                    self.opening_stats[opening_key]['wins'] += 1

            logger.info(
                f"MelissaKiller: {opening_key} -> {result} (score: {final_score:+.4f})"
            )

        if self.opening_mode == 'cycle':
            self.current_opening_index = (
                (self.current_opening_index + 1) % len(self.WINNING_OPENINGS)
            )

        self.solver.end_game(result, final_score)
        self._save_state()

        self.current_game_opening = None
        self.move_count = 0

    def get_stats(self) -> dict:
        """Get strategy statistics."""
        stats = self.solver.get_stats() if hasattr(self.solver, 'get_stats') else {}

        total_games = sum(s['games'] for s in self.opening_stats.values())
        total_wins = sum(s['wins'] for s in self.opening_stats.values())

        stats['melissa_killer'] = {
            'opening_mode': self.opening_mode,
            'total_games': total_games,
            'total_wins': total_wins,
            'win_rate': total_wins / total_games if total_games > 0 else 0,
            'opening_stats': self.opening_stats,
        }

        return stats


class AlphaQExplorerStrategy:
    """
    Thompson Sampling strategy for defeating AlphaQ Up with a closed learning loop.

    Uses Thompson Sampling to dynamically select first moves, balancing exploration
    and exploitation. The underlying solver is disabled (learning_rate=0.0) for the
    first MIN_GAMES_BEFORE_LEARNING games to gather clean opening data, then
    re-enabled for the rest.

    All 30 possible opening moves are tracked with win/draw/loss counts and
    converted to Beta distribution parameters. Each game, the opening with the
    highest Beta sample is played, ensuring safe openings (high draws, no losses)
    are preferred while allowing untried openings to be explored.
    """

    NUM_EDGES = 15
    MIN_GAMES_BEFORE_LEARNING = 10

    # All 30 possible opening moves (edge 0..14 × G/P)
    ALL_OPENINGS = [(edge, color) for edge in range(15) for color in ['G', 'P']]

    def __init__(
        self,
        time_limit: float = 10.0,
        minimax_depth: int = 4,
        mcts_iterations: int = 5000,
        player: int = 1,
        state_path: Optional[Path] = None,
        force_opening: Optional[str] = None,
        opening_mode: str = 'forced',
    ):
        """
        Initialize AlphaQExplorerStrategy.

        Args:
            time_limit: Time budget per move for underlying solver
            minimax_depth: Depth for minimax in underlying solver
            mcts_iterations: MCTS iterations in underlying solver
            player: Player perspective (1 or 2)
            state_path: Path to persist state across runs
            force_opening: Force specific opening (e.g., 'E7G') to override Thompson Sampling
            opening_mode: 'forced' (use force_opening), 'thompson' (Beta sampling),
                          'round_robin' (cycle all 30 openings systematically)
        """
        self.time_limit = time_limit
        self.minimax_depth = minimax_depth
        self.mcts_iterations = mcts_iterations
        self.player = player
        self.force_opening = force_opening
        self.opening_mode = opening_mode

        # Round-robin state: index into ALL_OPENINGS, persisted across games
        self.rr_index = 0
        self.rr_games_per_opening = 3  # Phase 1: 3 games per opening

        # Start with learning disabled; use SA LUT to match server adjudicator
        self.solver = HybridSolverStrategy(
            time_limit=time_limit,
            minimax_depth=minimax_depth,
            mcts_iterations=mcts_iterations,
            player=player,
            learning_rate=0.0,
        )
        self.solver.lut_file = 'terminal_scores_sa.mat'

        # Persistent state
        self.state_path = state_path or Path.home() / ".tangled" / "alphaq_explorer_state.json"

        # Thompson sampling state: each opening tracks W/D/L
        self.openings = {}  # {opening_key: {'wins': 0, 'draws': 0, 'losses': 0}}
        self.games_played = 0

        # Current game tracking
        self.current_game_opening = None
        self.move_count = 0
        self.thompson_sample = 0.0  # For observability
        self.thompson_alpha = 0.0
        self.thompson_beta = 0.0

        self._load_state()

    def _load_state(self):
        """Load persisted state from disk. Migrate v1 format to v2."""
        try:
            # Always initialize all 30 openings first
            self.openings = {f"E{e}{c}": {'wins': 0, 'draws': 0, 'losses': 0}
                             for e in range(15) for c in ['G', 'P']}

            if self.state_path.exists():
                import json
                with open(self.state_path) as f:
                    data = json.load(f)

                # Detect format version
                if 'version' in data:
                    # v2 format
                    loaded_openings = data.get('openings', {})
                    for key in self.openings:
                        if key in loaded_openings:
                            self.openings[key] = loaded_openings[key]
                    self.games_played = data.get('games_played', 0)
                    self.rr_index = data.get('rr_index', 0)
                    logger.info(
                        f"Loaded AlphaQ explorer state (v2): "
                        f"{self.games_played} games, {len([k for k, v in self.openings.items() if v['wins'] + v['draws'] + v['losses'] > 0])} tested openings"
                        f"{f', rr_index={self.rr_index}' if self.opening_mode == 'round_robin' else ''}"
                    )
                else:
                    # v1 format: migrate
                    logger.info("Migrating AlphaQ explorer state from v1 to v2")
                    exploration_results = data.get('exploration_results', {})

                    # Tally W/D/L for each opening
                    for key, results in exploration_results.items():
                        if key not in self.openings:
                            self.openings[key] = {'wins': 0, 'draws': 0, 'losses': 0}
                        for r in results:
                            result_type = r['result']
                            if result_type == 'win':
                                self.openings[key]['wins'] += 1
                            elif result_type == 'draw':
                                self.openings[key]['draws'] += 1
                            elif result_type == 'loss':
                                self.openings[key]['losses'] += 1

                    # Compute total games
                    self.games_played = sum(
                        v['wins'] + v['draws'] + v['losses']
                        for v in self.openings.values()
                    )
                    logger.info(
                        f"Migrated to v2: {self.games_played} games, "
                        f"openings={[k for k, v in self.openings.items() if v['wins'] + v['draws'] + v['losses'] > 0]}"
                    )
                    self._save_state()
            else:
                # Fresh start
                self.games_played = 0
        except Exception as e:
            logger.warning(f"Failed to load AlphaQ explorer state, starting fresh: {e}")
            self.openings = {f"E{e}{c}": {'wins': 0, 'draws': 0, 'losses': 0}
                             for e in range(15) for c in ['G', 'P']}
            self.games_played = 0

    def _save_state(self):
        """Persist state to disk in v2 format."""
        try:
            import json
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'version': 2,
                'openings': self.openings,
                'games_played': self.games_played,
                'rr_index': self.rr_index,
                'last_updated': __import__('datetime').datetime.now().isoformat(),
            }
            with open(self.state_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save AlphaQ explorer state: {e}")

    def initialize(self, opponent: Optional[str] = None) -> bool:
        """
        Initialize the underlying solver. Enable learning if we've played
        enough games to have reliable opening data.
        """
        result = self.solver.initialize(opponent)

        if self.games_played >= self.MIN_GAMES_BEFORE_LEARNING:
            # Enable learning
            self.solver.learning_rate = 0.03
            # Push any previously accumulated edge bias into MATLAB
            if any(a != 0.0 for a in self.solver.edge_adjustments):
                self._push_edge_bias()
                logger.info("AlphaQ explorer: Re-applied edge bias from previous session")

        return result

    def calculate_move(
        self,
        state: str,
        score: float = 0.0,
        score_history: list = None
    ) -> Optional[Tuple[int, str, dict]]:
        """
        Calculate move. On the first move of each game, use Thompson Sampling
        to select an opening. All subsequent moves delegate to the underlying solver.
        """
        grey_count = state.count('-')

        # First move of the game
        if grey_count == 15:
            self.move_count = 1

            if self.opening_mode == 'round_robin':
                # Round-robin: cycle through all 30 openings systematically
                opening_idx = self.rr_index // self.rr_games_per_opening
                game_within = self.rr_index % self.rr_games_per_opening
                if opening_idx >= len(self.ALL_OPENINGS):
                    logger.info("AlphaQ [round_robin]: All openings exhausted, wrapping around")
                    opening_idx = opening_idx % len(self.ALL_OPENINGS)
                edge_rr, color_rr = self.ALL_OPENINGS[opening_idx]
                best_opening = f"E{edge_rr}{color_rr}"
                self.thompson_sample = 0.0
                self.thompson_alpha = 0.0
                self.thompson_beta = 0.0
                logger.info(
                    f"AlphaQ [round_robin]: Opening {best_opening} "
                    f"(opening {opening_idx + 1}/30, game {game_within + 1}/{self.rr_games_per_opening}, "
                    f"rr_index={self.rr_index})"
                )

            elif self.opening_mode == 'forced' and self.force_opening:
                best_opening = self.force_opening
                # Get stats for the forced opening if it exists
                if best_opening in self.openings:
                    counts = self.openings[best_opening]
                    self.thompson_alpha = 1 + counts['wins'] + 0.5 * counts['draws']
                    self.thompson_beta = 1 + counts['losses'] + 0.5 * counts['draws']
                else:
                    self.thompson_alpha = 1.0
                    self.thompson_beta = 1.0
                self.thompson_sample = 1.0
                logger.info(f"AlphaQ [forced]: Opening {best_opening} (FORCED)")

            else:
                # Thompson sampling: pick opening with highest Beta sample
                import random
                best_opening = None
                best_sample = -1.0

                for key, counts in self.openings.items():
                    # Beta parameters: draws count as half-wins
                    alpha = 1 + counts['wins'] + 0.5 * counts['draws']
                    beta = 1 + counts['losses'] + 0.5 * counts['draws']

                    # Approach C: E7G bias - add virtual draws to favor proven best opening
                    if key == 'E7G':
                        alpha += 25  # Equivalent to 50 virtual draws (50 * 0.5 = 25)
                        beta += 25   # Keep ratio balanced

                    sample = random.betavariate(alpha, beta)

                    if sample > best_sample:
                        best_opening = key
                        best_sample = sample
                        self.thompson_alpha = alpha
                        self.thompson_beta = beta

                self.thompson_sample = best_sample
                logger.info(
                    f"AlphaQ [thompson]: Opening {best_opening} "
                    f"(sample={best_sample:.4f}, α={self.thompson_alpha:.2f}, β={self.thompson_beta:.2f})"
                )

            # Parse edge and color from opening key
            edge = int(best_opening[1:-1])
            color = best_opening[-1]
            self.current_game_opening = (edge, color)

            stats = {
                'strategy': 'alphaq_explorer_opening',
                'predicted_score': 0.0,
                'forced_opening': best_opening,
                'thompson_sample': self.thompson_sample,
                'thompson_alpha': self.thompson_alpha,
                'thompson_beta': self.thompson_beta,
            }
            return (edge, color, stats)

        # After first move, delegate to underlying solver
        self.move_count += 1

        # Approach E: Winning-push heuristic
        # When score >0.75, we're in winning territory - be more careful/thorough
        if score > 0.75:
            # Save original MCTS iterations
            original_iterations = self.solver.mcts_iterations

            # Increase MCTS iterations by 50% for critical winning positions
            boosted_iterations = int(original_iterations * 1.5)
            self.solver.mcts_iterations = boosted_iterations

            # Update MATLAB solver with boosted iterations
            if self.solver.solver_initialized and self.solver.engine:
                try:
                    self.solver.engine.eval(
                        f"hybridSolver.MCTSIterations = {boosted_iterations};",
                        nargout=0
                    )
                    logger.info(f"AlphaQ [winning-push]: Score {score:+.3f} >0.75, boosted MCTS {original_iterations} → {boosted_iterations}")
                except Exception as e:
                    logger.warning(f"Failed to boost MCTS iterations: {e}")

            # Calculate move with boosted iterations
            result = self.solver.calculate_move(state, score, score_history)

            # Restore original iterations
            self.solver.mcts_iterations = original_iterations
            if self.solver.solver_initialized and self.solver.engine:
                try:
                    self.solver.engine.eval(
                        f"hybridSolver.MCTSIterations = {original_iterations};",
                        nargout=0
                    )
                except Exception:
                    pass

            return result
        else:
            return self.solver.calculate_move(state, score, score_history)

    def _push_edge_bias(self):
        """
        Forward learned edge_adjustments into MATLAB via hybridSolver.setEdgeBias().
        This closes the learning loop: REINFORCE updates edge_adjustments in Python,
        and this method propagates them into the MCTS rollout priors in MATLAB.
        """
        if not self.solver.solver_initialized or self.solver.engine is None:
            logger.debug("AlphaQ: Cannot push edge bias — solver not initialized")
            return

        try:
            bias_str = ', '.join(f'{a:.6f}' for a in self.solver.edge_adjustments)
            self.solver.engine.eval(
                f"hybridSolver.setEdgeBias([{bias_str}]);",
                nargout=0
            )
            logger.info(
                f"AlphaQ: Pushed edge bias to MATLAB: "
                f"[{', '.join(f'{a:.3f}' for a in self.solver.edge_adjustments)}]"
            )
        except Exception as e:
            logger.warning(f"AlphaQ: Failed to push edge bias: {e}")

    def record_move(self, edge: int, color: str, score_after: float):
        """Record move for learning (delegate to solver for non-opening moves)."""
        if self.move_count > 1:
            self.solver.record_move(edge, color, score_after)

    def end_game(self, result: str, final_score: float):
        """
        Record game result and update opening counts. Trigger learning if
        we've passed the MIN_GAMES_BEFORE_LEARNING threshold.
        """
        # Normalise result
        if result not in ('win', 'loss', 'draw'):
            result = 'draw'

        if self.current_game_opening:
            edge, color = self.current_game_opening
            opening_key = f"E{edge}{color}"

            # Update opening counts: map result to the correct key
            result_key_map = {'wins': 'win', 'draws': 'draw', 'losses': 'loss'}
            for key_name, result_type in result_key_map.items():
                if result == result_type:
                    self.openings[opening_key][key_name] += 1
                    break

            self.games_played += 1

            # Advance round-robin counter
            if self.opening_mode == 'round_robin':
                self.rr_index += 1
                total_rr_games = len(self.ALL_OPENINGS) * self.rr_games_per_opening
                logger.info(
                    f"AlphaQ [round_robin]: {opening_key} -> {result} "
                    f"(score: {final_score:+.4f}, rr_index={self.rr_index}/{total_rr_games}, "
                    f"games_played={self.games_played})"
                )
            else:
                logger.info(
                    f"AlphaQ [{self.opening_mode}]: {opening_key} -> {result} "
                    f"(score: {final_score:+.4f}, games_played={self.games_played})"
                )

            # Learning gating
            if self.games_played >= self.MIN_GAMES_BEFORE_LEARNING:
                # Enable learning if this is the first game crossing the threshold
                if self.games_played == self.MIN_GAMES_BEFORE_LEARNING:
                    self.solver.learning_rate = 0.03
                    logger.info(
                        f"AlphaQ: Reached {self.MIN_GAMES_BEFORE_LEARNING} games, "
                        f"enabling learning (learning_rate=0.03)"
                    )

                # Trigger REINFORCE learning inside the solver
                self.solver.end_game(result, final_score)

                # Close the loop: push updated adjustments into MATLAB
                self._push_edge_bias()
            else:
                # Before MIN_GAMES_BEFORE_LEARNING, just clear solver history
                self.solver.move_history = []

        # Save state
        self._save_state()

        # Reset for next game
        self.current_game_opening = None
        self.move_count = 0

    def get_stats(self) -> dict:
        """Return strategy statistics including current Thompson parameters."""
        stats = self.solver.get_stats() if hasattr(self.solver, 'get_stats') else {}

        # Compute current α/β for all openings
        opening_stats = {}
        for key, counts in self.openings.items():
            alpha = 1 + counts['wins'] + 0.5 * counts['draws']
            beta = 1 + counts['losses'] + 0.5 * counts['draws']
            opening_stats[key] = {
                'wins': counts['wins'],
                'draws': counts['draws'],
                'losses': counts['losses'],
                'alpha': alpha,
                'beta': beta,
                'mean': alpha / (alpha + beta),
            }

        stats['alphaq_explorer'] = {
            'games_played': self.games_played,
            'learning_enabled': self.games_played >= self.MIN_GAMES_BEFORE_LEARNING,
            'thompson_sample': self.thompson_sample,
            'thompson_alpha': self.thompson_alpha,
            'thompson_beta': self.thompson_beta,
            'openings': opening_stats,
            'edge_adjustments': self.solver.edge_adjustments,
        }

        return stats
