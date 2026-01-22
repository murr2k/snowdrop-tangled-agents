"""
MATLAB MCTS Strategy for live play against MCTS Melissa.

Uses the TangledMCTS engine implemented in MATLAB with configurable
compute time to match or exceed Melissa's search depth.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

# Try to import MATLAB engine
MATLAB_AVAILABLE = False
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    logger.info("MATLAB Engine not available. MatlabMCTSStrategy will use fallback.")


@dataclass
class MCTSParams:
    """Tunable MCTS parameters."""

    # Search parameters
    iterations: int = 5000
    time_limit: float = 20.0  # Match Melissa's ~20-25s compute
    exploration: float = 1.414
    prior_weight: float = 2.0

    # Edge value adjustments (learned from game statistics)
    # Index 0-14 corresponds to E0-E14
    edge_green_bonus: list = field(default_factory=lambda: [0.0] * 15)
    edge_purple_bonus: list = field(default_factory=lambda: [0.0] * 15)

    # Opening preferences (which edges to prioritize early)
    opening_sequence: list = field(default_factory=lambda: [9, 10, 11, 5, 12, 13])
    opening_moves: int = 3  # How many moves to use opening sequence

    # Adaptive parameters
    losing_exploration_boost: float = 1.3  # Boost exploration when losing
    winning_exploration_reduction: float = 0.8  # Reduce when winning

    def save(self, path: str):
        """Save parameters to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'MCTSParams':
        """Load parameters from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class MatlabMCTSStrategy:
    """
    MCTS strategy using MATLAB TangledMCTS engine.

    Designed for live play against MCTS Melissa with:
    - High compute time (~20-25 seconds per move)
    - Tunable parameters based on game statistics
    - Adaptive exploration based on score momentum
    """

    def __init__(
        self,
        params: Optional[MCTSParams] = None,
        params_path: Optional[str] = None,
        fallback_to_python: bool = True,
    ):
        """
        Initialize MATLAB MCTS strategy.

        Args:
            params: MCTSParams instance (creates default if None)
            params_path: Path to load/save parameters
            fallback_to_python: Use Python MCTS if MATLAB unavailable
        """
        self.fallback_to_python = fallback_to_python
        self.params_path = params_path

        # Load or create parameters
        if params is not None:
            self.params = params
        elif params_path and Path(params_path).exists():
            self.params = MCTSParams.load(params_path)
            logger.info(f"Loaded MCTS params from {params_path}")
        else:
            self.params = MCTSParams()

        self.matlab_engine = None
        self.mcts_initialized = False
        self.move_history = []
        self.game_results = []

        # Initialize MATLAB if available
        if MATLAB_AVAILABLE:
            try:
                self._init_matlab()
            except Exception as e:
                logger.warning(f"Failed to initialize MATLAB: {e}")

    def _init_matlab(self):
        """Initialize MATLAB engine and TangledMCTS."""
        logger.info("Starting MATLAB engine for MCTS strategy...")
        self.matlab_engine = matlab.engine.start_matlab()

        # Add RL toolbox path
        rl_path = Path(__file__).parent.parent / "matlab" / "rl"
        self.matlab_engine.addpath(str(rl_path), nargout=0)

        # Create MCTS engine with high compute
        logger.info(f"Creating TangledMCTS (iterations={self.params.iterations}, time={self.params.time_limit}s)")
        self.matlab_engine.eval(f'''
            mcts_engine = TangledMCTS(...
                'Iterations', {self.params.iterations}, ...
                'TimeLimit', {self.params.time_limit}, ...
                'Exploration', {self.params.exploration}, ...
                'PriorWeight', {self.params.prior_weight}, ...
                'UseParallel', false, ...
                'Player', 1);
        ''', nargout=0)

        self.mcts_initialized = True
        logger.info("MATLAB MCTS engine ready")

    def calculate_move(
        self,
        state: str,
        score: float = 0.0,
        history: list = None
    ) -> Optional[tuple[int, str]]:
        """
        Calculate the best move using MATLAB MCTS.

        Args:
            state: 15-char board state string
            score: Current game score
            history: List of (edge, color, score) tuples

        Returns:
            (edge_index, color) or None
        """
        grey_count = state.count('-')
        if grey_count == 0:
            return None

        # Opening phase: use predefined sequence
        move_num = 15 - grey_count
        our_move_num = (move_num + 1) // 2

        if our_move_num < self.params.opening_moves:
            for edge in self.params.opening_sequence:
                if state[edge] == '-':
                    # Apply edge bonus adjustments
                    green_bonus = self.params.edge_green_bonus[edge]
                    purple_bonus = self.params.edge_purple_bonus[edge]

                    # Default color based on edge type
                    if edge in [9, 10, 11]:  # MY_EDGES
                        color = 'G'
                    elif edge in [5, 12, 13]:  # OPP_EDGES
                        color = 'P'
                    else:
                        color = 'G' if green_bonus >= purple_bonus else 'P'

                    logger.info(f"Opening move {our_move_num + 1}: E{edge} {color}")
                    return (edge, color)

        # Main MCTS search
        if self.mcts_initialized and self.matlab_engine is not None:
            try:
                return self._matlab_mcts_move(state, score, history)
            except Exception as e:
                logger.warning(f"MATLAB MCTS failed: {e}")

        # Fallback to Python MCTS
        if self.fallback_to_python:
            return self._python_mcts_fallback(state, score, history)

        # Last resort: heuristic
        return self._heuristic_fallback(state)

    def _matlab_mcts_move(
        self,
        state: str,
        score: float,
        history: list
    ) -> Optional[tuple[int, str]]:
        """Get move from MATLAB MCTS engine."""

        # Adaptive exploration based on momentum
        exploration = self.params.exploration
        if history and len(history) >= 2:
            recent_scores = [h[2] for h in history[-4:]]
            if len(recent_scores) >= 2:
                momentum = recent_scores[-1] - recent_scores[0]
                if momentum < -0.5:
                    exploration *= self.params.losing_exploration_boost
                    logger.info(f"Losing momentum ({momentum:.2f}), boosting exploration to {exploration:.2f}")
                elif momentum > 0.5:
                    exploration *= self.params.winning_exploration_reduction
                    logger.info(f"Winning momentum ({momentum:.2f}), reducing exploration to {exploration:.2f}")

        # Update exploration in MATLAB
        self.matlab_engine.eval(f'mcts_engine.Exploration = {exploration};', nargout=0)

        # Run MCTS search
        # Debug: Log the state being sent to MATLAB
        grey_indices = [i for i, c in enumerate(state) if c == '-']
        logger.debug(f"MCTS input state: {state} (grey: {grey_indices})")

        self.matlab_engine.workspace['state_str'] = state
        self.matlab_engine.eval('''
            [edge_out, color_out, info_out] = mcts_engine.search(state_str);
        ''', nargout=0)

        edge = int(self.matlab_engine.workspace['edge_out'])
        color = str(self.matlab_engine.workspace['color_out'])

        # Log search info with diagnostic compute metrics
        try:
            info = self.matlab_engine.workspace['info_out']
            iterations = int(info['iterations'])
            search_time = float(info['time'])

            # Extract diagnostic metrics if available
            cpu_time = float(info.get('cpuTime', 0))
            nodes = int(info.get('nodesExpanded', 0))
            sims = int(info.get('simulations', 0))
            depth = int(info.get('treeDepth', 0))
            cpu_eff = float(info.get('cpuEfficiency', 0))

            # Log with compute effort diagnostics
            logger.info(
                f"MCTS: E{edge} {color} ({iterations} iter in {search_time:.1f}s, "
                f"CPU={cpu_time:.2f}s [{cpu_eff:.0%}], nodes={nodes}, sims={sims}, depth={depth})"
            )
        except Exception:
            logger.info(f"MCTS: E{edge} {color}")

        # Validate move
        if 0 <= edge < 15 and state[edge] == '-':
            logger.debug(f"MCTS returning E{edge} {color} (state[{edge}]='{state[edge]}')")
            return (edge, color)

        logger.warning(f"MATLAB returned invalid move E{edge} (state[{edge}]='{state[edge]}'), falling back")
        return None

    def _python_mcts_fallback(
        self,
        state: str,
        score: float,
        history: list
    ) -> Optional[tuple[int, str]]:
        """Fallback to Python MCTS if MATLAB unavailable."""
        try:
            from snowdrop_tangled_agents.strategy.mcts_strategy import MCTSStrategy

            # Use Python MCTS with similar settings
            mcts = MCTSStrategy(
                time_limit=min(self.params.time_limit, 5.0),  # Cap at 5s for Python
                max_iterations=min(self.params.iterations, 10000),
                exploration=self.params.exploration,
                prior_weight=self.params.prior_weight
            )
            return mcts.calculate_move(state, score, history)
        except Exception as e:
            logger.warning(f"Python MCTS fallback failed: {e}")
            return None

    def _heuristic_fallback(self, state: str) -> Optional[tuple[int, str]]:
        """Simple heuristic fallback."""
        # Priority: MY_EDGES green, OPP_EDGES purple, HUB_EDGES green
        priority_edges = [
            (9, 'G'), (10, 'G'), (11, 'G'),  # MY_EDGES
            (5, 'P'), (12, 'P'), (13, 'P'),  # OPP_EDGES
            (2, 'G'),  # HUB
        ]

        for edge, color in priority_edges:
            if state[edge] == '-':
                return (edge, color)

        # Any available edge
        for i, c in enumerate(state):
            if c == '-':
                return (i, 'G')

        return None

    def record_move(self, edge: int, color: str, score_after: float):
        """Record a move for learning."""
        self.move_history.append({
            'edge': edge,
            'color': color,
            'score': score_after
        })

    def end_game(self, result: str, final_score: float):
        """
        Called at end of game to update parameters based on outcome.

        Args:
            result: 'win', 'loss', or 'draw'
            final_score: Final game score
        """
        if not self.move_history:
            self.move_history = []
            return

        # Store game result
        self.game_results.append({
            'result': result,
            'score': final_score,
            'moves': list(self.move_history)
        })

        # Update edge bonuses based on game outcome
        self._update_edge_bonuses(result, final_score)

        # Save parameters if path specified
        if self.params_path:
            self.params.save(self.params_path)
            logger.info(f"Saved updated params to {self.params_path}")

        # Clear move history for next game
        self.move_history = []

        logger.info(f"Game ended: {result} (score={final_score:.3f})")

    def _update_edge_bonuses(self, result: str, final_score: float):
        """Update edge value bonuses based on game outcome."""
        if not self.move_history:
            return

        # Learning rate based on result significance
        if result == 'win':
            lr = 0.05 * (1 + min(final_score, 2) / 2)  # 0.05 to 0.1
        elif result == 'loss':
            lr = -0.03 * (1 + min(-final_score, 2) / 2)  # -0.03 to -0.06
        else:  # draw
            lr = 0.01 if final_score > 0 else -0.01

        # Update bonuses for moves we made
        for move in self.move_history:
            edge = move['edge']
            color = move['color']

            if color == 'G':
                self.params.edge_green_bonus[edge] += lr
                # Clamp to reasonable range
                self.params.edge_green_bonus[edge] = max(-1.0, min(1.0, self.params.edge_green_bonus[edge]))
            else:
                self.params.edge_purple_bonus[edge] += lr
                self.params.edge_purple_bonus[edge] = max(-1.0, min(1.0, self.params.edge_purple_bonus[edge]))

        logger.info(f"Updated edge bonuses (lr={lr:.3f})")

    def get_stats(self) -> dict:
        """Return current statistics and parameters."""
        wins = sum(1 for g in self.game_results if g['result'] == 'win')
        losses = sum(1 for g in self.game_results if g['result'] == 'loss')
        draws = sum(1 for g in self.game_results if g['result'] == 'draw')

        stats = {
            'games': len(self.game_results),
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / len(self.game_results) if self.game_results else 0,
            'params': asdict(self.params),
            'edge_green_bonus': self.params.edge_green_bonus,
            'edge_purple_bonus': self.params.edge_purple_bonus,
        }

        # Add compute effort diagnostics if MATLAB is available
        compute_effort = self.get_compute_effort()
        if compute_effort:
            stats['compute_effort'] = compute_effort

        return stats

    def get_compute_effort(self) -> Optional[dict]:
        """Get compute effort diagnostics from MATLAB."""
        if not self.mcts_initialized or self.matlab_engine is None:
            return None

        try:
            self.matlab_engine.eval('effort_out = mcts_engine.getComputeEffort();', nargout=0)
            effort = self.matlab_engine.workspace['effort_out']

            return {
                'last_cpu_time': float(effort.get('cpuTime', 0)),
                'last_wall_time': float(effort.get('wallTime', 0)),
                'cpu_efficiency': float(effort.get('cpuEfficiency', 0)),
                'nodes_expanded': int(effort.get('nodesExpanded', 0)),
                'simulations': int(effort.get('simulations', 0)),
                'tree_depth': int(effort.get('treeDepth', 0)),
                'memory_mb': float(effort.get('memoryMB', 0)),
                'iterations_per_cpu_sec': float(effort.get('iterationsPerCPUSec', 0)),
                'session_total_cpu': float(effort.get('sessionTotalCPU', 0)),
                'session_total_iterations': int(effort.get('sessionTotalIterations', 0)),
            }
        except Exception as e:
            logger.debug(f"Could not get compute effort: {e}")
            return None

    def cleanup(self):
        """Clean up MATLAB engine."""
        if self.matlab_engine is not None:
            try:
                self.matlab_engine.quit()
            except:
                pass
            self.matlab_engine = None
