"""
Opponent modeling for Tangled game.

Builds probabilistic models of opponent behavior to improve MCTS rollouts.
Uses Bayesian updating with Dirichlet prior for smoothing.

See docs/OPPONENT_MODELING.md for detailed design.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import scipy.io
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .collector import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)


def grey_bucket(grey_count: int) -> str:
    """Convert grey count to game phase bucket.

    Phases:
    - early: 12-15 grey edges (opening moves)
    - mid: 8-11 grey edges (mid-game)
    - late: 4-7 grey edges (late game)
    - endgame: 0-3 grey edges (final moves)
    """
    if grey_count >= 12:
        return 'early'
    elif grey_count >= 8:
        return 'mid'
    elif grey_count >= 4:
        return 'late'
    else:
        return 'endgame'


def move_to_index(edge: int, color: str) -> int:
    """Convert (edge, color) to unique index 0-29.

    Index = edge * 2 + (0 if Green, 1 if Purple)
    """
    color_idx = 0 if color.upper() == 'G' else 1
    return edge * 2 + color_idx


def index_to_move(idx: int) -> tuple[int, str]:
    """Convert index 0-29 to (edge, color)."""
    edge = idx // 2
    color = 'G' if idx % 2 == 0 else 'P'
    return (edge, color)


class OpponentModel:
    """
    Probabilistic model of opponent behavior.

    Uses Bayesian updating with Dirichlet prior for smoothing.
    Tracks two types of patterns:
    1. Response patterns: P(opp_move | our_last_move)
    2. Phase patterns: P(opp_move | game_phase)

    Args:
        opponent_name: Name of the opponent (e.g., "melissa")
        smoothing: Laplace smoothing parameter (default 0.1)
                   Lower values give more weight to observed data.
                   With smoothing=0.1 and 30 actions, adds 3 pseudo-counts.
                   For context with N samples, data weight = N/(N+3).
    """

    NUM_MOVES = 30  # 15 edges * 2 colors
    PHASES = ['early', 'mid', 'late', 'endgame']

    def __init__(self, opponent_name: str, smoothing: float = 0.1):
        self.opponent_name = opponent_name
        self.smoothing = smoothing

        # Response counts: response_counts[our_move_idx][opp_move_idx] = count
        self.response_counts = np.zeros((self.NUM_MOVES, self.NUM_MOVES), dtype=np.float64)

        # Phase counts: phase_counts[phase_idx][opp_move_idx] = count
        self.phase_counts = np.zeros((len(self.PHASES), self.NUM_MOVES), dtype=np.float64)

        # Total observations per context
        self.response_totals = np.zeros(self.NUM_MOVES, dtype=np.float64)
        self.phase_totals = np.zeros(len(self.PHASES), dtype=np.float64)

        # Metadata
        self.total_games = 0
        self.total_moves = 0

    def _phase_idx(self, phase: str) -> int:
        """Convert phase name to index."""
        return self.PHASES.index(phase)

    def update(self, our_move: tuple[int, str], opp_move: tuple[int, str], grey_count: int, weight: float = 1.0):
        """Update model with observed opponent response.

        Args:
            our_move: (edge, color) of our last move
            opp_move: (edge, color) of opponent's response
            grey_count: Number of grey edges BEFORE opponent's move
            weight: Weight for this observation (0.0-1.0), used for policy decay
        """
        our_idx = move_to_index(our_move[0], our_move[1])
        opp_idx = move_to_index(opp_move[0], opp_move[1])
        phase = grey_bucket(grey_count)
        phase_idx = self._phase_idx(phase)

        # Update response counts (weighted)
        self.response_counts[our_idx, opp_idx] += weight
        self.response_totals[our_idx] += weight

        # Update phase counts (weighted)
        self.phase_counts[phase_idx, opp_idx] += weight
        self.phase_totals[phase_idx] += weight

        self.total_moves += 1

    def predict_response(
        self,
        our_last_move: tuple[int, str],
        available_moves: list[tuple[int, str]],
        grey_count: int,
        alpha: float = 0.7
    ) -> dict[tuple[int, str], float]:
        """Predict probability distribution over opponent's next move.

        Uses weighted combination of response-conditional and phase-conditional
        probabilities:

            P(opp_move) = alpha * P(opp_move | our_last_move)
                        + (1-alpha) * P(opp_move | phase)

        Args:
            our_last_move: (edge, color) of our last move
            available_moves: List of valid (edge, color) tuples
            grey_count: Number of grey edges before opponent's move
            alpha: Weight for response-conditional probability (0-1)
                   Higher = trust response patterns more

        Returns:
            Dictionary mapping (edge, color) to probability
        """
        if not available_moves:
            return {}

        our_idx = move_to_index(our_last_move[0], our_last_move[1])
        phase = grey_bucket(grey_count)
        phase_idx = self._phase_idx(phase)

        # Compute response-conditional probabilities with smoothing
        response_total = self.response_totals[our_idx]
        if response_total > 0:
            response_probs = (
                (self.response_counts[our_idx] + self.smoothing) /
                (response_total + self.smoothing * self.NUM_MOVES)
            )
            # Confidence based on sample size (saturates at 20 samples)
            response_confidence = min(response_total / 20.0, 1.0)
        else:
            # No data for this context - use uniform
            response_probs = np.ones(self.NUM_MOVES) / self.NUM_MOVES
            response_confidence = 0.0

        # Compute phase-conditional probabilities with smoothing
        phase_total = self.phase_totals[phase_idx]
        if phase_total > 0:
            phase_probs = (
                (self.phase_counts[phase_idx] + self.smoothing) /
                (phase_total + self.smoothing * self.NUM_MOVES)
            )
        else:
            # No data for this phase - use uniform
            phase_probs = np.ones(self.NUM_MOVES) / self.NUM_MOVES

        # Adjust alpha based on confidence in response data
        # When confidence is high, use mostly response-conditional (up to 95%)
        # When confidence is low, blend more with phase-conditional
        # Formula: effective_alpha = alpha + (0.95 - alpha) * confidence
        # At confidence=0: effective_alpha = alpha (use phase as fallback)
        # At confidence=1: effective_alpha = 0.95 (mostly response-conditional)
        effective_alpha = alpha + (0.95 - alpha) * response_confidence

        # Combine probabilities
        combined_probs = (
            effective_alpha * response_probs +
            (1 - effective_alpha) * phase_probs
        )

        # Filter to available moves and renormalize
        result = {}
        total_prob = 0.0
        for move in available_moves:
            idx = move_to_index(move[0], move[1])
            prob = combined_probs[idx]
            result[move] = prob
            total_prob += prob

        # Normalize
        if total_prob > 0:
            for move in result:
                result[move] /= total_prob
        else:
            # Uniform fallback
            uniform_prob = 1.0 / len(available_moves)
            for move in result:
                result[move] = uniform_prob

        return result

    def load_from_database(self, db_path: Optional[Path] = None, policy_weights: Optional[dict] = None):
        """Initialize model from historical game data.

        Reads opponent moves from the moves table and rebuilds the model.
        Applies policy-based weighting to decay games from older/different policies.

        Args:
            db_path: Path to database (default: ~/.tangled/game_stats.db)
            policy_weights: Dict mapping policy_id to weight (0.0-1.0).
                           Default weights if not specified:
                           - Current policy (from git): 1.0
                           - One version back: 0.5
                           - Older/unknown: 0.25
        """
        if db_path is None:
            db_path = DEFAULT_DB_PATH

        # Get current policy for default weighting
        try:
            from snowdrop_tangled_agents.utils.version import get_policy_id
            current_policy = get_policy_id()
        except Exception:
            current_policy = None

        # Default policy weights: current=1.0, others decayed
        if policy_weights is None:
            policy_weights = {
                current_policy: 1.0,
                'v0.6.0-bayesian-oracle': 0.8,  # Recent with opponent modeling
                'pre-opponent-modeling': 0.3,   # Different play style
                'legacy': 0.1,                  # Very old
                'abandoned': 0.0,               # Incomplete games
                None: 0.25,                     # Unknown
            }

        logger.info(f"Loading opponent model from {db_path}")

        from .collector import connect_db
        conn = connect_db(db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Get games against this opponent with policy info
            games = conn.execute('''
                SELECT id, policy_id FROM games
                WHERE LOWER(opponent) = LOWER(?)
            ''', (self.opponent_name,)).fetchall()

            self.total_games = len(games)
            logger.info(f"Found {self.total_games} games against {self.opponent_name}")

            # Reset counts
            self.response_counts.fill(0)
            self.phase_counts.fill(0)
            self.response_totals.fill(0)
            self.phase_totals.fill(0)
            self.total_moves = 0

            # Track weighted vs unweighted for logging
            weighted_moves = 0.0
            skipped_games = 0

            # Process each game with policy-based weighting
            for game in games:
                game_id = game['id']
                policy_id = game['policy_id']

                # Get weight for this policy
                weight = policy_weights.get(policy_id, policy_weights.get(None, 0.25))

                # Skip games with zero weight
                if weight <= 0:
                    skipped_games += 1
                    continue

                # Get all moves in this game ordered by move_number
                moves = conn.execute('''
                    SELECT move_number, player, edge, color, state_after
                    FROM moves
                    WHERE game_id = ?
                    ORDER BY move_number, player
                ''', (game_id,)).fetchall()

                # Track our last move to correlate with opponent response
                our_last_move = None
                last_grey_count = 15  # Start of game

                for move in moves:
                    player = move['player']
                    edge = move['edge']
                    color = move['color']
                    state_after = move['state_after'] or ''

                    if player == 'us':
                        our_last_move = (edge, color)
                        # Update grey count from state
                        if state_after:
                            last_grey_count = state_after.count('-')
                    elif player == 'opponent':
                        if our_last_move is not None:
                            # Grey count BEFORE opponent's move is after our move
                            grey_before_opp = last_grey_count
                            self.update(our_last_move, (edge, color), grey_before_opp, weight=weight)
                            weighted_moves += weight

                        # Update grey count from state
                        if state_after:
                            last_grey_count = state_after.count('-')

            logger.info(f"Loaded {self.total_moves} opponent moves (weighted: {weighted_moves:.1f}, skipped {skipped_games} games)")

        finally:
            conn.close()

    def save_mat(self, path: Optional[Path] = None):
        """Save model to .mat file for MATLAB consumption.

        Saves:
        - response_probs: 30x30 matrix of P(opp_move | our_move)
        - phase_probs: 4x30 matrix of P(opp_move | phase)
        - metadata: opponent name, total games, etc.

        Args:
            path: Output path (default: matlab/rl/data/opponent_model.mat)
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, cannot save .mat file")
            return

        if path is None:
            path = Path(__file__).parent.parent / 'matlab' / 'rl' / 'data' / 'opponent_model.mat'

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Compute probability matrices with smoothing
        response_probs = np.zeros((self.NUM_MOVES, self.NUM_MOVES))
        for i in range(self.NUM_MOVES):
            total = self.response_totals[i]
            if total > 0:
                response_probs[i] = (
                    (self.response_counts[i] + self.smoothing) /
                    (total + self.smoothing * self.NUM_MOVES)
                )
            else:
                response_probs[i] = 1.0 / self.NUM_MOVES

        phase_probs = np.zeros((len(self.PHASES), self.NUM_MOVES))
        for i in range(len(self.PHASES)):
            total = self.phase_totals[i]
            if total > 0:
                phase_probs[i] = (
                    (self.phase_counts[i] + self.smoothing) /
                    (total + self.smoothing * self.NUM_MOVES)
                )
            else:
                phase_probs[i] = 1.0 / self.NUM_MOVES

        # Save to .mat
        mdict = {
            'response_probs': response_probs,
            'phase_probs': phase_probs,
            'response_counts': self.response_counts,
            'phase_counts': self.phase_counts,
            'response_totals': self.response_totals,
            'phase_totals': self.phase_totals,
            'opponent_name': self.opponent_name,
            'total_games': self.total_games,
            'total_moves': self.total_moves,
            'smoothing': self.smoothing,
            'phases': np.array(self.PHASES, dtype=object),
        }

        scipy.io.savemat(str(path), mdict)
        logger.info(f"Saved opponent model to {path}")

    def save_json(self, path: Optional[Path] = None):
        """Save model to JSON file.

        Args:
            path: Output path (default: matlab/rl/data/opponent_model.json)
        """
        if path is None:
            path = Path(__file__).parent.parent / 'matlab' / 'rl' / 'data' / 'opponent_model.json'

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'opponent_name': self.opponent_name,
            'total_games': self.total_games,
            'total_moves': self.total_moves,
            'smoothing': self.smoothing,
            'response_counts': self.response_counts.tolist(),
            'phase_counts': self.phase_counts.tolist(),
            'response_totals': self.response_totals.tolist(),
            'phase_totals': self.phase_totals.tolist(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved opponent model to {path}")

    def get_stats(self) -> dict:
        """Get model statistics for display."""
        return {
            'opponent_name': self.opponent_name,
            'total_games': self.total_games,
            'total_moves': self.total_moves,
            'smoothing': self.smoothing,
            'response_contexts_with_data': int(np.sum(self.response_totals > 0)),
            'phase_contexts_with_data': int(np.sum(self.phase_totals > 0)),
        }

    def print_summary(self):
        """Print model summary to console."""
        stats = self.get_stats()

        print(f"\n=== Opponent Model: {stats['opponent_name']} ===")
        print(f"Games: {stats['total_games']}")
        print(f"Moves: {stats['total_moves']}")
        print(f"Smoothing: {stats['smoothing']}")
        print(f"Response contexts with data: {stats['response_contexts_with_data']}/30")
        print(f"Phase contexts with data: {stats['phase_contexts_with_data']}/4")

        # Show top responses per phase
        print("\nTop moves by phase:")
        for phase_idx, phase in enumerate(self.PHASES):
            total = self.phase_totals[phase_idx]
            if total > 0:
                counts = self.phase_counts[phase_idx]
                top_idx = np.argsort(counts)[::-1][:3]
                top_moves = []
                for idx in top_idx:
                    if counts[idx] > 0:
                        edge, color = index_to_move(idx)
                        top_moves.append(f"E{edge}{color}({int(counts[idx])})")
                print(f"  {phase:8s}: {', '.join(top_moves)}")
            else:
                print(f"  {phase:8s}: (no data)")


def get_opponent_model(opponent_name: str = "melissa", db_path: Optional[Path] = None) -> OpponentModel:
    """Get opponent model, loading from database.

    Args:
        opponent_name: Name of opponent to model
        db_path: Path to database

    Returns:
        Loaded OpponentModel instance
    """
    model = OpponentModel(opponent_name)
    model.load_from_database(db_path)
    return model
