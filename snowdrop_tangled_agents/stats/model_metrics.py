"""
Opponent model metrics for tracking learning progress.

Computes metrics that measure:
- How certain the model is (entropy)
- How accurate predictions are (top-3 hit rate, probability assigned to actual moves)

These metrics are stored per-game to enable learning trajectory analysis.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution.

    Args:
        probs: Probability array (should sum to 1)

    Returns:
        Entropy in bits. Higher = more uncertain.
        Max entropy for 30 actions = log2(30) ≈ 4.91 bits
    """
    # Filter out zeros to avoid log(0)
    probs = probs[probs > 0]
    if len(probs) == 0:
        return 0.0
    return -np.sum(probs * np.log2(probs))


def compute_model_entropy(model) -> float:
    """Compute average prediction entropy across all contexts with data.

    This measures how certain/uncertain the model is overall.
    Lower entropy = more confident predictions.

    Args:
        model: OpponentModel instance

    Returns:
        Average entropy across response contexts with sufficient data
    """
    entropies = []

    for our_move_idx in range(model.NUM_MOVES):
        total = model.response_totals[our_move_idx]
        if total >= 5:  # Only include contexts with meaningful data
            # Compute smoothed probabilities
            probs = (
                (model.response_counts[our_move_idx] + model.smoothing) /
                (total + model.smoothing * model.NUM_MOVES)
            )
            entropies.append(compute_entropy(probs))

    if not entropies:
        # No data - return max entropy (uniform distribution)
        return np.log2(model.NUM_MOVES)

    return np.mean(entropies)


def compute_prediction_metrics(
    model,
    our_move: tuple[int, str],
    actual_opp_move: tuple[int, str],
    grey_count: int
) -> dict:
    """Compute prediction accuracy metrics for a single opponent move.

    Args:
        model: OpponentModel instance
        our_move: (edge, color) of our previous move
        actual_opp_move: (edge, color) that opponent actually played
        grey_count: Grey edges before opponent's move

    Returns:
        Dict with:
        - predicted_prob: Probability we assigned to the actual move
        - top3_hit: 1 if actual move was in our top 3 predictions, 0 otherwise
        - rank: Rank of actual move in our prediction (1 = most likely)
    """
    from .opponent_model import move_to_index

    # Get all possible moves (all 30, we'll filter later)
    all_moves = [(e, c) for e in range(15) for c in ['G', 'P']]

    # Get predictions
    predictions = model.predict_response(
        our_last_move=our_move,
        available_moves=all_moves,
        grey_count=grey_count
    )

    # Sort by probability
    sorted_preds = sorted(predictions.items(), key=lambda x: -x[1])
    top3_moves = [m for m, p in sorted_preds[:3]]

    # Find actual move's rank and probability
    actual_idx = move_to_index(actual_opp_move[0], actual_opp_move[1])
    predicted_prob = predictions.get(actual_opp_move, 0.0)

    rank = None
    for i, (move, prob) in enumerate(sorted_preds):
        if move == actual_opp_move:
            rank = i + 1
            break

    return {
        'predicted_prob': predicted_prob,
        'top3_hit': 1 if actual_opp_move in top3_moves else 0,
        'rank': rank,
    }


class GameMetricsTracker:
    """Tracks opponent model metrics during a game.

    Use this to accumulate per-move metrics and compute game-level summaries.

    Usage:
        tracker = GameMetricsTracker(opponent_model)
        tracker.record_snapshot()  # At game start

        # After each opponent move:
        tracker.record_prediction(our_move, opp_move, grey_count)

        # At game end:
        metrics = tracker.get_game_metrics()
    """

    def __init__(self, model):
        """Initialize tracker with opponent model.

        Args:
            model: OpponentModel instance (or None if not available)
        """
        self.model = model
        self.start_entropy = None
        self.start_games_learned = None
        self.start_moves_learned = None
        self.prediction_results = []

    def record_snapshot(self):
        """Record model state at game start."""
        if self.model is None:
            return

        self.start_entropy = compute_model_entropy(self.model)
        self.start_games_learned = self.model.total_games
        self.start_moves_learned = self.model.total_moves

        logger.debug(
            f"Model snapshot: entropy={self.start_entropy:.3f}, "
            f"games={self.start_games_learned}, moves={self.start_moves_learned}"
        )

    def record_prediction(
        self,
        our_move: tuple[int, str],
        actual_opp_move: tuple[int, str],
        grey_count: int
    ):
        """Record prediction vs actual for one opponent move.

        Args:
            our_move: Our previous move (edge, color)
            actual_opp_move: Opponent's actual move (edge, color)
            grey_count: Grey edges before opponent moved
        """
        if self.model is None:
            return

        metrics = compute_prediction_metrics(
            self.model, our_move, actual_opp_move, grey_count
        )
        self.prediction_results.append(metrics)

        logger.debug(
            f"Prediction: our={our_move}, opp={actual_opp_move}, "
            f"prob={metrics['predicted_prob']:.3f}, rank={metrics['rank']}, "
            f"top3={'hit' if metrics['top3_hit'] else 'miss'}"
        )

    def get_game_metrics(self) -> dict:
        """Get aggregated metrics for the game.

        Returns:
            Dict with:
            - model_entropy: Entropy at game start
            - model_top3_hit: Fraction of opponent moves in our top 3
            - prediction_accuracy: Average probability assigned to actual moves
            - model_games_learned: Games in model at start
            - model_moves_learned: Moves in model at start
        """
        metrics = {
            'model_entropy': self.start_entropy,
            'model_games_learned': self.start_games_learned,
            'model_moves_learned': self.start_moves_learned,
            'model_top3_hit': None,
            'prediction_accuracy': None,
        }

        if self.prediction_results:
            metrics['model_top3_hit'] = np.mean([
                r['top3_hit'] for r in self.prediction_results
            ])
            metrics['prediction_accuracy'] = np.mean([
                r['predicted_prob'] for r in self.prediction_results
            ])

        return metrics
