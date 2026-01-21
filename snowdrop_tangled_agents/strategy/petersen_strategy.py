"""
Petersen Graph Strategy Engine for Tangled Game.

This module implements a parameterized strategy calculator for the Petersen graph,
with support for learning from game outcomes and parameter persistence.
"""

import json
import os
from pathlib import Path
from typing import Optional

# Petersen graph edge list (10 vertices, 15 edges)
# Inner vertices form a pentagram (star pattern), outer form a pentagon
PETERSEN_EDGES = [
    (0, 2), (0, 3), (0, 6), (1, 3), (1, 4),
    (1, 7), (2, 4), (2, 8), (3, 9), (4, 5),
    (5, 6), (5, 9), (6, 7), (7, 8), (8, 9),
]

# Default parameters tuned from gameplay vs Amara, Randy, and Melissa
DEFAULT_PARAMS = {
    # Priority weights (higher = do first)
    "w_my_edge": 10.0,
    "w_opp_edge": 8.0,
    "w_hub_edge": 5.0,
    "w_neutral": 1.0,

    # Hub preference (0.0 = ignore hub, 1.0 = always prefer hub)
    "hub_priority": 0.8,

    # Color decision thresholds
    "green_threshold": 1.0,
    "purple_threshold": -1.0,

    # Adaptive factors
    "momentum_weight": 0.5,
    "opp_pattern_weight": 0.3,

    # Edge-specific value adjustments (learned from score swings)
    # Edge list (pentagram inner + pentagon outer):
    # E0=(0,2), E1=(0,3), E2=(0,6)hub, E3=(1,3), E4=(1,4),
    # E5=(1,7)opp, E6=(2,4), E7=(2,8), E8=(3,9), E9=(4,5)MY,
    # E10=(5,6)MY+hub, E11=(5,9)MY, E12=(6,7)opp+hub, E13=(7,8)opp, E14=(8,9)
    "edge_values": [
        0.0,   # E0  (0-2)  inner pentagram
        0.0,   # E1  (0-3)  inner pentagram
        0.6,   # E2  (0-6)  hub edge (spoke)
        0.0,   # E3  (1-3)  inner pentagram
        0.0,   # E4  (1-4)  inner pentagram
        0.9,   # E5  (1-7)  opponent spoke - high priority attack
        0.0,   # E6  (2-4)  inner pentagram
        0.0,   # E7  (2-8)  spoke
        0.0,   # E8  (3-9)  spoke
        1.1,   # E9  (4-5)  MY edge - secure early
        1.0,   # E10 (5-6)  MY + hub edge
        1.1,   # E11 (5-9)  MY edge - secure early
        0.9,   # E12 (6-7)  opponent hub
        0.7,   # E13 (7-8)  opponent edge
        0.0,   # E14 (8-9)  outer pentagon
    ],

    # Strategy mode: "defensive", "aggressive", "adaptive"
    "strategy_mode": "adaptive",

    # Opening sequence override: [(edge, color), ...] for first N moves
    # Secure MY edges first (E9, E10, E11)
    "opening_sequence": [(9, 'G'), (10, 'G'), (11, 'G')],
}

# Vertex assignments for Petersen graph
MY_VERTEX = 5      # Player 1 (Red)
OPP_VERTEX = 7     # Player 2 (Blue)
HUB_VERTEX = 6     # Critical shared vertex


def edges_of_vertex(v: int) -> list[int]:
    """Return list of edge indices touching vertex v."""
    return [i for i, (a, b) in enumerate(PETERSEN_EDGES) if a == v or b == v]


# Pre-computed edge classifications
MY_EDGES = edges_of_vertex(MY_VERTEX)      # [9, 10, 11]
OPP_EDGES = edges_of_vertex(OPP_VERTEX)    # [5, 12, 13]
HUB_EDGES = edges_of_vertex(HUB_VERTEX)    # [2, 10, 12]


class PetersenStrategy:
    """
    Parameterized strategy calculator for Petersen graph Tangled games.

    Provides move calculation based on edge priorities, score thresholds,
    and adaptive factors. Supports learning from game outcomes.
    """

    def __init__(self, params: Optional[dict] = None, params_path: Optional[str] = None):
        """
        Initialize strategy with parameters.

        Args:
            params: Parameter dict (uses DEFAULT_PARAMS if None)
            params_path: Path to load/save parameters (optional)
        """
        self.params_path = params_path

        if params is not None:
            self.params = params.copy()
        elif params_path and os.path.exists(params_path):
            self.load_params(params_path)
        else:
            self.params = DEFAULT_PARAMS.copy()
            self.params["edge_values"] = DEFAULT_PARAMS["edge_values"].copy()

    def calculate_move(
        self,
        state: str,
        score: float,
        score_history: list[tuple[int, str, float]]
    ) -> Optional[tuple[int, str]]:
        """
        Calculate optimal move given game state.

        Args:
            state: 15-char string, 'G'/'P'/'-'/'?' for each edge
                   '-' = available, 'G' = green, 'P' = purple, '?' = unknown
            score: Current game score (positive = we're ahead)
            score_history: List of (edge_index, color, score_after) tuples

        Returns:
            (edge_index, color) or None if no moves available
        """
        grey_indices = [i for i, c in enumerate(state) if c == '-']
        if not grey_indices:
            return None

        # Check for opening sequence override
        move_num = 15 - len(grey_indices)
        opening = self.params.get("opening_sequence")
        if opening and move_num < len(opening):
            forced_edge, forced_color = opening[move_num]
            if state[forced_edge] == '-':
                return (forced_edge, forced_color)

        # Score all available edges
        edge_scores = []
        for idx in grey_indices:
            s = self._score_edge(idx, state, score, score_history)
            edge_scores.append((s, idx))

        # Pick highest scoring edge
        edge_scores.sort(reverse=True)
        best_edge = edge_scores[0][1]

        # Choose color
        color = self._choose_color(best_edge, score)

        return (best_edge, color)

    def _score_edge(
        self,
        idx: int,
        state: str,
        score: float,
        score_history: list[tuple[int, str, float]]
    ) -> float:
        """Score an edge for prioritization. Higher = play first."""
        if state[idx] != '-':
            return -999.0

        base = self.params["edge_values"][idx]

        # Category bonuses
        if idx in MY_EDGES:
            base += self.params["w_my_edge"]
        elif idx in OPP_EDGES:
            base += self.params["w_opp_edge"]
        elif idx in HUB_EDGES:
            base += self.params["w_hub_edge"]
        else:
            base += self.params["w_neutral"]

        # Hub preference within category
        if idx in HUB_EDGES:
            base += self.params["hub_priority"]

        # Momentum adjustment
        momentum = self._compute_momentum(score_history)
        base += momentum * self.params["momentum_weight"]

        # Opponent pattern adjustment
        opp_prefs = self._opponent_edge_preference(score_history)
        if idx in opp_prefs:
            base += opp_prefs[idx] * self.params["opp_pattern_weight"]

        return base

    def _choose_color(self, idx: int, score: float) -> str:
        """Decide Green or Purple for the chosen edge."""
        mode = self.params["strategy_mode"]

        # Fixed rules: my edges = Green, opponent edges = Purple
        if idx in MY_EDGES:
            return 'G'
        if idx in OPP_EDGES:
            return 'P'

        # Neutral edges: depends on mode and score
        if mode == "defensive":
            return 'G'
        elif mode == "aggressive":
            return 'P'
        else:  # adaptive
            if score > self.params["green_threshold"]:
                return 'G'
            elif score < self.params["purple_threshold"]:
                return 'P'
            else:
                return 'G'

    def _compute_momentum(
        self,
        score_history: list[tuple[int, str, float]],
        window: int = 4
    ) -> float:
        """Compute recent score trend. Positive = improving."""
        if len(score_history) < 2:
            return 0.0
        recent = score_history[-window:] if len(score_history) >= window else score_history
        scores = [s for _, _, s in recent]
        if len(scores) < 2:
            return 0.0
        return (scores[-1] - scores[0]) / len(scores)

    def _opponent_edge_preference(
        self,
        score_history: list[tuple[int, str, float]]
    ) -> dict[int, float]:
        """Analyze which edges opponent values (big negative swings after their moves)."""
        opp_valued: dict[int, float] = {}
        for i in range(1, len(score_history)):
            prev_score = score_history[i - 1][2]
            curr_edge, _, curr_score = score_history[i]
            delta = curr_score - prev_score
            if delta < -0.3:
                opp_valued[curr_edge] = opp_valued.get(curr_edge, 0) + abs(delta)
        return opp_valued

    def update_from_trial(
        self,
        result: str,
        score_history: list[tuple[int, str, float]],
        final_score: float
    ) -> None:
        """
        Update parameters based on trial outcome using REINFORCE-style learning.

        Args:
            result: 'win', 'loss', or 'draw'
            score_history: Full game history [(edge, color, score_after), ...]
            final_score: Final game score
        """
        if not score_history:
            return

        # Learning rates
        base_lr = 0.05
        outcome_bonus = {"win": 1.0, "draw": 0.0, "loss": -1.0}.get(result, 0.0)

        # Track statistics for this game
        if "game_stats" not in self.params:
            self.params["game_stats"] = {"wins": 0, "losses": 0, "draws": 0, "games": 0}

        self.params["game_stats"]["games"] += 1
        if result == "win":
            self.params["game_stats"]["wins"] += 1
        elif result == "loss":
            self.params["game_stats"]["losses"] += 1
        else:
            self.params["game_stats"]["draws"] += 1

        # Compute discounted returns for each move (REINFORCE-style)
        # Return = final_outcome + sum of future score improvements
        gamma = 0.95  # Discount factor
        returns = []
        G = outcome_bonus * 2.0  # Terminal reward based on outcome

        # Work backwards to compute returns
        for i in range(len(score_history) - 1, -1, -1):
            edge, color, score_after = score_history[i]

            # Immediate reward = score improvement from this move
            if i > 0:
                score_before = score_history[i - 1][2]
            else:
                score_before = 0.0
            immediate_reward = score_after - score_before

            G = immediate_reward + gamma * G
            returns.insert(0, G)

        # Normalize returns
        if len(returns) > 1:
            mean_return = sum(returns) / len(returns)
            std_return = max(0.01, (sum((r - mean_return) ** 2 for r in returns) / len(returns)) ** 0.5)
            returns = [(r - mean_return) / std_return for r in returns]

        # Update edge values based on returns
        for i, (edge, color, _) in enumerate(score_history):
            if color not in ('G', 'P'):
                continue

            advantage = returns[i] if i < len(returns) else 0.0

            # Update edge priority
            update = base_lr * advantage
            self.params["edge_values"][edge] = max(
                0.0, min(2.0, self.params["edge_values"][edge] + update)
            )

        # Learn color preferences per edge category
        if result == "win":
            # Reinforce color choices that led to wins
            for edge, color, _ in score_history:
                if edge in MY_EDGES and color == 'G':
                    # Good: Green on our edges
                    pass
                elif edge in OPP_EDGES and color == 'P':
                    # Good: Purple on opponent edges
                    pass
        elif result == "loss":
            # Analyze what went wrong
            # If we lost with negative score, we may need to be more aggressive
            if final_score < -1.0:
                self.params["w_opp_edge"] = min(15.0, self.params["w_opp_edge"] + 0.5)
            # If we lost with positive score (opponent outplayed us tactically)
            elif final_score > 0:
                self.params["w_my_edge"] = min(15.0, self.params["w_my_edge"] + 0.5)

        # Adjust opening sequence based on early game performance
        if len(score_history) >= 3:
            early_score = score_history[2][2]  # Score after 3rd move
            if result == "loss" and early_score < -0.5:
                # Our opening may be weak, consider shuffling
                opening = self.params.get("opening_sequence", [])
                if opening and len(opening) >= 2:
                    # Try swapping first two moves next time
                    self.params["opening_sequence"] = [opening[1], opening[0]] + opening[2:]

        # Clamp all edge values
        self.params["edge_values"] = [
            max(0.0, min(2.0, v)) for v in self.params["edge_values"]
        ]

        # Auto-save if path configured
        if self.params_path:
            self.save_params(self.params_path)

    def save_params(self, path: Optional[str] = None) -> None:
        """Save parameters to JSON file."""
        save_path = path or self.params_path
        if not save_path:
            return

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.params, f, indent=2)

    def load_params(self, path: str) -> None:
        """Load parameters from JSON file."""
        with open(path, 'r') as f:
            loaded = json.load(f)
        self.params = DEFAULT_PARAMS.copy()
        self.params.update(loaded)
        self.params_path = path

    def get_params(self) -> dict:
        """Return current parameters."""
        return self.params.copy()

    def set_opening_sequence(self, sequence: Optional[list[tuple[int, str]]]) -> None:
        """Set or clear opening sequence override."""
        self.params["opening_sequence"] = sequence
