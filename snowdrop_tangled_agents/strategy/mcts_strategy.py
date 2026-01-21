"""
Monte Carlo Tree Search Strategy for Tangled Game.

Implements MCTS with UCB1 selection and Progressive Bias to compete against
MCTS-based opponents like Melissa. Uses the official SimulatedAnnealingAdjudicator
for terminal state evaluation to match tangled-game.com scores.

Key improvements over basic MCTS:
- Progressive Bias: adds heuristic prior to guide early exploration
- Action prioritization: expands good moves first
- Domain-specific rollout policy: uses Tangled heuristics
"""

import math
import random
import time
from typing import Optional
from dataclasses import dataclass, field
from functools import lru_cache

from snowdrop_adjudicators import SimulatedAnnealingAdjudicator
from snowdrop_tangled_game_engine.game import Edge

# Petersen graph structure (must match petersen_strategy.py)
PETERSEN_EDGES = [
    (0, 2), (0, 3), (0, 6), (1, 3), (1, 4),
    (1, 7), (2, 4), (2, 8), (3, 9), (4, 5),
    (5, 6), (5, 9), (6, 7), (7, 8), (8, 9),
]

NUM_VERTICES = 10
NUM_EDGES = 15

# Player vertices
MY_VERTEX = 5      # Player 1 (Red)
OPP_VERTEX = 7     # Player 2 (Blue)
HUB_VERTEX = 6     # Hub (strategically important)

# Edge classifications
MY_EDGES = [9, 10, 11]      # Touch vertex 5
OPP_EDGES = [5, 12, 13]     # Touch vertex 7
HUB_EDGES = [2, 10, 12]     # Touch vertex 6


def compute_action_prior(edge: int, color: str, is_our_turn: bool) -> float:
    """
    Compute a heuristic prior for an action.

    Higher values indicate actions we believe are better a priori.
    This guides MCTS to explore good moves first.

    Returns value in [0, 1] range.

    Key insights from game analysis:
    - E0 Purple is often a strong mid-game move
    - E4 Purple tends to backfire (avoid it)
    - Inner edges (0,1,3) with Purple often work well
    """
    prior = 0.5  # Base prior

    # Edges that empirically work well/poorly based on 40+ game analysis
    GOOD_PURPLE_EDGES = [0, 1, 3, 5, 12, 13]  # Inner edges and opponent edges
    BAD_PURPLE_EDGES = [2, 4, 6, 7, 8, 14]  # These often backfire (E2, E7 cause big swings)

    if is_our_turn:
        # OUR TURN - stronger priors based on empirical analysis
        if edge in MY_EDGES:
            prior = 0.99 if color == 'G' else 0.01  # Always Green on our edges
        elif edge in OPP_EDGES:
            prior = 0.95 if color == 'P' else 0.05  # Purple on opponent edges
        elif edge in GOOD_PURPLE_EDGES and color == 'P':
            prior = 0.80  # Favor Purple on empirically good edges
        elif edge in BAD_PURPLE_EDGES and color == 'P':
            prior = 0.10  # Strongly avoid Purple on bad edges
        elif edge in BAD_PURPLE_EDGES and color == 'G':
            prior = 0.90  # Prefer Green on historically bad-for-purple edges
        elif edge in HUB_EDGES:
            prior = 0.70 if color == 'G' else 0.30
        else:
            prior = 0.60 if color == 'G' else 0.40
    else:
        # OPPONENT'S TURN: They prefer Green on their edges, Purple on ours
        if edge in OPP_EDGES:
            prior = 0.95 if color == 'G' else 0.05
        elif edge in MY_EDGES:
            prior = 0.85 if color == 'P' else 0.15
        elif edge in HUB_EDGES:
            prior = 0.65 if color == 'G' else 0.35
        else:
            prior = 0.55 if color == 'G' else 0.45

    return prior


@lru_cache(maxsize=1024)
def evaluate_terminal_state(state: str) -> float:
    """
    Evaluate a terminal state (all edges colored) using the official adjudicator.

    Uses SimulatedAnnealingAdjudicator to match tangled-game.com scoring exactly.
    Results are cached to avoid repeated expensive adjudication calls.

    Args:
        state: 15-char string, all 'G' or 'P' (no grey edges)

    Returns:
        Score from Player 1's perspective (positive = P1 wins)
        Typically in range [-5, +5] based on website observations.
    """
    if state.count('-') > 0:
        raise ValueError("Cannot evaluate non-terminal state")

    # Build edge list with colors for the adjudicator
    edges = []
    for i, (v1, v2) in enumerate(PETERSEN_EDGES):
        color = state[i]
        edge_state = Edge.State.FM.value if color == 'G' else Edge.State.AFM.value
        edges.append((v1, v2, edge_state))

    # Create game state dict for adjudicator
    game_state = {
        'num_nodes': NUM_VERTICES,
        'edges': edges,
        'graph_id': 11,  # Petersen graph
        'player1_id': 'p1',
        'player2_id': 'p2',
        'turn_count': NUM_EDGES,
        'current_player_index': 2,
        'player1_node': MY_VERTEX,
        'player2_node': OPP_VERTEX
    }

    # Use simulated annealing adjudicator (matches website)
    adj = SimulatedAnnealingAdjudicator()
    adj.setup(epsilon=0.0)
    result = adj.adjudicate(game_state)

    return float(result['score'])


def quick_evaluate(state: str) -> float:
    """
    Quick heuristic evaluation for non-terminal states.

    Estimates advantage based on:
    - Control of player vertices (edges touching MY_VERTEX vs OPP_VERTEX)
    - Color advantage (green on our side, purple on theirs)
    """
    my_green = 0
    my_purple = 0
    opp_green = 0
    opp_purple = 0

    for edge_idx, (v1, v2) in enumerate(PETERSEN_EDGES):
        color = state[edge_idx]
        if color == '-':
            continue

        touches_me = (v1 == MY_VERTEX or v2 == MY_VERTEX)
        touches_opp = (v1 == OPP_VERTEX or v2 == OPP_VERTEX)

        if touches_me:
            if color == 'G':
                my_green += 1
            else:
                my_purple += 1
        if touches_opp:
            if color == 'G':
                opp_green += 1
            else:
                opp_purple += 1

    # We want green on our edges, purple on opponent's edges
    score = (my_green - my_purple) + (opp_purple - opp_green)
    return score / 6.0  # Normalize (max 6 relevant edges)


@dataclass
class MCTSNode:
    """
    Node in the MCTS tree with Progressive Bias.

    Uses heuristic priors to guide exploration toward likely-good moves.
    """
    state: str  # Board state string
    is_our_turn: bool  # True if it's our turn to move
    parent: Optional['MCTSNode'] = None
    action: Optional[tuple] = None  # (edge_idx, color) that led to this node
    prior: float = 0.5  # Heuristic prior for this action
    children: dict = field(default_factory=dict)  # action -> MCTSNode
    visits: int = 0
    total_value: float = 0.0
    untried_actions: list = field(default_factory=list)
    action_priors: dict = field(default_factory=dict)  # action -> prior

    def __post_init__(self):
        if not self.untried_actions:
            self.untried_actions = self._get_prioritized_actions()

    def _get_prioritized_actions(self) -> list:
        """
        Get legal actions sorted by heuristic prior (best first).

        This ensures we expand promising moves before unlikely ones.
        """
        actions = []
        for i, c in enumerate(self.state):
            if c == '-':
                for color in ['G', 'P']:
                    prior = compute_action_prior(i, color, self.is_our_turn)
                    actions.append((i, color, prior))
                    self.action_priors[(i, color)] = prior

        # Sort by prior descending (best first), with small random tie-breaking
        actions.sort(key=lambda x: (-x[2], random.random()))

        # Return list of (edge, color) tuples without the prior
        return [(a[0], a[1]) for a in actions]

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.state.count('-') == 0

    def is_fully_expanded(self) -> bool:
        """Check if all children have been expanded."""
        return len(self.untried_actions) == 0

    def ucb1_value(self, exploration: float = 1.414, prior_weight: float = 1.0) -> float:
        """
        Calculate UCB1 value with Progressive Bias.

        The prior bonus decreases as visits increase, allowing MCTS to
        override the heuristic when data shows otherwise.

        Formula: Q/N + c*sqrt(ln(parent.N)/N) + w*(prior-0.5)/(N+1)

        All values are stored from P1's perspective. When selecting at opponent
        turn nodes, we negate exploitation to model opponent's minimizing choice.
        """
        if self.visits == 0:
            return float('inf')

        exploitation = self.total_value / self.visits
        exploration_term = exploration * math.sqrt(math.log(self.parent.visits) / self.visits)

        # Progressive bias: prior influence decreases with visits
        prior_bonus = prior_weight * (self.prior - 0.5) / (self.visits + 1)

        # Negate when PARENT is opponent's turn (parent minimizes our value)
        # Child's is_our_turn is opposite of parent's, so negate when child is OUR turn
        if self.is_our_turn:
            return -exploitation + exploration_term + prior_bonus
        return exploitation + exploration_term + prior_bonus

    def best_child(self, exploration: float = 1.414, prior_weight: float = 1.0) -> 'MCTSNode':
        """Select best child using UCB1 with Progressive Bias."""
        return max(self.children.values(),
                   key=lambda c: c.ucb1_value(exploration, prior_weight))

    def expand(self) -> 'MCTSNode':
        """Expand by adding a new child node (best unexplored action first)."""
        action = self.untried_actions.pop(0)  # Pop from front (highest priority)
        edge_idx, color = action

        # Create new state
        new_state = list(self.state)
        new_state[edge_idx] = color
        new_state = ''.join(new_state)

        # Get the prior we computed earlier
        prior = self.action_priors.get(action, 0.5)

        child = MCTSNode(
            state=new_state,
            is_our_turn=not self.is_our_turn,
            parent=self,
            action=action,
            prior=prior
        )
        self.children[action] = child
        return child

    def update(self, value: float):
        """Backpropagate value up the tree (P1 perspective throughout)."""
        self.visits += 1
        self.total_value += value
        if self.parent:
            self.parent.update(value)


class MCTSStrategy:
    """
    Monte Carlo Tree Search strategy for Petersen graph Tangled games.

    Uses UCB1 with Progressive Bias for selection, heuristic-guided rollouts
    for simulation, and full enumeration for terminal state evaluation.
    """

    def __init__(
        self,
        time_limit: float = 2.0,
        max_iterations: int = 10000,
        exploration: float = 1.414,
        prior_weight: float = 2.0,
        use_heuristic_rollout: bool = True
    ):
        """
        Initialize MCTS strategy.

        Args:
            time_limit: Maximum time per move in seconds
            max_iterations: Maximum MCTS iterations per move
            exploration: UCB1 exploration constant (sqrt(2) ≈ 1.414 is standard)
            prior_weight: Weight for heuristic prior in Progressive Bias
            use_heuristic_rollout: If True, use heuristic guidance in rollouts
        """
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.exploration = exploration
        self.prior_weight = prior_weight
        self.use_heuristic_rollout = use_heuristic_rollout

        # Statistics
        self.last_iterations = 0
        self.last_time = 0.0

    def _compute_momentum(self, score_history: list, window: int = 4) -> float:
        """Compute recent score trend. Positive = improving, negative = declining."""
        if not score_history or len(score_history) < 2:
            return 0.0
        recent = score_history[-window:] if len(score_history) >= window else score_history
        if len(recent) < 2:
            return 0.0
        # Extract scores (assuming format: (edge, color, score) or just scores)
        scores = []
        for item in recent:
            if isinstance(item, (int, float)):
                scores.append(float(item))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                scores.append(float(item[2]))
        if len(scores) < 2:
            return 0.0
        return (scores[-1] - scores[0]) / len(scores)

    def calculate_move(
        self,
        state: str,
        score: float = 0.0,
        score_history: list = None
    ) -> Optional[tuple[int, str]]:
        """
        Calculate the best move using MCTS.

        Args:
            state: 15-char string, 'G'/'P'/'-' for each edge
            score: Current game score (used for adaptive exploration)
            score_history: Previous moves (used for momentum-based exploration)

        Returns:
            (edge_index, color) or None if no moves available
        """
        if state.count('-') == 0:
            return None

        # Adaptive exploration based on game state
        # - When losing (negative momentum), explore more aggressively
        # - When winning (positive momentum), exploit more
        momentum = self._compute_momentum(score_history) if score_history else 0.0
        adaptive_exploration = self.exploration
        if momentum < -0.3:
            # Losing - explore more to find comebacks
            adaptive_exploration = min(2.0, self.exploration * 1.3)
        elif momentum > 0.3:
            # Winning - exploit more to secure lead
            adaptive_exploration = max(1.0, self.exploration * 0.8)

        # Create root node
        root = MCTSNode(state=state, is_our_turn=True)

        start_time = time.time()
        iterations = 0

        while iterations < self.max_iterations:
            # Check time limit
            if time.time() - start_time >= self.time_limit:
                break

            # Selection: traverse tree using UCB1 with Progressive Bias
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(adaptive_exploration, self.prior_weight)

            # Expansion: add a new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # Simulation: random rollout to terminal state
            value = self._simulate(node.state, node.is_our_turn)

            # Backpropagation: update values up the tree
            node.update(value)

            iterations += 1

        self.last_iterations = iterations
        self.last_time = time.time() - start_time

        # Select best action based on visit count (most robust)
        if not root.children:
            # No children expanded, return first legal action
            actions = root._get_legal_actions()
            return actions[0] if actions else None

        best_action = max(root.children.keys(), key=lambda a: root.children[a].visits)
        return best_action

    def _simulate(self, state: str, is_our_turn: bool) -> float:
        """
        Simulate a random game from the given state to terminal.

        Returns value from our perspective (positive = good for us).
        """
        current_state = list(state)
        current_turn = is_our_turn

        # Get available edges
        available = [i for i, c in enumerate(current_state) if c == '-']

        while available:
            # Select action
            if self.use_heuristic_rollout:
                action = self._heuristic_action(current_state, available, current_turn)
            else:
                edge = random.choice(available)
                color = random.choice(['G', 'P'])
                action = (edge, color)

            edge, color = action
            current_state[edge] = color
            available.remove(edge)
            current_turn = not current_turn

        # Evaluate terminal state
        terminal_state = ''.join(current_state)
        return evaluate_terminal_state(terminal_state)

    def _heuristic_action(
        self,
        state: list,
        available: list,
        is_our_turn: bool
    ) -> tuple[int, str]:
        """
        Select action using weighted stochastic selection based on priors.

        Uses domain knowledge about edge importance with randomization to
        ensure diverse rollouts. Actions are selected with probability
        proportional to their heuristic prior values.

        This prevents deterministic rollouts that can mislead MCTS.
        """
        # Build list of all possible actions with their weights
        actions = []
        weights = []

        for edge in available:
            for color in ['G', 'P']:
                prior = compute_action_prior(edge, color, is_our_turn)
                actions.append((edge, color))
                # Convert prior to weight (square to amplify differences)
                weights.append(prior ** 2)

        # Weighted random selection
        if actions and weights:
            total = sum(weights)
            if total > 0:
                # Normalize weights and select
                selected = random.choices(actions, weights=weights, k=1)[0]
                return selected

        # Fallback: uniform random (should rarely happen)
        edge = random.choice(available)
        color = random.choice(['G', 'P'])
        return (edge, color)

    def get_stats(self) -> dict:
        """Return statistics from the last move calculation."""
        return {
            'iterations': self.last_iterations,
            'time': self.last_time,
            'iterations_per_second': self.last_iterations / max(self.last_time, 0.001)
        }


class HybridStrategy:
    """
    Combines MCTS with heuristic opening, endgame play, and learning.

    Uses:
    - Heuristic opening (first few moves) to secure key edges quickly
    - MCTS with Progressive Bias for midgame decisions
    - Exhaustive minimax for endgame
    - Learning from game outcomes to adjust edge values
    """

    def __init__(
        self,
        mcts_time_limit: float = 2.0,
        mcts_iterations: int = 5000,
        opening_moves: int = 4,
        prior_weight: float = 4.0
    ):
        self.base_time_limit = mcts_time_limit  # Store base for reset
        self.mcts = MCTSStrategy(
            time_limit=mcts_time_limit,
            max_iterations=mcts_iterations,
            prior_weight=prior_weight
        )
        self.opening_moves = opening_moves

        # Opening sequence: secure our edges first, then attack opponent
        # Prioritize defense before offense for stability
        # These are the most critical edges for Player 1 (Red)
        self.opening_sequence = [
            (9, 'G'),   # E9 (4-5): MY edge - secure our spoke first
            (10, 'G'),  # E10 (5-6): MY edge + HUB - critical hub control
            (11, 'G'),  # E11 (5-9): MY edge - complete our vertex protection
            (5, 'P'),   # E5 (1-7): OPP edge - attack their spoke
            (12, 'P'),  # E12 (6-7): OPP edge + HUB - attack hub connection
            (13, 'P'),  # E13 (7-8): OPP edge - complete attack on vertex 7
        ]

        # Learning: track move history and outcomes
        self.move_history = []  # [(edge, color, score_after), ...]
        self.game_results = []  # List of (result, final_score, moves)

        # Learned edge value adjustments (start at 0, adjusted by learning)
        self.edge_adjustments = [0.0] * NUM_EDGES

        # Learning rate - reduced to prevent overreaction to losses
        self.learning_rate = 0.03

    def calculate_move(
        self,
        state: str,
        score: float = 0.0,
        score_history: list = None
    ) -> Optional[tuple[int, str]]:
        """Calculate the best move using hybrid strategy."""
        grey_count = state.count('-')

        if grey_count == 0:
            return None

        # Count how many moves have been made (our moves + opponent moves)
        total_moves = NUM_EDGES - grey_count

        # Opening phase: use heuristic sequence
        # But only for our first few moves, not total moves
        our_move_count = (total_moves + 1) // 2  # Approximate our move count
        if our_move_count < self.opening_moves:
            for edge, color in self.opening_sequence:
                if state[edge] == '-':
                    return (edge, color)

        # Endgame with very few options: use exhaustive minimax
        if grey_count <= 2:
            return self._exhaustive_endgame(state)

        # Late game with few options: use much more MCTS time
        if grey_count <= 4:
            self.mcts.time_limit = min(10.0, self.base_time_limit * 3)
        else:
            self.mcts.time_limit = self.base_time_limit  # Reset to base

        # Use MCTS for midgame and later
        return self.mcts.calculate_move(state, score, score_history)

    def _exhaustive_endgame(self, state: str) -> Optional[tuple[int, str]]:
        """
        Exhaustive minimax for endgame positions with ≤2 edges remaining.

        Evaluates all possible game completions and picks the best move.
        """
        available = [i for i, c in enumerate(state) if c == '-']
        if not available:
            return None

        best_move = None
        best_value = float('-inf')

        for edge in available:
            for color in ['G', 'P']:
                # Make our move
                new_state = list(state)
                new_state[edge] = color

                # Evaluate resulting position
                value = self._minimax_value(''.join(new_state), is_our_turn=False, depth=4)

                if value > best_value:
                    best_value = value
                    best_move = (edge, color)

        return best_move

    def _minimax_value(self, state: str, is_our_turn: bool, depth: int) -> float:
        """Minimax evaluation for endgame."""
        grey_count = state.count('-')

        # Terminal state - evaluate
        if grey_count == 0 or depth == 0:
            if grey_count == 0:
                return evaluate_terminal_state(state)
            else:
                return quick_evaluate(state)

        available = [i for i, c in enumerate(state) if c == '-']

        if is_our_turn:
            # Maximize
            best = float('-inf')
            for edge in available:
                for color in ['G', 'P']:
                    new_state = list(state)
                    new_state[edge] = color
                    value = self._minimax_value(''.join(new_state), False, depth - 1)
                    best = max(best, value)
            return best
        else:
            # Minimize (opponent's turn)
            best = float('inf')
            for edge in available:
                for color in ['G', 'P']:
                    new_state = list(state)
                    new_state[edge] = color
                    value = self._minimax_value(''.join(new_state), True, depth - 1)
                    best = min(best, value)
            return best

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
        if not self.move_history:
            return

        # Store game result
        self.game_results.append((result, final_score, list(self.move_history)))

        # Learn from this game using temporal difference
        self._learn_from_game(result, final_score)

        # Clear history for next game
        self.move_history = []

    def _learn_from_game(self, result: str, final_score: float):
        """
        Update edge adjustments based on game outcome.

        Uses a simplified policy gradient approach:
        - Winning moves get positive reinforcement
        - Losing moves get negative reinforcement
        - Magnitude depends on score margin
        """
        if not self.move_history:
            return

        # Reward signal based on result
        if result == 'win':
            reward = 1.0 + min(final_score, 2.0) / 2.0  # 1.0 to 2.0
        elif result == 'draw':
            reward = 0.1 if final_score >= 0 else -0.1
        else:  # loss
            reward = -1.0 + max(final_score, -2.0) / 2.0  # -2.0 to -1.0

        # Discount factor for temporal credit assignment
        gamma = 0.9
        n_moves = len(self.move_history)

        # Update edge adjustments with discounted rewards
        for i, (edge, color, score) in enumerate(self.move_history):
            # Later moves get more credit/blame (less discounting)
            discount = gamma ** (n_moves - i - 1)
            update = self.learning_rate * reward * discount

            # Color-specific learning
            if color == 'G':
                # Green move - adjust toward positive if winning
                self.edge_adjustments[edge] += update * 0.5
            else:
                # Purple move - adjust toward positive if winning
                self.edge_adjustments[edge] += update * 0.5

            # Clamp adjustments to reasonable range
            self.edge_adjustments[edge] = max(-1.0, min(1.0, self.edge_adjustments[edge]))

        # Log learning stats
        print(f"Learning: {result} (score {final_score:.2f}), reward={reward:.2f}")
        print(f"Edge adjustments: {[f'{a:.2f}' for a in self.edge_adjustments]}")

    def get_learned_adjustments(self) -> list:
        """Return current learned edge adjustments."""
        return self.edge_adjustments.copy()

    def get_game_stats(self) -> dict:
        """Return statistics from all games played."""
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

    def get_stats(self) -> dict:
        """Return MCTS statistics."""
        stats = self.mcts.get_stats()
        stats['game_stats'] = self.get_game_stats()
        return stats
