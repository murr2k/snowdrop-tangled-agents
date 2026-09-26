"""
Search-level replica of AlphaQ Up (stage 1 of docs/ALPHAQ_REPLICA_PLAN.md).

AlphaQ is an AlphaZero network with 1000 MCTS rollouts per move. This replica
keeps the search (PUCT, 1000 simulations, argmax visits with ties to the higher
prior) and swaps the network for what we have: the behaviour clone's move
distribution as the prior at every node, and the win/draw/loss class (+1/0/-1)
of the learned table's minimax value as the leaf value, since AlphaQ optimises
results rather than raw score. Recorded terminals use their true table values.

It measures behavioural fidelity only: with near-exact values it cannot make
AlphaQ's value-network mistakes (variant 1b in the plan swaps in a fitted value
approximator for that).

Sign convention: every node stores its value from the view of the player who
moved into it, so a parent picks the child with the highest mean value.
"""

import math
from typing import Callable, Optional

from snowdrop_tangled_agents.strategy import ternary_model as tm
from snowdrop_tangled_agents.strategy.line_planner import ValueOracle, play, to_move


def result_class(v: float) -> int:
    return 1 if v > tm.DRAW_EPSILON else -1 if v < -tm.DRAW_EPSILON else 0


class Node:
    __slots__ = ("state", "prior", "children", "n", "w")

    def __init__(self, state: str, prior: float):
        self.state = state
        self.prior = prior
        self.children: Optional[dict] = None     # move -> Node
        self.n = 0
        self.w = 0.0


class SearchReplica:
    """AlphaQ stand-in for positions from games where we hold `our_seat`."""

    def __init__(self, our_seat: int, prior_fn: Callable, sims: int = 1000, c_puct: float = 1.5,
                 model: str = "learned", known: Optional[dict] = None):
        """prior_fn(state, mover_is_us) -> {move: probability} over the mover's legal moves."""
        self.our_seat = our_seat
        self.prior_fn = prior_fn
        self.sims = sims
        self.c_puct = c_puct
        self.oracle = ValueOracle(model, our_seat)
        self.known = known or {}

    def _class_for(self, state: str, player: int) -> int:
        """Result class of `state` for `player` (+1 win, 0 draw, -1 loss) under the learned model."""
        if '-' not in state:
            p1 = self.known.get(state)
            p1 = float(self.oracle.lut[tm.terminal_index(state)]) if p1 is None else p1
            return result_class(p1 if player == 1 else -p1)
        ours = result_class(self.oracle.value(state)[0])
        return ours if player == self.our_seat else -ours

    def _select(self, node: Node):
        sq = math.sqrt(node.n)
        best, best_score = None, -1e18
        for ch in node.children.values():
            q = ch.w / ch.n if ch.n else 0.0                 # first-play value 0, as in AlphaZero
            score = q + self.c_puct * ch.prior * sq / (1 + ch.n)
            if score > best_score:
                best, best_score = ch, score
        return best

    def search(self, state: str) -> dict:
        """Visit counts {move: n} after `sims` simulations from `state` (AlphaQ to move)."""
        root = Node(state, 1.0)
        root.children = {mv: Node(play(state, *mv), p)
                         for mv, p in self.prior_fn(state, to_move(state) == self.our_seat).items()}
        root.n = 1
        for _ in range(self.sims):
            node, path = root, [root]
            while node.children is not None and '-' in node.state:
                node = self._select(node)
                path.append(node)
            mover_into = to_move(path[-2].state)             # the player whose move reached this node
            if '-' in node.state:
                mover_here = to_move(node.state)
                node.children = {mv: Node(play(node.state, *mv), p)
                                 for mv, p in self.prior_fn(node.state, mover_here == self.our_seat).items()}
            v = self._class_for(node.state, mover_into)     # value for the player who moved into the leaf
            for nd in reversed(path):
                nd.n += 1
                nd.w += v
                v = -v
        return {mv: ch.n for mv, ch in root.children.items()}

    def choose(self, state: str):
        """(visit counts, predicted move): argmax visits, ties to the higher prior."""
        root_priors = self.prior_fn(state, to_move(state) == self.our_seat)
        visits = self.search(state)
        mv = max(visits, key=lambda m: (visits[m], root_priors.get(m, 0.0)))
        return visits, mv
