"""
Line planning against a deterministic opponent.

AlphaQ Up answers a given position the same way every time, so every captured
game is a replayable line and its reply table (tools/alphaq_captures.py) says
exactly where each of our move sequences leads, as far as it has been played.
The planner values the explored tree with those known replies substituted for
minimax, the true lookup-table score at every adjudicated terminal, and plain
model minimax everywhere else (ValueOracle). Then it picks the next line:

exploit  if the tree offers a line whose value beats the draw band, follow it
         (known replies the model rates as AlphaQ mistakes raise it); a line
         that ends on a known terminal is only replayed when that terminal was
         a real win (to confirm it);
explore  otherwise, the shallowest of our decision points on the tree that
         still has an unplayed move within TOL of its best value, taking the
         first such move in rank order (value, then tiebreak). Every game
         therefore reaches new territory and adds a new terminal.

The plan maps positions to our moves up to and including the new move; the
strategy plays model minimax after that, or as soon as AlphaQ leaves the tree.
Values are from our seat's perspective.
"""

import logging
from typing import Optional

import numpy as np

from snowdrop_tangled_agents.strategy import ternary_model as tm
from snowdrop_tangled_agents.strategy.ternary_strategy import TOL, Solution, rank_moves

logger = logging.getLogger(__name__)

STORE_LAYERS = 6


def layer_store_path(beta, us: int):
    from snowdrop_tangled_agents.tools.solve_ternary_game import layer_store_path as p
    return p(beta, us)


def to_move(state: str) -> int:
    """Seat to move: P1 when an even number of edges is colored."""
    return 1 if (tm.NUM_EDGES - state.count('-')) % 2 == 0 else 2


def play(state: str, edge: int, color: str) -> str:
    return state[:edge] + color + state[edge + 1:]


class ValueOracle:
    """Model minimax (value, tiebreak) of any position, from seat `us`'s perspective.

    Positions with up to STORE_LAYERS colored edges come from the full-pass
    layer store (solve_ternary_game); deeper ones (at most 9 free edges) are
    solved on demand with Solution, which covers the whole subgame below.
    """

    def __init__(self, beta, us: int):
        self.us = us
        self.lut = tm.load_lut(beta)
        store = np.load(layer_store_path(beta, us))
        self.rows = {k: {int(m): i for i, m in enumerate(store[f"masks_{k}"])} for k in range(STORE_LAYERS + 1)}
        self.v = {k: store[f"v_{k}"] for k in range(STORE_LAYERS + 1)}
        self.w = {k: store[f"w_{k}"] for k in range(STORE_LAYERS + 1)}
        self._solutions: list = []

    def _solution_for(self, state: str) -> Solution:
        for s in self._solutions:
            if s.covers(state):
                return s
        s = Solution(state, self.lut, self.us)
        self._solutions.append(s)
        if len(self._solutions) > 64:
            self._solutions.pop(0)
        return s

    def value(self, state: str) -> tuple:
        k = tm.NUM_EDGES - state.count('-')
        if k == tm.NUM_EDGES:
            v = float(self.lut[tm.terminal_index(state)]) * (1 if self.us == 1 else -1)
            return v, v
        if k <= STORE_LAYERS:
            mask = sum(1 << e for e, c in enumerate(state) if c != '-')
            col = sum(tm.COLORS.index(c) * 3 ** i for i, c in enumerate(c for c in state if c != '-'))
            row = self.rows[k][mask]
            return float(self.v[k][row, col]), float(self.w[k][row, col])
        s = self._solution_for(state)
        i = s.index(state)
        return float(s.V[i]), float(s.W[i])


class LinePlanner:
    def __init__(self, oracle: ValueOracle, replies: dict, known: dict, played: dict, tol: float = TOL,
                 win_margin: float = tm.DRAW_EPSILON):
        """replies: AlphaQ's {state: (edge, color)}; known: {terminal: lut score, P1 perspective};
        played: {state: {(edge, color), ...}} our moves already played from each state (this seat)."""
        self.o = oracle
        self.us = oracle.us
        self.sign = 1.0 if self.us == 1 else -1.0
        self.replies = replies
        self.known = known
        self.played = played
        self.tol = tol
        self.win_margin = win_margin
        self._memo: dict = {}

    def value(self, state: str) -> float:
        """Value with known replies and known terminal scores substituted."""
        if state in self._memo:
            return self._memo[state]
        if '-' not in state:
            v = self.sign * self.known[state] if state in self.known else self.o.value(state)[0]
        elif to_move(state) == self.us:
            v = max(self.value(c) if self._on_tree(c) else self.o.value(c)[0] for c in self._children(state))
        elif state in self.replies:
            v = self.value(play(state, *self.replies[state]))
        else:
            v = self.o.value(state)[0]
        self._memo[state] = v
        return v

    def _on_tree(self, state: str) -> bool:
        return '-' not in state or state in self.replies

    def _children(self, state: str):
        return [play(state, e, c) for e in range(tm.NUM_EDGES) if state[e] == '-' for c in 'ZGP']

    def ranked(self, state: str) -> list:
        """Our moves from state, best first: [((edge, color), value, tiebreak)]."""
        moves = {}
        for e in range(tm.NUM_EDGES):
            if state[e] != '-':
                continue
            for c in 'ZGP':
                child = play(state, e, c)
                w = self.o.value(child)[1]
                moves[(e, c)] = (self.value(child) if self._on_tree(child) else self.o.value(child)[0], w)
        return [(mv, v, w) for mv, (v, w) in rank_moves(moves)]

    def _candidates(self, state: str) -> list:
        ranked = self.ranked(state)
        best = ranked[0][1]
        return [(mv, v, w) for mv, v, w in ranked if v >= best - self.tol]

    def plan(self) -> dict:
        """Next line: {'mode', 'moves': {state: (edge, color)}, 'depth', 'value', 'note'}."""
        root = '-' * tm.NUM_EDGES
        root_value = self.value(root)

        # Exploit: follow the best known line while it stays on the tree.
        if root_value > self.win_margin:
            moves, state = {}, root
            while '-' in state:
                if to_move(state) == self.us:
                    mv, v, _ = self.ranked(state)[0]
                    moves[state] = mv
                    state = play(state, *mv)
                elif state in self.replies:
                    state = play(state, *self.replies[state])
                else:
                    break
            replay = '-' not in state
            return {"mode": "confirm" if replay else "exploit", "moves": moves, "value": root_value,
                    "depth": len(moves), "note": f"known-reply value {root_value:+.6f}"}

        # Explore: breadth-first over our decision points on the tree.
        frontier = [(root, {})]
        visited = 0
        while frontier:
            nxt = []
            for state, prefix in frontier:
                if '-' not in state:
                    continue
                if to_move(state) != self.us:
                    if state in self.replies:
                        nxt.append((play(state, *self.replies[state]), prefix))
                    continue
                visited += 1
                cands = self._candidates(state)
                done = self.played.get(state, set())
                for mv, v, w in cands:
                    if mv not in done:
                        moves = dict(prefix)
                        moves[state] = mv
                        return {"mode": "explore", "moves": moves, "value": root_value, "depth": len(moves),
                                "note": f"new move E{mv[0]}{mv[1]} (value {v:+.6f}, tiebreak {w:+.4f}) "
                                        f"at our decision {len(moves)}, {len(done)} of {len(cands)} "
                                        f"candidates there already played"}
                for mv, v, w in cands:
                    moves = dict(prefix)
                    moves[state] = mv
                    nxt.append((play(state, *mv), moves))
            frontier = nxt
        if not visited:
            return {"mode": "book", "moves": {}, "value": root_value, "depth": 0,
                    "note": "no known line from this seat yet"}
        return {"mode": "exhausted", "moves": {}, "value": root_value, "depth": 0,
                "note": "every candidate on the explored tree has been played"}


def build_planner(beta, seat: int, games: Optional[list] = None, oracle: Optional[ValueOracle] = None,
                  **kw) -> LinePlanner:
    """Planner over every captured game vs AlphaQ from this seat (re-read from disk each call)."""
    from snowdrop_tangled_agents.tools import alphaq_captures as ac
    games = ac.load_games() if games is None else games
    played: dict = {}
    for g in games:
        if g["seat"] != seat:
            continue
        for i, (s, mv) in enumerate(zip(ac.states_of(g["moves"]), g["moves"])):
            if ac.mover(i) == seat:
                played.setdefault(s, set()).add(mv)
    return LinePlanner(oracle or ValueOracle(beta, seat), ac.reply_table(games, seat),
                       ac.known_terminals(games), played, **kw)


def reply_regrets(planner: LinePlanner) -> list:
    """AlphaQ's known replies with the model value before and after, from our perspective.

    A positive regret means the reply gave us more than minimax allows: a model mistake.
    """
    out = []
    for state, (e, c) in planner.replies.items():
        before = planner.o.value(state)[0]
        after = planner.o.value(play(state, e, c))[0]
        out.append((after - before, state, (e, c), before, after))
    return sorted(out, reverse=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Show the next planned line and AlphaQ's known replies")
    ap.add_argument("--seat", type=int, choices=(1, 2), default=1)
    ap.add_argument("--beta", type=tm.parse_model, default=tm.DEFAULT_BETA)
    ap.add_argument("--regrets", type=int, default=10, help="list the N largest model regrets")
    args = ap.parse_args()
    planner = build_planner(args.beta, args.seat)
    p = planner.plan()
    print(f"P{args.seat} plan: {p['mode']}, tree value {p['value']:+.6f}: {p['note']}")
    for state, (e, c) in sorted(p["moves"].items(), key=lambda kv: -kv[0].count('-')):
        print(f"  {state} -> E{e}{c}")
    print(f"AlphaQ replies known from P{args.seat}: {len(planner.replies)}; largest model regrets:")
    for regret, state, (e, c), before, after in reply_regrets(planner)[:args.regrets]:
        print(f"  {state} e{e}{c.lower()} regret {regret:+.6f} ({before:+.6f} -> {after:+.6f})")


if __name__ == "__main__":
    main()
