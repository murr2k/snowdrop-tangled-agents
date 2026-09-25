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


def parse_line(text: str) -> list:
    """'E7G E2G e0z ...' -> our moves [(7, 'G'), (2, 'G'), ...]; lower-case (opponent) tokens are ignored."""
    out = []
    for tok in text.replace(',', ' ').split():
        if tok[0] == 'E':
            out.append((int(tok[1:-1]), tok[-1].upper()))
    return out


def plan_from_line(our_moves: list, seat: int, replies: dict) -> dict:
    """Plan that plays our_moves in order along AlphaQ's known replies.

    The plan stops where AlphaQ's reply is not yet known; from there (or if
    AlphaQ deviates) the strategy plays minimax.
    """
    state, moves, i = '-' * tm.NUM_EDGES, {}, 0
    while '-' in state and i < len(our_moves):
        if to_move(state) == seat:
            e, c = our_moves[i]
            if state[e] != '-':
                raise ValueError(f"E{e}{c} is not legal at {state}")
            moves[state] = (e, c)
            state = play(state, e, c)
            i += 1
        elif state in replies:
            state = play(state, *replies[state])
        else:
            break
    return {"mode": "line", "moves": moves, "value": float('nan'), "depth": len(moves),
            "note": f"fixed line, {len(moves)} of {len(our_moves)} moves reachable along known replies"}


class EndgameProber:
    """Find a P1 win in the last three plies by replaying known lines.

    AlphaQ appears exact at its last move (move 14): from the position it
    chooses, every one of our final moves is at best a draw. A win therefore
    needs a position after our move 13 where all six of AlphaQ's replies leave
    us a winning final. Anchors are positions after AlphaQ's move 12 on known
    P1 lines (replayable, AlphaQ is deterministic). For each of our 9 move-13
    options the prober tracks what is known: AlphaQ's reply and the true value
    of each final. Near-tie terminals are decided by table noise, so their prior
    win chance is tie_prior; decisive ones follow the model.

    Each plan is one game: the known prefix, our move 13, and at the last move
    the untested final with the best prior (the strategy asks final_choice()).
    A known winning final on a reachable line is replayed at once.
    """

    def __init__(self, beta, games: list, tie_prior: float = 0.25, tie_band: float = 0.01):
        from snowdrop_tangled_agents.tools import alphaq_captures as ac
        self.lut = tm.load_lut(beta)
        self.replies = ac.reply_table(games, 1)
        self.known = ac.known_terminals(games)
        self.tie_prior = tie_prior
        self.tie_band = tie_band
        self.anchors = {}                       # position after AlphaQ's move 12 -> our prefix {state: move}
        for g in games:
            if g["seat"] != 1:
                continue
            prefix = {}
            for i, (s, mv) in enumerate(zip(ac.states_of(g["moves"]), g["moves"])):
                if i == 12:
                    self.anchors.setdefault(s, dict(prefix))
                    break
                if ac.mover(i) == 1:
                    prefix[s] = mv

    def prior(self, terminal: str) -> float:
        """Chance that this terminal is a real P1 win."""
        if terminal in self.known:
            return 1.0 if self.known[terminal] > tm.DRAW_EPSILON else 0.0
        v = float(self.lut[tm.terminal_index(terminal)])
        if v > self.tie_band:
            return 0.97
        if v < -self.tie_band:
            return 0.02
        return self.tie_prior

    def finals(self, p14: str) -> list:
        e = p14.index('-')
        return [play(p14, e, c) for c in 'ZGP']

    def final_choice(self, p14: str):
        """Our last move: a known win, else the untested final most likely to win, else the best known."""
        e = p14.index('-')
        opts = []
        for c in 'ZGP':
            t = play(p14, e, c)
            known = t in self.known
            win = known and self.known[t] > tm.DRAW_EPSILON
            value = self.known[t] if known else float(self.lut[tm.terminal_index(t)])
            opts.append((win, not known, self.prior(t), value, c))
        return e, max(opts)[4]

    def p_escape(self, p14: str) -> float:
        """Chance that AlphaQ's move 14 to p14 leaves us no winning final."""
        out = 1.0
        for t in self.finals(p14):
            out *= 1 - self.prior(t)
        return out

    def p_forced(self, s13: str) -> float:
        """Chance that s13 (after our move 13) is a forced win, given what is known."""
        if s13 in self.replies:
            return 1 - self.p_escape(play(s13, *self.replies[s13]))
        out = 1.0
        for e in range(tm.NUM_EDGES):
            if s13[e] == '-':
                for c in 'ZGP':
                    out *= 1 - self.p_escape(play(s13, e, c))
        return out

    def plan(self) -> dict:
        best = None
        for a, prefix in self.anchors.items():
            for e in range(tm.NUM_EDGES):
                if a[e] != '-':
                    continue
                for c in 'ZGP':
                    s13 = play(a, e, c)
                    p = self.p_forced(s13)
                    if s13 in self.replies:
                        fin = self.finals(play(s13, *self.replies[s13]))
                        if any(self.known.get(t, 0.0) > tm.DRAW_EPSILON for t in fin):
                            p = 2.0                   # known win: replay it
                        elif all(t in self.known for t in fin):
                            continue                  # refuted
                    if best is None or p > best[0]:
                        best = (p, a, prefix, (e, c))
        if best is None:
            return {"mode": "exhausted", "moves": {}, "value": 0.0, "depth": 0,
                    "note": "every move 13 at every anchor refuted"}
        p, a, prefix, mv = best
        moves = dict(prefix)
        moves[a] = mv
        mode = "confirm" if p >= 2.0 else "probe"
        return {"mode": mode, "moves": moves, "value": p, "depth": len(moves), "prober": self,
                "note": f"{mode}: move 13 E{mv[0]}{mv[1]} at anchor {a}, P(forced win) {min(p, 1.0):.3f}"}


class P2EndgameProber(EndgameProber):
    """P2 version: our move 14, then AlphaQ picks the final (exactly, as observed).

    One game per option gives that option's exact value: AlphaQ's final is
    its best of three, so the option wins for us only if all three finals are
    P2 wins. Anchors are positions after AlphaQ's move 13 on known P2 lines.
    """

    def __init__(self, beta, games: list, tie_prior: float = 0.25, tie_band: float = 0.01):
        from snowdrop_tangled_agents.tools import alphaq_captures as ac
        self.lut = tm.load_lut(beta)
        self.replies = ac.reply_table(games, 2)
        self.known = ac.known_terminals(games)
        self.tie_prior = tie_prior
        self.tie_band = tie_band
        self.anchors = {}
        for g in games:
            if g["seat"] != 2:
                continue
            prefix = {}
            for i, (s, mv) in enumerate(zip(ac.states_of(g["moves"]), g["moves"])):
                if i == 13:
                    self.anchors.setdefault(s, dict(prefix))
                    break
                if ac.mover(i) == 2:
                    prefix[s] = mv

    def p2_win(self, terminal: str) -> float:
        """Chance that this terminal is a real P2 win."""
        if terminal in self.known:
            return 1.0 if self.known[terminal] < -tm.DRAW_EPSILON else 0.0
        v = float(self.lut[tm.terminal_index(terminal)])
        if v < -self.tie_band:
            return 0.97
        if v > self.tie_band:
            return 0.02
        return self.tie_prior

    def plan(self) -> dict:
        best = None
        for a, prefix in self.anchors.items():
            for e in range(tm.NUM_EDGES):
                if a[e] != '-':
                    continue
                for c in 'ZGP':
                    s14 = play(a, e, c)
                    fin = self.finals(s14)
                    if s14 in self.replies:
                        t = play(s14, *self.replies[s14])
                        if t in self.known and self.known[t] < -tm.DRAW_EPSILON:
                            p = 2.0                   # known win: replay it
                        else:
                            continue                  # AlphaQ's best final is known and not a loss for it
                    else:
                        p = float(np.prod([self.p2_win(t) for t in fin]))
                    if best is None or p > best[0]:
                        best = (p, a, prefix, (e, c))
        if best is None:
            return {"mode": "exhausted", "moves": {}, "value": 0.0, "depth": 0,
                    "note": "every move 14 at every P2 anchor played"}
        p, a, prefix, mv = best
        moves = dict(prefix)
        moves[a] = mv
        mode = "confirm" if p >= 2.0 else "probe"
        return {"mode": mode, "moves": moves, "value": p, "depth": len(moves),
                "note": f"{mode}: move 14 E{mv[0]}{mv[1]} at anchor {a}, P(win) {min(p, 1.0):.3f}"}
