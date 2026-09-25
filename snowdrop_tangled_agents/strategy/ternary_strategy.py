"""
Exact minimax for three-color Tangled on the Petersen graph.

Each edge is uncolored ('-') or grey ('Z', zero coupling), green ('G') or
purple ('P'). From any position the solver enumerates every reachable
position (4^m for m uncolored edges), scores the terminals from the ternary
model table, and backs values up layer by layer (positions with k uncolored
edges depend only on positions with k-1). Values are from our perspective.

value     minimax model score: both sides play perfectly under the model.
tiebreak  expected model score when the opponent picks uniformly at random
          and we keep playing minimax. Among moves whose value is within TOL
          of the best, the solver prefers the one that leaves the opponent
          the most ways to go wrong. TOL (1e-4) sits above float32 noise
          (~2e-6) and below the thermal margins that separate outcomes
          (~1e-3; the site's draw band is 5e-4); the value itself stays the
          exact minimax.

A solve covers the whole subgame below the position it was run from, so
later positions in the same game are answered by lookup. Our first move is too
large to solve inside a turn (4^15 positions as P1, 4^14 as P2), so it comes
from move_overrides or a book built by tools/solve_ternary_game.py: the opening
book for P1, the reply book (best reply to each P1 opening) for P2.
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from snowdrop_tangled_agents.strategy import ternary_model as tm

logger = logging.getLogger(__name__)

SYMBOLS = '-ZGP'   # base-4 digit per free edge: 0 uncolored, 1 grey, 2 green, 3 purple
TOL = 1e-4
OPENING_BOOK_PATH = Path.home() / ".tangled" / "ternary_opening_book.json"
REPLY_BOOK_PATH = Path.home() / ".tangled" / "ternary_reply_book.json"


class Solution:
    """Minimax values for every position below a root position."""

    def __init__(self, root: str, lut: np.ndarray, us: int):
        self.root = root
        self.us = us
        self.free = [e for e, c in enumerate(root) if c == '-']
        m = len(self.free)
        base = sum(tm.COLORS.index(c) * 3 ** e for e, c in enumerate(root) if c != '-')

        # Per position (index = sum digit_p * 4^p): number of uncolored free
        # edges, and the terminal-table index (meaningful once all are colored).
        # Built by adding one more-significant digit at a time.
        uncolored = np.zeros(1, dtype=np.int8)
        term = np.zeros(1, dtype=np.int32)
        for p, e in enumerate(self.free):
            step = 3 ** e
            uncolored = np.concatenate([uncolored + 1, uncolored, uncolored, uncolored])
            term = np.concatenate([term, term, term + step, term + 2 * step])

        sign = 1.0 if us == 1 else -1.0
        n = 4 ** m
        self.V = np.empty(n, dtype=np.float32)
        self.W = np.empty(n, dtype=np.float32)
        leaf = uncolored == 0
        self.V[leaf] = sign * lut[base + term[leaf]]
        self.W[leaf] = self.V[leaf]
        del term, leaf

        for u in range(1, m + 1):
            S = np.flatnonzero(uncolored == u)
            our_turn = ((tm.NUM_EDGES - u) % 2 == 0) == (us == 1)
            if our_turn:
                bv = np.full(S.size, -np.inf, dtype=np.float32)
                bw = np.full(S.size, -np.inf, dtype=np.float32)
            else:
                bv = np.full(S.size, np.inf, dtype=np.float32)
                bw = np.zeros(S.size, dtype=np.float32)
            # Our turn takes two passes: exact max value first, then the best
            # tiebreak among children within TOL of it.
            for pass_ in ((0, 1) if our_turn else (0,)):
                for p in range(m):
                    pos = np.flatnonzero(((S >> (2 * p)) & 3) == 0)
                    if pos.size == 0:
                        continue
                    parents = S[pos]
                    for c in (1, 2, 3):
                        child = parents + (c << (2 * p))
                        v = self.V[child]
                        if not our_turn:
                            bv[pos] = np.minimum(bv[pos], v)
                            bw[pos] += self.W[child]
                        elif pass_ == 0:
                            bv[pos] = np.maximum(bv[pos], v)
                        else:
                            w = self.W[child]
                            cw = bw[pos]
                            bw[pos] = np.where((v >= bv[pos] - TOL) & (w > cw), w, cw)
            self.V[S] = bv
            self.W[S] = bw if our_turn else bw / (3 * u)

    def covers(self, state: str) -> bool:
        return all(state[e] == c for e, c in enumerate(self.root) if c != '-')

    def index(self, state: str) -> int:
        return sum(SYMBOLS.index(state[e]) << (2 * p) for p, e in enumerate(self.free))

    def moves(self, state: str) -> dict:
        """{(edge, color): (value, tiebreak)} for every legal move from state."""
        i = self.index(state)
        out = {}
        for p, e in enumerate(self.free):
            if state[e] == '-':
                for c in (1, 2, 3):
                    j = i + (c << (2 * p))
                    out[(e, SYMBOLS[c])] = (float(self.V[j]), float(self.W[j]))
        return out


def rank_moves(moves: dict) -> list:
    """Moves best first: those within TOL of the best value by tiebreak, then the rest by value."""
    vmax = max(v for v, _ in moves.values())
    return sorted(moves.items(), key=lambda kv: (kv[1][0] < vmax - TOL,
                                                  -kv[1][1] if kv[1][0] >= vmax - TOL else -kv[1][0],
                                                  -kv[1][1]))


def best_of(moves: dict) -> Tuple[Tuple[int, str], float, float]:
    """Best move: highest tiebreak among moves within TOL of the best value."""
    (edge, color), (v, w) = rank_moves(moves)[0]
    return (edge, color), v, w


class TernaryMinimaxStrategy:
    """
    Three-color strategy: exact minimax under the ternary terminal model.

    Same initialize / calculate_move / record_move / end_game / get_stats
    interface as SwitchbackStrategy. supports_grey tells the runner to pass
    the true board (with 'Z') instead of the legacy two-color view.
    """

    supports_grey = True

    def __init__(self, player: int = 1, move_overrides: Optional[dict] = None,
                 beta: Optional[float] = tm.DEFAULT_BETA, max_live_free: int = 14,
                 opening_book: Path = OPENING_BOOK_PATH, reply_book: Path = REPLY_BOOK_PATH,
                 plan_lines: bool = False, fixed_lines: Optional[list] = None, probe_endgame: bool = False):
        self.player = player
        self.probe_endgame = probe_endgame  # line_planner.EndgameProber (P1)
        self.fixed_lines = list(fixed_lines or [])   # one line of our moves per game, played before --plan-lines
        self.plan_lines = plan_lines        # line_planner: replay known lines, branch into new territory
        self._oracle = None
        self._plan: dict = {"moves": {}}
        self._move_overrides: dict = move_overrides or {}
        self.beta = beta
        self.max_live_free = max_live_free
        self.opening_book_path = opening_book
        self.reply_book_path = reply_book
        self._solution: Optional[Solution] = None
        self.moves_calculated = 0
        self.total_time = 0.0
        self.last_strategy = 'ternary'

    def initialize(self, opponent: str = '') -> bool:
        tm.load_lut(self.beta)
        self._solution = None   # new game
        self._plan = {"moves": {}}
        if self.fixed_lines:
            from snowdrop_tangled_agents.strategy import line_planner as lp
            from snowdrop_tangled_agents.tools import alphaq_captures as ac
            line = self.fixed_lines.pop(0)
            self._plan = lp.plan_from_line(lp.parse_line(line), self.player,
                                           ac.reply_table(ac.load_games(), self.player))
            logger.info(f"ternary plan: {line!r}: {self._plan['note']}")
        elif self.probe_endgame:
            from snowdrop_tangled_agents.strategy import line_planner as lp
            from snowdrop_tangled_agents.tools import alphaq_captures as ac
            prober = lp.EndgameProber if self.player == 1 else lp.P2EndgameProber
            self._plan = prober(self.beta, ac.load_games()).plan()
            logger.info(f"ternary plan: {self._plan['note']}")
        elif self.plan_lines:
            from snowdrop_tangled_agents.strategy import line_planner as lp
            if self._oracle is None:
                self._oracle = lp.ValueOracle(self.beta, self.player)
            planner = lp.build_planner(self.beta, self.player, oracle=self._oracle)
            self._plan = planner.plan()
            logger.info(f"ternary plan: {self._plan['mode']}, {self._plan['depth']} planned moves, "
                        f"tree value {self._plan['value']:+.6f}: {self._plan['note']}")
        return True

    def _book_moves(self, state: str) -> dict:
        """Book entry for this position: P1's first move, or P2's reply to the opening."""
        free = state.count('-')
        if self.player == 1 and free == tm.NUM_EDGES:
            path = self.opening_book_path
        elif self.player == 2 and free == tm.NUM_EDGES - 1:
            path = self.reply_book_path
        else:
            return {}
        per_model = Path(path).parent / f"ternary_{tm.model_name(self.beta)}_book_p{self.player}.json"
        try:
            book = json.loads((per_model if per_model.exists() else Path(path)).read_text())
        except (OSError, ValueError):
            return {}
        if book.get('beta') != self.beta or book.get('player') != self.player:
            return {}
        if self.player == 1:
            moves = book['moves']
        else:
            opening = next(f"E{e}{c}" for e, c in enumerate(state) if c != '-')
            moves = book['replies'].get(opening, {}).get('replies', [])
        return {(m['edge'], m['color']): (m['value'], m['tiebreak']) for m in moves}

    def calculate_move(self, state: str, score: float = 0.0,
                       score_history: list = None) -> Optional[Tuple[int, str, dict]]:
        start = time.time()
        free = state.count('-')
        if free == 0:
            return None
        stats = {'num_grey': free, 'beta': self.beta}

        if free in self._move_overrides:
            edge, color = self._move_overrides[free]
            stats['strategy'] = 'ternary/override'
            return self._done(edge, color, stats, start)

        if free == 1 and self._plan.get("prober") is not None:
            edge, color = self._plan["prober"].final_choice(state)
            stats.update(strategy="ternary/probe-final")
            logger.info(f"ternary: E{edge}{color} final probe")
            return self._done(edge, color, stats, start)

        planned = self._plan["moves"].get(state)
        if planned:
            edge, color = planned
            stats.update(strategy=f"ternary/plan-{self._plan['mode']}")
            logger.info(f"ternary: E{edge}{color} from plan ({self._plan['mode']})")
            return self._done(edge, color, stats, start)

        if self._solution is None or not self._solution.covers(state):
            book = self._book_moves(state)
            if book:
                (edge, color), v, w = best_of(book)
                stats.update(strategy='ternary/book', value=v, tiebreak=w)
                logger.info(f"ternary: E{edge}{color} from book, value={v:+.4f} tiebreak={w:+.4f}")
                return self._done(edge, color, stats, start)
            if free > self.max_live_free:
                edge, color = 7, 'G'
                logger.warning(f"{free} free edges exceeds the live solve limit and there is no "
                               f"opening book; falling back to E{edge}{color} "
                               f"(or pass --oracle-override {free} EDGE COLOR)")
                stats['strategy'] = 'ternary/fallback'
                return self._done(edge, color, stats, start)
            t = time.time()
            self._solution = Solution(state, tm.load_lut(self.beta), self.player)
            stats['solve_time'] = round(time.time() - t, 2)

        moves = self._solution.moves(state)
        (edge, color), v, w = best_of(moves)
        top = rank_moves(moves)[:3]
        stats.update(strategy='ternary', value=v, tiebreak=w,
                     top3=[f"E{e}{c}:{val:+.4f}/{tb:+.4f}" for (e, c), (val, tb) in top])
        logger.info(f"ternary: E{edge}{color} value={v:+.4f} tiebreak={w:+.4f} top3={stats['top3']}")
        return self._done(edge, color, stats, start)

    def _done(self, edge: int, color: str, stats: dict, start: float):
        elapsed = time.time() - start
        self.moves_calculated += 1
        self.total_time += elapsed
        stats['time'] = elapsed
        return edge, color, stats

    def record_move(self, edge: int, color: str, score_after: float) -> None:
        pass

    def end_game(self, result: str, final_score: float) -> None:
        self._solution = None

    def get_stats(self) -> dict:
        avg = self.total_time / self.moves_calculated if self.moves_calculated else 0.0
        return {'moves_calculated': self.moves_calculated, 'avg_move_time': avg, 'backend': 'python'}
