"""
Behaviour clone of AlphaQ Up, for steering games into positions it finds unfamiliar.

AlphaQ is an AlphaZero net with 1000 MCTS rollouts per move. It has been exact
wherever its search reaches the end of the game (moves 10-14 in every fully
probed position), so an error, if any, lives in the middlegame where the value
net decides. The adversarial-policy literature (Wang et al. 2023, "Adversarial
Policies Beat Superhuman Go AIs") finds such errors in positions the victim's
self-play rarely visits. This clone, a gradient-boosting model over (position,
candidate move) trained on every recorded AlphaQ decision, stands in for its
policy: where the clone is unsure what AlphaQ plays, the position is likely far
from its self-play distribution. Held-out top-1 accuracy 0.60 overall, 0.67-0.86
at moves 2-6 (logs/aq_clone.py, 2026-09-25).

Features per candidate move: learned-table value relative to the best (AlphaQ
minimises our value), tiebreak, colour, edge index, whether it touches vertices
5 / 6 / 7, free edges, seat, colour counts on the board.

Usage:
    python -m snowdrop_tangled_agents.tools.alphaq_clone          # train and save
"""

import pickle

import numpy as np

from snowdrop_tangled_agents.strategy import ternary_model as tm

MODEL_PATH = tm.LUT_DIR / "alphaq_clone.pkl"
MODEL = "learned"


def option_features(state: str, seat: int, oracle, mover_is_us: bool = False) -> tuple:
    """(options [(edge, colour)], feature matrix) at `state`; oracle is ValueOracle(MODEL, seat).

    The first feature is how much worse each option is than the mover's best, from the
    mover's point of view: AlphaQ minimises the oracle value, we (mover_is_us) maximise it.
    """
    from snowdrop_tangled_agents.strategy.line_planner import play
    opts = [(e, c) for e in range(tm.NUM_EDGES) if state[e] == '-' for c in 'ZGP']
    if 6 <= tm.NUM_EDGES - state.count('-') and state.count('-') <= 9:
        oracle._solution_for(state)          # one subgame solve covers every child lookup below
    vals = [oracle.value(play(state, e, c)) for e, c in opts]
    if mover_is_us:                          # negate so that lower is better for the mover in both cases
        vals = [(-v, w) for v, w in vals]
    best = min(v for v, _ in vals)
    free = state.count('-')
    nz, ng, np_ = state.count('Z'), state.count('G'), state.count('P')
    rows = []
    for (e, c), (v, w) in zip(opts, vals):
        a, b = tm.EDGES[e]
        rows.append([v - best, w, 'ZGP'.index(c), e, int(5 in (a, b)), int(7 in (a, b)), int(6 in (a, b)),
                     free, seat, nz, ng, np_])
    return opts, np.array(rows, dtype=float)


def train():
    from sklearn.ensemble import GradientBoostingClassifier
    from snowdrop_tangled_agents.strategy.line_planner import ValueOracle
    from snowdrop_tangled_agents.tools import alphaq_captures as ac
    games = ac.load_games()
    X, y = [], []
    for seat in (1, 2):
        oracle = ValueOracle(MODEL, seat)
        for s, mv in ac.reply_table(games, seat).items():
            opts, F = option_features(s, seat, oracle)
            X.append(F)
            y += [int(o == mv) for o in opts]
    clf = GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=0)
    clf.fit(np.vstack(X), np.array(y))
    MODEL_PATH.write_bytes(pickle.dumps(clf))
    return clf


class ClonePolicy:
    def __init__(self, seat: int):
        from snowdrop_tangled_agents.strategy.line_planner import ValueOracle
        self.seat = seat
        self.oracle = ValueOracle(MODEL, seat)
        self.clf = pickle.loads(MODEL_PATH.read_bytes())

    def distribution(self, state: str) -> dict:
        """AlphaQ's predicted reply distribution {(edge, colour): p} at `state` (AlphaQ to move)."""
        opts, F = option_features(state, self.seat, self.oracle)
        s = self.clf.predict_proba(F)[:, 1]
        s = s / s.sum() if s.sum() > 0 else np.full(len(s), 1 / len(s))
        return dict(zip(opts, s))

    def unfamiliarity(self, state: str) -> float:
        """1 - the clone's top probability: high where AlphaQ's reply is hard to predict."""
        return 1.0 - max(self.distribution(state).values())


def main():
    clf = train()
    print(f"trained on {clf.n_features_in_} features; saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
