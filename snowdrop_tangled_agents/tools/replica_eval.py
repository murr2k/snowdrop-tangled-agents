"""
Stage-1 fidelity check for the search-level AlphaQ replica (docs/ALPHAQ_REPLICA_PLAN.md).

Held-out agreement with AlphaQ's recorded moves, grouped by line family (the
position after move 4) so near-duplicate positions never straddle folds. Each
fold's clone is trained on its training folds only and serves as the replica's
prior. Reports, per move number: clone and replica top-1 / top-3 and result-class
agreement; plus the false-alarm rate on AlphaQ decisions we verified exact
(P1-seat moves 10, 12 and 14: a replica move in a worse result class for AlphaQ
than the move it actually played).

Usage:
    python -m snowdrop_tangled_agents.tools.replica_eval [--sims 1000] [--cpuct 1.5] [--limit N] [--workers 6]
"""

import argparse
import pickle
import random
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from snowdrop_tangled_agents.strategy import ternary_model as tm
from snowdrop_tangled_agents.strategy.line_planner import ValueOracle, play
from snowdrop_tangled_agents.tools import alphaq_captures as ac
from snowdrop_tangled_agents.tools.alphaq_clone import MODEL, option_features


def decisions() -> list:
    """[(our_seat, state, alphaq_move, family, move_number)] for every recorded AlphaQ decision."""
    games = ac.load_games()
    family = {}
    for g in games:
        states = ac.states_of(g["moves"])
        for s in states:
            family.setdefault((g["seat"], s), states[4])
    out = []
    for seat in (1, 2):
        for s, mv in ac.reply_table(games, seat).items():
            out.append((seat, s, mv, family.get((seat, s), s), tm.NUM_EDGES - s.count('-') + 1))
    return out


def fit_clone(train: list):
    from sklearn.ensemble import GradientBoostingClassifier
    oracles = {1: ValueOracle(MODEL, 1), 2: ValueOracle(MODEL, 2)}
    X, y = [], []
    for seat, s, mv, _, _ in train:
        opts, F = option_features(s, seat, oracles[seat])
        X.append(F)
        y += [int(o == mv) for o in opts]
    return GradientBoostingClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                      random_state=0).fit(np.vstack(X), np.array(y))


_W = {}


def _init(clone_bytes, sims, cpuct):
    from snowdrop_tangled_agents.strategy.alphaq_replica import SearchReplica
    _W["clf"] = pickle.loads(clone_bytes)
    _W["oracles"] = {1: ValueOracle(MODEL, 1, max_solutions=8), 2: ValueOracle(MODEL, 2, max_solutions=8)}
    known = ac.known_terminals(ac.load_games())

    def prior_fn_for(seat):
        def prior_fn(state, mover_is_us):
            opts, F = option_features(state, seat, _W["oracles"][seat], mover_is_us=mover_is_us)
            p = _W["clf"].predict_proba(F)[:, 1]
            p = p / p.sum() if p.sum() > 0 else np.full(len(p), 1 / len(p))
            return dict(zip(opts, p))
        return prior_fn

    _W["replicas"] = {seat: SearchReplica(seat, prior_fn_for(seat), sims=sims, c_puct=cpuct, known=known,
                                          oracle=_W["oracles"][seat]) for seat in (1, 2)}


def _evaluate(d):
    seat, s, mv, _, mvno = d
    rep = _W["replicas"][seat]
    prior = rep.prior_fn(s, False)
    clone_rank = sorted(prior, key=lambda m: -prior[m])
    visits, pred = rep.choose(s)
    rep_rank = sorted(visits, key=lambda m: (-visits[m], -prior.get(m, 0)))
    aq = rep._class_for(play(s, *mv), 3 - seat)              # AlphaQ's class after its actual move
    rp = rep._class_for(play(s, *pred), 3 - seat)
    return {"seat": seat, "mvno": mvno, "clone1": clone_rank[0] == mv, "clone3": mv in clone_rank[:3],
            "rep1": pred == mv, "rep3": mv in rep_rank[:3], "class_same": aq == rp, "false_alarm": rp < aq}


def run_fold(train, test, sims, cpuct, workers):
    clf = fit_clone(train)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(pickle.dumps(clf), sims, cpuct)) as pool:
        return list(pool.map(_evaluate, test, chunksize=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sims", type=int, default=1000)
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="held-out decisions per fold (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    from sklearn.model_selection import GroupKFold
    D = decisions()
    groups = np.array([hash(d[3]) % 1000003 for d in D])
    rows, alarms = [], []
    t0 = time.time()
    for k, (tr, te) in enumerate(GroupKFold(args.folds).split(D, groups=groups)):
        train = [D[i] for i in tr]
        test = [D[i] for i in te if 2 <= D[i][4] <= 9]
        safe = [D[i] for i in te if D[i][0] == 1 and D[i][4] in (10, 12, 14)]
        random.Random(k).shuffle(test)
        random.Random(k).shuffle(safe)
        if args.limit:
            test, safe = test[:args.limit], safe[:max(args.limit // 2, 1)]
        res = run_fold(train, test + safe, args.sims, args.cpuct, args.workers)
        rows += res[:len(test)]
        alarms += res[len(test):]
        print(f"fold {k}: {len(test)} middlegame + {len(safe)} verified decisions, {time.time() - t0:.0f} s", flush=True)
    by = defaultdict(list)
    for r in rows:
        by[r["mvno"]].append(r)
    print("move   n  clone@1 clone@3  replica@1 replica@3  class-agree")
    for m in sorted(by):
        rs = by[m]
        f = lambda k: np.mean([r[k] for r in rs])
        print(f"{m:4d} {len(rs):4d}   {f('clone1'):.2f}    {f('clone3'):.2f}      {f('rep1'):.2f}      "
              f"{f('rep3'):.2f}       {f('class_same'):.2f}")
    f = lambda k: np.mean([r[k] for r in rows])
    print(f"all  {len(rows):4d}   {f('clone1'):.2f}    {f('clone3'):.2f}      {f('rep1'):.2f}      "
          f"{f('rep3'):.2f}       {f('class_same'):.2f}")
    print(f"false alarms on verified decisions: {np.mean([a['false_alarm'] for a in alarms]):.3f} (n={len(alarms)})")
    rep3 = np.mean([r['rep3'] for r in rows])
    gate = f('rep1') >= f('clone1') and rep3 >= 0.85 and np.mean([a['false_alarm'] for a in alarms]) <= 0.02
    print(f"stage-1 gate (replica@1 >= clone@1, replica@3 >= 0.85, false alarms <= 2%): {'PASS' if gate else 'FAIL'}")


if __name__ == "__main__":
    main()
