"""
Learned terminal table: physics for decisive boards, a classifier for ties.

The quantum models (q40) get decisive boards right but cannot say how the
site's table breaks model ties: it stores residues from its hardware samples
that decide the result inside the 0.0005 draw band. Those residues turn out to
be largely predictable from structure (grouped cross-validation 0.82 against
0.59 for always-draw on 362 observed ties, 2026-09-25). This tool trains a
gradient-boosting classifier on the adjudicated terminals of played games only,
predicts win/draw/loss for every q40 tie, and writes

    ternary_learned_lut.npy    decisive: q40 score; ties: TIE_SCALE * (P(P1) - P(P2))

so solve_ternary_game --beta learned and the planners can use it. The tie value
is the expected result in units above the draw band, which keeps minimax
steering toward ties the classifier rates as ours.

Usage:
    python -m snowdrop_tangled_agents.tools.learned_table [--cv]
"""

import argparse
import time

import numpy as np

from snowdrop_tangled_agents.strategy import ternary_model as tm
from snowdrop_tangled_agents.tools import alphaq_captures as ac

TIE_BAND = 0.01          # |q40| below this is a model tie
TIE_SCALE = 0.002        # learned tie value: TIE_SCALE * (P(P1 win) - P(P2 win)), 4x the draw band
BETAS = (1.0, 2.0, 4.0, 8.0)
LUT_NAMES = ('q40', 'q10', 'qb52')
FEATURE_NAMES = ([f"L{l}_{k}" for l in (0, 2, 4) for k in ('n', 'sum', 'min', 'max', 'abs')] +
                 [f"b{b:g}" for b in BETAS] + list(LUT_NAMES) + ['nZ', 'nG', 'nP'] +
                 [f"e{e}" for e in (9, 10, 11, 5, 12, 13)])
_POW3 = np.array([3 ** e for e in range(tm.NUM_EDGES)], dtype=np.int64)
_COUPLING_OF_DIGIT = np.array([tm.COUPLING[c] for c in tm.COLORS], dtype=np.float32)


def features_from_index(idx: np.ndarray, luts: dict) -> np.ndarray:
    """Feature rows for terminal indices (same layout as FEATURE_NAMES)."""
    digits = (idx[:, None] // _POW3[None, :]) % 3                         # (n, 15) 0=Z 1=G 2=P
    J = _COUPLING_OF_DIGIT[digits]                                       # (n, 15)
    E = J @ tm._PRODUCTS.T                                               # (n, 512)
    e0 = E.min(axis=1, keepdims=True)
    W = tm._WEIGHTS[None, :]
    cols = []
    for lvl in (0, 2, 4):
        sel = np.abs(E - (e0 + lvl)) < 1e-3
        n = sel.sum(axis=1)
        w = np.where(sel, W, 0.0)
        cols += [n, w.sum(axis=1), np.where(sel, W, np.inf).min(axis=1), np.where(sel, W, -np.inf).max(axis=1),
                 np.abs(w).sum(axis=1)]
    cols[2] = np.where(np.isfinite(cols[2]), cols[2], 0.0)
    cols[3] = np.where(np.isfinite(cols[3]), cols[3], 0.0)
    for k in (7, 8, 12, 13):                                             # min/max of empty levels
        cols[k] = np.where(np.isfinite(cols[k]), cols[k], 0.0)
    for b in BETAS:
        cols.append(luts[f"b{b:g}"][idx])
    for name in LUT_NAMES:
        cols.append(luts[name][idx])
    cols += [(digits == 0).sum(1), (digits == 1).sum(1), (digits == 2).sum(1)]
    for e in (9, 10, 11, 5, 12, 13):
        cols.append(J[:, e])
    return np.stack([np.asarray(c, dtype=np.float32) for c in cols], axis=1)


def load_luts() -> dict:
    luts = {f"b{b:g}": tm.load_lut(b) for b in BETAS}
    luts.update({name: tm.load_lut(name) for name in LUT_NAMES})
    return luts


def training_set(luts: dict):
    known = ac.known_terminals(ac.load_games())
    T = [t for t in known if abs(luts['q40'][tm.terminal_index(t)]) < TIE_BAND]
    idx = np.array([tm.terminal_index(t) for t in T], dtype=np.int64)
    y = np.array([1 if known[t] > tm.DRAW_EPSILON else -1 if known[t] < -tm.DRAW_EPSILON else 0 for t in T])
    return T, idx, y


def censored_not_p1(luts: dict) -> list:
    """Ties that cannot be P1 wins, from AlphaQ's own last move as P1.

    AlphaQ's move 15 chooses among three finals (its search is trivially exact
    there). When it settles for a draw, neither declined final can be a P1 win.
    (The analogous inference from its move 14 in our P1 games would assume it was
    never in zugzwang, which is what the probes look for, so it is not used.)
    """
    games = ac.load_games()
    known = ac.known_terminals(games)
    out = set()
    for g in games:
        if g["seat"] != 2 or g["winner"] != "draw":
            continue
        pen = ac.states_of(g["moves"])[-1]
        e = pen.index('-')
        for c in 'ZGP':
            t = pen[:e] + c + pen[e + 1:]
            if t != g["terminal"] and t not in known and abs(luts['q40'][tm.terminal_index(t)]) < TIE_BAND:
                out.add(t)
    return sorted(out)


def fit_p1_classifier(luts: dict, T: list, idx: np.ndarray, y: np.ndarray):
    """Binary P1-win classifier: observed ties plus the censored 'not a P1 win' ties."""
    extra = censored_not_p1(luts)
    eidx = np.array([tm.terminal_index(t) for t in extra], dtype=np.int64)
    X = features_from_index(np.concatenate([idx, eidx]), luts)
    yb = np.concatenate([(y == 1).astype(int), np.zeros(len(extra), dtype=int)])
    return make_classifier().fit(X, yb)


def make_classifier():
    from sklearn.ensemble import GradientBoostingClassifier
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0)


def build(chunk: int = 100_000) -> tuple:
    """(value table, P(P1 win) table, P(P2 win) table), all indexed by terminal_index()."""
    luts = load_luts()
    T, idx, y = training_set(luts)
    clf = make_classifier().fit(features_from_index(idx, luts), y)
    clf1 = fit_p1_classifier(luts, T, idx, y)
    classes = list(clf.classes_)
    q40 = luts['q40']
    lut = q40.astype(np.float32).copy()
    win1 = (q40 > TIE_BAND).astype(np.float16)                          # decisive boards follow the physics
    win2 = (q40 < -TIE_BAND).astype(np.float16)
    known = ac.known_terminals(ac.load_games())
    ties = np.flatnonzero(np.abs(q40) < TIE_BAND)
    for s in range(0, len(ties), chunk):
        part = ties[s:s + chunk]
        F = features_from_index(part, luts)
        pr = clf.predict_proba(F)
        p1 = clf1.predict_proba(F)[:, 1]
        p2 = pr[:, classes.index(-1)] if -1 in classes else 0.0
        lut[part] = TIE_SCALE * (p1 - p2)
        win1[part] = p1
        win2[part] = p2
    for t, v in known.items():                                           # observed terminals: the truth
        i = tm.terminal_index(t)
        lut[i] = v
        win1[i] = float(v > tm.DRAW_EPSILON)
        win2[i] = float(v < -tm.DRAW_EPSILON)
    return lut, win1, win2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cv", action="store_true", help="only report grouped cross-validation")
    args = ap.parse_args()
    luts = load_luts()
    T, idx, y = training_set(luts)
    print(f"{len(T)} observed ties: P1 {np.sum(y == 1)}, draw {np.sum(y == 0)}, P2 {np.sum(y == -1)}")
    if args.cv:
        from sklearn.model_selection import GroupKFold, cross_val_predict
        fam = {}
        for g in ac.load_games():
            fam.setdefault(g["terminal"], ac.states_of(g["moves"])[10])
        groups = np.array([hash(fam.get(t, t)) % 100000 for t in T])
        pred = cross_val_predict(make_classifier(), features_from_index(idx, luts), y, cv=GroupKFold(5), groups=groups)
        print(f"grouped CV accuracy {np.mean(pred == y):.3f} (always-draw {np.mean(y == 0):.3f})")
        # P1-win ranking on held-out real labels, without and with the censored negatives
        from sklearn.metrics import roc_auc_score
        X = features_from_index(idx, luts)
        extra = censored_not_p1(luts)
        eX = features_from_index(np.array([tm.terminal_index(t) for t in extra], dtype=np.int64), luts)
        for use_extra in (False, True):
            p1 = np.zeros(len(T))
            for tr, te in GroupKFold(5).split(X, y, groups):
                Xt, yt = X[tr], (y[tr] == 1).astype(int)
                if use_extra:
                    Xt, yt = np.vstack([Xt, eX]), np.concatenate([yt, np.zeros(len(eX), dtype=int)])
                p1[te] = make_classifier().fit(Xt, yt).predict_proba(X[te])[:, 1]
            top = np.argsort(-p1)[:20]
            print(f"P1-win classifier {'with' if use_extra else 'without'} {len(extra)} censored negatives: "
                  f"AUC {roc_auc_score(y == 1, p1):.3f}, real P1 wins in top 20: {int(np.sum(y[top] == 1))}")
        return
    t0 = time.time()
    lut, win1, win2 = build()
    path = tm.lut_path("learned")
    np.save(path, lut)
    np.save(tm.LUT_DIR / "ternary_learned_p1win.npy", win1)
    np.save(tm.LUT_DIR / "ternary_learned_p2win.npy", win2)
    print(f"saved {path} in {time.time() - t0:.0f} s; draws {np.mean(np.abs(lut) <= tm.DRAW_EPSILON):.3f}, "
          f"P1 wins {np.mean(lut > tm.DRAW_EPSILON):.3f}, P2 wins {np.mean(lut < -tm.DRAW_EPSILON):.3f}")


if __name__ == "__main__":
    main()
