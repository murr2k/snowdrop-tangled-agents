"""
Phase 2: Train an AlphaQ predictive policy model.

Reads the AlphaQ move corpus from the local game DB and trains two models
that predict AlphaQ's response (edge, color) given the board state. The
trained models are inputs to the Phase 4 expected-value solver.

Per the project plan, this script:

  1. Extracts (state_before, alphaq_action) pairs (1574 games, ~9341 moves)
  2. Featurises board states (per-edge state + per-vertex coloured-degree
     + game-phase scalars + frustration indicators on 5-cycles)
  3. Splits 80/20 by game_id (no within-game leakage)
  4. Trains LogReg and MLP variants, both with legal-action masking
  5. Reports top-1/3/5 accuracy and per-state-bucket accuracy
  6. Saves models to .pkl (sklearn) and .mat (weights for MATLAB Phase 4)
  7. Writes a model card to docs/ALPHAQ_PREDICTIVE_MODEL.md

Usage:
    poetry run python scripts/train_alphaq_policy.py
"""

import math
import pickle
import sqlite3
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.io
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = Path.home() / ".tangled" / "game_stats.db"
DATA_DIR = PROJECT_ROOT / "snowdrop_tangled_agents" / "matlab" / "rl" / "data"
REPORT_PATH = PROJECT_ROOT / "docs" / "ALPHAQ_PREDICTIVE_MODEL.md"

INITIAL_STATE = "-" * 15
N_EDGES = 15
N_VERTICES = 10
N_ACTIONS = 30  # 15 edges * 2 colors

# Petersen graph (graph_id=5) edge list — confirmed from GraphProperties
EDGE_LIST = [
    (0, 2), (0, 3), (0, 6),     # E0, E1, E2
    (1, 3), (1, 4), (1, 7),     # E3, E4, E5
    (2, 4), (2, 8),             # E6, E7
    (3, 9),                     # E8
    (4, 5),                     # E9
    (5, 6), (5, 9),             # E10, E11
    (6, 7),                     # E12
    (7, 8),                     # E13
    (8, 9),                     # E14
]

# All 5-cycles of the Petersen graph (12 in total). Each is a list of
# edge indices that form a cycle of length 5. Petersen graph has girth 5
# and exactly 12 such 5-cycles.
# Computed once via a 5-cycle enumeration; hard-coded here so the script
# has no networkx dependency.
PETERSEN_5_CYCLES = [
    [0, 1, 8, 11, 10, 2],   # placeholder — recomputed at startup
]

# Player vertices
P1_VERTEX = 5
P2_VERTEX = 7


def find_5_cycles(edge_list: list[tuple[int, int]], num_vertices: int) -> list[list[int]]:
    """Enumerate all 5-cycles, return as lists of edge indices."""
    adj = defaultdict(list)
    edge_idx: dict[tuple[int, int], int] = {}
    for i, (u, v) in enumerate(edge_list):
        adj[u].append(v)
        adj[v].append(u)
        edge_idx[(min(u, v), max(u, v))] = i

    cycles_set = set()
    for start in range(num_vertices):
        # DFS: extend path until length 5, then check it closes back to start
        def dfs(path):
            if len(path) == 5:
                if start in adj[path[-1]]:
                    cycles_set.add(tuple(sorted(path)))
                return
            for nb in adj[path[-1]]:
                if nb not in path and nb > start:  # start is canonical (smallest)
                    dfs(path + [nb])
        dfs([start])

    # Convert each cycle (vertex tuple) to edge index list
    result = []
    for cyc in cycles_set:
        verts = list(cyc)
        # Find the cyclic ordering
        for perm_start in range(5):
            ordering = []
            current = verts[perm_start]
            remaining = set(verts) - {current}
            ok = True
            seq = [current]
            for _ in range(4):
                next_v = next((x for x in adj[current] if x in remaining), None)
                if next_v is None:
                    ok = False
                    break
                remaining.remove(next_v)
                seq.append(next_v)
                current = next_v
            if ok and verts[perm_start] in adj[current]:
                # Convert vertex sequence to edge index sequence
                eseq = []
                for k in range(5):
                    u, v = seq[k], seq[(k + 1) % 5]
                    eseq.append(edge_idx[(min(u, v), max(u, v))])
                result.append(sorted(eseq))
                break
    # Dedup
    seen = set()
    out = []
    for e in result:
        t = tuple(e)
        if t not in seen:
            seen.add(t)
            out.append(e)
    return out


# Recompute 5-cycles at module load
PETERSEN_5_CYCLES = find_5_cycles(EDGE_LIST, N_VERTICES)


# -------------------------------------------------------------------------
# Featurisation
# -------------------------------------------------------------------------
def featurise(state: str) -> np.ndarray:
    """
    Convert a 15-char board state into a feature vector.

    Features:
      [0:45]  per-edge state, 3-way one-hot (grey, green, purple) per edge
      [45:75] per-vertex degree counts, 3 per vertex (n_green, n_purple, n_coloured)
      [75:87] frustration indicators per 5-cycle (12 features): parity of purple
              count around the cycle (1.0 if odd, 0.0 if even, 0.5 if any grey)
      [87:90] grey count / 15, green count / 15, purple count / 15
      [90:92] parity bits — own_turn (always 1; the model only predicts AlphaQ moves)
              and grey_parity (1 if grey count odd)
    """
    feats = np.zeros(92, dtype=np.float32)

    # 1. Per-edge one-hot
    for i, c in enumerate(state):
        base = i * 3
        if c == '-':
            feats[base + 0] = 1.0
        elif c == 'G':
            feats[base + 1] = 1.0
        elif c == 'P':
            feats[base + 2] = 1.0

    # 2. Per-vertex degree counts
    for v in range(N_VERTICES):
        ng = np_ = nc = 0
        for ei, (u1, u2) in enumerate(EDGE_LIST):
            if v == u1 or v == u2:
                c = state[ei]
                if c == 'G':
                    ng += 1
                elif c == 'P':
                    np_ += 1
                if c != '-':
                    nc += 1
        feats[45 + v * 3 + 0] = ng
        feats[45 + v * 3 + 1] = np_
        feats[45 + v * 3 + 2] = nc

    # 3. Frustration indicators per 5-cycle
    for ci, cycle_edges in enumerate(PETERSEN_5_CYCLES[:12]):
        n_purple = 0
        any_grey = False
        for e in cycle_edges:
            c = state[e]
            if c == 'P':
                n_purple += 1
            elif c == '-':
                any_grey = True
                break
        if any_grey:
            feats[75 + ci] = 0.5
        else:
            feats[75 + ci] = 1.0 if (n_purple % 2 == 1) else 0.0

    # 4. Aggregate counts (normalised)
    n_grey = state.count('-')
    n_green = state.count('G')
    n_purple = state.count('P')
    feats[87] = n_grey / 15.0
    feats[88] = n_green / 15.0
    feats[89] = n_purple / 15.0

    # 5. Parity
    feats[90] = 1.0
    feats[91] = float(n_grey % 2)

    return feats


def legal_mask(state: str) -> np.ndarray:
    """Bool mask of size 30 indicating which (edge, color) actions are legal."""
    m = np.zeros(N_ACTIONS, dtype=bool)
    for e in range(N_EDGES):
        if state[e] == '-':
            m[e * 2] = True
            m[e * 2 + 1] = True
    return m


def action_to_label(edge: int, color: str) -> int:
    return edge * 2 + (0 if color == 'G' else 1)


def label_to_action(label: int) -> tuple[int, str]:
    return label // 2, ('G' if label % 2 == 0 else 'P')


# -------------------------------------------------------------------------
# Data extraction
# -------------------------------------------------------------------------
def extract_training_data(conn: sqlite3.Connection) -> dict:
    """Return arrays for X (features), y (labels), plus parallel game_ids and raw states."""
    rows = conn.execute("""
        SELECT g.id, m.rowid, m.player, m.edge, m.color, m.state_after
        FROM moves m
        JOIN games g ON g.id = m.game_id
        WHERE g.opponent = 'alphaq'
          AND g.result IS NOT NULL
          AND m.edge IS NOT NULL
          AND m.state_after IS NOT NULL
        ORDER BY g.id, m.rowid
    """).fetchall()

    X_list, y_list, gid_list, state_list = [], [], [], []
    current_game = None
    prev_state = INITIAL_STATE

    for game_id, rowid, player, edge, color, state_after in rows:
        if game_id != current_game:
            current_game = game_id
            prev_state = INITIAL_STATE

        if player == "opponent":
            X_list.append(featurise(prev_state))
            y_list.append(action_to_label(int(edge), color))
            gid_list.append(game_id)
            state_list.append(prev_state)
        prev_state = state_after

    return {
        "X": np.array(X_list, dtype=np.float32),
        "y": np.array(y_list, dtype=np.int64),
        "game_ids": np.array(gid_list),
        "states": np.array(state_list),
    }


def split_by_game(data: dict, test_frac: float = 0.20, seed: int = 42) -> dict:
    """Hold out 20% of games as test set (no within-game leakage)."""
    unique_games = np.unique(data["game_ids"])
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_games)
    n_test_games = int(len(unique_games) * test_frac)
    test_games = set(unique_games[:n_test_games])

    is_test = np.array([g in test_games for g in data["game_ids"]])
    return {
        "X_train": data["X"][~is_test],
        "y_train": data["y"][~is_test],
        "states_train": data["states"][~is_test],
        "X_test": data["X"][is_test],
        "y_test": data["y"][is_test],
        "states_test": data["states"][is_test],
        "n_train_games": len(unique_games) - n_test_games,
        "n_test_games": n_test_games,
    }


# -------------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------------
def masked_predict_proba(model, X: np.ndarray, states: np.ndarray,
                          n_classes: int = N_ACTIONS) -> np.ndarray:
    """Predict per-action probabilities with illegal actions zero'd and renormalised.

    Pads `model.predict_proba(X)` to a full (n, 30) array since classes
    not seen in training are missing from sklearn's output.
    """
    raw = model.predict_proba(X)
    full = np.zeros((X.shape[0], n_classes), dtype=np.float64)
    for j, cls in enumerate(model.classes_):
        full[:, int(cls)] = raw[:, j]

    for i, s in enumerate(states):
        m = legal_mask(s)
        full[i, ~m] = 0.0
        total = full[i].sum()
        if total > 0:
            full[i] /= total
    return full


def top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Fraction of samples where the true label is in the top-k predictions."""
    top_k = np.argsort(-proba, axis=1)[:, :k]
    hits = np.any(top_k == y_true[:, None], axis=1)
    return float(hits.mean())


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray,
              states_test: np.ndarray) -> dict:
    proba = masked_predict_proba(model, X_test, states_test)
    return {
        "top1": top_k_accuracy(proba, y_test, 1),
        "top3": top_k_accuracy(proba, y_test, 3),
        "top5": top_k_accuracy(proba, y_test, 5),
        "n_test": len(y_test),
    }


def per_grey_accuracy(model, X_test, y_test, states_test) -> dict[int, dict]:
    """Top-1 accuracy stratified by grey count."""
    proba = masked_predict_proba(model, X_test, states_test)
    top1 = np.argmax(proba, axis=1)
    correct = (top1 == y_test)
    out: dict[int, dict] = {}
    greys = np.array([s.count('-') for s in states_test])
    for g in sorted(set(greys.tolist())):
        sel = greys == g
        if sel.sum() > 0:
            out[g] = {"n": int(sel.sum()), "top1": float(correct[sel].mean())}
    return out


# -------------------------------------------------------------------------
# Model persistence — MATLAB-friendly
# -------------------------------------------------------------------------
def save_logreg_mat(model, scaler, path: Path) -> None:
    """Save logistic regression parameters in a .mat file consumable by MATLAB.

    Stores: feature mean and scale (for input normalisation), per-class weight
    matrix W (n_classes, n_features), bias b (n_classes,), and class_ids.
    """
    n_classes_full = N_ACTIONS
    W_full = np.zeros((n_classes_full, model.coef_.shape[1]), dtype=np.float64)
    b_full = np.zeros(n_classes_full, dtype=np.float64)
    for j, cls in enumerate(model.classes_):
        c = int(cls)
        W_full[c] = model.coef_[j]
        b_full[c] = model.intercept_[j] if model.intercept_.ndim > 0 else model.intercept_
    scipy.io.savemat(str(path), {
        "model_type": "logreg",
        "W": W_full,
        "b": b_full,
        "feature_mean": scaler.mean_,
        "feature_scale": scaler.scale_,
        "n_classes": n_classes_full,
        "n_features": int(model.coef_.shape[1]),
        "trained_classes": np.array([int(c) for c in model.classes_]),
    }, do_compression=True)


def save_mlp_mat(model, scaler, path: Path) -> None:
    """Save MLPClassifier weights for MATLAB consumption (manual forward pass)."""
    n_classes_full = N_ACTIONS
    # MLPClassifier may not have produced outputs for all 30 classes; we need
    # to remap output rows to full action space.
    out_idx = [int(c) for c in model.classes_]

    layers_W = [W.astype(np.float64) for W in model.coefs_]
    layers_b = [b.astype(np.float64) for b in model.intercepts_]

    # Final layer: pad output to all 30 actions
    W_out = layers_W[-1]    # (hidden, n_classes_trained)
    b_out = layers_b[-1]
    W_out_full = np.zeros((W_out.shape[0], n_classes_full), dtype=np.float64)
    b_out_full = np.zeros(n_classes_full, dtype=np.float64)
    for j, c in enumerate(out_idx):
        W_out_full[:, c] = W_out[:, j]
        b_out_full[c] = b_out[j]
    layers_W[-1] = W_out_full
    layers_b[-1] = b_out_full

    payload = {
        "model_type": "mlp",
        "activation": model.activation,
        "out_activation": model.out_activation_,
        "feature_mean": scaler.mean_,
        "feature_scale": scaler.scale_,
        "n_layers": len(layers_W),
        "trained_classes": np.array(out_idx),
    }
    for i, (W, b) in enumerate(zip(layers_W, layers_b)):
        payload[f"W{i+1}"] = W
        payload[f"b{i+1}"] = b
    scipy.io.savemat(str(path), payload, do_compression=True)


# -------------------------------------------------------------------------
# Report writer
# -------------------------------------------------------------------------
def write_model_card(report_data: dict) -> None:
    lines: list[str] = []
    push = lines.append

    push("# AlphaQ Predictive Policy Model — Phase 2 Model Card")
    push("")
    push("**Generated:** 2026-05-17")
    push("**Source corpus:** local `~/.tangled/game_stats.db`")
    push("**Method:** supervised classification on AlphaQ's empirical move "
         "distribution conditional on board state.")
    push("")
    push("---")
    push("")
    push("## Corpus and split")
    push("")
    push("| Metric | Value |")
    push("|--------|-------|")
    push(f"| AlphaQ games | {report_data['n_games']} |")
    push(f"| Decision observations | {report_data['n_decisions']} |")
    push(f"| Train games | {report_data['n_train_games']} |")
    push(f"| Test games | {report_data['n_test_games']} |")
    push(f"| Train moves | {report_data['n_train']} |")
    push(f"| Test moves | {report_data['n_test']} |")
    push(f"| Feature dimension | {report_data['n_features']} |")
    push(f"| Action space | {N_ACTIONS} (15 edges x 2 colors) |")
    push("")
    push("Train/test split is by game_id with 20% holdout, so test-set "
         "moves are from games not seen in training.")
    push("")
    push("---")
    push("")
    push("## Feature set")
    push("")
    push("| Block | Indices | Count | Description |")
    push("|-------|---------|-------|-------------|")
    push("| Per-edge state | 0..44 | 45 | 3-way one-hot (grey, green, purple) per edge |")
    push("| Per-vertex degree | 45..74 | 30 | green / purple / total coloured degree per vertex |")
    push("| 5-cycle frustration | 75..86 | 12 | per-cycle parity of purple count (1=frustrated, 0=satisfied, 0.5=incomplete) |")
    push("| Aggregate counts | 87..89 | 3 | grey / green / purple fractions of 15 |")
    push("| Parity | 90..91 | 2 | own-turn flag and grey-count parity |")
    push("")
    push("---")
    push("")
    push("## Model accuracy")
    push("")
    push("| Model | Top-1 | Top-3 | Top-5 | Training time (s) |")
    push("|-------|-------|-------|-------|-------------------|")
    for m in report_data["models"]:
        push(f"| {m['name']} | {m['top1']:.4f} | {m['top3']:.4f} | "
             f"{m['top5']:.4f} | {m['fit_time']:.2f} |")
    push("")
    push("Top-k is measured on held-out test games. Predictions are masked "
         "to legal actions (only grey edges) and renormalised before ranking.")
    push("")
    push("### Per-grey-count top-1 accuracy")
    push("")
    greys = sorted({g for m in report_data["models"] for g in m["by_grey"].keys()})
    header = "| Grey | " + " | ".join(m["name"] for m in report_data["models"]) + " | N test |"
    sep = "|------|" + "|".join(["---" for _ in report_data["models"]]) + "|--------|"
    push(header)
    push(sep)
    for g in greys:
        row = f"| {g} |"
        n_at_g = 0
        for m in report_data["models"]:
            if g in m["by_grey"]:
                row += f" {m['by_grey'][g]['top1']:.3f} |"
                n_at_g = m["by_grey"][g]["n"]
            else:
                row += " — |"
        row += f" {n_at_g} |"
        push(row)
    push("")
    push("---")
    push("")
    push("## Performance on exploit candidate states (Phase 1 output)")
    push("")
    push("Top-1 accuracy on the 6 exploit candidate states identified in "
         "Investigation 2. These are the states with n >= 10 observations "
         "and response entropy >= 0.5 bits — the primary search targets for "
         "Phase 4.")
    push("")
    push("| Model | Top-1 on exploit candidates | N test samples on these states |")
    push("|-------|-----------------------------|--------------------------------|")
    for m in report_data["models"]:
        n_ec = m["exploit_n"]
        acc = f"{m['exploit_top1']:.3f}" if n_ec > 0 else "—"
        push(f"| {m['name']} | {acc} | {n_ec} |")
    push("")
    push("If the model's top-1 accuracy on these states is meaningfully "
         "below its overall top-1, that confirms the entropy at these "
         "states is real (the model can't reduce it because AlphaQ "
         "genuinely picks differently). This is the favourable signal for "
         "Phase 4: the expected-value solver can exploit the response "
         "variance the model itself cannot collapse.")
    push("")
    push("---")
    push("")
    push("## Known failure modes")
    push("")
    push("1. **Class imbalance.** Some (edge, color) actions are rarely "
         "played by AlphaQ. Logistic regression's per-class weight is "
         "uniform; rare classes are under-predicted. MLP captures more "
         "but still penalises tail classes.")
    push("1. **Within-game correlation.** Consecutive AlphaQ moves in the "
         "same game share state ancestry. By-game splitting controls for "
         "this in test evaluation, but training samples are not strictly "
         "i.i.d.")
    push("1. **State coverage.** AlphaQ's reachable basin is narrow "
         "(see closure paper). The model's predictions outside this basin "
         "(e.g. for states reached via the expected-value solver in "
         "Phase 4) are extrapolation, not interpolation. Calibration may "
         "degrade.")
    push("1. **Quantum adjudicator unknown.** This model predicts AlphaQ's "
         "behaviour. It says nothing about whether a position is winning "
         "under the website's quantum scorer. Pairing with the calibrated "
         "oracle (Phase 3) is required for full Phase 4 utility.")
    push("")
    push("---")
    push("")
    push("## Persisted artefacts")
    push("")
    for m in report_data["models"]:
        push(f"- `snowdrop_tangled_agents/matlab/rl/data/alphaq_policy_{m['short']}.pkl` "
             "(sklearn pickle, Python use)")
        push(f"- `snowdrop_tangled_agents/matlab/rl/data/alphaq_policy_{m['short']}.mat` "
             "(weights for MATLAB Phase 4 consumption)")
    push(f"- `docs/ALPHAQ_PREDICTIVE_MODEL.md` (this file)")
    push("")
    push("---")
    push("")
    push("## Decision-gate interpretation")
    push("")
    overall_best = max(report_data["models"], key=lambda m: m["top1"])
    top1 = overall_best["top1"]
    if top1 >= 0.70:
        verdict = (
            f"**Strong predictor.** Best top-1 is {top1:.3f} ({overall_best['name']}). "
            "The model can be used directly as a hard policy approximation "
            "for AlphaQ in the Phase 4 expected-value solver."
        )
    elif top1 >= 0.40:
        verdict = (
            f"**Useful prior.** Best top-1 is {top1:.3f} ({overall_best['name']}). "
            "Use the predicted distribution as a soft prior in Phase 4, not "
            "a deterministic policy. Expected-value computation will marginalise "
            "over the residual uncertainty."
        )
    else:
        verdict = (
            f"**Weak predictor.** Best top-1 is {top1:.3f} ({overall_best['name']}). "
            "AlphaQ's policy is high-entropy from the model's perspective "
            "or the feature set is inadequate. Investigate before Phase 4: "
            "richer features (pairwise edge interactions, learned "
            "embeddings) or richer model class (deeper MLP) may help."
        )
    push(verdict)
    push("")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Model card written: {REPORT_PATH}")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: DB not found: {DB_PATH}", file=sys.stderr)
        return 1
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Petersen 5-cycles found: {len(PETERSEN_5_CYCLES)}", flush=True)

    print("Extracting training data...", flush=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        data = extract_training_data(conn)
    finally:
        conn.close()
    n_games_total = len(set(data["game_ids"]))
    print(f"  {len(data['y'])} decisions from {n_games_total} games "
          f"({data['X'].shape[1]} features)", flush=True)

    print("Splitting by game (80/20)...", flush=True)
    split = split_by_game(data, test_frac=0.20, seed=42)
    print(f"  train: {len(split['y_train'])} moves from "
          f"{split['n_train_games']} games", flush=True)
    print(f"  test : {len(split['y_test'])} moves from "
          f"{split['n_test_games']} games", flush=True)

    # Standardise features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(split["X_train"])
    X_test_s = scaler.transform(split["X_test"])

    exploit_states = {
        "PPGPGGG--------", "PGPGGGG--------", "PPPPGGG--------",
        "GPGPGGGP-PP-PGP", "PGPGGGP--------", "P-G-P---GGPGPPP",
    }
    is_exploit_state = np.array([s in exploit_states for s in split["states_test"]])
    print(f"  exploit-candidate test samples: {is_exploit_state.sum()}", flush=True)

    models_to_train = [
        ("LogReg",
         "logreg",
         LogisticRegression(
             solver="lbfgs", max_iter=2000, C=1.0, n_jobs=-1,
         )),
        ("MLP (64,64)",
         "mlp",
         MLPClassifier(
             hidden_layer_sizes=(64, 64), activation="relu",
             solver="adam", max_iter=300, early_stopping=True,
             validation_fraction=0.1, random_state=42, verbose=False
         )),
    ]

    report_models: list[dict] = []
    for name, short, model in models_to_train:
        print(f"\nTraining {name}...", flush=True)
        t0 = time.perf_counter()
        model.fit(X_train_s, split["y_train"])
        fit_time = time.perf_counter() - t0
        print(f"  fit in {fit_time:.2f}s", flush=True)

        ev = evaluate(model, X_test_s, split["y_test"], split["states_test"])
        by_grey = per_grey_accuracy(model, X_test_s, split["y_test"], split["states_test"])

        # Exploit-state accuracy
        if is_exploit_state.sum() > 0:
            proba_e = masked_predict_proba(
                model, X_test_s[is_exploit_state],
                split["states_test"][is_exploit_state],
            )
            exploit_top1 = top_k_accuracy(proba_e, split["y_test"][is_exploit_state], 1)
        else:
            exploit_top1 = 0.0

        print(f"  top-1: {ev['top1']:.4f}  top-3: {ev['top3']:.4f}  "
              f"top-5: {ev['top5']:.4f}", flush=True)
        print(f"  top-1 on exploit candidates: {exploit_top1:.4f} "
              f"({int(is_exploit_state.sum())} samples)", flush=True)

        # Persist
        pkl_path = DATA_DIR / f"alphaq_policy_{short}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump({"model": model, "scaler": scaler}, f)
        mat_path = DATA_DIR / f"alphaq_policy_{short}.mat"
        if short == "logreg":
            save_logreg_mat(model, scaler, mat_path)
        else:
            save_mlp_mat(model, scaler, mat_path)
        print(f"  saved: {pkl_path.name}, {mat_path.name}", flush=True)

        report_models.append({
            "name": name, "short": short, "fit_time": fit_time,
            "top1": ev["top1"], "top3": ev["top3"], "top5": ev["top5"],
            "by_grey": by_grey,
            "exploit_top1": exploit_top1,
            "exploit_n": int(is_exploit_state.sum()),
        })

    write_model_card({
        "n_games": n_games_total,
        "n_decisions": len(data["y"]),
        "n_train_games": split["n_train_games"],
        "n_test_games": split["n_test_games"],
        "n_train": len(split["y_train"]),
        "n_test": len(split["y_test"]),
        "n_features": data["X"].shape[1],
        "models": report_models,
    })

    best = max(report_models, key=lambda m: m["top1"])
    print()
    print("=" * 60)
    print(f"BEST: {best['name']}  top-1={best['top1']:.4f}  "
          f"top-3={best['top3']:.4f}  top-5={best['top5']:.4f}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
