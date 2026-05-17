"""Sanity-check that loading the .mat policy file and doing the forward
pass by hand (numpy, the same algebra as AlphaQPolicy.predict) produces
the same probabilities as sklearn's predict_proba on the .pkl version.

If this passes, the MATLAB code path will produce the same numbers
provided the MATLAB featuriser matches the Python featuriser (verified
separately by the unit test against hand-checked reference values)."""

import pickle
from pathlib import Path

import numpy as np
import scipy.io

import sys
sys.path.insert(0, str(Path(__file__).parent))
from train_alphaq_policy import featurise, legal_mask, masked_predict_proba

DATA_DIR = Path(__file__).parent.parent / "snowdrop_tangled_agents" / "matlab" / "rl" / "data"


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def mlp_forward(mat: dict, x_raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mean = mat["feature_mean"].ravel()
    scale = mat["feature_scale"].ravel()
    scale = np.where(scale < 1e-12, 1.0, scale)
    x = (x_raw - mean) / scale
    h1 = np.maximum(0, x @ mat["W1"] + mat["b1"].ravel())
    h2 = np.maximum(0, h1 @ mat["W2"] + mat["b2"].ravel())
    z = h2 @ mat["W3"] + mat["b3"].ravel()
    z_masked = np.where(mask, z, -np.inf)
    return softmax(z_masked)


def logreg_forward(mat: dict, x_raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mean = mat["feature_mean"].ravel()
    scale = mat["feature_scale"].ravel()
    scale = np.where(scale < 1e-12, 1.0, scale)
    x = (x_raw - mean) / scale
    W = mat["W"]  # (30, 92)
    b = mat["b"].ravel()  # (30,)
    z = W @ x + b
    z_masked = np.where(mask, z, -np.inf)
    return softmax(z_masked)


def main() -> int:
    states = [
        "PPGPGGG--------",
        "GPGPGGGP-PP-PGP",
        "P-G-P---GGPGPPP",
    ]

    for short, fwd, pkl_name, mat_name in [
        ("MLP", mlp_forward, "alphaq_policy_mlp.pkl", "alphaq_policy_mlp.mat"),
        ("LogReg", logreg_forward, "alphaq_policy_logreg.pkl", "alphaq_policy_logreg.mat"),
    ]:
        print(f"\n=== {short}")
        with open(DATA_DIR / pkl_name, "rb") as f:
            pk = pickle.load(f)
        model, scaler = pk["model"], pk["scaler"]
        mat = scipy.io.loadmat(str(DATA_DIR / mat_name))

        max_err = 0.0
        for s in states:
            x_raw = featurise(s)
            mask = legal_mask(s)

            # sklearn path
            X = np.atleast_2d(x_raw)
            Xs = scaler.transform(X)
            sk_proba = masked_predict_proba(model, Xs, np.array([s]))[0]

            # numpy-on-.mat path
            ny_proba = fwd(mat, x_raw, mask)

            err = float(np.max(np.abs(sk_proba - ny_proba)))
            max_err = max(max_err, err)
            top_sk = int(np.argmax(sk_proba))
            top_ny = int(np.argmax(ny_proba))
            agree = top_sk == top_ny
            print(f"  state={s!r}  max|sk-ny|={err:.3e}  argmax sk={top_sk} ny={top_ny}  "
                  f"agree={agree}")

        print(f"  worst max abs error: {max_err:.3e}")
        # 1e-5 tolerates float32/float64 round-trip drift; argmax
        # decisions must always agree, which is what matters for the solver.
        assert max_err < 1e-5, f"{short} mat path disagrees with sklearn by {max_err}"
    print("\nAll parity checks PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
