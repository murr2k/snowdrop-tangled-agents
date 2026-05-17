"""Quick inspection of the Phase 2 .mat files to confirm shape/dtype match
what AlphaQPolicy.m expects."""

import scipy.io


def main() -> int:
    for path in (
        "snowdrop_tangled_agents/matlab/rl/data/alphaq_policy_mlp.mat",
        "snowdrop_tangled_agents/matlab/rl/data/alphaq_policy_logreg.mat",
    ):
        print(f"=== {path}")
        m = scipy.io.loadmat(path)
        for k, v in m.items():
            if k.startswith("__"):
                continue
            sh = getattr(v, "shape", None)
            dt = getattr(v, "dtype", None)
            print(f"  {k}: shape={sh} dtype={dt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
