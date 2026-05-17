"""Dump Petersen 5-cycle list and reference featurisation values for the
MATLAB-side AlphaQPolicy implementation. Used to hand-verify that the
MATLAB featuriser produces identical output to the Python one."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_alphaq_policy import (
    EDGE_LIST,
    PETERSEN_5_CYCLES,
    featurise,
    find_5_cycles,
    N_VERTICES,
)

import numpy as np


def main() -> int:
    print("Edge list:")
    for i, (u, v) in enumerate(EDGE_LIST):
        print(f"  E{i}: ({u},{v})")
    print()
    print(f"Petersen 5-cycles enumerated ({len(PETERSEN_5_CYCLES)}):")
    for i, c in enumerate(PETERSEN_5_CYCLES):
        print(f"  C{i+1}: {c}")
    print()
    samples = ["-" * 15, "PPGPGGG--------", "GPGPGGGP-PP-PGP"]
    for s in samples:
        f = featurise(s)
        print(f"State {s!r}:")
        print(f"  edge_ohe (0..44) sum={float(f[:45].sum()):.1f}")
        print(f"  vertex_deg (45..74)={f[45:75].astype(int).tolist()}")
        print(f"  cycle_parity (75..86)={f[75:87].tolist()}")
        print(f"  aggregates (87..89)={[round(x,5) for x in f[87:90].tolist()]}")
        print(f"  parity bits (90..91)={f[90:92].tolist()}")
        print(f"  sum={float(f.sum()):.5f}  norm={float(np.linalg.norm(f)):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
