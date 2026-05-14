#!/usr/bin/env python3
"""
generate_sa_oracle.py — Autonomous offline SA oracle generator for Tangled.

Computes exact minimax-optimal scores for all Tangled game positions with
0 through 9 grey edges using retrograde dynamic programming (backward induction).

ALGORITHM
---------
Starting from SA-adjudicated terminal scores (level 0), build level k from
level k-1 using the minimax recurrence:

  V(state, k grey) = max_{move} V(child)   if k is ODD  (P1 maximizes)
                   = min_{move} V(child)   if k is EVEN (P2 minimizes)

Turn order (P1 plays first in 15-move game):
  grey=15 -> P1 | grey=14 -> P2 | grey=13 -> P1 | ... | grey=1 -> P1 | grey=0 terminal

STATE ENCODING
--------------
A state with k grey edges is represented by:
  - combo: which k edge positions (0-14) are grey, stored as sorted tuple
  - base_idx: 15-bit integer, bit j = 1 if edge j is Green; grey bits = 0 (P)
  - flat index = base_idx * C(15,k) + combo_idx  (matches MATLAB ExpandedLUT convention)

Array layout: level_k has shape (32768, C(15,k)) float32.
  Axis 0 = base_idx, Axis 1 = combo_idx.
  Flat index formula ensures MATLAB ExpandedLUT.m can load directly.

KNOWN BUG IN EXISTING DATA (levels 1 and 2)
--------------------------------------------
expanded_lut_sa.mat levels 1-2 have WRONG turn conventions:
  - Level 1 uses min() but P1 should maximize at grey=1 (k=1, odd)
  - Level 2 has P1 maximize first but P2 should move first at grey=2 (k=2, even)
This script recomputes levels 0-9 correctly from terminal_scores_sa.mat.

OUTPUT
------
Checkpoints: data/oracle_sa_level_{k}.npy  (one per level, resumable)
Final:       data/oracle_sa.npz            (all levels, pure numpy)
             data/expanded_lut_sa.mat      (overwritten with corrected + extended data)

USAGE
-----
    python scripts/generate_sa_oracle.py [--start-level N] [--end-level N] [--validate]

    --start-level N   Resume from level N (default: auto-detect from checkpoints)
    --end-level N     Stop after level N (default: 9)
    --validate        After generation, validate values against known positions
    --skip-mat        Skip writing the MATLAB .mat file (faster, numpy-only output)
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

# Optional: scipy for reading MATLAB v5 files, h5py for v7.3 (HDF5)
try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "snowdrop_tangled_agents" / "matlab" / "rl" / "data"

TERMINAL_SA_MAT = DATA_DIR / "terminal_scores_sa.mat"
EXPANDED_LUT_SA_MAT = DATA_DIR / "expanded_lut_sa.mat"
ORACLE_NPZ = DATA_DIR / "oracle_sa.npz"

CHECKPOINT_PATTERN = str(DATA_DIR / "oracle_sa_level_{k}.npy")


# ---------------------------------------------------------------------------
# State encoding utilities
# ---------------------------------------------------------------------------

NUM_EDGES = 15
NUM_TERMINAL = 32768  # 2^15


def state_str_to_base_idx(state: str) -> int:
    """Convert 15-char state string (G/P/-) to 0-indexed base integer.

    Grey edges (-) are treated as Purple (bit=0), consistent with ExpandedLUT.m.
    """
    idx = 0
    for j, c in enumerate(state):
        if c == 'G':
            idx |= (1 << j)
    return idx


def base_idx_to_state_str(base_idx: int) -> str:
    """Convert 0-indexed base integer to 15-char state string (G/P only)."""
    return ''.join('G' if (base_idx >> j) & 1 else 'P' for j in range(NUM_EDGES))


def combo_idx_map(k: int) -> dict[tuple, int]:
    """Return mapping from sorted combo tuple -> 0-indexed position in combinations(range(15), k)."""
    return {c: i for i, c in enumerate(itertools.combinations(range(NUM_EDGES), k))}


def combos_list(k: int) -> list[tuple]:
    """Return list of all C(15,k) combinations of grey positions (0-indexed, sorted)."""
    return list(itertools.combinations(range(NUM_EDGES), k))


# ---------------------------------------------------------------------------
# Terminal LUT loading
# ---------------------------------------------------------------------------

def load_terminal_lut() -> np.ndarray:
    """Load SA terminal scores from terminal_scores_sa.mat.

    Returns float32 array of shape (32768,) indexed by base_idx.
    """
    if not TERMINAL_SA_MAT.exists():
        raise FileNotFoundError(f"Terminal LUT not found: {TERMINAL_SA_MAT}")

    # Try scipy first (MATLAB v5 format)
    if HAS_SCIPY:
        try:
            data = scipy.io.loadmat(str(TERMINAL_SA_MAT))
            # Key is 'terminal_scores' based on generate_expanded_lut.m convention
            for key in ('terminal_scores', 'terminalLUT', 'terminal_lut'):
                if key in data:
                    scores = data[key].flatten().astype(np.float32)
                    if len(scores) == NUM_TERMINAL:
                        print(f"  Loaded terminal LUT via scipy ({key}): "
                              f"{len(scores)} entries, "
                              f"range [{scores.min():.3f}, {scores.max():.3f}]")
                        return scores
        except Exception as e:
            print(f"  scipy.io failed ({e}), trying h5py...")

    # Fall back to h5py (MATLAB v7.3 / HDF5)
    if HAS_H5PY:
        try:
            with h5py.File(str(TERMINAL_SA_MAT), 'r') as f:
                for key in ('terminal_scores', 'terminalLUT', 'terminal_lut'):
                    if key in f:
                        scores = f[key][()].flatten().astype(np.float32)
                        if len(scores) == NUM_TERMINAL:
                            print(f"  Loaded terminal LUT via h5py ({key}): "
                                  f"{len(scores)} entries, "
                                  f"range [{scores.min():.3f}, {scores.max():.3f}]")
                            return scores
        except Exception as e:
            print(f"  h5py failed: {e}")

    # Last resort: try to extract terminalLUT from expanded_lut_sa.mat
    if EXPANDED_LUT_SA_MAT.exists() and HAS_H5PY:
        print(f"  Falling back to terminalLUT from {EXPANDED_LUT_SA_MAT.name}")
        try:
            with h5py.File(str(EXPANDED_LUT_SA_MAT), 'r') as f:
                if 'terminalLUT' in f:
                    scores = f['terminalLUT'][()].flatten().astype(np.float32)
                    if len(scores) == NUM_TERMINAL:
                        print(f"  Loaded terminalLUT from expanded_lut_sa.mat: "
                              f"range [{scores.min():.3f}, {scores.max():.3f}]")
                        return scores
        except Exception as e:
            print(f"  h5py fallback failed: {e}")

    raise RuntimeError(
        "Could not load terminal LUT. Need scipy or h5py, and either "
        "terminal_scores_sa.mat or expanded_lut_sa.mat (with terminalLUT key)."
    )


# ---------------------------------------------------------------------------
# Retrograde DP — core computation
# ---------------------------------------------------------------------------

def compute_level(k: int, prev_level: np.ndarray, prev_combo_map: dict) -> np.ndarray:
    """Compute minimax oracle for level k (k grey edges) from level k-1.

    Parameters
    ----------
    k : int
        Current level (1 to 9).
    prev_level : np.ndarray
        Oracle for level k-1. Shape depends on k:
          k=1: (32768,) flat terminal array (level 0)
          k≥2: (32768, C(15,k-1)) array
    prev_combo_map : dict
        Mapping from child combo tuple -> combo_idx in prev_level.
        For k=1: ignored (prev_level is flat terminal array).

    Returns
    -------
    np.ndarray of shape (32768, C(15,k)), dtype float32.
        level_k[base_idx, combo_idx] = minimax value for state where:
          - grey edges are at positions combos_k[combo_idx] (0-indexed)
          - non-grey edges are encoded in base_idx (bit j=1 -> Green)
    """
    combos_k = combos_list(k)
    n_combos = len(combos_k)
    is_p1_turn = (k % 2 == 1)  # True -> P1 maximizes, False -> P2 minimizes

    level_k = np.empty((NUM_TERMINAL, n_combos), dtype=np.float32)

    # Pre-build green_child_bases lookup: for each edge position ei,
    # setting bit ei to 1. Vectorized over all 32768 base patterns at once.
    all_bases = np.arange(NUM_TERMINAL, dtype=np.int32)

    t0 = time.perf_counter()
    report_every = max(1, n_combos // 20)

    for combo_idx, c in enumerate(combos_k):
        # Collect minimax child values across all 2k moves (k positions × 2 colors)
        # Each entry is a (32768,) float32 array
        child_value_arrays: list[np.ndarray] = []

        for ei in c:
            child_c = tuple(e for e in c if e != ei)  # remove ei from grey set

            if k == 1:
                # prev_level is flat (32768,) terminal array
                # Green child: set bit ei to 1
                green_bases = all_bases | (1 << ei)
                child_value_arrays.append(prev_level[green_bases])

                # Purple child: bit ei already 0 for valid bases
                child_value_arrays.append(prev_level[all_bases])  # = prev_level[:]
            else:
                child_combo_idx = prev_combo_map[child_c]

                # Green child: base_idx with bit ei = 1
                green_bases = all_bases | (1 << ei)
                child_value_arrays.append(prev_level[green_bases, child_combo_idx])

                # Purple child: base_idx unchanged (bit ei stays 0 for valid bases)
                child_value_arrays.append(prev_level[:, child_combo_idx])

        # Stack all 2k child value arrays -> (2k, 32768), then reduce
        stacked = np.stack(child_value_arrays, axis=0)  # (2k, 32768)

        if is_p1_turn:
            level_k[:, combo_idx] = stacked.max(axis=0)
        else:
            level_k[:, combo_idx] = stacked.min(axis=0)

        if (combo_idx + 1) % report_every == 0 or combo_idx == n_combos - 1:
            elapsed = time.perf_counter() - t0
            pct = (combo_idx + 1) / n_combos * 100
            eta = elapsed / (combo_idx + 1) * (n_combos - combo_idx - 1)
            print(f"    {combo_idx+1:5d}/{n_combos} ({pct:5.1f}%)  "
                  f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s", end='\r', flush=True)

    print()  # newline after progress
    return level_k


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def checkpoint_path(k: int) -> Path:
    return Path(CHECKPOINT_PATTERN.format(k=k))


def load_checkpoint(k: int) -> np.ndarray | None:
    p = checkpoint_path(k)
    if p.exists():
        data = np.load(str(p))
        print(f"  Loaded checkpoint: level {k} from {p.name}")
        return data
    return None


def save_checkpoint(k: int, level_data: np.ndarray) -> None:
    p = checkpoint_path(k)
    np.save(str(p), level_data)
    size_mb = level_data.nbytes / 1024 / 1024
    print(f"  Checkpoint saved: {p.name} ({size_mb:.1f} MB)")


def detect_start_level(end_level: int) -> int:
    """Find highest completed checkpoint to resume from."""
    for k in range(end_level, -1, -1):
        if checkpoint_path(k).exists():
            return k
    return -1


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_npz(levels: dict[int, np.ndarray]) -> None:
    """Write all levels to a single .npz file."""
    arrays = {}
    for k, data in levels.items():
        arrays[f'level_{k}'] = data
        combos = np.array(list(itertools.combinations(range(NUM_EDGES), k)), dtype=np.int8)
        arrays[f'combos_{k}'] = combos

    # Level 0 is flat; combos_0 is a single empty combo
    if 0 in levels:
        arrays['combos_0'] = np.zeros((1, 0), dtype=np.int8)

    np.savez_compressed(str(ORACLE_NPZ), **arrays)
    size_mb = ORACLE_NPZ.stat().st_size / 1024 / 1024
    print(f"  Saved {ORACLE_NPZ.name} ({size_mb:.1f} MB)")


def write_mat(levels: dict[int, np.ndarray]) -> None:
    """Write extended expanded_lut_sa.mat with corrected levels 0-9.

    Overwrites the existing file. Uses h5py for MATLAB v7.3 HDF5 format
    to handle files > 2 GB.
    """
    if not HAS_H5PY:
        print("  SKIP: h5py not available, cannot write .mat file")
        return

    print(f"  Writing {EXPANDED_LUT_SA_MAT.name} ...")

    with h5py.File(str(EXPANDED_LUT_SA_MAT), 'w') as f:
        # MATLAB v7.3 marker
        f.attrs['MATLAB_class'] = np.bytes_('double')

        def write_dataset(name: str, data: np.ndarray, chunks: bool = True) -> None:
            opts = {}
            if chunks and data.nbytes > 1024 * 1024:
                opts['compression'] = 'gzip'
                opts['compression_opts'] = 4
                opts['chunks'] = True
            f.create_dataset(name, data=data, **opts)

        def write_combo_array(name: str, combos_1indexed: np.ndarray) -> None:
            # MATLAB reads HDF5 with dimensions reversed (column-major convention).
            # Write combos.T so MATLAB sees shape (n_combos, k) instead of (k, n_combos).
            write_dataset(name, combos_1indexed.T.astype(np.float64), chunks=False)

        # Level 0 — terminal LUT (shape: (32768,) stored flat)
        if 0 in levels:
            terminal = levels[0].flatten()
            write_dataset('terminalLUT', terminal.astype(np.float64))

        # Level 1 — one grey (shape: (32768*15,) = (491520,))
        if 1 in levels:
            one_grey = levels[1].flatten(order='C').astype(np.float32)
            write_dataset('oneGreyScores', one_grey)

        # Level 2 — two grey (shape: (32768*105,) = (3440640,))
        if 2 in levels:
            two_grey = levels[2].flatten(order='C').astype(np.float32)
            write_dataset('twoGreyScores', two_grey)
            pairs = np.array(list(itertools.combinations(range(1, NUM_EDGES + 1), 2)))
            write_combo_array('greyPairs', pairs)

        # Level 3 — three grey (shape: (32768*455,))
        if 3 in levels:
            three_grey = levels[3].flatten(order='C').astype(np.float32)
            write_dataset('threeGreyScores', three_grey)
            triples = np.array(list(itertools.combinations(range(1, NUM_EDGES + 1), 3)))
            write_combo_array('greyTriples', triples)

        # Level 4 — four grey (shape: (32768*1365,))
        if 4 in levels:
            four_grey = levels[4].flatten(order='C').astype(np.float32)
            write_dataset('fourGreyScores', four_grey)
            quads = np.array(list(itertools.combinations(range(1, NUM_EDGES + 1), 4)))
            write_combo_array('greyQuads', quads)

        # Levels 5-9 — new extensions beyond ExpandedLUT.m's current coverage
        level_names = {
            5: ('fiveGreyScores', 'greyFives'),
            6: ('sixGreyScores', 'greySixes'),
            7: ('sevenGreyScores', 'greySevens'),
            8: ('eightGreyScores', 'greyEights'),
            9: ('nineGreyScores', 'greyNines'),
        }
        for k, (scores_key, combos_key) in level_names.items():
            if k in levels:
                scores_flat = levels[k].flatten(order='C').astype(np.float32)
                write_dataset(scores_key, scores_flat)
                combo_arr = np.array(
                    list(itertools.combinations(range(1, NUM_EDGES + 1), k))
                )
                write_combo_array(combos_key, combo_arr)

        # Metadata
        import datetime
        meta = f"oracle_sa v2.0 generated {datetime.datetime.now().isoformat()} | " \
               f"turn-order: k-odd=P1-max, k-even=P2-min | " \
               f"levels: {sorted(levels.keys())}"
        f.create_dataset('metadata', data=np.bytes_(meta))

    size_mb = EXPANDED_LUT_SA_MAT.stat().st_size / 1024 / 1024
    print(f"  Wrote {EXPANDED_LUT_SA_MAT.name} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_oracle(levels: dict[int, np.ndarray]) -> None:
    """Spot-check oracle values for internal consistency."""
    print("\n=== Validation ===")
    errors = 0

    # Check 1: level 1 value ≥ min(level 0) and ≤ max(level 0)
    if 0 in levels and 1 in levels:
        l0 = levels[0]
        l1 = levels[1]
        out_of_range = np.sum((l1 < l0.min()) | (l1 > l0.max()))
        print(f"  Level 1 values out of terminal range: {out_of_range}")
        if out_of_range:
            errors += 1

    # Check 2: for a specific terminal state, verify level 0 matches expectation
    # All-Green state (base_idx = 32767): all edges G -> check it's loaded
    if 0 in levels:
        all_green_score = levels[0][32767]
        all_purple_score = levels[0][0]
        print(f"  Level 0 all-Green score: {all_green_score:.4f}")
        print(f"  Level 0 all-Purple score: {all_purple_score:.4f}")

    # Check 3: level 1 should equal max(G, P) completions (P1 maximizes at k=1)
    # Verify for a specific combo
    if 0 in levels and 1 in levels:
        # Combo 0 = grey at edge 0
        # For base_idx = 0 (all edges P, edge 0 grey):
        # Green completion: set edge 0 to G -> base_idx = 1
        # Purple completion: edge 0 stays P -> base_idx = 0
        green_score = float(levels[0][1])  # base_idx=1: edge 0 = G
        purple_score = float(levels[0][0])  # base_idx=0: edge 0 = P
        expected = max(green_score, purple_score)  # P1 maximizes at k=1
        combo_idx_0 = 0  # combo (0,) is the first combination
        actual = float(levels[1][0, combo_idx_0])  # base_idx=0, combo_idx=0
        match = abs(actual - expected) < 1e-4
        print(f"  Level 1 sanity check (grey=edge0, all-P base): "
              f"expected={expected:.4f}, got={actual:.4f} -> {'OK' if match else 'FAIL'}")
        if not match:
            errors += 1

    # Check 4: level 2 should equal min(over P2 moves) of max(over P1 responses)
    # At k=2, P2 minimizes; then at k=1, P1 maximizes.
    if 0 in levels and 1 in levels and 2 in levels:
        combo_map_1 = combo_idx_map(1)
        # Combo (0,1) = grey at edges 0 and 1, base_idx=0 (all non-grey = P)
        # P2 moves (k=2 even -> minimize): colors one of {edge0, edge1} with G or P
        # Then P1 moves (k=1 odd -> maximize): colors the remaining edge
        #
        # P2 options from base_idx=0, grey={0,1}:
        #   E0->G: child base=1, grey={1} -> P1 maximizes -> level_1[1, combo_idx(1,)]
        #   E0->P: child base=0, grey={1} -> P1 maximizes -> level_1[0, combo_idx(1,)]
        #   E1->G: child base=2, grey={0} -> P1 maximizes -> level_1[2, combo_idx(0,)]
        #   E1->P: child base=0, grey={0} -> P1 maximizes -> level_1[0, combo_idx(0,)]
        #
        # level_1[b, combo_idx(ei,)] = max(level_0[b|(1<<ei)], level_0[b])

        l0 = levels[0]
        l1 = levels[1]
        ci_0 = combo_map_1[(0,)]
        ci_1 = combo_map_1[(1,)]

        p2_opts = [
            l1[1, ci_1],   # E0->G: child base=1, grey={1}
            l1[0, ci_1],   # E0->P: child base=0, grey={1}
            l1[2, ci_0],   # E1->G: child base=2, grey={0}
            l1[0, ci_0],   # E1->P: child base=0, grey={0}
        ]
        expected = min(float(v) for v in p2_opts)  # P2 minimizes

        combo_idx_01 = list(itertools.combinations(range(15), 2)).index((0, 1))
        actual = float(levels[2][0, combo_idx_01])
        match = abs(actual - expected) < 1e-4
        print(f"  Level 2 sanity check (grey=edges 0,1, all-P base): "
              f"expected={expected:.4f}, got={actual:.4f} -> {'OK' if match else 'FAIL'}")
        if not match:
            errors += 1

    # Check 5: monotonicity — with more grey edges, uncertainty should not artificially
    # inflate range beyond level 0 bounds
    for k in sorted(levels.keys()):
        lk = levels[k]
        l0 = levels[0]
        out_of_range = np.sum((lk < l0.min() - 0.01) | (lk > l0.max() + 0.01))
        pct = 100 * out_of_range / lk.size
        print(f"  Level {k}: out-of-terminal-range entries: {out_of_range} ({pct:.1f}%)")

    print(f"\n  Validation {'PASSED' if errors == 0 else f'FAILED ({errors} errors)'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start-level', type=int, default=None,
                   help='Force start from this level (default: auto-detect from checkpoints)')
    p.add_argument('--end-level', type=int, default=9,
                   help='Stop after computing this level (default: 9)')
    p.add_argument('--validate', action='store_true',
                   help='Run validation checks after generation')
    p.add_argument('--skip-mat', action='store_true',
                   help='Skip writing expanded_lut_sa.mat (write only .npz)')
    p.add_argument('--skip-npz', action='store_true',
                   help='Skip writing oracle_sa.npz (write only .mat)')
    return p.parse_args()


def print_size_table(end_level: int) -> None:
    """Print expected array sizes for each level."""
    from math import comb
    print("\nExpected sizes (32768 × C(15,k) × 4 bytes float32):")
    total_mb = 0
    for k in range(0, end_level + 1):
        n = comb(15, k)
        mb = 32768 * n * 4 / 1024 / 1024
        if k == 0:
            mb = 32768 * 8 / 1024 / 1024  # float64 for terminal
        total_mb += mb
        player = "P1 max" if k % 2 == 1 else ("P2 min" if k > 0 else "terminal")
        print(f"  Level {k:2d} (k={k:2d}, C(15,{k:2d})={n:5d}): {mb:7.1f} MB  [{player}]")
    print(f"  TOTAL: {total_mb:.1f} MB")


def main() -> None:
    args = parse_args()
    end_level = args.end_level

    print("=" * 60)
    print("SA Oracle Generator for Tangled (levels 0-9)")
    print("=" * 60)
    print_size_table(end_level)
    print()

    if not HAS_SCIPY and not HAS_H5PY:
        print("ERROR: Need scipy or h5py to load terminal LUT.")
        print("  pip install scipy   # OR")
        print("  pip install h5py")
        sys.exit(1)

    if not HAS_H5PY and not args.skip_mat:
        print("WARNING: h5py not available — cannot write .mat file.")
        print("  Install with: pip install h5py")
        print("  Proceeding with --skip-mat mode.")
        args.skip_mat = True

    # Load all completed checkpoints into memory
    loaded_levels: dict[int, np.ndarray] = {}

    # Auto-detect resume point
    if args.start_level is not None:
        resume_from = args.start_level - 1
    else:
        resume_from = detect_start_level(end_level)

    print(f"\nResume detection: highest completed checkpoint = level {resume_from}")

    # Load checkpoints for levels 0..resume_from
    for k in range(0, resume_from + 1):
        ck = load_checkpoint(k)
        if ck is not None:
            loaded_levels[k] = ck
        else:
            print(f"  WARNING: checkpoint for level {k} not found, will recompute from level 0")
            resume_from = -1
            loaded_levels = {}
            break

    # If level 0 not in loaded_levels, load terminal LUT
    if 0 not in loaded_levels:
        print("\n[Level 0] Loading terminal SA scores...")
        terminal = load_terminal_lut()
        loaded_levels[0] = terminal
        save_checkpoint(0, terminal)

    # Main retrograde loop
    for k in range(1, end_level + 1):
        if k in loaded_levels:
            print(f"[Level {k}] Skipping (checkpoint exists)")
            continue

        player = "P1 (maximizes)" if k % 2 == 1 else "P2 (minimizes)"
        from math import comb
        n_combos = comb(NUM_EDGES, k)
        n_bases = NUM_TERMINAL
        est_mb = n_bases * n_combos * 4 / 1024 / 1024

        print(f"\n[Level {k}] Computing {n_bases}×{n_combos} states "
              f"({est_mb:.1f} MB) — {player}")

        prev_level = loaded_levels[k - 1]

        if k == 1:
            prev_combo_map = {}  # Not used for k=1
        else:
            prev_combo_map = combo_idx_map(k - 1)

        t_start = time.perf_counter()
        level_k = compute_level(k, prev_level, prev_combo_map)
        elapsed = time.perf_counter() - t_start

        print(f"  Computed in {elapsed:.1f}s | "
              f"value range: [{level_k.min():.3f}, {level_k.max():.3f}]")

        save_checkpoint(k, level_k)
        loaded_levels[k] = level_k

        # Free previous level from memory if no longer needed
        # Keep level k-1 in case of forward reference, but release k-2
        if k >= 2 and (k - 2) in loaded_levels and k - 2 > 0:
            del loaded_levels[k - 2]
            print(f"  Released level {k-2} from memory")

    print(f"\n=== Generation complete (levels 0-{end_level}) ===")

    # Reload all levels for output (may need to reload freed ones)
    all_levels: dict[int, np.ndarray] = {}
    for k in range(0, end_level + 1):
        if k in loaded_levels:
            all_levels[k] = loaded_levels[k]
        else:
            ck = load_checkpoint(k)
            if ck is not None:
                all_levels[k] = ck
            else:
                print(f"  WARNING: missing level {k}, skipping in output")

    if args.validate:
        validate_oracle(all_levels)

    # Write outputs
    print("\n=== Writing output files ===")
    if not args.skip_npz:
        print(f"Writing {ORACLE_NPZ.name}...")
        write_npz(all_levels)

    if not args.skip_mat:
        print(f"Writing {EXPANDED_LUT_SA_MAT.name} (overwrite with corrected + extended)...")
        write_mat(all_levels)

    print("\nDone.")
    print(f"Checkpoint files: {DATA_DIR}/oracle_sa_level_{{0..{end_level}}}.npy")
    if not args.skip_npz:
        print(f"Oracle (numpy):   {ORACLE_NPZ}")
    if not args.skip_mat:
        print(f"Oracle (MATLAB):  {EXPANDED_LUT_SA_MAT}")
    print()
    print("Next step: update ExpandedLUT.m to load levels 5-9 from the new .mat keys.")


if __name__ == '__main__':
    main()
