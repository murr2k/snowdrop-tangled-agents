"""Phase 5A.2 — Launch the joint (anneal_time, s_max, sched_red) sweep via MATLAB.

Connects to an existing MATLAB session, ensures the parpool is up, and
invokes parameter_sweep_schrodinger.m on the full 1452-board calibration
set. The sweep runs in the foreground; progress prints per combo.

After completion, this script loads the .mat result, prints the best
parameter combo, and tabulates the top 10 combos by R².

Usage:
    poetry run python scripts/calib_parameter_sweep.py
    poetry run python scripts/calib_parameter_sweep.py --analyse-only
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import h5py
import numpy as np

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
MATLAB_DIR = REPO_ROOT / "snowdrop_tangled_agents" / "matlab" / "rl"
CALIB_MAT = MATLAB_DIR / "data" / "calibration_boards.mat"
OUTPUT_MAT = MATLAB_DIR / "data" / "phase5a2_sweep.mat"


def run_sweep() -> None:
    print("=" * 78)
    print("Phase 5A.2 — Joint Schrödinger parameter sweep via MATLAB")
    print("=" * 78)

    if not CALIB_MAT.exists():
        sys.exit(f"ERROR: calibration_boards.mat not found at {CALIB_MAT}. "
                 f"Run `poetry run python scripts/calibrate_adjudicator.py --export-mat` first.")

    print("\nImporting matlab.engine and connecting to running MATLAB session...")
    import matlab.engine
    sessions = matlab.engine.find_matlab()
    print(f"  Found MATLAB sessions: {sessions}")
    if not sessions:
        print("  No running session — starting a new MATLAB engine (slower)...")
        eng = matlab.engine.start_matlab()
    else:
        eng = matlab.engine.connect_matlab(sessions[0])
        print(f"  Connected to {sessions[0]}")

    eng.addpath(str(MATLAB_DIR), nargout=0)

    print("\nEnsuring parallel pool is up...")
    try:
        eng.eval("p = gcp('nocreate'); if isempty(p), parpool(); end; fprintf('Pool: %d workers\\n', gcp().NumWorkers);", nargout=0)
    except Exception as e:
        print(f"  WARNING: pool startup error: {e}")

    print("\nLaunching parameter_sweep_schrodinger (this is a long run)...")
    t0 = time.time()
    eng.parameter_sweep_schrodinger(str(CALIB_MAT), str(OUTPUT_MAT), nargout=0)
    elapsed = time.time() - t0
    print(f"\nMATLAB sweep finished in {elapsed/3600:.2f} hr")

    analyse_results()


def analyse_results() -> None:
    print("\n" + "=" * 78)
    print("Loading sweep results")
    print("=" * 78)
    if not OUTPUT_MAT.exists():
        sys.exit(f"ERROR: results file not found: {OUTPUT_MAT}")

    with h5py.File(OUTPUT_MAT, "r") as f:
        # NOTE: MATLAB -v7.3 stores variables in column-major order; the
        # 3D arrays from MATLAB show as (sched_reds, s_max, anneal_times) here.
        r2 = np.array(f["r2_grid"]).astype(np.float64)
        rmse = np.array(f["rmse_grid"]).astype(np.float64)
        mae = np.array(f["mae_grid"]).astype(np.float64)
        bias = np.array(f["bias_grid"]).astype(np.float64)
        at = np.array(f["anneal_times"]).flatten()
        sm = np.array(f["s_max_values"]).flatten()
        sr = np.array(f["sched_reds"]).flatten()

    print(f"\nGrid shape (anneal_times, s_max, sched_red): MATLAB = ({len(at)}, {len(sm)}, {len(sr)})")
    print(f"  In Python h5py we get shape {r2.shape} (reversed axes)")
    # Reorder to (anneal_times, s_max, sched_red) to match the MATLAB code
    r2 = r2.transpose()
    rmse = rmse.transpose()
    mae = mae.transpose()
    bias = bias.transpose()
    print(f"  After transpose: {r2.shape}")

    print(f"\nSweep dimensions:")
    print(f"  anneal_times: {at}")
    print(f"  s_max_values: {sm}")
    print(f"  sched_reds:   {sr}")

    # Baseline R² at (1.85, 0.999, 0.5)
    def closest_idx(arr, target):
        return int(np.argmin(np.abs(arr - target)))
    bat = closest_idx(at, 1.85)
    bsm = closest_idx(sm, 0.999)
    bsr = closest_idx(sr, 0.5)
    print(f"\nBaseline (calib parameters): tf={at[bat]}, s_max={sm[bsm]}, sched={sr[bsr]}")
    print(f"  R² = {r2[bat, bsm, bsr]:+.4f}")

    # Best combo
    best_flat = np.argmax(r2)
    best_idx = np.unravel_index(best_flat, r2.shape)
    print(f"\nBest combo:")
    print(f"  anneal_time   = {at[best_idx[0]]}")
    print(f"  s_max         = {sm[best_idx[1]]}")
    print(f"  sched_red     = {sr[best_idx[2]]}")
    print(f"  R²            = {r2[best_idx]:+.4f}")
    print(f"  RMSE          = {rmse[best_idx]:.4f}")
    print(f"  MAE           = {mae[best_idx]:.4f}")
    print(f"  bias          = {bias[best_idx]:+.4f}")
    print(f"  Δ R² vs baseline: {r2[best_idx] - r2[bat, bsm, bsr]:+.4f}")

    # Top 15 combos
    print("\nTop 15 combos by R²:")
    print(f"  {'rank':>4}  {'tf(ns)':>8}  {'s_max':>7}  {'sched':>7}  {'R²':>9}  {'RMSE':>7}  {'MAE':>7}  {'bias':>9}")
    flat = r2.flatten()
    order = np.argsort(-flat)[:15]
    for rk, idx_flat in enumerate(order, 1):
        idx = np.unravel_index(idx_flat, r2.shape)
        print(f"  {rk:>4d}  {at[idx[0]]:>8.3f}  {sm[idx[1]]:>7.3f}  {sr[idx[2]]:>7.3f}  "
              f"{r2[idx]:+9.4f}  {rmse[idx]:>7.4f}  {mae[idx]:>7.4f}  {bias[idx]:+9.4f}")

    # Best per (s_max, sched_red) — what is the best anneal_time for each combo?
    print("\nBest anneal_time per (s_max, sched_red):")
    print(f"  {'sched\\s_max':>14s}", end="")
    for sm_val in sm:
        print(f"  s_max={sm_val:<6.3f}", end="")
    print()
    for k_sr, sr_val in enumerate(sr):
        print(f"  sched_red={sr_val:<5.3f}", end="")
        for k_sm in range(len(sm)):
            slice_r2 = r2[:, k_sm, k_sr]
            best_at_idx = int(np.argmax(slice_r2))
            print(f"   {at[best_at_idx]:>5.2f}/R²={slice_r2[best_at_idx]:+.2f}", end="")
        print()

    # Marginals: best R² by anneal_time (max over sm, sr)
    print("\nBest R² vs anneal_time (max over s_max, sched_red):")
    for k_at, at_val in enumerate(at):
        m_r2 = r2[k_at].max()
        m_idx = np.unravel_index(np.argmax(r2[k_at]), r2[k_at].shape)
        print(f"  tf={at_val:>7.3f}  max R²={m_r2:+.4f}  (at s_max={sm[m_idx[0]]}, sched={sr[m_idx[1]]})")

    print("\nBest R² vs sched_red (max over anneal_time, s_max):")
    for k_sr, sr_val in enumerate(sr):
        m_r2 = r2[:, :, k_sr].max()
        m_idx = np.unravel_index(np.argmax(r2[:, :, k_sr]), r2[:, :, k_sr].shape)
        print(f"  sched_red={sr_val:>5.3f}  max R²={m_r2:+.4f}  (at tf={at[m_idx[0]]}, s_max={sm[m_idx[1]]})")

    print("\nBest R² vs s_max (max over anneal_time, sched_red):")
    for k_sm, sm_val in enumerate(sm):
        m_r2 = r2[:, k_sm, :].max()
        m_idx = np.unravel_index(np.argmax(r2[:, k_sm, :]), r2[:, k_sm, :].shape)
        print(f"  s_max={sm_val:>5.3f}  max R²={m_r2:+.4f}  (at tf={at[m_idx[0]]}, sched={sr[m_idx[1]]})")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--analyse-only", action="store_true",
                   help="Skip the MATLAB sweep, just analyse the existing .mat file")
    args = p.parse_args()
    if args.analyse_only:
        analyse_results()
    else:
        run_sweep()


if __name__ == "__main__":
    main()
