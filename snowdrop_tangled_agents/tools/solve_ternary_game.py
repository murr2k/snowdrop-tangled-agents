"""
Solve the whole three-color Petersen game in one retrograde pass.

One pass over all 4^15 positions (instead of 27 overlapping 14-free-edge subgame
solves, which took ~50 min), grouped by which edges are colored.
For a colored-edge set C (|C| = k) the values form a base-3 array of 3^k entries,
digit per colored edge in ascending edge order (Z=0, G=1, P=2); for C = all 15
edges that is exactly the ternary_model terminal table. Coloring edge e takes one
slice along e's axis of the child array, so every update is a contiguous,
bandwidth-bound numpy operation. Groups within a layer are independent and run on
a thread pool; only two adjacent layers are resident (peak ~4 GB, at k = 12/11).

Values follow TernaryMinimaxStrategy (P1 perspective): value is the minimax model
score, tiebreak the expected score against a uniformly random P2. The output is
the P1 opening book in the format TernaryMinimaxStrategy reads.

Progress telemetry goes to <telemetry-dir>/progress.jsonl (one event per line)
and <telemetry-dir>/status.json (latest state, replaced atomically), plus
throttled console lines. `--status` prints the latest status from any shell.

Usage:
    python -m snowdrop_tangled_agents.tools.solve_ternary_game [--beta 4.0] [--threads 6]
    python -m snowdrop_tangled_agents.tools.solve_ternary_game --status
"""

import argparse
import ctypes
import itertools
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from math import comb
from pathlib import Path

import numpy as np

from snowdrop_tangled_agents.strategy import ternary_model as tm
from snowdrop_tangled_agents.strategy.ternary_strategy import OPENING_BOOK_PATH, TOL, rank_moves

N = tm.NUM_EDGES
FULL = (1 << N) - 1
TELEMETRY_DIR = Path.home() / ".tangled" / "ternary_solver"


def _memory_gb() -> tuple:
    """(working set, peak working set) of this process in GB; (None, None) off Windows."""
    if os.name != "nt":
        return None, None
    import ctypes.wintypes as wt

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

    c = Counters()
    c.cb = ctypes.sizeof(c)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wt.HANDLE
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [wt.HANDLE, ctypes.POINTER(Counters), wt.DWORD]
    if not get_info(get_process(), ctypes.byref(c), c.cb):
        return None, None
    return round(c.WorkingSetSize / 1e9, 2), round(c.PeakWorkingSetSize / 1e9, 2)


class Telemetry:
    """JSONL event log + atomically replaced status file + throttled console lines."""

    def __init__(self, directory: Path, run_info: dict, console_every: float = 5.0):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = open(self.dir / "progress.jsonl", "a", encoding="utf-8")
        self.status_path = self.dir / "status.json"
        self.run_info = run_info
        self.start = time.time()
        self.console_every = console_every
        self._last_console = 0.0

    def emit(self, event: str, console: bool = False, **fields):
        now = time.time()
        mem, peak = _memory_gb()
        record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event,
                  "elapsed_s": round(now - self.start, 1), "mem_gb": mem, "peak_mem_gb": peak,
                  **self.run_info, **fields}
        self.log.write(json.dumps(record) + "\n")
        self.log.flush()
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=1))
        os.replace(tmp, self.status_path)
        if console or now - self._last_console >= self.console_every:
            self._last_console = now
            pct = fields.get("work_pct")
            eta = fields.get("eta_s")
            print(f"[{record['elapsed_s']:7.1f}s] {event:10s} "
                  f"layer k={fields.get('layer', '-')} "
                  f"{fields.get('layer_groups_done', '')}/{fields.get('layer_groups', '')} "
                  f"work {pct if pct is not None else '-'}% "
                  f"ETA {eta if eta is not None else '-'}s mem {mem}/{peak} GB", flush=True)

    def close(self):
        self.log.close()


def layer_work(k: int) -> int:
    """Child-slice elements read to compute every group with k colored edges."""
    return comb(N, k) * (N - k) * 3 ** (k + 1)


def solve_group(mask: int, k: int, child: dict, our_turn: bool) -> tuple:
    """Values and tiebreaks for every position whose colored-edge set is `mask`."""
    colored = [e for e in range(N) if mask >> e & 1]
    free = [e for e in range(N) if not mask >> e & 1]
    size = 3 ** k
    bv = np.full(size, -np.inf if our_turn else np.inf, dtype=np.float32)
    bw = np.full(size, -np.inf if our_turn else 0.0, dtype=np.float32)
    # Our turn takes two passes: exact max value first, then the best tiebreak
    # among children within TOL of it (same rule as ternary_strategy.Solution).
    for pass_ in ((0, 1) if our_turn else (0,)):
        for e in free:
            r = sum(1 for c in colored if c < e)      # e's digit position in the child
            hi, lo = 3 ** (k - r), 3 ** r
            cv, cw = child[mask | (1 << e)]
            cv3, cw3 = cv.reshape(hi, 3, lo), cw.reshape(hi, 3, lo)
            pv, pw = bv.reshape(hi, lo), bw.reshape(hi, lo)
            for d in range(3):
                v, w = cv3[:, d, :], cw3[:, d, :]
                if not our_turn:
                    np.minimum(pv, v, out=pv)
                    np.add(pw, w, out=pw)
                elif pass_ == 0:
                    np.maximum(pv, v, out=pv)
                else:
                    np.copyto(pw, w, where=(v >= pv - TOL) & (w > pw))
    if not our_turn:
        bw /= 3 * len(free)
    return bv, bw


def solve(beta, threads: int, telemetry: Telemetry, keep_layers=()) -> tuple:
    """Run the full pass for P1. Returns (layer-1 values, root value, kept layers)."""
    total_work = sum(layer_work(k) for k in range(N))
    done_work = 0
    t0 = time.time()

    lut = tm.load_lut(beta)
    layer = {FULL: (lut, lut)}                        # k = 15: terminals, value = tiebreak
    kept = {}
    telemetry.emit("start", console=True, total_work=total_work, positions=4 ** N)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for k in range(N - 1, -1, -1):
            our_turn = k % 2 == 0                     # P1 moves when an even number is colored
            masks = [sum(1 << e for e in c) for c in itertools.combinations(range(N), k)]
            group_work = (N - k) * 3 ** (k + 1)
            new_layer = {}
            futures = {pool.submit(solve_group, m, k, layer, our_turn): m for m in masks}
            last_emit = 0.0
            for i, f in enumerate(as_completed(futures), 1):
                new_layer[futures[f]] = f.result()
                done_work += group_work
                now = time.time()
                if now - last_emit >= 1.0:            # throttle: middle layers have thousands of groups
                    last_emit = now
                    rate = done_work / (now - t0)
                    telemetry.emit("progress", layer=k, layer_groups_done=i, layer_groups=len(masks),
                                   work_pct=round(100 * done_work / total_work, 2),
                                   eta_s=round((total_work - done_work) / rate) if rate else None,
                                   elements_per_s=round(rate))
            layer = new_layer
            if k in keep_layers:
                kept[k] = layer
            telemetry.emit("layer_done", console=True, layer=k, layer_groups_done=len(masks),
                           layer_groups=len(masks), work_pct=round(100 * done_work / total_work, 2),
                           positions=comb(N, k) * 3 ** k)
            if k == 1:
                first_moves = layer
    root_v, root_w = layer[0]
    return first_moves, (float(root_v[0]), float(root_w[0])), kept


def verify(kept: dict, beta, samples: int, telemetry: Telemetry) -> float:
    """Compare kept-layer values with the brute-force-verified Solution class."""
    from snowdrop_tangled_agents.strategy.ternary_strategy import Solution
    lut = tm.load_lut(beta)
    rng = np.random.default_rng(0)
    worst = 0.0
    for k, layer in kept.items():
        for _ in range(samples):
            mask = int(rng.choice(list(layer)))
            colored = [e for e in range(N) if mask >> e & 1]
            digits = rng.integers(0, 3, len(colored))
            state = ['-'] * N
            for e, d in zip(colored, digits):
                state[e] = tm.COLORS[d]
            state = ''.join(state)
            ref = Solution(state, lut, us=1)
            v, w = layer[mask]
            idx = int(sum(int(d) * 3 ** i for i, d in enumerate(digits)))
            worst = max(worst, abs(float(v[idx]) - float(ref.V[0])), abs(float(w[idx]) - float(ref.W[0])))
    telemetry.emit("verified", console=True, samples=samples * len(kept), max_abs_diff=worst)
    return worst


def print_status(directory: Path) -> None:
    path = directory / "status.json"
    if not path.exists():
        print(f"No status at {path}")
        return
    print(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--beta", type=lambda s: None if s.lower() == "inf" else float(s),
                        default=tm.DEFAULT_BETA, help="terminal model beta (default %(default)s; 'inf' = ground states)")
    parser.add_argument("--threads", type=int, default=6,
                        help="worker threads (default %(default)s: memory bandwidth saturates around the 6 P-cores)")
    parser.add_argument("--out", type=Path, default=OPENING_BOOK_PATH, help="opening book path (default %(default)s)")
    parser.add_argument("--telemetry-dir", type=Path, default=TELEMETRY_DIR,
                        help="where progress.jsonl and status.json go (default %(default)s)")
    parser.add_argument("--verify", type=int, default=0, metavar="N",
                        help="check N random mid-game positions against the Solution class")
    parser.add_argument("--status", action="store_true", help="print the latest status and exit")
    args = parser.parse_args()

    if args.status:
        print_status(args.telemetry_dir)
        return

    telemetry = Telemetry(args.telemetry_dir, {"pid": os.getpid(), "beta": args.beta, "threads": args.threads})
    try:
        first_moves, (root_v, root_w), kept = solve(args.beta, args.threads, telemetry,
                                                    keep_layers=(6,) if args.verify else ())
        if args.verify:
            verify(kept, args.beta, args.verify, telemetry)
            del kept
        values = {(mask.bit_length() - 1, tm.COLORS[d]): (float(v[d]), float(w[d]))
                  for mask, (v, w) in first_moves.items() for d in range(3)}
        moves = [{"edge": e, "color": c, "value": v, "tiebreak": w}
                 for (e, c), (v, w) in rank_moves(values)]
        book = {"beta": args.beta, "player": 1, "solver": "solve_ternary_game (full pass)",
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "root_value": root_v, "root_tiebreak": root_w, "moves": moves}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(book, indent=1))
        telemetry.emit("done", console=True, work_pct=100.0, out=str(args.out), root_value=root_v,
                       root_tiebreak=root_w, best=[f"E{m['edge']}{m['color']}" for m in moves[:3]])
    except BaseException as exc:
        telemetry.emit("error", console=True, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        telemetry.close()


if __name__ == "__main__":
    main()
