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

Values follow TernaryMinimaxStrategy, from the perspective of the seat being
solved (`--player`): value is the minimax model score, tiebreak the expected
score against a uniformly random opponent. For P1 the output is the opening book
(all 45 first moves ranked); for P2 it is the reply book (for each of the 45 P1
openings, all 42 replies ranked), both in the formats TernaryMinimaxStrategy reads.

Progress telemetry goes to <telemetry-dir>/progress.jsonl (one event per line)
and <telemetry-dir>/status.json (latest state, replaced atomically), plus
throttled console lines. `--status` prints the latest status from any shell.

Usage:
    python -m snowdrop_tangled_agents.tools.solve_ternary_game [--player 1|2] [--beta 4.0] [--threads 6]
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
from snowdrop_tangled_agents.strategy.ternary_strategy import (
    OPENING_BOOK_PATH, REPLY_BOOK_PATH, TOL, rank_moves)

N = tm.NUM_EDGES
FULL = (1 << N) - 1
TELEMETRY_DIR = Path.home() / ".tangled" / "ternary_solver"
STORE_LAYERS = 6                                      # layers 0..6 (4.5M positions, ~36 MB) kept for the planner


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


def solve(beta, threads: int, telemetry: Telemetry, us: int = 1, keep_layers=()) -> tuple:
    """Run the full pass from seat `us`'s perspective.

    Returns (root (value, tiebreak), kept layers). Layers 1 and 2 (openings and
    opening + reply) are always kept; they are tiny.
    """
    total_work = sum(layer_work(k) for k in range(N))
    done_work = 0
    t0 = time.time()

    lut = tm.load_lut(beta)
    terminal = lut if us == 1 else -lut               # values from our seat's perspective
    layer = {FULL: (terminal, terminal)}              # k = 15: terminals, value = tiebreak
    kept = {}
    keep_layers = set(keep_layers) | {1, 2}
    telemetry.emit("start", console=True, total_work=total_work, positions=4 ** N, player=us)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for k in range(N - 1, -1, -1):
            our_turn = (k % 2 == 0) == (us == 1)      # P1 moves when an even number is colored
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
    root_v, root_w = layer[0]
    return (float(root_v[0]), float(root_w[0])), kept


def index_of(colored_digits: dict) -> int:
    """Array index of a position within its colored-edge group: {edge: digit} -> base-3 index."""
    return sum(d * 3 ** i for i, (_, d) in enumerate(sorted(colored_digits.items())))


def verify(kept: dict, beta, samples: int, telemetry: Telemetry, us: int = 1) -> float:
    """Compare kept-layer values with the brute-force-verified Solution class."""
    from snowdrop_tangled_agents.strategy.ternary_strategy import Solution
    lut = tm.load_lut(beta)
    rng = np.random.default_rng(0)
    worst = 0.0
    for k, layer in kept.items():
        if k < 6:                                     # Solution below 9 free edges only; small layers checked elsewhere
            continue
        for _ in range(samples):
            mask = int(rng.choice(list(layer)))
            colored = [e for e in range(N) if mask >> e & 1]
            digits = rng.integers(0, 3, len(colored))
            state = ['-'] * N
            for e, d in zip(colored, digits):
                state[e] = tm.COLORS[d]
            state = ''.join(state)
            ref = Solution(state, lut, us=us)
            v, w = layer[mask]
            idx = index_of(dict(zip(colored, (int(d) for d in digits))))
            worst = max(worst, abs(float(v[idx]) - float(ref.V[0])), abs(float(w[idx]) - float(ref.W[0])))
    telemetry.emit("verified", console=True, samples=samples, max_abs_diff=worst)
    return worst


def opening_book(kept: dict) -> list:
    """P1: all 45 first moves ranked."""
    values = {(mask.bit_length() - 1, tm.COLORS[d]): (float(v[d]), float(w[d]))
              for mask, (v, w) in kept[1].items() for d in range(3)}
    return [{"edge": e, "color": c, "value": v, "tiebreak": w} for (e, c), (v, w) in rank_moves(values)]


def reply_book(kept: dict) -> dict:
    """P2: for each P1 opening, all 42 replies ranked (values from P2's perspective)."""
    book = {}
    for e1 in range(N):
        v1, w1 = kept[1][1 << e1]
        for d1 in range(3):
            replies = {}
            for e2 in range(N):
                if e2 == e1:
                    continue
                v2, w2 = kept[2][(1 << e1) | (1 << e2)]
                for d2 in range(3):
                    i = index_of({e1: d1, e2: d2})
                    replies[(e2, tm.COLORS[d2])] = (float(v2[i]), float(w2[i]))
            book[f"E{e1}{tm.COLORS[d1]}"] = {
                "value": float(v1[d1]), "tiebreak": float(w1[d1]),
                "replies": [{"edge": e, "color": c, "value": v, "tiebreak": w}
                            for (e, c), (v, w) in rank_moves(replies)]}
    return book


def layer_store_path(beta, us: int) -> Path:
    return TELEMETRY_DIR / f"layers_{tm.model_name(beta)}_p{us}.npz"


def save_layers(kept: dict, beta, us: int) -> Path:
    """Write layers 0..STORE_LAYERS for the line planner.

    Layer k is stored as masks_k (ascending colored-edge masks) and v_k / w_k,
    shape (len(masks_k), 3^k), rows in mask order, columns indexed as index_of().
    """
    arrays = {}
    for k in range(STORE_LAYERS + 1):
        masks = sorted(kept[k])
        arrays[f"masks_{k}"] = np.array(masks, dtype=np.int32)
        arrays[f"v_{k}"] = np.stack([kept[k][m][0] for m in masks])
        arrays[f"w_{k}"] = np.stack([kept[k][m][1] for m in masks])
    path = layer_store_path(beta, us)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def print_status(directory: Path) -> None:
    path = directory / "status.json"
    if not path.exists():
        print(f"No status at {path}")
        return
    print(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--beta", type=tm.parse_model, default=tm.DEFAULT_BETA,
                        help="terminal model: beta (default %(default)s), 'inf' = ground states, "
                             "or the name of a fitted table (tools/refit_ternary_model.py)")
    parser.add_argument("--threads", type=int, default=6,
                        help="worker threads (default %(default)s: memory bandwidth saturates around the 6 P-cores)")
    parser.add_argument("--player", type=int, choices=(1, 2), default=1,
                        help="seat to solve for: 1 writes the opening book, 2 the reply book (default %(default)s)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"book path (default {OPENING_BOOK_PATH} for P1, {REPLY_BOOK_PATH} for P2)")
    parser.add_argument("--telemetry-dir", type=Path, default=TELEMETRY_DIR,
                        help="where progress.jsonl and status.json go (default %(default)s)")
    parser.add_argument("--verify", type=int, default=0, metavar="N",
                        help="check N random mid-game positions against the Solution class")
    parser.add_argument("--status", action="store_true", help="print the latest status and exit")
    args = parser.parse_args()

    if args.status:
        print_status(args.telemetry_dir)
        return

    out = args.out or (OPENING_BOOK_PATH if args.player == 1 else REPLY_BOOK_PATH)
    telemetry = Telemetry(args.telemetry_dir, {"pid": os.getpid(), "beta": args.beta, "threads": args.threads,
                                               "player": args.player})
    try:
        (root_v, root_w), kept = solve(args.beta, args.threads, telemetry, us=args.player,
                                       keep_layers=range(STORE_LAYERS + 1))
        if args.verify:
            verify(kept, args.beta, args.verify, telemetry, us=args.player)
        store = save_layers(kept, args.beta, args.player)
        telemetry.emit("stored", console=True, out=str(store))
        book = {"beta": args.beta, "player": args.player, "solver": "solve_ternary_game (full pass)",
                "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "root_value": root_v, "root_tiebreak": root_w}
        if args.player == 1:
            book["moves"] = opening_book(kept)
            best = [f"E{m['edge']}{m['color']}" for m in book["moves"][:3]]
        else:
            book["replies"] = reply_book(kept)
            best = {o: f"E{r['replies'][0]['edge']}{r['replies'][0]['color']}"
                    for o, r in list(book["replies"].items())[:3]}
        del kept
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(book, indent=1))
        telemetry.emit("done", console=True, work_pct=100.0, out=str(out), root_value=root_v,
                       root_tiebreak=root_w, best=best)
    except BaseException as exc:
        telemetry.emit("error", console=True, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        telemetry.close()


if __name__ == "__main__":
    main()
