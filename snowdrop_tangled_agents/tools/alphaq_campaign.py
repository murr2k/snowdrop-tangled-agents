"""
Run batches of games vs AlphaQ Up in sequence and stop once a win is confirmed.

One browser session at a time: each batch is a separate play_tangled.py run.
After any win, one more probe game of that seat is played (the endgame probers
replay a known win first), so the campaign ends with the win confirmed or shown
to be a one-off.

Usage:
    python -m snowdrop_tangled_agents.tools.alphaq_campaign ROUNDS kind:games [kind:games ...]
    kinds: p1explore p2explore p1probe p2probe p1deep p1learn p2learn p1ood p2ood, and refit:0 (retrain the
    learned table and re-solve both seats; see tools/learned_table.py)
e.g. `... alphaq_campaign 10 p1deep:15` plays 150 DeepProber games.
"""

import subprocess
import sys
import time
from pathlib import Path

from snowdrop_tangled_agents.tools import alphaq_captures as ac

ROOT = Path(__file__).resolve().parents[2]
BASE = [sys.executable, "play_tangled.py", "--strategy", "ternary", "--opponent", "alphaq", "--headless",
        "--no-dashboard", "--ternary-beta", "q40"]
KINDS = {
    "p1explore": ["--seat", "1", "--plan-lines"],
    "p2explore": ["--seat", "2", "--plan-lines"],
    "p1probe": ["--seat", "1", "--probe-endgame"],
    "p2probe": ["--seat", "2", "--probe-endgame"],
    "p1deep": ["--seat", "1", "--probe-deep"],
    "p1learn": ["--seat", "1", "--plan-lines", "--ternary-beta", "learned"],
    "p2learn": ["--seat", "2", "--plan-lines", "--ternary-beta", "learned"],
    "p1ood": ["--seat", "1", "--ood-steer", "--ternary-beta", "learned"],
    "p2ood": ["--seat", "2", "--ood-steer", "--ternary-beta", "learned"],
}
REFIT = [[sys.executable, "-m", "snowdrop_tangled_agents.tools.learned_table"],
         [sys.executable, "-m", "snowdrop_tangled_agents.tools.alphaq_clone"]] + [
    [sys.executable, "-m", "snowdrop_tangled_agents.tools.solve_ternary_game", "--player", str(p), "--beta", "learned",
     "--out", str(Path.home() / ".tangled" / f"ternary_learned_book_p{p}.json")] for p in (1, 2)]


def wins_since(stamp: str) -> list:
    return [g for g in ac.load_games() if g["stamp"] >= stamp and ac.our_result(g) == "win"]


def run(kind: str, games: int, tag: str) -> None:
    log = ROOT / "logs" / f"campaign_{tag}_{kind}.log"
    with open(log, "w", encoding="utf-8") as f:
        if kind == "refit":                   # retrain the learned table on every game so far, re-solve both seats
            for cmd in REFIT:
                subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            return
        subprocess.run(BASE + KINDS[kind] + ["--games", str(games)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)


def main(rounds: int, plan: list) -> None:
    start = time.strftime("%Y%m%d_%H%M%S")
    for r in range(rounds):
        for kind, games in plan:
            print(f"[{time.strftime('%H:%M:%S')}] round {r} {kind} x{games}", flush=True)
            run(kind, games, f"{start}_r{r:02d}")
            w = wins_since(start)
            if w:
                seat = w[0]["seat"]
                print(f"WIN: {w[0]['file']} P{seat} {w[0]['lut_score']:+.6f} {ac.line_string(w[0])}", flush=True)
                print("replaying once to confirm", flush=True)
                run(f"p{seat}probe", 1, f"{start}_confirm")
                for g in wins_since(start):
                    print(f"WIN: {g['file']} P{g['seat']} {g['lut_score']:+.6f} {ac.line_string(g)}", flush=True)
                return
            n = sum(1 for g in ac.load_games() if g["stamp"] >= start)
            print(f"   {n} games this campaign, no win", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), [(a.split(':')[0], int(a.split(':')[1])) for a in sys.argv[2:]])
