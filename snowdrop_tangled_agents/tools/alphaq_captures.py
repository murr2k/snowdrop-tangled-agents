"""
Games vs AlphaQ Up as recorded by the game-end network captures.

Every game play_tangled.py finishes writes logs/game_end_capture/*.json with
the network trace. The /api/make_move responses give the full move sequence
(edge_index, color label 1 grey / 2 green / 3 purple), /api/adjudicate the
lookup-table winner, score and the server's terminal, /api/games/complete the
ELO. This module turns those files into game records, AlphaQ's reply table
(AlphaQ is deterministic, so one reply per position it has faced) and the
goal ledger (logs/goal_ledger.md).

Captures hold the account email: they and the ledger stay in logs/, which is
never committed.

Usage:
    python -m snowdrop_tangled_agents.tools.alphaq_captures            # summary
    python -m snowdrop_tangled_agents.tools.alphaq_captures --ledger   # rewrite the ledger's game table
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from snowdrop_tangled_agents.strategy import ternary_model as tm

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_DIR = ROOT / "logs" / "game_end_capture"
LEDGER_PATH = ROOT / "logs" / "goal_ledger.md"
GOAL_START = "20260925_000000"          # captures stamped at or after this count toward the goal
LABEL = {1: 'Z', 2: 'G', 3: 'P'}        # site color labels
GAME_BEGIN = "<!-- games:begin -->"
GAME_END = "<!-- games:end -->"


def _body(ev: dict):
    try:
        return json.loads(ev.get("body") or "")
    except ValueError:
        return None


def parse_capture(path: Path) -> Optional[dict]:
    """One game record from a capture file, or None if the trace is incomplete."""
    d = json.loads(path.read_text(encoding="utf-8"))
    moves, adjudication, complete = [], None, None
    for ev in d.get("net", []):
        url = ev.get("url", "").split("?")[0]
        body = _body(ev)
        if not isinstance(body, dict):
            continue
        if url.endswith("/api/make_move") and "edge_index" in body:
            moves.append((int(body["edge_index"]), LABEL[int(body["color"])]))
        elif url.endswith("/api/adjudicate"):
            adjudication = body
        elif url.endswith("/api/games/complete"):
            complete = body
    seat = d.get("seat")
    board = ['-'] * tm.NUM_EDGES
    for e, c in moves:
        board[e] = c
    terminal = ''.join(board)
    # Captures from before 2026-09-24 23:00 lack the top-level fields; the trace has them.
    symbols = {0: '-', 1: 'Z', 2: 'G', 3: 'P'}
    server_edges = ((adjudication or {}).get("game_state") or {}).get("edges") or []
    server_terminal = d.get("server_terminal") or ''.join(symbols.get(e[2], '?') for e in server_edges) or None
    lut_score = d.get("lut_score", (adjudication or {}).get("score"))
    rec = {
        "file": path.name,
        "stamp": path.name[:15],
        "game_id": d.get("game_id"),
        "run_id": d.get("run_id"),
        "seat": seat,
        "opponent": d.get("opponent"),
        "moves": moves,                          # in play order, P1 first
        "terminal": terminal,
        "server_terminal": server_terminal,
        "lut_score": lut_score,
        "winner": (adjudication or {}).get("winner"),
        "result": d.get("result"),
        "site_result": d.get("site_result"),
        "elo_before": (complete or {}).get("player_elo_before"),
        "elo_after": (complete or {}).get("player_elo_after"),
        "opp_elo_after": (complete or {}).get("opponent_elo_after"),
        "challenge": d.get("challenge"),
        "complete": len(moves) == tm.NUM_EDGES and server_terminal == terminal and lut_score is not None,
    }
    return rec


def load_games(capture_dir: Path = CAPTURE_DIR, opponent: str = "alphaq", complete_only: bool = True) -> list:
    """Game records in time order (complete traces only by default)."""
    games = []
    for p in sorted(capture_dir.glob("*.json")):
        try:
            g = parse_capture(p)
        except (OSError, ValueError, KeyError):
            continue
        if g is None or (opponent and g["opponent"] not in (None, opponent)):
            continue
        if complete_only and not g["complete"]:
            continue
        games.append(g)
    return games


def states_of(moves: list) -> list:
    """Board before each move: states[i] is the position move i was played from."""
    board = ['-'] * tm.NUM_EDGES
    out = []
    for e, c in moves:
        out.append(''.join(board))
        board[e] = c
    return out


def mover(i: int) -> int:
    """Seat that plays move i (0-based): P1 plays the even moves."""
    return 1 if i % 2 == 0 else 2


def reply_table(games: list, seat: int) -> dict:
    """AlphaQ's reply {state: (edge, color)} in games where we held `seat`.

    Raises if AlphaQ ever answered the same position two ways, which would
    break the deterministic-opponent assumption the planner relies on.
    """
    table = {}
    for g in games:
        if g["seat"] != seat:
            continue
        for i, (s, mv) in enumerate(zip(states_of(g["moves"]), g["moves"])):
            if mover(i) == seat:
                continue
            if table.get(s, mv) != mv:
                raise ValueError(f"AlphaQ answered {s} with both {table[s]} and {mv} ({g['file']})")
            table[s] = mv
    return table


def known_terminals(games: list) -> dict:
    """{terminal: lookup-table score (P1 perspective)} for every adjudicated game."""
    return {g["terminal"]: g["lut_score"] for g in games if g["lut_score"] is not None}


def our_result(g: dict) -> str:
    ours = {1: "red", 2: "blue"}[g["seat"]]
    w = g["winner"]
    return "draw" if w == "draw" else "win" if w == ours else "loss" if w in ("red", "blue") else "?"


def line_string(g: dict) -> str:
    """Moves in play order, ours in upper case: 'E12P e10p E4G ...'."""
    out = []
    for i, (e, c) in enumerate(g["moves"]):
        tok = f"E{e}{c}"
        out.append(tok if mover(i) == g["seat"] else tok.lower())
    return ' '.join(out)


def goal_games(games: list) -> list:
    return [g for g in games if g["stamp"] >= GOAL_START]


def ledger_table(games: list) -> str:
    rows = ["| # | Capture | Seat | Mode | Line (ours upper case) | Terminal | LUT score | Model b4 | Result | ELO after |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for n, g in enumerate(goal_games(games), 1):
        model = tm.score_state(g["terminal"])
        rows.append(f"| {n} | {g['stamp']} | P{g['seat']} | {'challenge' if g['challenge'] else 'pick'} | "
                    f"`{line_string(g)}` | `{g['terminal']}` | {g['lut_score']:+.6f} | {model:+.6f} | "
                    f"{our_result(g)} | {g['elo_after']} |")
    return '\n'.join(rows)


def write_ledger(games: list, path: Path = LEDGER_PATH) -> None:
    """Replace the generated game table in the ledger, keeping the hand-written notes."""
    table = ledger_table(games)
    gg = goal_games(games)
    w = sum(our_result(g) == "win" for g in gg)
    l = sum(our_result(g) == "loss" for g in gg)
    d = sum(our_result(g) == "draw" for g in gg)
    elo = next((g["elo_after"] for g in reversed(games) if g["elo_after"] is not None), None)
    summary = (f"Generated {datetime.now():%Y-%m-%d %H:%M}: {len(gg)} ranked games under the goal "
               f"({w}W {d}D {l}L), account ELO {elo}.")
    block = f"{GAME_BEGIN}\n{summary}\n\n{table}\n{GAME_END}"
    text = path.read_text(encoding="utf-8") if path.exists() else "# AlphaQ goal ledger\n\n## Games\n\n" + GAME_BEGIN + GAME_END + "\n\n## Notes\n"
    text = re.sub(re.escape(GAME_BEGIN) + ".*?" + re.escape(GAME_END), lambda _: block, text, flags=re.S)
    path.write_text(text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ledger", action="store_true", help=f"rewrite the game table in {LEDGER_PATH}")
    ap.add_argument("--all", action="store_true", help="list pre-goal games too")
    args = ap.parse_args()
    games = load_games()
    for g in (games if args.all else goal_games(games)):
        print(f"{g['stamp']} P{g['seat']} {our_result(g):5s} {g['lut_score']:+.6f} {g['terminal']} "
              f"elo={g['elo_after']}  {line_string(g)}")
    for seat in (1, 2):
        print(f"AlphaQ replies known as P{seat}: {len(reply_table(games, seat))}")
    print(f"Adjudicated terminals: {len(known_terminals(games))}")
    if args.ledger:
        write_ledger(games)
        print(f"Ledger written: {LEDGER_PATH}")


if __name__ == "__main__":
    main()
