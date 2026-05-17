"""
Fetch the tangled-game.com leaderboard and report stats for our Investigation 4 accounts.

The leaderboard is served as public JSON — no login required.

Usage:
    poetry run python scripts/leaderboard_report.py
"""

import sys
import requests

# UTF-8 output so emoji / special chars don't crash on Windows consoles
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_URL = "https://snowdrop-tangled-web-backend-production.up.railway.app/api/leaderboard"

# Investigation 4 account display names (set during account enrollment).
# Exact match (case-insensitive) to avoid catching murr2k, murr2k@gmail.com, etc.
OUR_NAMES = {
    "murr1", "murr2", "murr3", "murr4", "murr5",
    "murr6", "murr7", "murr8", "murr9", "murr10",
}


def fetch_leaderboard() -> list[dict]:
    resp = requests.get(API_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()


def is_ours(player: dict) -> bool:
    return player.get("name", "").lower() in {n.lower() for n in OUR_NAMES}


def fmt_int(v) -> str:
    return str(int(v)) if v is not None else "-"

def fmt_pct(wins, total) -> str:
    if not total:
        return "-"
    return f"{100 * wins / total:.1f}%"

def fmt_elo(v) -> str:
    return str(int(v)) if v is not None else "-"


def print_report(data: list[dict]) -> None:
    # API returns players sorted by ELO descending; assign ranks
    players = []
    for rank, p in enumerate(data, start=1):
        players.append({
            "rank":   rank,
            "name":   p.get("username") or p.get("name", "?"),
            "elo":    p.get("current_elo"),
            "wins":   p.get("wins", 0),
            "losses": p.get("losses", 0),
            "draws":  p.get("draws", 0),
            "games":  p.get("total_games", 0),
        })

    total = len(players)
    ours  = [p for p in players if is_ours(p)]

    hdr = (f"{'Rank':>5}  {'Player':<18}  {'ELO':>5}  "
           f"{'W':>5}  {'L':>5}  {'D':>5}  {'Win%':>6}  {'Games':>6}")
    sep = "-" * len(hdr)

    def print_row(p):
        print(f"{p['rank']:>5}  {p['name']:<18}  "
              f"{fmt_elo(p['elo']):>5}  "
              f"{fmt_int(p['wins']):>5}  "
              f"{fmt_int(p['losses']):>5}  "
              f"{fmt_int(p['draws']):>5}  "
              f"{fmt_pct(p['wins'], p['games']):>6}  "
              f"{fmt_int(p['games']):>6}")

    print("=" * len(hdr))
    print("  TANGLED-GAME.COM LEADERBOARD REPORT")
    print("=" * len(hdr))
    print(f"  Total players on board : {total}")
    print(f"  Our accounts found     : {len(ours)}")
    print()

    if ours:
        print("OUR ACCOUNTS")
        print(hdr)
        print(sep)
        for p in ours:
            print_row(p)
        print()

        wins   = sum(p["wins"]   for p in ours)
        losses = sum(p["losses"] for p in ours)
        draws  = sum(p["draws"]  for p in ours)
        games  = sum(p["games"]  for p in ours)
        best_elo   = max(p["elo"] or 0 for p in ours)
        worst_elo  = min(p["elo"] or 0 for p in ours)
        best_rank  = min(p["rank"] for p in ours)
        worst_rank = max(p["rank"] for p in ours)

        print("AGGREGATE (all our accounts combined)")
        print("-" * 40)
        print(f"  Total games  : {games}")
        print(f"  Wins         : {wins}")
        print(f"  Losses       : {losses}")
        print(f"  Draws        : {draws}")
        print(f"  Win %        : {fmt_pct(wins, games)}")
        print(f"  ELO range    : {worst_elo} – {best_elo}")
        print(f"  Rank range   : #{best_rank} – #{worst_rank} of {total}")
        print()
    else:
        print("[!] No Investigation 4 accounts found.")
        print(f"    Expected names: {sorted(OUR_NAMES)}")
        print()


if __name__ == "__main__":
    import argparse, time, os
    parser = argparse.ArgumentParser(description="Fetch tangled-game.com leaderboard")
    parser.add_argument("--watch", "-w", type=int, default=0, metavar="SECONDS",
                        help="Re-fetch every N seconds (like Unix watch)")
    args = parser.parse_args()

    first = True
    while True:
        if args.watch:
            if first:
                os.system("cls" if os.name == "nt" else "clear")
            else:
                print("\033[H", end="", flush=True)  # move cursor to top without clearing
        try:
            data = fetch_leaderboard()
            print_report(data)
        except Exception as e:
            print(f"[error] {e}")
        first = False
        if not args.watch:
            break
        print(f"  Refreshing in {args.watch}s — Ctrl+C to stop", flush=True)
        time.sleep(args.watch)
