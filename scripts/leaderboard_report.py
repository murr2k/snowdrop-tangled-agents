"""
Fetch the tangled-game.com leaderboard and report stats for our Investigation 4 accounts.

Usage:
    poetry run python scripts/leaderboard_report.py [--no-headless] [--dump-html]

Reads TANGLED_USERNAME / TANGLED_PASSWORD from .env (login required for full board).
Columns: Rank | Player | ELO | W | L | D | Win% | Games
"""

import os
import re
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright

# UTF-8 output so emoji / special chars don't crash on Windows consoles
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

BASE_URL = "https://tangled-game.com"
LEADERBOARD_URL = f"{BASE_URL}/leaderboard"

# Match any leaderboard name containing these substrings (case-insensitive).
# murr1-10 are the display names the user chose during account enrollment.
OUR_TAGS = ["murr1", "murr2", "murr3", "murr4", "murr5",
            "murr6", "murr7", "murr8", "murr9", "murr10",
            "tangled"]

MEDAL_RANK = {"🥇": 1, "🥈": 2, "🥉": 3}


def fetch_leaderboard_html(headless: bool = True) -> tuple[str, str]:
    """Return (body_text, full_html) after logging in and loading the leaderboard."""
    username = os.getenv("TANGLED_USERNAME")
    password = os.getenv("TANGLED_PASSWORD")
    if not username or not password:
        sys.exit("TANGLED_USERNAME / TANGLED_PASSWORD not set in environment or .env")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        page = browser.new_page()

        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle", timeout=60000)
        time.sleep(2)

        try:
            btn = page.locator("text=/log.?in/i").first
            if btn.is_visible(timeout=3000):
                btn.click()
                time.sleep(2)
                page.wait_for_load_state("networkidle", timeout=60000)
        except Exception:
            pass

        time.sleep(3)
        try:
            page.locator("input[name='username']").fill(username)
            page.locator("input[name='password']").fill(password)
            page.locator("button[name='action']").click()
        except Exception as e:
            print(f"[warn] Login form: {e}", file=sys.stderr)

        time.sleep(3)
        page.wait_for_load_state("networkidle", timeout=60000)

        page.goto(LEADERBOARD_URL)
        page.wait_for_load_state("networkidle", timeout=60000)
        time.sleep(3)

        # Scroll to ensure all rows are rendered
        for _ in range(5):
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(0.8)

        text = page.inner_text("body")
        html = page.content()
        browser.close()

    return text, html


def parse_leaderboard(html: str) -> list[dict]:
    """
    Parse the leaderboard table from HTML.
    Columns: Rank | Player | ELO | W | L | D | Win% | Games
    Ranks 1-3 use medal emoji instead of numbers.
    """
    players = []
    rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)

    for row in rows:
        cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
        cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        cells = [c for c in cells if c]

        if len(cells) < 7:
            continue

        # Determine rank (medal emoji or integer)
        rank_raw = cells[0]
        if rank_raw in MEDAL_RANK:
            rank = MEDAL_RANK[rank_raw]
        elif rank_raw.isdigit():
            rank = int(rank_raw)
        else:
            continue  # header or unparseable row

        name = cells[1]

        def to_int(s):
            s = s.replace(',', '').strip()
            try:
                return int(s)
            except ValueError:
                return None

        def to_float(s):
            s = s.replace('%', '').replace(',', '').strip()
            try:
                return float(s)
            except ValueError:
                return None

        elo    = to_int(cells[2])
        wins   = to_int(cells[3])
        losses = to_int(cells[4])
        draws  = to_int(cells[5])
        win_pct = to_float(cells[6])
        games  = to_int(cells[7]) if len(cells) > 7 else None

        players.append({
            "rank": rank, "name": name,
            "elo": elo, "wins": wins, "losses": losses, "draws": draws,
            "win_pct": win_pct, "games": games,
        })

    return players


def is_ours(player: dict) -> bool:
    name = player.get("name", "").lower()
    return any(tag.lower() in name for tag in OUR_TAGS)


def fmt_int(v) -> str:
    return str(int(v)) if v is not None else "-"

def fmt_pct(v) -> str:
    return f"{v:.1f}%" if v is not None else "-"

def fmt_elo(v) -> str:
    return str(int(v)) if v is not None else "-"


def print_report(players: list[dict]) -> None:
    total = len(players)
    ours  = [p for p in players if is_ours(p)]

    print("=" * 68)
    print("  TANGLED-GAME.COM LEADERBOARD REPORT")
    print("=" * 68)
    print(f"  Total players on board : {total}")
    print(f"  Our accounts found     : {len(ours)}")
    print()

    if not players:
        print("[!] No rows parsed. Re-run with --dump-html and inspect leaderboard_dump.html.")
        return

    hdr = f"{'Rank':>5}  {'Player':<26}  {'ELO':>5}  {'W':>5}  {'L':>5}  {'D':>5}  {'Win%':>6}  {'Games':>6}"
    sep = "-" * len(hdr)

    def print_row(p):
        print(f"{p['rank']:>5}  {p['name']:<26}  "
              f"{fmt_elo(p['elo']):>5}  "
              f"{fmt_int(p['wins']):>5}  "
              f"{fmt_int(p['losses']):>5}  "
              f"{fmt_int(p['draws']):>5}  "
              f"{fmt_pct(p['win_pct']):>6}  "
              f"{fmt_int(p['games']):>6}")

    if ours:
        print("OUR ACCOUNTS")
        print(hdr)
        print(sep)
        for p in ours:
            print_row(p)
        print()

        wins   = sum(p["wins"]   or 0 for p in ours)
        losses = sum(p["losses"] or 0 for p in ours)
        draws  = sum(p["draws"]  or 0 for p in ours)
        games  = sum(p["games"]  or 0 for p in ours)
        win_pct = 100 * wins / games if games else 0.0
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
        print(f"  Win %        : {win_pct:.1f}%")
        print(f"  ELO range    : {worst_elo} – {best_elo}")
        print(f"  Rank range   : #{best_rank} – #{worst_rank} of {total}")
        print()
    else:
        print("[!] No accounts matching our tags found.")
        print(f"    Tags searched: {OUR_TAGS}")
        print()

    print("FULL LEADERBOARD")
    print(hdr)
    print(sep)
    for p in players:
        marker = " <--" if is_ours(p) else ""
        print(f"{p['rank']:>5}  {p['name']:<26}  "
              f"{fmt_elo(p['elo']):>5}  "
              f"{fmt_int(p['wins']):>5}  "
              f"{fmt_int(p['losses']):>5}  "
              f"{fmt_int(p['draws']):>5}  "
              f"{fmt_pct(p['win_pct']):>6}  "
              f"{fmt_int(p['games']):>6}"
              f"{marker}")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch tangled-game.com leaderboard")
    parser.add_argument("--no-headless", action="store_true", help="Show browser window")
    parser.add_argument("--dump-html", action="store_true", help="Write raw HTML to leaderboard_dump.html")
    args = parser.parse_args()

    print("Fetching leaderboard...", flush=True)
    text, html = fetch_leaderboard_html(headless=not args.no_headless)

    if args.dump_html:
        Path("leaderboard_dump.html").write_text(html, encoding="utf-8")
        print("HTML written to leaderboard_dump.html")

    players = parse_leaderboard(html)
    print_report(players)
