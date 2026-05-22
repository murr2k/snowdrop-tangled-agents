"""
Analyze results from scripts/sweep_p2_responses.ps1.

Reads logs/sweep_p2_obs.log.err (observation game) and
logs/sweep_p2_E{N}{C}.log.err for each P2 response, then prints
a sorted table: wins first, draws, losses.

Usage:
    poetry run python scripts/_analyze_p2_sweep.py --alphaq-opening E7G
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR      = PROJECT_ROOT / "logs"

_ORBIT = (0, 0, 1, 2, 3, 4, 5, 2, 5, 6, 7, 6, 8, 4, 3)

_ORBIT_EDGES: dict[int, list[int]] = {}
for _e, _o in enumerate(_ORBIT):
    _ORBIT_EDGES.setdefault(_o, []).append(_e)

_ANSI = re.compile(r'\x1b\[[0-9;]*m')


def strip_ansi(text: str) -> str:
    return _ANSI.sub('', text)


def parse_log(path: Path) -> dict:
    if not path.exists():
        return {'result': 'MISSING', 'score': None, 'our_move': None,
                'opp_move1': None, 'error': None}

    text = strip_ansi(path.read_text(encoding='utf-8', errors='replace'))

    result = 'NO_RESULT'
    m = re.search(r'GAME OVER: (WIN|DRAW|LOSS)', text)
    if m:
        result = m.group(1)

    score = None
    m = re.search(r'Final Score:\s*([+-]?\d+\.\d+)', text)
    if m:
        score = float(m.group(1))

    # AlphaQ's P1 opening (first move, opponent)
    opp_move1 = None
    m = re.search(r'1\.\s+\[OPP\s*\]\s+E(\d+)\s+(Green|Purple)', text)
    if m:
        c = 'G' if m.group(2) == 'Green' else 'P'
        opp_move1 = f"E{m.group(1)}{c}"

    # Our P2 first response (second move, us)
    our_move = None
    m = re.search(r'2\.\s+\[US\s*\]\s+E(\d+)\s+(Green|Purple)', text)
    if m:
        c = 'G' if m.group(2) == 'Green' else 'P'
        our_move = f"E{m.group(1)}{c}"

    error = None
    if 'Traceback' in text:
        m = re.search(r'(Traceback.{0,300})', text, re.DOTALL)
        if m:
            error = m.group(1)[:200]

    return {'result': result, 'score': score, 'our_move': our_move,
            'opp_move1': opp_move1, 'error': error}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--alphaq-opening', default=None,
                        help='AlphaQ P1 opening observed (e.g. E7G)')
    args = parser.parse_args()

    # Parse observation game
    obs = parse_log(LOG_DIR / "sweep_p2_obs.log.err")
    alphaq_opening = args.alphaq_opening or obs.get('opp_move1') or '?'

    print()
    print("=" * 70)
    print("P2 Response Sweep — Results")
    print(f"  AlphaQ (P1) opened: {alphaq_opening}")
    if obs['result'] != 'MISSING':
        score_s = f"{obs['score']:+.4f}" if obs['score'] is not None else '?'
        print(f"  Observation game:   {obs['result']} ({score_s})")
    print("=" * 70)
    print()

    # Determine which edge AlphaQ used (exclude from sweep)
    alphaq_edge = None
    m = re.match(r'E(\d+)[GP]', alphaq_opening)
    if m:
        alphaq_edge = int(m.group(1))

    # Collect all sweep results
    rows = []
    for edge in range(15):
        if edge == alphaq_edge:
            continue
        for color in ('G', 'P'):
            label = f"E{edge}{color}"
            log   = LOG_DIR / f"sweep_p2_{label}.log.err"
            data  = parse_log(log)

            override_ok = (data['our_move'] == label) if data['our_move'] else None

            rows.append({
                'response': label,
                'edge':     edge,
                'color':    color,
                'orbit':    _ORBIT[edge],
                'orbit_peers': [f"E{e}" for e in _ORBIT_EDGES[_ORBIT[edge]]
                                if e != edge and e != alphaq_edge],
                'override_ok': override_ok,
                **data,
            })

    # Sort: WIN < DRAW < LOSS < NO_RESULT < MISSING; within group score desc
    _rank = {'WIN': 0, 'DRAW': 1, 'LOSS': 2, 'NO_RESULT': 3, 'MISSING': 4}

    rows.sort(key=lambda r: (_rank.get(r['result'], 9), -(r['score'] or -999.0)))

    counts: dict[str, int] = {}
    for r in rows:
        counts[r['result']] = counts.get(r['result'], 0) + 1

    for result, count in sorted(counts.items(), key=lambda kv: _rank.get(kv[0], 9)):
        print(f"  {result:<12}: {count}")
    print()

    hdr = f"{'Response':<10} {'Orbit':<6} {'Peers':<10} {'Result':<10} {'Score':>8}  Notes"
    print(hdr)
    print("-" * 68)

    prev_result = None
    for r in rows:
        if r['result'] != prev_result:
            if prev_result is not None:
                print()
            prev_result = r['result']

        peers   = ','.join(r['orbit_peers']) if r['orbit_peers'] else '—'
        score_s = f"{r['score']:+.4f}" if r['score'] is not None else '?'

        notes = []
        if r['result'] not in ('MISSING',) and r['override_ok'] is False:
            notes.append(f"OVERRIDE FAIL (played {r['our_move'] or '?'})")
        if r['result'] == 'WIN':
            notes.append('<<< WIN! <<<')
        if r.get('error'):
            notes.append('ERROR')

        print(f"{r['response']:<10} {r['orbit']:<6} {peers:<10} {r['result']:<10} {score_s:>8}  {'  '.join(notes)}")

    print()

    # Orbit summary
    print(f"Orbit summary (AlphaQ opened {alphaq_opening}, orbit {_ORBIT[alphaq_edge] if alphaq_edge is not None else '?'}):")
    print(f"  {'Orbit':<6} {'Edge':<8} {'G':<10} {'G score':>8}   {'P':<10} {'P score':>8}")
    print(f"  {'-'*6} {'-'*8} {'-'*10} {'-'*8}   {'-'*10} {'-'*8}")

    by_resp = {r['response']: r for r in rows}
    shown: set[int] = set()
    for orbit in sorted(set(_ORBIT[e] for e in range(15) if e != alphaq_edge)):
        for edge in _ORBIT_EDGES[orbit]:
            if edge == alphaq_edge:
                continue
            rg = by_resp.get(f"E{edge}G")
            rp = by_resp.get(f"E{edge}P")
            g_res   = rg['result'] if rg else '?'
            g_score = f"{rg['score']:+.4f}" if rg and rg['score'] is not None else '?'
            p_res   = rp['result'] if rp else '?'
            p_score = f"{rp['score']:+.4f}" if rp and rp['score'] is not None else '?'
            print(f"  {orbit:<6} E{edge:<7} {g_res:<10} {g_score:>8}   {p_res:<10} {p_score:>8}")

    print()

    wins = [r for r in rows if r['result'] == 'WIN']
    if wins:
        print("!" * 70)
        print(f"  {len(wins)} WIN(S) FOUND as P2:")
        for r in wins:
            print(f"    AlphaQ: {alphaq_opening}  Our response: {r['response']}"
                  f"  orbit={r['orbit']}  score={r['score']:+.4f}")
        print("!" * 70)
    else:
        print("No wins found. AlphaQ draws or wins against all tested P2 responses.")

    print()


if __name__ == '__main__':
    main()
