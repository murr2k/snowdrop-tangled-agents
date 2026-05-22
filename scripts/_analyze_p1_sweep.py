"""
Analyze results from scripts/sweep_p1_openings.ps1.

Two modes:
  (default)           Reads logs/sweep_E{N}{C}.log.err (one file per opening).
  --combined-log FILE Reads a single combined log from a one-session sweep,
                      paired with --sequence-file to map games to openings.

Usage:
    poetry run python scripts/_analyze_p1_sweep.py
    poetry run python scripts/_analyze_p1_sweep.py \\
        --combined-log logs/sweep_p1_combined.log.err \\
        --sequence-file logs/sweep_p1_sequence.txt
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

# Edge orbit IDs under Stab(p1=5, p2=7) — 9 orbits, order 2 stabilizer
# Index = edge number 0-14
_ORBIT = (0, 0, 1, 2, 3, 4, 5, 2, 5, 6, 7, 6, 8, 4, 3)

_ORBIT_EDGES: dict[int, list[int]] = {}
for _e, _o in enumerate(_ORBIT):
    _ORBIT_EDGES.setdefault(_o, []).append(_e)

_ANSI = re.compile(r'\x1b\[[0-9;]*m')


def strip_ansi(text: str) -> str:
    return _ANSI.sub('', text)


def parse_log(path: Path) -> dict:
    """Extract result, score, and actual opening from one log file."""
    if not path.exists():
        return {'result': 'MISSING', 'score': None, 'actual': None, 'error': None}

    text = strip_ansi(path.read_text(encoding='utf-8', errors='replace'))

    result = 'NO_RESULT'
    m = re.search(r'GAME OVER: (WIN|DRAW|LOSS)', text)
    if m:
        result = m.group(1)

    score = None
    m = re.search(r'Final Score:\s*([+-]?\d+\.\d+)', text)
    if m:
        score = float(m.group(1))

    # Actual P1 opening = first [US] move in the move history block
    actual = None
    m = re.search(r'1\.\s+\[US\s*\]\s+E(\d+)\s+(Green|Purple)', text)
    if m:
        c = 'G' if m.group(2) == 'Green' else 'P'
        actual = f"E{m.group(1)}{c}"

    # Capture any Python exception for debugging
    error = None
    m = re.search(r'(Traceback.*?\n(?:\S.*\n)*)', text, re.DOTALL)
    if m:
        error = m.group(1)[:200]

    return {'result': result, 'score': score, 'actual': actual, 'error': error}


def parse_combined_log(log_path: Path, seq_path: Path) -> list[dict]:
    """Parse a single combined log produced by a one-session sweep."""
    text = strip_ansi(log_path.read_text(encoding='utf-8', errors='replace'))

    # Load the opening sequence: "15 EDGE COLOR" per line
    openings = []
    for line in seq_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        _, edge_s, color = line.split()
        color = 'G' if color.upper() == 'G' else 'P'
        openings.append(f"E{edge_s}{color}")

    # Extract all GAME OVER results in order
    results_found = re.findall(r'GAME OVER: (WIN|DRAW|LOSS)', text)

    # Extract all Final Score values in order
    scores_found = re.findall(r'Final Score:\s*([+-]?\d+\.\d+)', text)

    rows = []
    for i, label in enumerate(openings):
        edge  = int(re.match(r'E(\d+)', label).group(1))
        color = label[-1]
        result = results_found[i] if i < len(results_found) else 'MISSING'
        score  = float(scores_found[i]) if i < len(scores_found) else None
        rows.append({
            'opening': label,
            'edge':    edge,
            'color':   color,
            'orbit':   _ORBIT[edge],
            'orbit_peers': [f"E{e}" for e in _ORBIT_EDGES[_ORBIT[edge]] if e != edge],
            'result':  result,
            'score':   score,
            'actual':  label,  # sequence-driven so override always matches
            'error':   None,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--combined-log', type=Path, default=None,
                        help='Combined log file from single-session sweep')
    parser.add_argument('--sequence-file', type=Path, default=None,
                        help='Sequence file listing openings in order')
    args = parser.parse_args()

    if args.combined_log and args.sequence_file:
        rows = parse_combined_log(args.combined_log, args.sequence_file)
    else:
        rows = []
        for edge in range(15):
            for color in ('G', 'P'):
                label = f"E{edge}{color}"
                log   = LOG_DIR / f"sweep_{label}.log.err"
                data  = parse_log(log)
                rows.append({
                    'opening': label,
                    'edge':    edge,
                    'color':   color,
                    'orbit':   _ORBIT[edge],
                    'orbit_peers': [f"E{e}" for e in _ORBIT_EDGES[_ORBIT[edge]] if e != edge],
                    **data,
                })

    # Sort: WIN < DRAW < LOSS < NO_RESULT < MISSING; within group by score desc
    _rank = {'WIN': 0, 'DRAW': 1, 'LOSS': 2, 'NO_RESULT': 3, 'MISSING': 4}

    def sort_key(r):
        return (_rank.get(r['result'], 9), -(r['score'] or -999.0))

    rows.sort(key=sort_key)

    # Count outcomes
    counts = {'WIN': 0, 'DRAW': 0, 'LOSS': 0, 'NO_RESULT': 0, 'MISSING': 0}
    for r in rows:
        counts[r['result']] = counts.get(r['result'], 0) + 1

    # Print summary header
    print()
    print("=" * 70)
    print("P1 Opening Sweep — Results")
    print(f"  Total openings: {len(rows)}")
    for result, count in counts.items():
        if count:
            print(f"  {result:<12}: {count}")
    print("=" * 70)
    print()

    # Print table
    hdr = f"{'Opening':<9} {'Orbit':<6} {'Peers':<12} {'Result':<10} {'Score':>8}  Notes"
    print(hdr)
    print("-" * 70)

    prev_result = None
    for r in rows:
        if r['result'] != prev_result:
            if prev_result is not None:
                print()
            prev_result = r['result']

        actual  = r['actual'] or '?'
        peers   = ','.join(r['orbit_peers']) if r['orbit_peers'] else '—'
        score_s = f"{r['score']:+.4f}" if r['score'] is not None else '?'

        # Flag override failures and wins
        notes = []
        if r['result'] != 'MISSING' and actual != r['opening']:
            notes.append(f"OVERRIDE FAIL (played {actual})")
        if r['result'] == 'WIN':
            notes.append('<<< WIN! <<<')
        if r['error']:
            notes.append('ERROR')

        note_s = '  '.join(notes)
        print(f"{r['opening']:<9} {r['orbit']:<6} {peers:<12} {r['result']:<10} {score_s:>8}  {note_s}")

    print()

    # Orbit-level summary: for each orbit, show G and P results side by side
    print("Orbit-level summary (G vs P for each orbit):")
    print(f"  {'Orbit':<6} {'Edges':<16} {'G result':<10} {'G score':>8}   {'P result':<10} {'P score':>8}")
    print(f"  {'-'*6} {'-'*16} {'-'*10} {'-'*8}   {'-'*10} {'-'*8}")

    by_opening = {r['opening']: r for r in rows}
    seen_orbits: set[int] = set()

    # Iterate in orbit order
    for orbit in sorted(set(_ORBIT)):
        edges_in_orbit = _ORBIT_EDGES[orbit]
        seen_orbits.add(orbit)

        for edge in edges_in_orbit:
            r_g = by_opening.get(f"E{edge}G")
            r_p = by_opening.get(f"E{edge}P")

            g_res   = r_g['result'] if r_g else '?'
            g_score = f"{r_g['score']:+.4f}" if r_g and r_g['score'] is not None else '?'
            p_res   = r_p['result'] if r_p else '?'
            p_score = f"{r_p['score']:+.4f}" if r_p and r_p['score'] is not None else '?'

            peers = [e for e in edges_in_orbit if e != edge]
            peers_s = ','.join(f"E{e}" for e in peers) if peers else '(singleton)'
            edge_s  = f"E{edge} (orbit {orbit})"

            print(f"  {orbit:<6} {edge_s:<16} {g_res:<10} {g_score:>8}   {p_res:<10} {p_score:>8}")

    print()

    # Highlight any wins
    wins = [r for r in rows if r['result'] == 'WIN']
    if wins:
        print("!" * 70)
        print(f"  {len(wins)} WIN(S) FOUND — CANDIDATE EXPLOIT(S):")
        for r in wins:
            print(f"    {r['opening']}  orbit={r['orbit']}  score={r['score']:+.4f}")
        print("!" * 70)
    else:
        print("No wins found. AlphaQ draws or wins against all 30 P1 openings.")

    print()


if __name__ == "__main__":
    main()
