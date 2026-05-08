"""
Phase A CLI: Build AlphaQ Oracle, enumerate game tree, score terminals.

Usage:
    poetry run python -m snowdrop_tangled_agents.oracle.run_phase_a [--confidence 0.9] [--db-path PATH]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from .build_oracle import build_oracle, print_report as print_oracle_report
from .enumerate_tree import (
    enumerate_game_tree, save_results as save_enum_results,
    print_report as print_enum_report,
)
from .score_terminals import (
    load_lut, load_known_outcomes, score_terminals,
    save_scored, print_report as print_score_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="Phase A: Build AlphaQ Oracle and enumerate reachable terminal states."
    )
    parser.add_argument(
        "--confidence", type=float, default=0.9,
        help="Oracle confidence threshold for deterministic branching (default: 0.9)"
    )
    parser.add_argument(
        "--db-path", type=Path, default=None,
        help="Path to game_stats.db (default: ~/.tangled/game_stats.db)"
    )
    parser.add_argument(
        "--opponent", type=str, default="alphaq",
        help="Opponent name pattern for SQL LIKE (default: alphaq)"
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.5,
        help="Minimum LUT score for win candidates (default: 0.5)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    total_start = time.time()

    # ── Step A1: Build Oracle Response Table ──
    print("\n" + "=" * 60)
    print("  STEP A1: Building Oracle Response Table")
    print("=" * 60)

    oracle = build_oracle(args.db_path, args.opponent)
    print_oracle_report(oracle)

    from .build_oracle import DATA_DIR
    oracle.save(DATA_DIR / "oracle_responses.json")

    # ── Step A2: Enumerate Game Tree ──
    print("\n" + "=" * 60)
    print(f"  STEP A2: Enumerating Game Tree (confidence >= {args.confidence})")
    print("=" * 60)

    results, enum_stats = enumerate_game_tree(
        oracle,
        confidence_threshold=args.confidence,
    )
    print_enum_report(results, enum_stats)
    save_enum_results(results, DATA_DIR / "reachable_terminals.json")

    # ── Step A3: Score Terminals ──
    print("\n" + "=" * 60)
    print("  STEP A3: Scoring Terminal States")
    print("=" * 60)

    lut = load_lut()
    known_outcomes = load_known_outcomes(args.db_path, args.opponent)

    all_scored, win_candidates = score_terminals(
        results, lut, known_outcomes,
        win_candidate_score_threshold=args.score_threshold,
    )
    print_score_report(all_scored, win_candidates, known_outcomes)

    save_scored(all_scored, DATA_DIR / "scored_terminals.json")
    save_scored(win_candidates, DATA_DIR / "win_candidates.json")

    # ── Final Summary ──
    elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print("  PHASE A COMPLETE")
    print("=" * 60)
    print(f"  Oracle states:       {oracle.stats()['total_states']}")
    print(f"  Avg confidence:      {oracle.stats()['avg_confidence']:.3f}")
    print(f"  Terminals found:     {enum_stats.unique_terminals}")
    print(f"  Oracle gaps:         {enum_stats.oracle_gaps}")
    print(f"  Win candidates:      {len(win_candidates)}")
    print(f"  Total time:          {elapsed:.1f}s")
    print("=" * 60)

    if win_candidates:
        print(f"\n  Next: Review win_candidates.json and proceed to Phase B/C")
    else:
        print(f"\n  No candidates found. Consider:")
        print(f"    - Lower --score-threshold (currently {args.score_threshold})")
        print(f"    - Lower --confidence (currently {args.confidence})")
        print(f"    - Play more games to fill oracle gaps ({enum_stats.oracle_gaps} gaps)")


if __name__ == "__main__":
    main()
