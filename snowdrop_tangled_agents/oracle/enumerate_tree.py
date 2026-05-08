"""
Step A2: Enumerate reachable terminal states using the AlphaQ Oracle.

Performs depth-first enumeration from each of 30 openings (15 edges x 2 colors),
using the oracle to predict AlphaQ's responses. When the oracle has high confidence,
opponent branches collapse to a single path. Oracle gaps (no data) terminate that
branch early and are reported.
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .build_oracle import OracleResponseTable, DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class TerminalResult:
    """A reachable terminal (or gap) state found during enumeration."""
    state: str                          # 15-char board state
    path: list                          # [(player, edge, color, confidence?), ...]
    opening: tuple                      # (edge, color) of first move
    oracle_gap: bool = False            # True if stopped due to missing oracle data
    gap_grey_count: int = 0             # grey edges remaining at gap
    min_path_confidence: float = 1.0    # lowest oracle confidence along path


@dataclass
class EnumerationStats:
    """Aggregated statistics from tree enumeration."""
    total_terminals: int = 0
    unique_terminals: int = 0
    oracle_gaps: int = 0
    total_paths_explored: int = 0
    terminals_per_opening: dict = field(default_factory=dict)
    gaps_per_opening: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def enumerate_game_tree(
    oracle: OracleResponseTable,
    confidence_threshold: float = 0.9,
    min_branch_prob: float = 0.05,
    max_gap_branches: int = 0,
    progress_interval: int = 100000,
) -> tuple[list[TerminalResult], EnumerationStats]:
    """Enumerate all reachable terminal states from all 30 openings.

    Args:
        oracle: The AlphaQ oracle response table.
        confidence_threshold: Above this, treat opponent response as deterministic.
        min_branch_prob: Minimum probability to explore a non-deterministic branch.
        max_gap_branches: When oracle has no data, try this many random branches
                          (0 = stop at gap, which is the conservative default).
        progress_interval: Print progress every N paths explored.

    Returns:
        Tuple of (results list, stats).
    """
    all_results: list[TerminalResult] = []
    stats = EnumerationStats()
    start_time = time.time()

    # Enumerate from each of 30 openings
    for opening_edge in range(15):
        for opening_color in ("G", "P"):
            opening = (opening_edge, opening_color)
            opening_key = f"E{opening_edge}{opening_color}"

            # Apply opening move (it's our turn first)
            initial_state = list("---------------")
            initial_state[opening_edge] = opening_color
            initial_state_str = "".join(initial_state)

            results_before = len(all_results)

            # DFS enumeration — opponent's turn after our opening
            _enumerate_dfs(
                state=initial_state_str,
                is_our_turn=False,
                oracle=oracle,
                path=[("us", opening_edge, opening_color)],
                results=all_results,
                opening=opening,
                confidence_threshold=confidence_threshold,
                min_branch_prob=min_branch_prob,
                max_gap_branches=max_gap_branches,
                min_conf_so_far=1.0,
                stats=stats,
                progress_interval=progress_interval,
                visited=set(),
            )

            opening_results = len(all_results) - results_before
            terminals = sum(1 for r in all_results[results_before:] if not r.oracle_gap)
            gaps = sum(1 for r in all_results[results_before:] if r.oracle_gap)
            stats.terminals_per_opening[opening_key] = terminals
            stats.gaps_per_opening[opening_key] = gaps

            logger.info(f"  {opening_key}: {terminals} terminals, {gaps} gaps")

    stats.elapsed_seconds = time.time() - start_time
    stats.total_terminals = sum(1 for r in all_results if not r.oracle_gap)
    stats.oracle_gaps = sum(1 for r in all_results if r.oracle_gap)

    # Count unique terminal states
    unique = set()
    for r in all_results:
        if not r.oracle_gap:
            unique.add(r.state)
    stats.unique_terminals = len(unique)

    return all_results, stats


def _enumerate_dfs(
    state: str,
    is_our_turn: bool,
    oracle: OracleResponseTable,
    path: list,
    results: list[TerminalResult],
    opening: tuple,
    confidence_threshold: float,
    min_branch_prob: float,
    max_gap_branches: int,
    min_conf_so_far: float,
    stats: EnumerationStats,
    progress_interval: int,
    visited: set,
):
    """Recursive DFS enumeration."""
    grey_edges = [i for i, c in enumerate(state) if c == "-"]

    # Terminal state — no grey edges remain
    if not grey_edges:
        results.append(TerminalResult(
            state=state,
            path=list(path),
            opening=opening,
            oracle_gap=False,
            gap_grey_count=0,
            min_path_confidence=min_conf_so_far,
        ))
        stats.total_paths_explored += 1
        if stats.total_paths_explored % progress_interval == 0:
            logger.info(f"    ...explored {stats.total_paths_explored} paths, "
                        f"{len(results)} results so far")
        return

    if is_our_turn:
        # Explore all our legal moves (grey edges x 2 colors)
        for edge in grey_edges:
            for color in ("G", "P"):
                new_state = state[:edge] + color + state[edge + 1:]
                path.append(("us", edge, color))
                _enumerate_dfs(
                    new_state, False, oracle, path, results, opening,
                    confidence_threshold, min_branch_prob, max_gap_branches,
                    min_conf_so_far, stats, progress_interval, visited,
                )
                path.pop()
    else:
        # Opponent's turn — use oracle
        if oracle.has_data(state):
            conf = oracle.confidence(state)

            if conf >= confidence_threshold:
                # Deterministic: follow single branch
                (edge, color), _ = oracle.predict(state)
                new_state = state[:edge] + color + state[edge + 1:]
                new_conf = min(min_conf_so_far, conf)
                path.append(("opp", edge, color, conf))
                _enumerate_dfs(
                    new_state, True, oracle, path, results, opening,
                    confidence_threshold, min_branch_prob, max_gap_branches,
                    new_conf, stats, progress_interval, visited,
                )
                path.pop()
            else:
                # Non-deterministic: branch over top responses
                top = oracle.top_responses(state, n=3)
                for (edge, color), prob in top:
                    if prob >= min_branch_prob:
                        new_state = state[:edge] + color + state[edge + 1:]
                        new_conf = min(min_conf_so_far, prob)
                        path.append(("opp", edge, color, prob))
                        _enumerate_dfs(
                            new_state, True, oracle, path, results, opening,
                            confidence_threshold, min_branch_prob, max_gap_branches,
                            new_conf, stats, progress_interval, visited,
                        )
                        path.pop()
        else:
            # No oracle data — record as gap
            results.append(TerminalResult(
                state=state,
                path=list(path),
                opening=opening,
                oracle_gap=True,
                gap_grey_count=len(grey_edges),
                min_path_confidence=min_conf_so_far,
            ))
            stats.total_paths_explored += 1


def save_results(results: list[TerminalResult], path: Optional[Path] = None):
    """Save enumeration results as JSON."""
    if path is None:
        path = DATA_DIR / "reachable_terminals.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable = []
    for r in results:
        d = {
            "state": r.state,
            "path": r.path,
            "opening": list(r.opening),
            "oracle_gap": r.oracle_gap,
            "gap_grey_count": r.gap_grey_count,
            "min_path_confidence": r.min_path_confidence,
        }
        serializable.append(d)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=1)
    logger.info(f"Saved {len(results)} results to {path}")


def print_report(results: list[TerminalResult], stats: EnumerationStats):
    """Print human-readable enumeration report."""
    print("\n" + "=" * 60)
    print("  Game Tree Enumeration — Summary")
    print("=" * 60)
    print(f"  Total paths explored:           {stats.total_paths_explored}")
    print(f"  Terminal states found:          {stats.total_terminals}")
    print(f"  Unique terminal states:         {stats.unique_terminals}")
    print(f"  Oracle gaps (no data):          {stats.oracle_gaps}")
    print(f"  Elapsed time:                   {stats.elapsed_seconds:.1f}s")

    # Terminals per opening
    print("\n  Terminals per opening (top 10):")
    sorted_openings = sorted(
        stats.terminals_per_opening.items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    for opening, count in sorted_openings:
        gaps = stats.gaps_per_opening.get(opening, 0)
        print(f"    {opening:4s}: {count:6d} terminals, {gaps:4d} gaps")

    # Gaps per opening (worst)
    print("\n  Openings with most oracle gaps (top 5):")
    sorted_gaps = sorted(
        stats.gaps_per_opening.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]
    for opening, gaps in sorted_gaps:
        terminals = stats.terminals_per_opening.get(opening, 0)
        print(f"    {opening:4s}: {gaps:4d} gaps, {terminals:6d} terminals")

    # Confidence distribution of terminals
    if results:
        confs = [r.min_path_confidence for r in results if not r.oracle_gap]
        if confs:
            avg_conf = sum(confs) / len(confs)
            high_conf = sum(1 for c in confs if c >= 0.9)
            print(f"\n  Path confidence (terminals only):")
            print(f"    Average min-path confidence:  {avg_conf:.3f}")
            print(f"    Paths with conf >= 0.9:       {high_conf} ({high_conf/len(confs)*100:.1f}%)")

    # Gap depth distribution
    gap_results = [r for r in results if r.oracle_gap]
    if gap_results:
        grey_dist = {}
        for r in gap_results:
            grey_dist[r.gap_grey_count] = grey_dist.get(r.gap_grey_count, 0) + 1
        print(f"\n  Oracle gap depth distribution:")
        for grey, count in sorted(grey_dist.items()):
            print(f"    {grey:2d} grey edges remaining: {count} gaps")

    print("=" * 60)


def main(oracle: Optional[OracleResponseTable] = None,
         confidence_threshold: float = 0.9,
         db_path: Optional[Path] = None):
    """Run game tree enumeration."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if oracle is None:
        oracle = OracleResponseTable.load()

    print(f"\nEnumerating game tree (confidence threshold: {confidence_threshold})...")
    results, stats = enumerate_game_tree(
        oracle,
        confidence_threshold=confidence_threshold,
    )

    print_report(results, stats)

    output_path = DATA_DIR / "reachable_terminals.json"
    save_results(results, output_path)
    print(f"\nSaved to {output_path}")

    return results, stats


if __name__ == "__main__":
    main()
