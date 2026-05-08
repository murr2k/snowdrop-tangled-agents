"""
Step A3: Score enumerated terminal states using LUT and known outcomes.

Loads the 32,768-entry terminal score LUT, scores each reachable terminal state,
cross-references with known server outcomes from the database, and filters for
win candidates.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .build_oracle import DEFAULT_DB_PATH, DATA_DIR
from .enumerate_tree import TerminalResult

logger = logging.getLogger(__name__)

LUT_PATH = Path(__file__).parent.parent / "matlab" / "rl" / "data" / "terminal_scores.mat"


def state_to_index(state: str) -> int:
    """Convert 15-char state to 1-based LUT index (matching MATLAB stateToIndexStatic)."""
    idx = 1
    for j in range(15):
        if state[j] == "G":
            idx += 2 ** j
    return idx


def load_lut(lut_path: Optional[Path] = None) -> np.ndarray:
    """Load the terminal score LUT from .mat file (supports v7.3 HDF5 format)."""
    if lut_path is None:
        lut_path = LUT_PATH

    if not lut_path.exists():
        raise FileNotFoundError(f"LUT file not found: {lut_path}")

    # Try scipy first (v5/v7), fall back to h5py (v7.3 HDF5)
    try:
        import scipy.io
        data = scipy.io.loadmat(str(lut_path))
        lut = data["terminal_scores"].flatten()
    except NotImplementedError:
        import h5py
        with h5py.File(str(lut_path), "r") as f:
            lut = np.array(f["terminal_scores"]).flatten()

    logger.info(f"Loaded LUT: {len(lut)} entries from {lut_path}")
    return lut


def load_known_outcomes(db_path: Optional[Path] = None,
                        opponent: str = "alphaq") -> dict[str, dict]:
    """Load known terminal state -> server outcome mappings from database.

    Returns:
        {terminal_state: {"result": str, "count": int, "avg_score": float}}
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    outcomes = {}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("""
            SELECT m.state_after as terminal_state,
                   g.result,
                   COUNT(*) as games,
                   AVG(g.final_score) as avg_score
            FROM moves m
            JOIN games g ON m.game_id = g.id
            WHERE g.result IN ('win', 'draw', 'loss')
              AND LOWER(g.opponent) LIKE LOWER(?)
              AND m.state_after NOT LIKE '%-%'
              AND LENGTH(m.state_after) = 15
            GROUP BY m.state_after, g.result
            ORDER BY games DESC
        """, (f"%{opponent}%",)).fetchall()

        for state, result, count, avg_score in rows:
            # If a state has multiple outcomes, keep the most frequent
            if state not in outcomes or count > outcomes[state]["count"]:
                outcomes[state] = {
                    "result": result,
                    "count": count,
                    "avg_score": avg_score,
                }

        logger.info(f"Loaded {len(outcomes)} known terminal state outcomes")
    finally:
        conn.close()

    return outcomes


@dataclass
class ScoredTerminal:
    """A terminal state with LUT score, known outcomes, and reachability info."""
    state: str
    lut_score: float
    lut_index: int
    known_outcome: Optional[str] = None
    known_outcome_count: int = 0
    known_avg_score: Optional[float] = None
    num_paths: int = 0
    min_path_confidence: float = 0.0
    max_path_confidence: float = 0.0
    openings: list = field(default_factory=list)
    best_opening: Optional[str] = None


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two board state strings."""
    return sum(1 for x, y in zip(a, b) if x != y)


def score_terminals(
    results: list[TerminalResult],
    lut: np.ndarray,
    known_outcomes: dict[str, dict],
    win_candidate_score_threshold: float = 0.5,
    win_candidate_confidence_threshold: float = 0.8,
    win_candidate_hamming_threshold: int = 3,
) -> tuple[list[ScoredTerminal], list[ScoredTerminal]]:
    """Score all reachable terminal states and identify win candidates.

    Args:
        results: Enumeration results from Step A2.
        lut: Terminal score LUT (32768 entries, 1-based indexing).
        known_outcomes: Known outcome mappings from database.
        win_candidate_score_threshold: Minimum LUT score for candidates.
        win_candidate_confidence_threshold: Minimum path confidence for candidates.
        win_candidate_hamming_threshold: Minimum Hamming distance from known losses.

    Returns:
        Tuple of (all_scored, win_candidates).
    """
    # Group results by terminal state
    terminal_groups: dict[str, list[TerminalResult]] = defaultdict(list)
    for r in results:
        if not r.oracle_gap:
            terminal_groups[r.state].append(r)

    # Known loss states for Hamming distance filtering
    known_losses = [s for s, o in known_outcomes.items() if o["result"] == "loss"]

    # Score each unique terminal state
    all_scored: list[ScoredTerminal] = []
    for state, paths in terminal_groups.items():
        idx = state_to_index(state)
        # LUT uses 1-based indexing, Python array is 0-based
        lut_score = float(lut[idx - 1]) if idx <= len(lut) else 0.0

        # Known outcome
        outcome = known_outcomes.get(state)
        known_result = outcome["result"] if outcome else None
        known_count = outcome["count"] if outcome else 0
        known_avg = outcome["avg_score"] if outcome else None

        # Path statistics
        confidences = [p.min_path_confidence for p in paths]
        openings_set = set()
        for p in paths:
            openings_set.add(f"E{p.opening[0]}{p.opening[1]}")

        scored = ScoredTerminal(
            state=state,
            lut_score=lut_score,
            lut_index=idx,
            known_outcome=known_result,
            known_outcome_count=known_count,
            known_avg_score=known_avg,
            num_paths=len(paths),
            min_path_confidence=min(confidences),
            max_path_confidence=max(confidences),
            openings=sorted(openings_set),
            best_opening=max(
                openings_set,
                key=lambda o: sum(
                    1 for p in paths
                    if f"E{p.opening[0]}{p.opening[1]}" == o
                ),
            ),
        )
        all_scored.append(scored)

    # Sort all scored by LUT score descending
    all_scored.sort(key=lambda s: s.lut_score, reverse=True)

    # Filter for win candidates
    win_candidates = []
    for s in all_scored:
        # Must be novel (not observed) or not a known loss
        if s.known_outcome in ("loss",):
            continue

        # Must be novel (never observed before) — that's the whole point
        if s.known_outcome is not None:
            continue

        # LUT score threshold
        if s.lut_score < win_candidate_score_threshold:
            continue

        # Path confidence threshold
        if s.max_path_confidence < win_candidate_confidence_threshold:
            continue

        # Hamming distance from known losses
        if known_losses:
            min_hamming = min(hamming_distance(s.state, loss) for loss in known_losses)
            if min_hamming < win_candidate_hamming_threshold:
                continue

        win_candidates.append(s)

    win_candidates.sort(key=lambda s: s.lut_score, reverse=True)

    return all_scored, win_candidates


def save_scored(scored: list[ScoredTerminal], path: Path):
    """Save scored terminals as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for s in scored:
        serializable.append({
            "state": s.state,
            "lut_score": round(s.lut_score, 6),
            "lut_index": s.lut_index,
            "known_outcome": s.known_outcome,
            "known_outcome_count": s.known_outcome_count,
            "known_avg_score": round(s.known_avg_score, 4) if s.known_avg_score else None,
            "num_paths": s.num_paths,
            "min_path_confidence": round(s.min_path_confidence, 4),
            "max_path_confidence": round(s.max_path_confidence, 4),
            "openings": s.openings,
            "best_opening": s.best_opening,
        })

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved {len(scored)} scored terminals to {path}")


def print_report(all_scored: list[ScoredTerminal],
                 win_candidates: list[ScoredTerminal],
                 known_outcomes: dict[str, dict]):
    """Print human-readable scoring report."""
    print("\n" + "=" * 60)
    print("  Terminal State Scoring — Summary")
    print("=" * 60)

    total = len(all_scored)
    known = sum(1 for s in all_scored if s.known_outcome is not None)
    novel = total - known
    known_draws = sum(1 for s in all_scored if s.known_outcome == "draw")
    known_losses = sum(1 for s in all_scored if s.known_outcome == "loss")
    known_wins = sum(1 for s in all_scored if s.known_outcome == "win")

    print(f"  Total unique terminal states:   {total}")
    print(f"  Known outcomes:                 {known} ({known_draws}D / {known_losses}L / {known_wins}W)")
    print(f"  Novel (never observed):         {novel}")
    print(f"  Win candidates:                 {len(win_candidates)}")

    if all_scored:
        scores = [s.lut_score for s in all_scored]
        print(f"\n  LUT score distribution:")
        print(f"    Min:    {min(scores):.4f}")
        print(f"    Max:    {max(scores):.4f}")
        print(f"    Mean:   {sum(scores)/len(scores):.4f}")
        print(f"    Median: {sorted(scores)[len(scores)//2]:.4f}")

    # Score distribution of known outcomes
    print(f"\n  Known outcome LUT scores:")
    for outcome_type in ("draw", "loss", "win"):
        subset = [s for s in all_scored if s.known_outcome == outcome_type]
        if subset:
            scores = [s.lut_score for s in subset]
            print(f"    {outcome_type:5s}: n={len(subset):3d}, "
                  f"score=[{min(scores):.3f}, {max(scores):.3f}], "
                  f"avg={sum(scores)/len(scores):.3f}")

    # Top win candidates
    if win_candidates:
        print(f"\n  Top 20 win candidates:")
        print(f"    {'State':<17s} {'LUT':>7s} {'Paths':>6s} {'Conf':>5s} {'Opening'}")
        print(f"    {'-'*17} {'-'*7} {'-'*6} {'-'*5} {'-'*8}")
        for s in win_candidates[:20]:
            print(f"    {s.state}  {s.lut_score:+.4f}  {s.num_paths:5d}  "
                  f"{s.max_path_confidence:.2f}  {s.best_opening}")
    else:
        print(f"\n  No win candidates found with current thresholds.")
        print(f"  Consider lowering the score threshold or confidence threshold.")

    print("=" * 60)


def main(results: Optional[list[TerminalResult]] = None,
         db_path: Optional[Path] = None):
    """Score terminals and identify win candidates."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load enumeration results if not provided
    if results is None:
        results_path = DATA_DIR / "reachable_terminals.json"
        with open(results_path) as f:
            raw = json.load(f)
        results = [
            TerminalResult(
                state=r["state"],
                path=r["path"],
                opening=tuple(r["opening"]),
                oracle_gap=r["oracle_gap"],
                gap_grey_count=r["gap_grey_count"],
                min_path_confidence=r["min_path_confidence"],
            )
            for r in raw
        ]
        logger.info(f"Loaded {len(results)} enumeration results")

    # Load LUT and known outcomes
    lut = load_lut()
    known_outcomes = load_known_outcomes(db_path)

    # Score and filter
    all_scored, win_candidates = score_terminals(results, lut, known_outcomes)

    print_report(all_scored, win_candidates, known_outcomes)

    # Save outputs
    save_scored(all_scored, DATA_DIR / "scored_terminals.json")
    save_scored(win_candidates, DATA_DIR / "win_candidates.json")
    print(f"\nSaved scored_terminals.json ({len(all_scored)} states)")
    print(f"Saved win_candidates.json ({len(win_candidates)} candidates)")

    return all_scored, win_candidates


if __name__ == "__main__":
    main()
