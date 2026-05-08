"""
Step A1: Build the AlphaQ Oracle response table from game history.

Extracts opponent move patterns from game_stats.db and builds a lookup table
mapping board states to predicted AlphaQ responses with confidence scores.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".tangled" / "game_stats.db"
DATA_DIR = Path(__file__).parent / "data"


class OracleResponseTable:
    """AlphaQ response predictor keyed by board state.

    Stores observation counts for each (board_state, edge, color) tuple
    and provides prediction/confidence methods.
    """

    def __init__(self):
        # {board_state: {(edge, color): observation_count}}
        self.responses: dict[str, dict[tuple[int, str], int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def add_observation(self, board_state: str, edge: int, color: str, count: int = 1):
        self.responses[board_state][(edge, color)] += count

    def has_data(self, state: str) -> bool:
        return state in self.responses and len(self.responses[state]) > 0

    def total_observations(self, state: str) -> int:
        if state not in self.responses:
            return 0
        return sum(self.responses[state].values())

    def confidence(self, state: str) -> float:
        """Return confidence = top_count / total for a state."""
        if not self.has_data(state):
            return 0.0
        counts = self.responses[state]
        total = sum(counts.values())
        if total == 0:
            return 0.0
        top_count = max(counts.values())
        return top_count / total

    def predict(self, state: str) -> Optional[tuple[tuple[int, str], float]]:
        """Return (best_response, confidence) for a board state, or None."""
        if not self.has_data(state):
            return None
        counts = self.responses[state]
        total = sum(counts.values())
        if total == 0:
            return None
        best_move = max(counts, key=counts.get)
        conf = counts[best_move] / total
        return (best_move, conf)

    def top_responses(self, state: str, n: int = 3) -> list[tuple[tuple[int, str], float]]:
        """Return top-n responses with probabilities."""
        if not self.has_data(state):
            return []
        counts = self.responses[state]
        total = sum(counts.values())
        if total == 0:
            return []
        sorted_moves = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return [(move, cnt / total) for move, cnt in sorted_moves[:n]]

    def save(self, path: Optional[Path] = None):
        """Save as JSON."""
        if path is None:
            path = DATA_DIR / "oracle_responses.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert tuple keys to strings for JSON serialization
        serializable = {}
        for state, moves in self.responses.items():
            serializable[state] = {
                f"{edge},{color}": count
                for (edge, color), count in moves.items()
            }

        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, sort_keys=True)
        logger.info(f"Saved oracle response table to {path}")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "OracleResponseTable":
        """Load from JSON."""
        if path is None:
            path = DATA_DIR / "oracle_responses.json"

        with open(path) as f:
            data = json.load(f)

        oracle = cls()
        for state, moves in data.items():
            for move_key, count in moves.items():
                edge_str, color = move_key.split(",")
                oracle.responses[state][(int(edge_str), color)] = count
        return oracle

    def stats(self) -> dict:
        """Compute summary statistics."""
        total_states = len(self.responses)
        if total_states == 0:
            return {"total_states": 0}

        confidences = [self.confidence(s) for s in self.responses]
        obs_counts = [self.total_observations(s) for s in self.responses]
        deterministic = sum(1 for c in confidences if c >= 1.0)
        states_2plus = sum(1 for o in obs_counts if o >= 2)

        # Phase distribution
        phase_counts = defaultdict(int)
        for state in self.responses:
            grey = state.count("-")
            phase_counts[grey] += 1

        return {
            "total_states": total_states,
            "states_observed_2plus": states_2plus,
            "fully_deterministic": deterministic,
            "deterministic_pct": deterministic / total_states * 100 if total_states else 0,
            "avg_confidence": sum(confidences) / len(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "total_observations": sum(obs_counts),
            "avg_observations_per_state": sum(obs_counts) / total_states,
            "phase_distribution": dict(sorted(phase_counts.items())),
        }


def build_oracle(db_path: Optional[Path] = None, opponent: str = "alphaq") -> OracleResponseTable:
    """Build oracle response table from game database.

    Uses both opponent_history (has board_state_before directly) and moves table
    (reconstructs board_state_before from prior move's state_after).

    Args:
        db_path: Path to game_stats.db
        opponent: Opponent name pattern (SQL LIKE)

    Returns:
        Populated OracleResponseTable
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    oracle = OracleResponseTable()
    conn = sqlite3.connect(db_path)

    try:
        # Source 1: opponent_history table (has board_state_before directly)
        hist_count = 0
        try:
            rows = conn.execute("""
                SELECT board_state_before, edge, color, COUNT(*) as cnt
                FROM opponent_history
                WHERE opponent_name LIKE ?
                  AND board_state_before IS NOT NULL
                  AND LENGTH(board_state_before) = 15
                GROUP BY board_state_before, edge, color
            """, (f"%{opponent}%",)).fetchall()

            for state, edge, color, cnt in rows:
                oracle.add_observation(state, edge, color, cnt)
                hist_count += cnt

            logger.info(f"opponent_history: {hist_count} observations from {len(rows)} state-action pairs")
        except sqlite3.OperationalError as e:
            logger.warning(f"opponent_history table not available: {e}")

        # Source 2: moves table (reconstruct board_state_before)
        moves_count = 0
        try:
            rows = conn.execute("""
                SELECT m_prev.state_after as board_state_before,
                       m_opp.edge, m_opp.color
                FROM moves m_opp
                JOIN moves m_prev ON m_prev.game_id = m_opp.game_id
                                 AND m_prev.move_number = m_opp.move_number
                                 AND m_prev.player = 'us'
                JOIN games g ON g.id = m_opp.game_id
                WHERE m_opp.player = 'opponent'
                  AND LOWER(g.opponent) LIKE LOWER(?)
                  AND m_prev.state_after IS NOT NULL
                  AND LENGTH(m_prev.state_after) = 15
            """, (f"%{opponent}%",)).fetchall()

            for state, edge, color in rows:
                oracle.add_observation(state, edge, color)
                moves_count += 1

            logger.info(f"moves table: {moves_count} observations reconstructed")
        except sqlite3.OperationalError as e:
            logger.warning(f"moves table query failed: {e}")

        logger.info(f"Total: {hist_count + moves_count} observations, "
                     f"{len(oracle.responses)} distinct board states")

    finally:
        conn.close()

    return oracle


def print_report(oracle: OracleResponseTable):
    """Print human-readable oracle statistics."""
    s = oracle.stats()

    print("\n" + "=" * 60)
    print("  AlphaQ Oracle Response Table — Summary")
    print("=" * 60)
    print(f"  Distinct board states:          {s['total_states']}")
    print(f"  States observed 2+ times:       {s['states_observed_2plus']}")
    print(f"  Fully deterministic (conf=1.0): {s['fully_deterministic']} "
          f"({s['deterministic_pct']:.1f}%)")
    print(f"  Average confidence:             {s['avg_confidence']:.3f}")
    print(f"  Min / Max confidence:           {s['min_confidence']:.3f} / {s['max_confidence']:.3f}")
    print(f"  Total observations:             {s['total_observations']}")
    print(f"  Avg observations per state:     {s['avg_observations_per_state']:.1f}")

    print("\n  Phase distribution (grey edges -> state count):")
    for grey, count in sorted(s["phase_distribution"].items()):
        phase = ("opening" if grey >= 12 else "mid" if grey >= 8
                 else "late" if grey >= 4 else "endgame")
        print(f"    {grey:2d} grey ({phase:8s}): {count} states")

    # Top-5 most observed states
    print("\n  Top 10 most-observed board states:")
    sorted_states = sorted(
        oracle.responses.items(),
        key=lambda x: sum(x[1].values()),
        reverse=True
    )[:10]
    for state, moves in sorted_states:
        total = sum(moves.values())
        best = max(moves, key=moves.get)
        conf = moves[best] / total
        print(f"    {state}  obs={total:4d}  best=E{best[0]}{best[1]}  conf={conf:.2f}")

    print("=" * 60)


def main(db_path: Optional[Path] = None):
    """Build and save the oracle response table."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    oracle = build_oracle(db_path)
    print_report(oracle)

    output_path = DATA_DIR / "oracle_responses.json"
    oracle.save(output_path)
    print(f"\nSaved to {output_path}")

    return oracle


if __name__ == "__main__":
    main()
