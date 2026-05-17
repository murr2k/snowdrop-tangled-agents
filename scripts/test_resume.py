"""Unit tests for Investigation 4 resume mechanism.

Tests:
1. lut_variant stored in runs and used for matching
2. Abandoned games not counted as completed
3. Opening index restored from DB on resume
4. get_next_game_number advances past abandoned slots
5. Resume joins existing run, new config creates new run
"""
import uuid
import sys
from pathlib import Path
from snowdrop_tangled_agents.stats.collector import StatsCollector
from snowdrop_tangled_agents.strategy.terminal_explorer_strategy import (
    TerminalExplorerStrategy, ALL_OPENINGS,
)

c = StatsCollector()
FAILURES = []


def check(label, got, expected):
    if got == expected:
        print(f"  PASS  {label}: {got!r}")
    else:
        print(f"  FAIL  {label}: got {got!r}, expected {expected!r}")
        FAILURES.append(label)


def fake_game(run_id, result, game_number):
    gid = str(uuid.uuid4())[:8]
    conn = c.get_connection()
    conn.execute(
        "INSERT INTO games (id, opponent, strategy, result, final_score, "
        "total_moves, run_id, game_number) VALUES (?,?,?,?,?,?,?,?)",
        (gid, "melissa", "terminal_explorer", result, 0.0, 15, run_id, game_number),
    )
    conn.commit()
    conn.close()
    return gid


def cleanup_run(run_id):
    conn = c.get_connection()
    conn.execute("DELETE FROM games WHERE run_id=?", (run_id,))
    conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
    conn.commit()
    conn.close()


print("\n=== Test 1: lut_variant stored and used for matching ===")
r1_id = c.start_run(planned_games=6, strategy="terminal_explorer",
                    opponent="melissa", seat=1, lut_variant="calib")
conn = c.get_connection()
row = conn.execute("SELECT lut_variant FROM runs WHERE id=?", (r1_id,)).fetchone()
conn.close()
check("lut_variant stored as 'calib'", row[0], "calib")

# Same config -> joins existing run
rid, gnum = c.get_or_create_run(planned_games=6, strategy="terminal_explorer",
                                 opponent="melissa", seat=1, lut_variant="calib")
check("same config joins existing run", rid, r1_id)
check("no games yet -> game_number=1", gnum, 1)

# Different lut_variant -> creates new run
rid2, _ = c.get_or_create_run(planned_games=6, strategy="terminal_explorer",
                               opponent="melissa", seat=1, lut_variant="sa")
check("different lut_variant creates new run", rid2 != r1_id, True)
cleanup_run(rid2)

# None / default -> treated as 'sa', does NOT match calib run
rid3, _ = c.get_or_create_run(planned_games=6, strategy="terminal_explorer",
                               opponent="melissa", seat=1, lut_variant=None)
check("lut_variant=None treated as 'sa', new run", rid3 != r1_id, True)
cleanup_run(rid3)


print("\n=== Test 2: abandoned games not counted as completed ===")
g1 = fake_game(r1_id, "draw", 1)
g2 = fake_game(r1_id, "draw", 2)
g3 = fake_game(r1_id, "abandoned", 3)
c.update_run_completed(r1_id)

conn = c.get_connection()
completed = conn.execute("SELECT completed_games FROM runs WHERE id=?", (r1_id,)).fetchone()[0]
conn.close()
check("2 draws + 1 abandoned -> completed=2", completed, 2)

# Run is still incomplete (2 < 6), so a resume joins it
rid_resume, gnum_resume = c.get_or_create_run(
    planned_games=6, strategy="terminal_explorer",
    opponent="melissa", seat=1, lut_variant="calib",
)
check("resume joins incomplete run", rid_resume, r1_id)
check("next game_number is 4 (past abandoned slot 3)", gnum_resume, 4)


print("\n=== Test 3: opening index restored from completed_games ===")
# completed_games=2, so opening_index should be 2 % 30 = 2
opening_index = completed % len(ALL_OPENINGS)
s = TerminalExplorerStrategy(opening_index_start=opening_index)
check("opening_index=2", s.opening_index, 2)
check("games_played=2", s.games_played, 2)
check("first opening at index 2 is E1G", ALL_OPENINGS[s.opening_index], (1, "G"))

# After end_game() it advances to index 3
s.current_opening = ALL_OPENINGS[s.opening_index]
s.end_game("draw", 0.0)
check("after 1 end_game, opening_index=3", s.opening_index, 3)
check("games_played advances to 3", s.games_played, 3)

# Wrap-around: 29 more games -> index wraps back to 2
s2 = TerminalExplorerStrategy(opening_index_start=29)
check("opening_index_start=29 wraps to 29", s2.opening_index, 29)
s2.current_opening = ALL_OPENINGS[s2.opening_index]
s2.end_game("draw", 0.0)
check("after end_game at 29, wraps to 0", s2.opening_index, 0)


print("\n=== Test 4: get_next_game_number skips abandoned slots ===")
gnum_next = c.get_next_game_number(r1_id)
check("next game_number=4 (max is 3)", gnum_next, 4)

# Add one more game at number 4
g4 = fake_game(r1_id, "draw", 4)
c.update_run_completed(r1_id)
conn = c.get_connection()
completed2 = conn.execute("SELECT completed_games FROM runs WHERE id=?", (r1_id,)).fetchone()[0]
conn.close()
check("completed_games after 3 draws + 1 abandoned = 3", completed2, 3)
gnum_next2 = c.get_next_game_number(r1_id)
check("next game_number=5", gnum_next2, 5)


print("\n=== Cleanup ===")
cleanup_run(r1_id)
print(f"  Removed test run {r1_id} and all its games")


print("\n" + "=" * 50)
if FAILURES:
    print(f"RESULT: {len(FAILURES)} FAILURE(S): {', '.join(FAILURES)}")
    sys.exit(1)
else:
    print(f"RESULT: ALL TESTS PASSED")
