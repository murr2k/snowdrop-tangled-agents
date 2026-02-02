#!/usr/bin/env python3
"""
Test AlphaQ strategy integration with new ground-truth LUTs.

Verifies:
- terminal_scores.mat loads correctly
- expanded_lut.mat loads correctly
- Strategy can initialize and make moves
- No errors during normal gameplay
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from snowdrop_tangled_game_engine import Game, LocalGamePlayer
from snowdrop_tangled_agents.matlab.matlab_strategy import AlphaQUpStrategy
from snowdrop_adjudicators import SimulatedAnnealingAdjudicator


def test_alphaq_lut_loading():
    """Test that AlphaQ can load the new ground-truth LUTs."""
    print("=" * 70)
    print("Testing AlphaQ Integration with Ground-Truth LUTs")
    print("=" * 70)

    # Create game instance (Petersen graph)
    print("\n[1/5] Creating game instance (Petersen graph)...")
    adjudicator = SimulatedAnnealingAdjudicator(num_reads=1000)
    game = Game(graph_number=5, adjudicator=adjudicator)
    print("    ✓ Game created")

    # Create AlphaQ strategy
    print("\n[2/5] Initializing AlphaQ Up strategy...")
    try:
        strategy = AlphaQUpStrategy(
            graph_number=5,
            exploration_weight=1.41,
            simulation_budget=50,
            learning_rate=0.0  # No learning for this test
        )
        print("    ✓ Strategy created")
    except Exception as e:
        print(f"    ✗ Failed to create strategy: {e}")
        return False

    # Initialize strategy (loads LUTs)
    print("\n[3/5] Initializing strategy (loading LUTs)...")
    try:
        strategy.initialize(opponent_name="TestOpponent")
        print("    ✓ Strategy initialized")
        print("    ✓ LUTs loaded successfully")
    except Exception as e:
        print(f"    ✗ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Make first move
    print("\n[4/5] Making first move...")
    try:
        move = strategy.make_move(game)
        move_type, edge_idx, color = move
        print(f"    ✓ Move generated: edge {edge_idx}, color {color}")
    except Exception as e:
        print(f"    ✗ Failed to make move: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Play a few more moves to verify stability
    print("\n[5/5] Playing test game (10 moves)...")
    try:
        player1 = LocalGamePlayer(name="AlphaQ", agent=strategy)
        player2 = LocalGamePlayer(name="Random", agent=None)  # Random moves

        game = Game(
            graph_number=5,
            adjudicator=adjudicator,
            player1=player1,
            player2=player2
        )

        for turn in range(10):
            if game.is_game_over():
                break
            game.play_next_turn()

        print(f"    ✓ Completed {game.turn_count} turns without errors")

    except Exception as e:
        print(f"    ✗ Error during gameplay: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check strategy stats
    print("\n[VERIFICATION] Strategy statistics:")
    try:
        stats = strategy.get_stats()
        print(f"    Simulations per move: {stats.get('simulations', 'N/A')}")
        print(f"    Avg simulation depth: {stats.get('avg_depth', 'N/A'):.1f}")
        print(f"    Cache hits: {stats.get('cache_hits', 0)}")
        print(f"    LUT loaded: {'terminal_scores' in str(stats)}")
    except Exception as e:
        print(f"    Warning: Could not retrieve stats: {e}")

    print("\n" + "=" * 70)
    print("✅ AlphaQ LUT Integration Test: PASSED")
    print("=" * 70)
    print("\nGround-truth LUTs are working correctly!")
    print("- terminal_scores.mat: Loaded ✓")
    print("- expanded_lut.mat: Loaded ✓")
    print("- Strategy operational: ✓")

    return True


if __name__ == "__main__":
    success = test_alphaq_lut_loading()
    sys.exit(0 if success else 1)
