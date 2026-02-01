"""
Tests for MATLAB integration components.

Tests:
- Database migrations
- Stats collector model/opponent methods
- Unified bridge functionality
- Training orchestrator
- MATLAB strategy integration
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path


class TestDatabaseMigrations:
    """Test database schema migrations."""

    def test_migrations_import(self):
        """Test that migrations module imports correctly."""
        from snowdrop_tangled_agents.stats.migrations import (
            run_migrations,
            get_migration_status,
            MIGRATIONS,
        )
        assert len(MIGRATIONS) >= 4  # v2, v3, v4, v5

    def test_run_migrations_on_fresh_db(self):
        """Test running migrations on a fresh database."""
        from snowdrop_tangled_agents.stats.migrations import (
            run_migrations,
            get_migration_status,
        )

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            migrations_applied = run_migrations(conn)
            assert migrations_applied >= 4  # Should apply v2-v5

            status = get_migration_status(conn)
            assert status['current_version'] >= 5
            assert status['is_up_to_date']
            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_migration_creates_tables(self):
        """Test that migrations create expected tables."""
        from snowdrop_tangled_agents.stats.migrations import run_migrations

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            conn = sqlite3.connect(db_path)
            run_migrations(conn)

            # Check tables exist
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}

            assert 'models' in tables
            assert 'opponents' in tables
            assert 'training_data' in tables
            assert 'opponent_history' in tables
            assert 'schema_version' in tables

            conn.close()
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestStatsCollector:
    """Test stats collector model/opponent methods."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        import gc
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = Path(f.name)

        yield db_path

        # Force garbage collection to close any lingering connections
        gc.collect()
        try:
            db_path.unlink(missing_ok=True)
        except PermissionError:
            pass  # Windows may still hold the file

    def test_collector_init_runs_migrations(self, temp_db):
        """Test that collector initialization runs migrations."""
        from snowdrop_tangled_agents.stats.collector import StatsCollector

        collector = StatsCollector(temp_db)
        status = collector.get_migration_status()
        assert status['current_version'] >= 5

    def test_save_and_get_model(self, temp_db):
        """Test saving and retrieving model metadata."""
        from snowdrop_tangled_agents.stats.collector import StatsCollector

        collector = StatsCollector(temp_db)

        # Save a model
        model_id = collector.save_model(
            name='test_value_net',
            model_type='value',
            training_games=100,
            validation_loss=0.05,
            file_path='/path/to/model.mat',
            hyperparameters={'epochs': 50, 'lr': 0.001},
            set_active=True,
        )

        assert model_id > 0

        # Get active model
        model = collector.get_active_model('value')
        assert model is not None
        assert model['name'] == 'test_value_net'
        assert model['type'] == 'value'
        assert model['training_games'] == 100
        assert model['validation_loss'] == 0.05
        assert model['hyperparameters']['epochs'] == 50

    def test_save_and_get_opponent(self, temp_db):
        """Test saving and retrieving opponent profiles."""
        from snowdrop_tangled_agents.stats.collector import StatsCollector

        collector = StatsCollector(temp_db)

        # Save an opponent
        features = [0.1] * 20
        opp_id = collector.save_opponent(
            name='test_opponent',
            cluster_id=1,
            features=features,
            win_rate=0.6,
            notes='Test opponent',
        )

        assert opp_id > 0

        # Get opponent
        opponent = collector.get_opponent('test_opponent')
        assert opponent is not None
        assert opponent['name'] == 'test_opponent'
        assert opponent['cluster_id'] == 1
        assert len(opponent['features']) == 20
        assert opponent['win_rate'] == 0.6

    def test_increment_opponent_games(self, temp_db):
        """Test incrementing opponent game counts."""
        from snowdrop_tangled_agents.stats.collector import StatsCollector

        collector = StatsCollector(temp_db)

        # Create opponent
        collector.save_opponent(name='test_opp')

        # Increment games
        collector.increment_opponent_games('test_opp', won=True)
        collector.increment_opponent_games('test_opp', won=False)
        collector.increment_opponent_games('test_opp', won=True)

        opponent = collector.get_opponent('test_opp')
        assert opponent['games_played'] == 3
        assert abs(opponent['win_rate'] - 2/3) < 0.01


class TestUnifiedBridge:
    """Test unified bridge functionality."""

    def test_unified_bridge_import(self):
        """Test that unified bridge imports correctly."""
        from snowdrop_tangled_agents.matlab.unified_bridge import (
            UnifiedMatlabBridge,
            get_unified_bridge,
        )

    def test_unified_bridge_connect(self):
        """Test unified bridge connection."""
        from snowdrop_tangled_agents.matlab.unified_bridge import UnifiedMatlabBridge

        # Create fresh bridge instance (don't use singleton to avoid state issues)
        # Note: In test environment, compiled packages may conflict with MATLAB Engine
        # so we force heuristic mode for reliable testing
        bridge = UnifiedMatlabBridge()

        # Skip compiled packages in test to avoid DLL conflicts
        bridge._prefer_existing = False
        bridge.compiled = None

        backend = bridge.connect()

        # Should return one of: 'compiled', 'engine', 'heuristic'
        assert backend in ('compiled', 'engine', 'heuristic')
        assert bridge.is_available()
        assert bridge.get_backend() == backend

    def test_unified_bridge_heuristic_eval(self):
        """Test heuristic position evaluation."""
        from snowdrop_tangled_agents.matlab.unified_bridge import UnifiedMatlabBridge

        bridge = UnifiedMatlabBridge()
        bridge.backend = 'heuristic'  # Force heuristic mode

        state = '---------------'  # All grey
        value, policy = bridge.evaluate_position(state, is_our_turn=True)

        # Value should be near 0 for empty board
        assert -1.0 <= value <= 1.0

        # Policy should have entries for all grey edges
        assert len(policy) == 30  # 15 edges x 2 colors

        # All probabilities should sum to ~1
        assert abs(sum(policy.values()) - 1.0) < 0.01

    def test_unified_bridge_heuristic_classify(self):
        """Test heuristic opponent classification."""
        from snowdrop_tangled_agents.matlab.unified_bridge import UnifiedMatlabBridge

        bridge = UnifiedMatlabBridge()

        # Test aggressive features
        features = [0.0] * 20
        features[16] = 0.5  # High opening aggression
        style, conf = bridge._heuristic_classify(features)
        assert style == 1  # Aggressive

        # Test defensive features
        features = [0.0] * 20
        features[17] = 0.7  # High response rate
        style, conf = bridge._heuristic_classify(features)
        assert style == 2  # Defensive

    def test_uniform_priors(self):
        """Test uniform prior generation."""
        from snowdrop_tangled_agents.matlab.unified_bridge import UnifiedMatlabBridge

        bridge = UnifiedMatlabBridge()

        # Fully grey board
        state = '---------------'
        priors = bridge._uniform_priors(state)
        assert len(priors) == 30
        assert abs(sum(priors.values()) - 1.0) < 0.01

        # Partially filled board
        state = 'GP-------------'
        priors = bridge._uniform_priors(state)
        assert len(priors) == 26  # 13 grey edges x 2 colors
        assert abs(sum(priors.values()) - 1.0) < 0.01


class TestCompiledBridge:
    """Test compiled bridge functionality."""

    def test_packages_available(self):
        """Test packages_available function."""
        from snowdrop_tangled_agents.matlab.compiled_bridge import packages_available

        available = packages_available()
        assert isinstance(available, dict)
        assert 'value_network' in available
        assert 'opponent_model' in available
        assert 'training' in available


class TestTrainingOrchestrator:
    """Test training orchestrator."""

    def test_orchestrator_import(self):
        """Test that training orchestrator imports correctly."""
        from snowdrop_tangled_agents.matlab.training import (
            TrainingOrchestrator,
            get_training_orchestrator,
            print_training_status,
        )

    def test_orchestrator_status(self):
        """Test orchestrator status."""
        from snowdrop_tangled_agents.matlab.training import get_training_orchestrator

        orchestrator = get_training_orchestrator()
        status = orchestrator.get_status()

        assert 'database_path' in status
        assert 'total_games' in status
        assert 'compiled_training' in status
        assert 'matlab_engine' in status
        assert 'ready_for_training' in status


class TestMatlabStrategy:
    """Test MATLAB strategy integration."""

    def test_strategy_import(self):
        """Test that MATLAB strategy imports correctly."""
        from snowdrop_tangled_agents.matlab.matlab_strategy import (
            MatlabEnhancedStrategy,
        )

    def test_strategy_init(self):
        """Test strategy initialization."""
        from snowdrop_tangled_agents.matlab.matlab_strategy import MatlabEnhancedStrategy

        strategy = MatlabEnhancedStrategy(
            mcts_time_limit=1.0,
            mcts_iterations=100,
            use_nn_priors=True,
            use_opponent_adaptation=True,
        )

        assert strategy.use_nn_priors is True
        assert strategy.use_opponent_adaptation is True
        assert strategy.backend is None  # Not initialized yet

    def test_strategy_initialize(self):
        """Test strategy initialization with backend."""
        from snowdrop_tangled_agents.matlab.matlab_strategy import MatlabEnhancedStrategy

        strategy = MatlabEnhancedStrategy(
            mcts_time_limit=1.0,
            mcts_iterations=100,
        )
        result = strategy.initialize()

        # Should return True (even if using heuristic fallback)
        assert strategy.backend in ('compiled', 'engine', 'heuristic')

    def test_strategy_calculate_move_opening(self):
        """Test opening move calculation."""
        from snowdrop_tangled_agents.matlab.matlab_strategy import MatlabEnhancedStrategy

        strategy = MatlabEnhancedStrategy(
            mcts_time_limit=0.5,
            mcts_iterations=50,
            opening_moves=3,
        )
        strategy.initialize()

        # Empty board - should use opening book
        state = '---------------'
        move = strategy.calculate_move(state, score=0.0)

        assert move is not None
        edge, color = move
        assert 0 <= edge <= 14
        assert color in ('G', 'P')

        # Should be from opening sequence
        assert (edge, color) in strategy.opening_sequence

    def test_strategy_get_stats(self):
        """Test strategy statistics."""
        from snowdrop_tangled_agents.matlab.matlab_strategy import MatlabEnhancedStrategy

        strategy = MatlabEnhancedStrategy(mcts_time_limit=0.5)
        strategy.initialize()

        stats = strategy.get_stats()
        assert 'backend' in stats
        assert 'nn_calls' in stats
        assert 'adapt_calls' in stats
        assert 'matlab_available' in stats


class TestAlphaQExplorerThompson:
    """Test Thompson Sampling opening selection in AlphaQExplorerStrategy."""

    def test_thompson_favours_safe_opening(self):
        """
        Test that Thompson Sampling favours openings with wins
        over openings with losses. When two openings are compared directly
        (all others set to equal losses), the one with wins is favored.
        """
        from snowdrop_tangled_agents.matlab.matlab_strategy import AlphaQExplorerStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            strategy = AlphaQExplorerStrategy(state_path=state_path)

            # Set up: E0G has 5 wins (mean=0.75), all others untried
            strategy.openings['E0G'] = {'wins': 5, 'draws': 0, 'losses': 0}
            for key in strategy.openings.keys():
                if key != 'E0G':
                    strategy.openings[key] = {'wins': 0, 'draws': 0, 'losses': 0}
            strategy.games_played = 5

            # Sample 1000 times with correct initial state (15 dashes)
            selected = {}
            for _ in range(1000):
                move = strategy.calculate_move("---------------")
                if move:
                    edge, color, stats = move
                    opening = f"E{edge}{color}"
                    selected[opening] = selected.get(opening, 0) + 1
                    strategy.current_game_opening = None  # Reset for next iteration

            # E0G (mean=0.857, alpha=6, beta=1) should be favored significantly
            # over untried openings (mean=0.5, alpha=1, beta=1).
            # With 29 untried openings each getting ~34 selections on average,
            # E0G should get substantially more (roughly 2-3x more per sample).
            # Conservatively expect >150 selections out of 1000.
            e0g_count = selected.get('E0G', 0)
            avg_untried_count = sum(c for k, c in selected.items() if k != 'E0G') / 29 if len(selected) > 1 else 0
            assert e0g_count > 150, \
                f"E0G selected {e0g_count}/1000, expected >150. Avg untried: {avg_untried_count:.1f}"

    def test_thompson_explores_untried(self):
        """
        Test that Thompson Sampling explores untried openings at a reasonable rate
        even when all other openings have losses.
        """
        from snowdrop_tangled_agents.matlab.matlab_strategy import AlphaQExplorerStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            strategy = AlphaQExplorerStrategy(state_path=state_path)

            # Set up: 29 openings with 10 losses each, 1 untried
            for key in strategy.openings.keys():
                if key != 'E0G':
                    strategy.openings[key] = {'wins': 0, 'draws': 0, 'losses': 10}
            strategy.openings['E0G'] = {'wins': 0, 'draws': 0, 'losses': 0}
            strategy.games_played = 290

            # Sample 1000 times with correct initial state
            selected = {}
            for _ in range(1000):
                move = strategy.calculate_move("---------------")
                if move:
                    edge, color, stats = move
                    opening = f"E{edge}{color}"
                    selected[opening] = selected.get(opening, 0) + 1
                    strategy.current_game_opening = None

            # E0G (untried) should be selected >5% of the time
            assert selected.get('E0G', 0) > 50, \
                f"E0G selected {selected.get('E0G', 0)}/1000, expected >50"

    def test_migration_v1_to_v2(self):
        """Test migration from v1 (exploration_results) to v2 (openings counts)."""
        from snowdrop_tangled_agents.matlab.matlab_strategy import AlphaQExplorerStrategy
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"

            # Write v1 format state
            v1_data = {
                'phase': 'exploration',
                'exploration_results': {
                    'E0G': [
                        {'score': 0.75, 'result': 'draw'},
                        {'score': 0.71, 'result': 'draw'},
                    ],
                    'E11G': [
                        {'score': 0.58, 'result': 'loss'},
                        {'score': 0.60, 'result': 'loss'},
                    ],
                },
                'exploitation_openings': [],
                'exploitation_index': 0,
            }
            with open(state_path, 'w') as f:
                json.dump(v1_data, f)

            # Load with new code
            strategy = AlphaQExplorerStrategy(state_path=state_path)

            # Check migration
            assert strategy.openings['E0G'] == {'wins': 0, 'draws': 2, 'losses': 0}
            assert strategy.openings['E11G'] == {'wins': 0, 'draws': 0, 'losses': 2}
            assert strategy.games_played == 4

            # Check that state was saved in v2 format
            with open(state_path) as f:
                saved_data = json.load(f)
            assert saved_data['version'] == 2
            assert saved_data['openings']['E0G'] == {'wins': 0, 'draws': 2, 'losses': 0}

    def test_migration_missing_file(self):
        """Test that missing state file initializes all 30 openings cleanly."""
        from snowdrop_tangled_agents.matlab.matlab_strategy import AlphaQExplorerStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "nonexistent" / "state.json"
            strategy = AlphaQExplorerStrategy(state_path=state_path)

            # Should have 30 openings, all zeros
            assert len(strategy.openings) == 30
            for key, counts in strategy.openings.items():
                assert counts == {'wins': 0, 'draws': 0, 'losses': 0}
            assert strategy.games_played == 0

    def test_end_game_normalises_none(self):
        """Test that end_game handles None result gracefully."""
        from snowdrop_tangled_agents.matlab.matlab_strategy import AlphaQExplorerStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            strategy = AlphaQExplorerStrategy(state_path=state_path)
            strategy.initialize()

            # Force an opening
            strategy.current_game_opening = (0, 'G')

            # Call end_game with None (should be normalized to 'draw')
            strategy.end_game(None, 0.5)

            # Check that draws incremented
            assert strategy.openings['E0G']['draws'] == 1
            assert strategy.openings['E0G']['wins'] == 0
            assert strategy.openings['E0G']['losses'] == 0

    def test_end_game_updates_correct_opening(self):
        """Test that end_game updates only the correct opening."""
        from snowdrop_tangled_agents.matlab.matlab_strategy import AlphaQExplorerStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            strategy = AlphaQExplorerStrategy(state_path=state_path)
            strategy.initialize()

            # Force E3G
            strategy.current_game_opening = (3, 'G')
            strategy.end_game('loss', 0.3)

            # Check only E3G updated
            assert strategy.openings['E3G']['losses'] == 1
            for key, counts in strategy.openings.items():
                if key != 'E3G':
                    assert counts['losses'] == 0

    def test_learning_gating(self):
        """Test that learning is disabled until MIN_GAMES_BEFORE_LEARNING."""
        from snowdrop_tangled_agents.matlab.matlab_strategy import AlphaQExplorerStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"
            strategy = AlphaQExplorerStrategy(state_path=state_path)
            strategy.initialize()

            # For games 1-9, learning_rate should be 0.0
            for i in range(9):
                strategy.current_game_opening = (0, 'G')
                strategy.end_game('draw', 0.5)
                assert strategy.solver.learning_rate == 0.0, \
                    f"Game {i+1}: learning_rate should be 0.0, got {strategy.solver.learning_rate}"

            # On game 10, learning_rate should flip to 0.03
            strategy.current_game_opening = (0, 'G')
            strategy.end_game('draw', 0.5)
            assert strategy.solver.learning_rate == 0.03, \
                f"Game 10: learning_rate should be 0.03, got {strategy.solver.learning_rate}"

    def test_state_roundtrip(self):
        """Test that state persists correctly across save/load cycles."""
        from snowdrop_tangled_agents.matlab.matlab_strategy import AlphaQExplorerStrategy

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "state.json"

            # Create and populate first instance
            strategy1 = AlphaQExplorerStrategy(state_path=state_path)
            strategy1.openings['E0G'] = {'wins': 2, 'draws': 5, 'losses': 1}
            strategy1.openings['E5P'] = {'wins': 0, 'draws': 3, 'losses': 2}
            strategy1.games_played = 13
            strategy1._save_state()

            # Load in fresh instance
            strategy2 = AlphaQExplorerStrategy(state_path=state_path)

            # Check roundtrip
            assert strategy2.openings['E0G'] == {'wins': 2, 'draws': 5, 'losses': 1}
            assert strategy2.openings['E5P'] == {'wins': 0, 'draws': 3, 'losses': 2}
            assert strategy2.games_played == 13


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
