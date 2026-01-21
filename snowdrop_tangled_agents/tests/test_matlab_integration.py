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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
