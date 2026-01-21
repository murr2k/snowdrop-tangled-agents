# Regression Test Suite

This document describes the regression test suite for the MATLAB toolbox integration.

## Overview

The test suite verifies all components of the MATLAB integration:
- Database migrations
- Stats collector extensions
- Unified bridge functionality
- Compiled bridge interface
- Training orchestrator
- MATLAB strategy integration

## Running Tests

### Full Test Suite

```bash
# Run all tests with coverage
python -m pytest snowdrop_tangled_agents/tests/test_matlab_integration.py -v

# Run without coverage (faster)
python -m pytest snowdrop_tangled_agents/tests/test_matlab_integration.py -v --no-cov

# Run specific test class
python -m pytest snowdrop_tangled_agents/tests/test_matlab_integration.py::TestUnifiedBridge -v

# Run specific test
python -m pytest snowdrop_tangled_agents/tests/test_matlab_integration.py::TestMatlabStrategy::test_strategy_calculate_move_opening -v
```

### Quick Verification

```bash
# Just check imports work
python -c "from snowdrop_tangled_agents.matlab import get_unified_bridge; print('OK')"

# Check training status
python play_tangled.py --training-status
```

---

## Test Categories

### 1. Database Migrations (3 tests)

| Test | Description | Verifies |
|------|-------------|----------|
| `test_migrations_import` | Import migrations module | Module structure |
| `test_run_migrations_on_fresh_db` | Run migrations on new DB | Migration execution |
| `test_migration_creates_tables` | Verify tables created | Schema correctness |

**Key Assertions:**
- At least 4 migrations defined (v2-v5)
- Tables `models`, `opponents`, `training_data`, `opponent_history` exist
- Schema version reaches 5

### 2. Stats Collector (4 tests)

| Test | Description | Verifies |
|------|-------------|----------|
| `test_collector_init_runs_migrations` | Collector runs migrations | Auto-migration |
| `test_save_and_get_model` | Save/retrieve model metadata | Model CRUD |
| `test_save_and_get_opponent` | Save/retrieve opponent profiles | Opponent CRUD |
| `test_increment_opponent_games` | Update opponent game counts | Win rate calculation |

**Key Assertions:**
- Model metadata persists correctly
- Hyperparameters stored as JSON
- Opponent features stored as JSON
- Win rate calculated correctly (2/3 after WWL)

### 3. Unified Bridge (5 tests)

| Test | Description | Verifies |
|------|-------------|----------|
| `test_unified_bridge_import` | Import unified bridge | Module structure |
| `test_unified_bridge_connect` | Connect to backend | Backend selection |
| `test_unified_bridge_heuristic_eval` | Heuristic position evaluation | Fallback evaluation |
| `test_unified_bridge_heuristic_classify` | Heuristic opponent classification | Style inference |
| `test_uniform_priors` | Uniform prior generation | Prior normalization |

**Key Assertions:**
- Backend returns 'compiled', 'engine', or 'heuristic'
- Value in [-1, 1] range
- Policy sums to ~1.0
- Aggressive style (17=0.5) → cluster 1
- Defensive style (18=0.7) → cluster 2

### 4. Compiled Bridge (1 test)

| Test | Description | Verifies |
|------|-------------|----------|
| `test_packages_available` | Check package availability | Import detection |

**Key Assertions:**
- Returns dict with keys: `value_network`, `opponent_model`, `training`
- Values are boolean

### 5. Training Orchestrator (2 tests)

| Test | Description | Verifies |
|------|-------------|----------|
| `test_orchestrator_import` | Import training module | Module structure |
| `test_orchestrator_status` | Get training status | Status reporting |

**Key Assertions:**
- Status contains `database_path`, `total_games`, `ready_for_training`
- Status contains `compiled_training`, `matlab_engine` flags

### 6. MATLAB Strategy (5 tests)

| Test | Description | Verifies |
|------|-------------|----------|
| `test_strategy_import` | Import strategy class | Module structure |
| `test_strategy_init` | Initialize strategy | Configuration |
| `test_strategy_initialize` | Connect to backend | Backend selection |
| `test_strategy_calculate_move_opening` | Opening book moves | Opening sequence |
| `test_strategy_get_stats` | Get strategy statistics | Stats tracking |

**Key Assertions:**
- `use_nn_priors` and `use_opponent_adaptation` configurable
- Backend set after `initialize()`
- Opening move from `opening_sequence` list
- Stats include `backend`, `nn_calls`, `adapt_calls`

---

## Test File Structure

```python
# snowdrop_tangled_agents/tests/test_matlab_integration.py

class TestDatabaseMigrations:
    """Test database schema migrations."""
    def test_migrations_import(self): ...
    def test_run_migrations_on_fresh_db(self): ...
    def test_migration_creates_tables(self): ...

class TestStatsCollector:
    """Test stats collector model/opponent methods."""
    @pytest.fixture
    def temp_db(self): ...  # Temporary database fixture

    def test_collector_init_runs_migrations(self, temp_db): ...
    def test_save_and_get_model(self, temp_db): ...
    def test_save_and_get_opponent(self, temp_db): ...
    def test_increment_opponent_games(self, temp_db): ...

class TestUnifiedBridge:
    """Test unified bridge functionality."""
    def test_unified_bridge_import(self): ...
    def test_unified_bridge_connect(self): ...
    def test_unified_bridge_heuristic_eval(self): ...
    def test_unified_bridge_heuristic_classify(self): ...
    def test_uniform_priors(self): ...

class TestCompiledBridge:
    """Test compiled bridge functionality."""
    def test_packages_available(self): ...

class TestTrainingOrchestrator:
    """Test training orchestrator."""
    def test_orchestrator_import(self): ...
    def test_orchestrator_status(self): ...

class TestMatlabStrategy:
    """Test MATLAB strategy integration."""
    def test_strategy_import(self): ...
    def test_strategy_init(self): ...
    def test_strategy_initialize(self): ...
    def test_strategy_calculate_move_opening(self): ...
    def test_strategy_get_stats(self): ...
```

---

## Expected Test Output

### All Tests Passing

```
============================= test session starts =============================
platform win32 -- Python 3.13.6, pytest-9.0.2
collected 20 items

test_matlab_integration.py::TestDatabaseMigrations::test_migrations_import PASSED
test_matlab_integration.py::TestDatabaseMigrations::test_run_migrations_on_fresh_db PASSED
test_matlab_integration.py::TestDatabaseMigrations::test_migration_creates_tables PASSED
test_matlab_integration.py::TestStatsCollector::test_collector_init_runs_migrations PASSED
test_matlab_integration.py::TestStatsCollector::test_save_and_get_model PASSED
test_matlab_integration.py::TestStatsCollector::test_save_and_get_opponent PASSED
test_matlab_integration.py::TestStatsCollector::test_increment_opponent_games PASSED
test_matlab_integration.py::TestUnifiedBridge::test_unified_bridge_import PASSED
test_matlab_integration.py::TestUnifiedBridge::test_unified_bridge_connect PASSED
test_matlab_integration.py::TestUnifiedBridge::test_unified_bridge_heuristic_eval PASSED
test_matlab_integration.py::TestUnifiedBridge::test_unified_bridge_heuristic_classify PASSED
test_matlab_integration.py::TestUnifiedBridge::test_uniform_priors PASSED
test_matlab_integration.py::TestCompiledBridge::test_packages_available PASSED
test_matlab_integration.py::TestTrainingOrchestrator::test_orchestrator_import PASSED
test_matlab_integration.py::TestTrainingOrchestrator::test_orchestrator_status PASSED
test_matlab_integration.py::TestMatlabStrategy::test_strategy_import PASSED
test_matlab_integration.py::TestMatlabStrategy::test_strategy_init PASSED
test_matlab_integration.py::TestMatlabStrategy::test_strategy_initialize PASSED
test_matlab_integration.py::TestMatlabStrategy::test_strategy_calculate_move_opening PASSED
test_matlab_integration.py::TestMatlabStrategy::test_strategy_get_stats PASSED

======================= 20 passed in 2.95s =======================
```

---

## Integration Tests (Manual)

These tests verify end-to-end functionality and require MATLAB Engine:

### Test 1: MATLAB Engine Connection

```python
from snowdrop_tangled_agents.matlab import get_unified_bridge

bridge = get_unified_bridge()
backend = bridge.connect()
assert backend in ('compiled', 'engine', 'heuristic')
print(f'Backend: {backend}')
```

**Expected:** Backend is 'engine' if MATLAB installed, 'heuristic' otherwise.

### Test 2: Position Evaluation

```python
from snowdrop_tangled_agents.matlab import get_unified_bridge

bridge = get_unified_bridge()
bridge.connect()

# Empty board
value, policy = bridge.evaluate_position('---------------', True)
assert -1.0 <= value <= 1.0
assert len(policy) == 30
assert abs(sum(policy.values()) - 1.0) < 0.01

# Partially filled
value, policy = bridge.evaluate_position('GP-------------', True)
assert len(policy) == 26  # 13 grey edges × 2 colors
```

### Test 3: Opponent Adaptation

```python
from snowdrop_tangled_agents.stats import get_collector
from snowdrop_tangled_agents.matlab import MatlabEnhancedStrategy

# Create opponent profile
collector = get_collector()
features = [0.1] * 15 + [0.0, 0.4, 0.5, 0.2, 0.1]
collector.save_opponent('test_opp', cluster_id=1, features=features)

# Test strategy
strategy = MatlabEnhancedStrategy(use_opponent_adaptation=True)
strategy.initialize(opponent='test_opp')

assert strategy.opponent_style == 1
assert strategy.opponent_features is not None

move = strategy.calculate_move('GP-GP-GP-------', 0.5)
stats = strategy.get_stats()
assert stats['adapt_calls'] >= 1
```

### Test 4: Training Status

```bash
python play_tangled.py --training-status
```

**Expected Output:**
```
=== MATLAB Training System Status ===
Database: C:\Users\murr2\.tangled\game_stats.db
Games in database: 44 (4W / 33L)
Schema version: 5

Backend availability:
  Compiled packages: No
  MATLAB Engine:     Yes

Status: Need more games (6 more)
========================================
```

---

## Coverage Report

Current test coverage (as of implementation):

| Module | Coverage |
|--------|----------|
| `stats/migrations.py` | 89% |
| `stats/collector.py` | 53% |
| `matlab/unified_bridge.py` | 50% |
| `matlab/bridge.py` | 35% |
| `matlab/matlab_strategy.py` | 35% |
| `matlab/training.py` | 28% |
| `matlab/compiled_bridge.py` | 25% |

**Total:** ~37% (core MATLAB integration paths covered)

---

## Adding New Tests

### Template for New Test

```python
class TestNewFeature:
    """Test description."""

    def test_feature_import(self):
        """Test that module imports correctly."""
        from snowdrop_tangled_agents.module import Feature
        # No assertion needed - import is the test

    def test_feature_basic(self):
        """Test basic functionality."""
        from snowdrop_tangled_agents.module import Feature

        feature = Feature()
        result = feature.do_something()

        assert result is not None
        assert result.property == expected_value

    @pytest.fixture
    def setup_data(self):
        """Fixture for test data."""
        # Setup
        data = create_test_data()
        yield data
        # Teardown
        cleanup(data)

    def test_with_fixture(self, setup_data):
        """Test using fixture."""
        result = process(setup_data)
        assert result.is_valid()
```

### Running New Tests

```bash
# Run just the new test class
python -m pytest snowdrop_tangled_agents/tests/test_matlab_integration.py::TestNewFeature -v

# Run with print statements visible
python -m pytest snowdrop_tangled_agents/tests/test_matlab_integration.py::TestNewFeature -v -s
```

---

## Continuous Integration

### GitHub Actions Workflow (suggested)

```yaml
name: MATLAB Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install pytest pytest-cov
          pip install -e .

      - name: Run tests
        run: |
          python -m pytest snowdrop_tangled_agents/tests/test_matlab_integration.py -v --no-cov
```

Note: Full MATLAB Engine tests require a MATLAB installation and license, so CI typically runs only the heuristic fallback tests.
