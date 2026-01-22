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
- **MATLAB RL System** (PPO agent, environment, parallel training)

## Running Tests

### Python Tests (No MATLAB Required)

```bash
# Run all Python tests
pytest -v -m "not matlab"

# Run MATLAB integration tests (Python-side)
pytest snowdrop_tangled_agents/tests/test_matlab_integration.py -v
```

### MATLAB RL Tests (Requires MATLAB)

```bash
# Run MATLAB RL tests via pytest
pytest -v -m matlab

# Run only quick MATLAB tests
pytest -v -m matlab -k "Quick"

# Run full MATLAB test suite (includes training)
pytest -v -m matlab -k "Full"
```

### Direct MATLAB Testing

```matlab
% In MATLAB, from snowdrop_tangled_agents/matlab/rl directory:
>> results = run_all_tests('quick')   % Fast tests (~30s)
>> results = run_all_tests('full')    % All tests including training (~60s)
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

### 7. MATLAB RL System (23 tests)

These tests require MATLAB installation and are marked with `@pytest.mark.matlab`.

#### Phase 0: Dependencies (5 tests)

| Test | Description | Verifies |
|------|-------------|----------|
| DEP-01 | RL Toolbox available | Toolbox installation |
| DEP-02 | Deep Learning Toolbox | Toolbox installation |
| DEP-03 | Database Toolbox (optional) | SQLite access |
| DEP-04 | Parallel Computing Toolbox (optional) | Parallel workers |
| DEP-05 | GPU detection | Graceful CPU fallback |

#### Phase 2: RL Environment (7 tests)

| Test | Description | Verifies |
|------|-------------|----------|
| ENV-01 | TangledEnvironment instantiation | Environment creation |
| ENV-02 | Observation space (50 elements) | Feature vector size |
| ENV-03 | Action space (30 discrete) | 15 edges × 2 colors |
| ENV-04 | Action masking | Valid move filtering |
| ENV-05 | Environment step function | State transitions |
| ENV-06 | Environment reset | Episode initialization |
| ENV-07 | Episode completion | Terminal state detection |

#### Phase 3: PPO Agent (6 tests)

| Test | Description | Verifies |
|------|-------------|----------|
| PPO-01 | PPO agent creation | Agent initialization |
| PPO-02 | Actor network forward pass | Policy inference |
| PPO-03 | Critic network forward pass | Value estimation |
| PPO-04 | Masked action selection | Valid action sampling |
| PPO-05 | SQLite experience buffer | Replay buffer storage |
| PPO-06 | Single training step | Gradient update |

#### Phase 4: Parallel Self-Play (5 tests)

| Test | Description | Verifies |
|------|-------------|----------|
| PAR-01 | Parallel environment creation | Worker setup |
| PAR-02 | Episode collection | Experience gathering |
| PAR-03 | Worker initialization | Pool management |
| PAR-04 | GPU enable (graceful fallback) | Device selection |
| PAR-05 | Short parallel training | End-to-end training |

#### Phase 5: Deployment Pipeline (9 tests)

Run via `test_deployment.m`:

| Test | Description | Verifies |
|------|-------------|----------|
| Test 1 | ModelRegistry creation | SQLite registry setup |
| Test 2 | Model registration | Version management |
| Test 3 | Model deployment | Hot-swap deployment |
| Test 4 | Load deployed model | Model loading |
| Test 5 | List versions | Version enumeration |
| Test 6 | Inference function | tangled_agent_inference |
| Test 7 | Auto-deploy (new) | First deployment |
| Test 8 | Auto-deploy (skip) | No improvement threshold |
| Test 9 | Auto-deploy (improve) | Significant improvement |

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

### GitHub Actions Workflow

The project includes a GitHub Actions workflow (`.github/workflows/test.yml`) that:

1. **Python Tests** - Run on every push/PR across Python 3.11, 3.12, 3.13
2. **MATLAB Tests** - Run on self-hosted runner with MATLAB (manual trigger)

```yaml
name: Tests

on:
  push:
    branches: [main, develop, 'feature/**']
  pull_request:
    branches: [main, develop]
  workflow_dispatch:
    inputs:
      run_matlab_tests:
        description: 'Run MATLAB tests (requires self-hosted runner)'
        required: false
        default: 'false'
        type: boolean

jobs:
  python-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.11', '3.12', '3.13']
    steps:
      # ... installs Poetry, runs pytest -m "not matlab"

  matlab-tests:
    runs-on: self-hosted  # Requires MATLAB installation
    if: github.event.inputs.run_matlab_tests == 'true'
    needs: python-tests
    steps:
      # ... runs pytest -m matlab
```

### Running Locally

```bash
# Python tests only (CI default)
pytest -v -m "not matlab"

# MATLAB tests (requires MATLAB)
pytest -v -m matlab

# All tests
pytest -v
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MATLAB_PATH` | Override MATLAB executable path |
| `SKIP_MATLAB_TESTS` | Set to "1" to skip MATLAB tests entirely |

Note: Full MATLAB tests require a MATLAB installation and license. CI runs Python tests by default; MATLAB tests require manual trigger on self-hosted runner.
