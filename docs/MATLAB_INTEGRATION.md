# MATLAB Toolbox Integration

This document describes the enhanced MATLAB integration for the Tangled game agent system, providing neural network position evaluation, opponent modeling, and training pipelines.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PYTHON ORCHESTRATION LAYER                              │
│  (Coordinates game play, stats collection, strategy selection)               │
└─────────────────────────────────────────────────────────────────────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐         ┌─────────────────┐         ┌─────────────────┐
│ Quantum SDK   │         │ MATLAB Backend  │         │ Pure Python     │
│ (D-Wave, etc) │         │                 │         │ Fallback        │
│               │         │ ┌─────────────┐ │         │                 │
│ - Annealing   │         │ │ Compiled    │ │         │ - Heuristics    │
│ - Sampling    │         │ │ Packages    │ │         │ - Basic MCTS    │
└───────────────┘         │ └─────────────┘ │         │ - SQLite only   │
                          │ ┌─────────────┐ │         └─────────────────┘
                          │ │ MATLAB      │ │
                          │ │ Engine API  │ │
                          │ └─────────────┘ │
                          └─────────────────┘
                                  │
                                  ▼
                          ┌─────────────────┐
                          │ SQLite Database │
                          │                 │
                          │ - games         │
                          │ - moves         │
                          │ - models        │
                          │ - opponents     │
                          │ - training_data │
                          └─────────────────┘
```

## Backend Fallback Chain

The unified bridge automatically selects the best available backend:

1. **Compiled Packages** (fastest, no license required)
   - Requires MATLAB Runtime (free download)
   - No MATLAB installation needed
   - Best for production deployment

2. **MATLAB Engine API** (full functionality)
   - Requires MATLAB license
   - Direct access to all toolboxes
   - Best for development and training

3. **Pure Python Heuristics** (always available)
   - No external dependencies
   - Basic position evaluation
   - Fallback when MATLAB unavailable

## MATLAB Toolboxes Used

| Toolbox | Purpose |
|---------|---------|
| Deep Learning Toolbox | Neural network training and inference |
| Statistics and ML Toolbox | Opponent clustering and classification |
| Database Toolbox | Direct SQLite access from MATLAB |
| MATLAB Compiler SDK | Package MATLAB code for Python |

---

## Python Components

### File Structure

```
snowdrop_tangled_agents/
├── matlab/
│   ├── __init__.py           # Module exports
│   ├── bridge.py             # MATLAB Engine API bridge
│   ├── compiled_bridge.py    # Compiled packages bridge
│   ├── unified_bridge.py     # Unified interface with fallback
│   ├── matlab_strategy.py    # MCTS + MATLAB strategy
│   └── training.py           # Training orchestration
├── stats/
│   ├── __init__.py           # Module exports
│   ├── collector.py          # Stats collection + model/opponent management
│   ├── migrations.py         # Database schema migrations
│   └── queries.py            # Analysis queries
└── tests/
    └── test_matlab_integration.py  # Regression test suite
```

### MATLAB Source Files

#### Directory Setup (Required)

MATLAB source files are stored in the repository under `snowdrop_tangled_agents/matlab/` but MATLAB IDE expects them in MATLAB Drive. To maintain a single source of truth, we use a **directory junction** (Windows) or **symlink** (macOS/Linux).

**Windows Setup:**
```powershell
# Create junction from MATLAB Drive to repo (run in PowerShell)
New-Item -ItemType Junction -Path "$env:USERPROFILE\MATLAB Drive\tangled_rl" `
         -Target "C:\path\to\snowdrop-tangled-agents\snowdrop_tangled_agents\matlab\rl"

New-Item -ItemType Junction -Path "$env:USERPROFILE\MATLAB Drive\tangled_strategies" `
         -Target "C:\path\to\snowdrop-tangled-agents\snowdrop_tangled_agents\matlab"
```

**macOS/Linux Setup:**
```bash
# Create symlink from MATLAB Drive to repo
ln -s /path/to/snowdrop-tangled-agents/snowdrop_tangled_agents/matlab/rl \
      ~/MATLAB\ Drive/tangled_rl

ln -s /path/to/snowdrop-tangled-agents/snowdrop_tangled_agents/matlab \
      ~/MATLAB\ Drive/tangled_strategies
```

**Note:** Paths are platform-specific. Adjust `C:\path\to\` or `/path/to/` to match your local clone location.

#### Strategy Files

Located in `snowdrop_tangled_agents/matlab/` (symlinked to `MATLAB Drive/tangled_strategies/`):

| File | Purpose |
|------|---------|
| `db_utils.m` | Database connection utilities (connect, fetch, exec, close) |
| `build_features.m` | 50-element feature vector construction for NN input |
| `prepare_dataset.m` | Training data preparation with augmentation |
| `train_value_network.m` | Value network training (FC 128→64→32→tanh) |
| `train_policy_network.m` | Policy network training (FC 128→64→30 softmax) |
| `evaluate_position_nn.m` | Neural network inference for position evaluation |
| `extract_opponent_features.m` | 20-element opponent feature extraction |
| `cluster_opponents.m` | K-means clustering with KNN classifier output |
| `classify_opponent.m` | Opponent style classification using trained model |
| `adapt_to_opponent.m` | Prior adaptation based on opponent style |
| `build_packages.m` | Compiler SDK packaging script |
| `build_remaining_packages.m` | Build opponent_model + training packages |
| `evaluate_position.m` | Heuristic position evaluation (non-NN) |
| `sa_evaluate.m` | Simulated annealing evaluation helper |
| `identify_opponent.m` | Opponent identification helper |

#### Reinforcement Learning Files

Located in `snowdrop_tangled_agents/matlab/rl/` (symlinked to `MATLAB Drive/tangled_rl/`):

| File | Purpose |
|------|---------|
| `TangledEnvironment.m` | Custom RL environment class for RL Toolbox |
| `getActionMask.m` | Valid action masking (30-action space) |
| `buildRLFeatures.m` | 50-element observation vector builder |
| `SimulatedOpponent.m` | Opponent models (random, heuristic, mcts, defensive, aggressive) |
| `test_environment.m` | Unit tests for RL environment |

These files implement Phase 2 of the Dynamic Learning plan (see `docs/DYNAMIC_LEARNING_PLAN.md`).

### Key Classes

#### UnifiedMatlabBridge (`unified_bridge.py`)

Primary interface for MATLAB functionality with automatic fallback.

```python
from snowdrop_tangled_agents.matlab import get_unified_bridge

bridge = get_unified_bridge()
backend = bridge.connect()  # Returns 'compiled', 'engine', or 'heuristic'

# Position evaluation
value, policy = bridge.evaluate_position(state, is_our_turn=True)

# Opponent classification
style, confidence = bridge.classify_opponent(features=opponent_features)

# Prior adaptation
adapted = bridge.adapt_priors(state, opponent_features, base_priors)
```

#### MatlabEnhancedStrategy (`matlab_strategy.py`)

MCTS strategy enhanced with neural network priors and opponent adaptation.

```python
from snowdrop_tangled_agents.matlab import MatlabEnhancedStrategy

strategy = MatlabEnhancedStrategy(
    mcts_time_limit=2.0,
    mcts_iterations=5000,
    use_nn_priors=True,
    use_opponent_adaptation=True,
)
strategy.initialize(opponent='melissa')

move = strategy.calculate_move(state, score=0.0)
```

#### TrainingOrchestrator (`training.py`)

Manages neural network training from Python.

```python
from snowdrop_tangled_agents.matlab.training import get_training_orchestrator

orchestrator = get_training_orchestrator()
status = orchestrator.get_status()

# Train networks (requires MATLAB)
metrics = orchestrator.train_value_network(epochs=100)
metrics = orchestrator.train_policy_network(epochs=100)

# Cluster opponents
result = orchestrator.cluster_opponents(k=3)
```

#### StatsCollector (`collector.py`)

Extended with model and opponent management.

```python
from snowdrop_tangled_agents.stats import get_collector

collector = get_collector()

# Model management
collector.save_model('value_net_v1', 'value', 100, 0.05, '/path/to/model.mat')
model = collector.get_active_model('value')

# Opponent profiles
collector.save_opponent('melissa', cluster_id=1, features=[...])
opponent = collector.get_opponent('melissa')
collector.increment_opponent_games('melissa', won=True)
```

### Database Schema (v5)

```sql
-- Core tables (v1)
games, moves, calibration

-- Model metadata (v2)
CREATE TABLE models (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    type TEXT,              -- 'value', 'policy', 'opponent'
    training_games INTEGER,
    validation_loss REAL,
    file_path TEXT,
    hyperparameters TEXT,   -- JSON
    active BOOLEAN
);

-- Opponent profiles (v3)
CREATE TABLE opponents (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    cluster_id INTEGER,     -- From k-means
    games_played INTEGER,
    win_rate REAL,
    features TEXT,          -- JSON: 20-element vector
    last_updated DATETIME
);

-- Training data (v4)
CREATE TABLE training_data (
    id INTEGER PRIMARY KEY,
    version INTEGER,
    features TEXT,          -- JSON: 50-element vector
    target_value REAL,
    target_policy TEXT,     -- JSON: 30-element vector
    quality_score REAL
);

-- Opponent move history (v5)
CREATE TABLE opponent_history (
    id INTEGER PRIMARY KEY,
    opponent_name TEXT,
    game_id TEXT,
    move_number INTEGER,
    edge INTEGER,
    color TEXT,
    score_before REAL,
    score_after REAL
);
```

---

## MATLAB Components

### File Structure

```
C:\Users\murr2\MATLAB Drive\tangled_strategies\
├── db_utils.m                    # Database connection utilities
├── build_features.m              # 50-element feature vector
├── prepare_dataset.m             # Training data preparation
├── train_value_network.m         # Value network training
├── train_policy_network.m        # Policy network training
├── evaluate_position_nn.m        # Neural network inference
├── extract_opponent_features.m   # 20-element opponent features
├── cluster_opponents.m           # K-means clustering
├── classify_opponent.m           # Style classification
├── adapt_to_opponent.m           # Prior adaptation
└── build_packages.m              # Compiler SDK packaging
```

### Neural Network Architecture

#### Value Network

```
Input Layer:     50 features
                 ├─ Board state (15): G=1, P=-1, grey=0
                 ├─ Turn indicator (1): +1 us, -1 opponent
                 ├─ Edge categories (15): MY/OPP/HUB encoding
                 ├─ Grey count (1): Normalized 0-1
                 ├─ Score momentum (3): Last 3 deltas
                 └─ Game phase (15): One-hot encoding

Hidden Layers:   FC(128) → ReLU → Dropout(0.3)
                 FC(64) → ReLU → Dropout(0.2)
                 FC(32) → ReLU

Output Layer:    FC(1) → Tanh → value ∈ [-1, +1]
```

#### Policy Network

```
Input Layer:     Same 50 features

Hidden Layers:   FC(128) → ReLU → Dropout(0.3)
                 FC(64) → ReLU

Output Layer:    FC(30) → Softmax → P(edge, color)
                 [P(E0,G), P(E0,P), P(E1,G), ..., P(E14,P)]
```

### Opponent Feature Vector (20 elements)

| Index | Feature | Description |
|-------|---------|-------------|
| 1-15 | edge_frequency | Play frequency per edge (normalized) |
| 16 | color_bias | (green - purple) / total |
| 17 | opening_aggression | Attack rate in moves 1-3 |
| 18 | response_rate | Purple rate on MY_EDGES |
| 19 | hub_priority | HUB_EDGES play frequency |
| 20 | endgame_aggression | Attack rate in last 3 moves |

### Opponent Clusters

| Cluster | Style | Characteristics | Counter-Strategy |
|---------|-------|-----------------|------------------|
| 1 | Aggressive | High opening aggression, attacks our edges | Prioritize defense |
| 2 | Defensive | Low aggression, focuses on own edges | Early attack |
| 3 | Hub-focused | Prioritizes hub control | Compete for hub |

---

## Command Line Interface

### Training Commands

```bash
# Show training system status
python play_tangled.py --training-status

# Train value network (requires 50+ games)
python play_tangled.py --train-value-network --epochs 100

# Train policy network
python play_tangled.py --train-policy-network --epochs 100

# Cluster opponents
python play_tangled.py --cluster-opponents --clusters 3
```

### Play with MATLAB Enhancement

```bash
# Use MATLAB strategy with neural network priors
python play_tangled.py --strategy matlab --use-nn

# Enable opponent adaptation
python play_tangled.py --strategy matlab --adapt-opponent

# Full MATLAB enhancement
python play_tangled.py --strategy matlab --use-nn --adapt-opponent --games 10
```

---

## Installation

### Deployment Options

There are three ways to use the MATLAB-enhanced features:

| Option | Requirements | Best For |
|--------|--------------|----------|
| **Full MATLAB** | MATLAB R2026a + Toolboxes | Development, training new models |
| **Compiled Packages** | MATLAB Runtime R2026a (free) | Production deployment |
| **Pure Python** | None | Fallback when MATLAB unavailable |

### Option 1: Full MATLAB Installation (Development)

For training neural networks and building compiled packages:

1. **MATLAB R2026a** with toolboxes:
   - Deep Learning Toolbox
   - Statistics and Machine Learning Toolbox
   - Database Toolbox
   - MATLAB Compiler SDK

2. **Python packages:**
   ```bash
   pip install matlabengine==26.1
   ```

### Option 2: MATLAB Runtime Only (Deployment)

For users **without** a MATLAB license who want to use pre-compiled packages:

1. **Download MATLAB Runtime R2026a** (free, ~3GB):
   - Visit: https://www.mathworks.com/products/compiler/matlab-runtime.html
   - Select **R2026a** (must match the version used to compile packages)
   - Download and install for your platform

2. **Install compiled packages:**
   ```bash
   pip install tangled_value_network
   pip install tangled_opponent_model
   ```

**Important:** The MATLAB Runtime version must **exactly match** the MATLAB version
used to compile the packages. Packages compiled with R2026a require Runtime R2026a.

### Option 3: Pure Python Fallback

If neither MATLAB nor Runtime is available, the system automatically falls back to
pure Python heuristics. No additional installation required.

### Setup Verification

```bash
# Check what's available
python play_tangled.py --training-status

# Or in Python:
python -c "
from snowdrop_tangled_agents.matlab import get_unified_bridge
bridge = get_unified_bridge()
backend = bridge.connect()
print(f'Backend: {backend}')  # 'compiled', 'engine', or 'heuristic'
"
```

### Building Compiled Packages

The MATLAB Compiler SDK creates Python packages that run without a MATLAB license
(only the free MATLAB Runtime is required for deployment).

**Available Packages:**

| Package | Functions | Purpose |
|---------|-----------|---------|
| `tangled_value_network` | `evaluate_position_nn`, `build_features` | Neural network inference |
| `tangled_opponent_model` | `classify_opponent`, `adapt_to_opponent` | Opponent modeling |
| `tangled_training` | `train_value_network`, `cluster_opponents` | Model training |

**Build All Packages (in MATLAB):**
```matlab
cd 'C:\Users\murr2\MATLAB Drive\tangled_strategies'
build_packages('C:\Users\murr2\.tangled\compiled')
```

**Build Individual Packages:**
```matlab
cd 'C:\Users\murr2\MATLAB Drive\tangled_strategies'
output_dir = 'C:\Users\murr2\.tangled\compiled';

% Value network package
files = {'evaluate_position_nn.m', 'build_features.m'};
compiler.build.pythonPackage(files, ...
    'PackageName', 'tangled_value_network', ...
    'OutputDir', fullfile(output_dir, 'tangled_value_network'), ...
    'Verbose', 'on');

% Opponent model package
files = {'classify_opponent.m', 'adapt_to_opponent.m', 'extract_opponent_features.m', 'db_utils.m'};
compiler.build.pythonPackage(files, ...
    'PackageName', 'tangled_opponent_model', ...
    'OutputDir', fullfile(output_dir, 'tangled_opponent_model'), ...
    'Verbose', 'on');
```

**Install Compiled Packages:**
```bash
pip install C:\Users\murr2\.tangled\compiled\tangled_value_network
pip install C:\Users\murr2\.tangled\compiled\tangled_opponent_model
```

**Verify Installation:**
```python
from snowdrop_tangled_agents.matlab.compiled_bridge import packages_available
print(packages_available())
# {'value_network': True, 'opponent_model': True, 'training': False}
```

---

## Workflow

### Training Workflow

1. **Collect Game Data**
   ```bash
   python play_tangled.py --strategy hybrid --games 50
   ```

2. **Check Training Status**
   ```bash
   python play_tangled.py --training-status
   ```

3. **Train Networks** (in MATLAB or via Python)
   ```matlab
   % In MATLAB
   [net, metrics] = train_value_network('', struct('epochs', 100));
   [net, metrics] = train_policy_network('', struct('epochs', 100));
   ```

4. **Cluster Opponents**
   ```bash
   python play_tangled.py --cluster-opponents --clusters 3
   ```

5. **Play with Enhanced Strategy**
   ```bash
   python play_tangled.py --strategy matlab --use-nn --adapt-opponent
   ```

### Development Workflow

1. **Modify MATLAB functions** in `MATLAB Drive/tangled_strategies/`
2. **Test via Python bridge**:
   ```python
   from snowdrop_tangled_agents.matlab import get_bridge
   bridge = get_bridge()
   bridge.connect()
   result = bridge.call_function('your_function', arg1, arg2)
   ```
3. **Run regression tests**:
   ```bash
   python -m pytest snowdrop_tangled_agents/tests/test_matlab_integration.py -v
   ```

---

## MATLAB Engine Connection

The bridge automatically handles MATLAB Engine connections without requiring manual setup.

### Connection Modes

| Mode | Behavior | Startup Time |
|------|----------|--------------|
| **Shared Session** | Connects to existing MATLAB with `shareEngine` | Instant |
| **Headless** (default) | Starts new MATLAB without GUI | ~10-15 seconds |
| **Desktop** | Starts new MATLAB with full GUI | ~20-30 seconds |

### Automatic Connection Flow

```
1. Check for existing shared sessions (instant if found)
2. If none found, start new headless MATLAB instance
3. Add tangled_strategies directory to MATLAB path
4. Ready for use
```

**No manual `matlab.engine.shareEngine` command is required!**

### Configuration Options

```python
from snowdrop_tangled_agents.matlab import get_unified_bridge

# Default: prefer existing sessions, headless if starting new
bridge = get_unified_bridge()
backend = bridge.connect()

# Always start fresh instance (useful for clean state)
bridge = get_unified_bridge(prefer_existing_session=False, force_new=True)

# Start with MATLAB desktop for debugging
bridge = get_unified_bridge(headless=False, force_new=True)
```

### Optional: Pre-sharing MATLAB Session

For faster repeated connections during development, you can optionally share
your MATLAB session:

```matlab
% In MATLAB Command Window (optional, for faster reconnection)
matlab.engine.shareEngine
```

This allows instant reconnection instead of waiting for a new instance to start.

---

## Troubleshooting

### MATLAB Engine Won't Connect

```python
# Check MATLAB is on PATH
import os
print(os.environ.get('PATH'))

# Verify matlabengine is installed
pip show matlabengine

# Check for connection issues
from snowdrop_tangled_agents.matlab import get_unified_bridge
bridge = get_unified_bridge()
backend = bridge.connect()
print(f"Backend: {backend}")  # Should be 'engine' if MATLAB working
```

### Database Migrations Failed

```python
from snowdrop_tangled_agents.stats import get_collector
collector = get_collector()
status = collector.get_migration_status()
print(status)
```

### Neural Network Returns 0.0

This is expected when no trained model exists. Train a model first:
```bash
python play_tangled.py --train-value-network
```

### Opponent Adaptation Not Working

Verify opponent profile exists:
```python
from snowdrop_tangled_agents.stats import get_collector
collector = get_collector()
print(collector.get_opponent('melissa'))
```
