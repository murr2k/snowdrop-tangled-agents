# RL + Monte Carlo Ensemble Implementation

This document describes the hybrid Reinforcement Learning and Monte Carlo ensemble system for the Tangled quantum game.

## Overview

The ensemble combines two complementary approaches:

1. **RL Neural Network** - Fast pattern recognition via trained PPO policy
2. **Monte Carlo Rollouts** - Ground-truth value estimation through simulation

This is similar to the AlphaZero architecture: the neural network provides intuition (priors), while Monte Carlo search provides verification and refinement.

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE DECISION FLOW                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Current State ──┬──► RL Policy ──► Prior P(action)            │
│                   │       │              │                      │
│                   │       │              ▼                      │
│                   │       │         Top-K Filter                │
│                   │       │              │                      │
│                   │       ▼              ▼                      │
│                   └──► MC Rollouts ◄── Candidates               │
│                            │                                    │
│                            ▼                                    │
│                    Value Estimates V(a)                         │
│                            │                                    │
│                            ▼                                    │
│              Combined Score = P(a)^α × softmax(V(a))^β          │
│                            │                                    │
│                            ▼                                    │
│                      Best Action                                │
└─────────────────────────────────────────────────────────────────┘
```

## Architecture Components

### 1. MCRollout (MATLAB)

**File:** `snowdrop_tangled_agents/matlab/rl/MCRollout.m`

Parallel Monte Carlo rollout engine using MATLAB's Parallel Computing Toolbox.

**Key Features:**
- Configurable worker count (default: 22 threads)
- Smart color selection during rollouts (domain knowledge)
- Heuristic terminal state evaluation

**Usage:**
```matlab
mc = MCRollout('NumWorkers', 22, 'RolloutsPerAction', 50);
values = mc.evaluateActions(state, candidateActions);
```

**Rollout Strategy:**
- Random move selection for exploration
- Informed color choice based on edge type:
  - Our turn: Green on MY_EDGES, Purple on OPP_EDGES
  - Opponent turn: Opposite strategy

### 2. EnsemblePolicy (MATLAB)

**File:** `snowdrop_tangled_agents/matlab/rl/EnsemblePolicy.m`

Combines RL neural network priors with MC value estimates.

**Parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| TopK | 5 | Number of top candidates to evaluate with MC |
| RolloutsPerAction | 50 | MC simulations per candidate |
| NumWorkers | 22 | Parallel threads |
| AlphaPrior | 0.5 | Weight for RL prior (0-1) |
| BetaMC | 0.5 | Weight for MC value (0-1) |
| Temperature | 1.0 | Softmax temperature |

**Selection Algorithm:**
```
1. Get RL prior probabilities P(a) for all 30 actions
2. Mask invalid actions (already-colored edges)
3. Select top-K candidates by prior probability
4. Run parallel MC rollouts on candidates → V(a)
5. Compute combined score: score(a) = P(a)^α × softmax(V(a)/τ)^β
6. Return action with highest combined score
```

### 3. TangledEnvironment (MATLAB)

**File:** `snowdrop_tangled_agents/matlab/rl/TangledEnvironment.m`

RL environment with reward shaping for strategic learning.

**Reward Structure:**

| Event | Reward | Rationale |
|-------|--------|-----------|
| Green on MY_EDGES (E9,E10,E11) | +0.15 | Securing our territory |
| Purple on OPP_EDGES (E5,E12,E13) | +0.10 | Attacking opponent |
| Green on HUB_EDGES (E2,E10,E12) | +0.05 | Controlling connectivity |
| Early strategic move (moves 1-3) | +0.10 | Urgency bonus |
| Purple on MY_EDGES | -0.10 | Self-sabotage penalty |
| Opponent takes MY_EDGES with Purple | -0.15 | Failed defense |
| Invalid action | -0.50 | Learn valid moves |
| Terminal win (score > 2) | tanh(score/2) + 0.2 | Decisive victory bonus |
| Terminal loss (score < -2) | tanh(score/2) - 0.2 | Decisive loss penalty |

### 4. Training Pipeline

**File:** `snowdrop_tangled_agents/matlab/rl/train_curriculum_ensemble.m`

Curriculum learning with three levels:

```
Level 1: Random Opponent
├── Episodes: 200
├── Target: 70% win rate
└── Purpose: Learn basic valid moves and scoring

Level 2: Petersen Opponent (CRITICAL)
├── Episodes: 500
├── Target: 50% win rate
└── Purpose: Learn strategic play against deterministic optimal

Level 3: Self-Play with Ensemble
├── Episodes: 300
├── Sync Interval: 50 episodes
└── Purpose: Refinement against strong adaptive opponent
```

**Running Training:**
```matlab
% In MATLAB, from the rl/ directory:
addpath('.')
results = train_curriculum_ensemble('NumWorkers', 22, 'RolloutsPerAction', 50);
```

### 5. Python Integration

**File:** `snowdrop_tangled_agents/strategy/rl_strategy.py`

**Classes:**
- `RLStrategy` - Pure RL or ensemble mode
- `EnsembleStrategy` - Convenience class for ensemble mode

**Usage:**
```python
# Pure RL (fast)
from snowdrop_tangled_agents.strategy import RLStrategy
strategy = RLStrategy()

# Ensemble (slower but more accurate)
from snowdrop_tangled_agents.strategy import EnsembleStrategy
strategy = EnsembleStrategy(num_workers=22, rollouts_per_action=50)

# Calculate move
edge, color = strategy.calculate_move(state, score, history)
```

**Command Line:**
```bash
# Pure RL
python play_tangled.py --strategy rl --games 5

# Ensemble (RL + MC)
python play_tangled.py --strategy ensemble --games 5
```

## Edge Classifications

The Petersen graph edges are classified for strategic purposes:

```
MY_EDGES (Player 1 / Red):
  E9  (V4-V5) ─┐
  E10 (V5-V6) ─┼─ Edges connected to our vertex (V5)
  E11 (V5-V9) ─┘

OPP_EDGES (Player 2 / Blue):
  E5  (V1-V7) ─┐
  E12 (V6-V7) ─┼─ Edges connected to opponent vertex (V7)
  E13 (V7-V8) ─┘

HUB_EDGES (High connectivity):
  E2  (V0-V6) ─┐
  E10 (V5-V6) ─┼─ Central hub edges
  E12 (V6-V7) ─┘
```

**Optimal Strategy:**
1. Secure MY_EDGES with Green (ferromagnetic)
2. Attack OPP_EDGES with Purple (antiferromagnetic)
3. Control HUB_EDGES for board influence

## Performance Characteristics

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| Pure RL | ~10ms/move | Moderate | Real-time play, training |
| Ensemble (K=5, R=50) | ~500ms/move | High | Important games |
| Ensemble (K=10, R=100) | ~2s/move | Very High | Analysis, debugging |

## Files Summary

```
snowdrop_tangled_agents/
├── matlab/
│   ├── rl/
│   │   ├── MCRollout.m              # Parallel MC engine
│   │   ├── EnsemblePolicy.m         # RL + MC combination
│   │   ├── EnsembleSelfPlayOpponent.m  # Training opponent
│   │   ├── TangledEnvironment.m     # RL environment with rewards
│   │   ├── train_curriculum_ensemble.m  # Full training script
│   │   ├── SimulatedOpponent.m      # Various opponent styles
│   │   └── buildRLFeatures.m        # Observation builder
│   └── models/
│       ├── agent_selfplay_v1.mat    # Previous model
│       └── agent_ensemble_v1.mat    # Ensemble-trained model
├── strategy/
│   ├── rl_strategy.py               # Python RL/Ensemble strategy
│   └── __init__.py                  # Exports
└── agents/
    └── rl_agent.py                  # GameAgentBase wrapper
```

## Future Improvements

1. **True MCTS Integration** - Replace random rollouts with UCT tree search
2. **Python Adjudicator Bridge** - Use actual SimulatedAnnealingAdjudicator for terminal evaluation
3. **Online Learning** - Update policy from live game results
4. **Opponent Modeling** - Adapt ensemble weights based on detected opponent style
5. **GPU Acceleration** - Use MATLAB's GPU arrays for faster neural network inference
