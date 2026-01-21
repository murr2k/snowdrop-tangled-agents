# Theory of Operation

This document provides comprehensive documentation of the Snowdrop Tangled Agents
project, including the game mechanics, strategy algorithms, web automation system,
and learning mechanisms.

## Table of Contents

1. [Project Overview](#project-overview)
2. [The Tangled Game](#the-tangled-game)
3. [Petersen Graph Structure](#petersen-graph-structure)
4. [Strategy Engine](#strategy-engine)
5. [Web Automation](#web-automation)
6. [Learning System](#learning-system)
7. [Architecture](#architecture)

---

## Project Overview

This project implements intelligent agents for the Tangled quantum game, a two-player
combinatorial game played on graph structures. The primary focus is the Petersen graph
(Graph #11), with a parameterized strategy engine that learns from game outcomes.

### Components

```
snowdrop-tangled-agents/
├── snowdrop_tangled_agents/
│   ├── agents/
│   │   ├── petersen_agent.py    # SDK-compatible agent wrapper
│   │   └── random_randy.py      # Baseline random agent
│   ├── strategy/
│   │   └── petersen_strategy.py # Core strategy engine
│   └── playing_games/
│       └── run_local_parallel_tournament.py
├── play_tangled.py              # Web automation for tangled-game.com
└── docs/
    └── tangled-bot-v28.txt      # Reference JS implementation
```

---

## The Tangled Game

### Game Mechanics

Tangled is a two-player game where players take turns coloring edges of a graph:

- **Green (Ferromagnetic/FM)**: Rewards vertices having the same spin
- **Purple (Antiferromagnetic/AFM)**: Rewards vertices having opposite spins

### Scoring

After all edges are colored, a quantum adjudicator determines the optimal spin
configuration for each vertex. The score is computed as:

```
Score = (Green edges with matching spins) - (Green edges with opposite spins)
      + (Purple edges with opposite spins) - (Purple edges with matching spins)
```

Player 1 (Red) wins if score > 0, Player 2 (Blue) wins if score < 0.

### Player Vertices

Each player has a "home" vertex:
- **Player 1 (Red)**: Vertex 5 (left side of outer pentagon)
- **Player 2 (Blue)**: Vertex 7 (right side of outer pentagon)
- **Hub Vertex**: Vertex 6 (top, strategically important)

---

## Petersen Graph Structure

The Petersen graph has 10 vertices and 15 edges, arranged as:
- **Outer Pentagon**: Vertices 5, 6, 7, 8, 9
- **Inner Pentagram**: Vertices 0, 1, 2, 3, 4 (star pattern)

### Edge List

```python
PETERSEN_EDGES = [
    (0, 2), (0, 3), (0, 6),  # E0-E2: Inner vertex 0 connections
    (1, 3), (1, 4), (1, 7),  # E3-E5: Inner vertex 1 connections
    (2, 4), (2, 8),          # E6-E7: Inner vertex 2 connections
    (3, 9),                   # E8: Inner vertex 3 connection
    (4, 5),                   # E9: Spoke to Player 1 vertex
    (5, 6), (5, 9),          # E10-E11: Player 1 vertex edges
    (6, 7),                   # E12: Hub edge (connects P1 and P2 regions)
    (7, 8), (8, 9),          # E13-E14: Remaining outer edges
]
```

### Visual Layout

```
                    V6 (HUB)
                      /\
                     /  \
                    /    \
              V5 (RED)    V7 (BLUE)
               /\          /\
              /  \        /  \
             /    \      /    \
           V9------V8---+
            \      /
             \    /
          [Inner Pentagram]
             V0-V4
```

### Strategic Edge Classifications

| Category | Edges | Description |
|----------|-------|-------------|
| MY_EDGES | E9, E10, E11 | Touch Player 1's vertex (5) |
| OPP_EDGES | E5, E12, E13 | Touch Player 2's vertex (7) |
| HUB_EDGES | E2, E10, E12 | Touch the hub vertex (6) |

---

## Strategy Engine

### PetersenStrategy Class

Located in `snowdrop_tangled_agents/strategy/petersen_strategy.py`

#### Parameters

```python
DEFAULT_PARAMS = {
    # Priority weights (higher = play first)
    "w_my_edge": 10.0,      # Edges touching our vertex
    "w_opp_edge": 8.0,      # Edges touching opponent's vertex
    "w_hub_edge": 5.0,      # Edges touching hub vertex
    "w_neutral": 1.0,       # Other edges

    # Hub preference multiplier
    "hub_priority": 0.8,

    # Color decision thresholds
    "green_threshold": 1.0,   # Play green if score > threshold
    "purple_threshold": -1.0, # Play purple if score < threshold

    # Adaptive factors
    "momentum_weight": 0.5,      # Weight of recent score trend
    "opp_pattern_weight": 0.3,   # Weight of opponent analysis

    # Per-edge learned values
    "edge_values": [0.0] * 15,   # Adjusted by learning

    # Strategy mode
    "strategy_mode": "adaptive", # "defensive", "aggressive", or "adaptive"

    # Opening book
    "opening_sequence": [(9, 'G'), (10, 'G'), (11, 'G')],
}
```

#### Move Calculation Algorithm

```
calculate_move(state, score, score_history):
    1. Check opening sequence override
    2. Score all available (grey) edges:
       - Base score from edge_values[i]
       - Add category bonus (my/opp/hub/neutral)
       - Add hub priority bonus if applicable
       - Add momentum adjustment
       - Add opponent pattern adjustment
    3. Select highest-scoring edge
    4. Choose color based on:
       - MY_EDGES → Green (always)
       - OPP_EDGES → Purple (always)
       - Others → Based on mode and score thresholds
```

#### Edge Scoring Formula

```
score(edge) = edge_values[edge]
            + category_weight(edge)
            + hub_bonus(edge)
            + momentum * momentum_weight
            + opp_preference[edge] * opp_pattern_weight
```

#### Color Selection Logic

```python
def choose_color(edge, score):
    if edge in MY_EDGES:
        return 'G'  # Always green on our edges
    if edge in OPP_EDGES:
        return 'P'  # Always purple on opponent edges

    if mode == "defensive":
        return 'G'
    elif mode == "aggressive":
        return 'P'
    else:  # adaptive
        if score > green_threshold:
            return 'G'  # Protect lead
        elif score < purple_threshold:
            return 'P'  # Attack when behind
        else:
            return 'G'  # Default defensive
```

---

## Web Automation

### play_tangled.py

Automates gameplay on tangled-game.com using Playwright.

### Dynamic Vertex Discovery

The website renders the Petersen graph as SVG `<line>` elements. Vertex positions
vary, so we dynamically discover them:

```
1. Collect all line endpoints
2. Cluster points within 5px tolerance → 10 unique vertices
3. Calculate center of mass
4. Separate by distance: outer 5 (far) vs inner 5 (close)
5. Sort each group by angle from center
6. Rotate outer so leftmost (angle ≈ π) is vertex 5
7. Rotate inner so topmost (angle ≈ -π/2) is vertex 0
8. Build vertex map: VTX[0-4] = inner, VTX[5-9] = outer
```

### Angle Wrap-Around Handling

```javascript
const angleDist = (a1, a2) => {
    let d = a1 - a2;
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    return Math.abs(d);
};
```

### Edge-to-Line Mapping

```
For each SVG line:
    1. Get endpoints (x1,y1) and (x2,y2)
    2. Find nearest vertex for each endpoint
    3. Look up edge index in PETERSEN_EDGES
    4. Extract color from stroke attribute
```

### Move Execution

```
execute_move(edge, color):
    1. Discover vertices dynamically
    2. Find line connecting edge vertices
    3. Verify line is grey (available)
    4. Dispatch click event on line center
    5. Wait for color dialog
    6. Click color button (Green/Purple/FM/AFM)
```

### Turn Detection

```python
def is_our_turn():
    text = page.inner_text("body").lower()
    if "your turn" in text:
        return True
    if "player 1" in text and "turn" in text:
        return True
    if "opponent" in text and "turn" in text:
        return False
    if "waiting" in text:
        return False
    return False
```

### Browser Lifecycle Management

```python
# Signal handlers for cleanup
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)
atexit.register(_cleanup_on_exit)

# Context manager support
with WebPlayer() as player:
    player.login()
    player.play_game("melissa")
# Browser automatically closed on exit
```

---

## Learning System

### REINFORCE-Style Policy Gradient

After each game, the strategy updates based on outcomes using a simplified
REINFORCE algorithm:

#### Discounted Returns

```python
gamma = 0.95  # Discount factor
G = outcome_bonus * 2.0  # Terminal reward (+2 win, -2 loss)

# Work backwards through moves
for i in range(len(history) - 1, -1, -1):
    immediate_reward = score_after - score_before
    G = immediate_reward + gamma * G
    returns[i] = G

# Normalize returns
returns = (returns - mean) / std
```

#### Parameter Updates

```python
base_lr = 0.05

for edge, color, _ in history:
    advantage = returns[i]
    update = base_lr * advantage
    edge_values[edge] = clamp(edge_values[edge] + update, 0.0, 2.0)
```

#### Adaptive Weight Adjustment

```python
if result == "loss":
    if final_score < -1.0:
        # Lost badly → increase opponent edge priority
        w_opp_edge += 0.5
    elif final_score > 0:
        # Lost despite positive score → increase own edge priority
        w_my_edge += 0.5
```

#### Opening Sequence Adaptation

```python
if result == "loss" and early_score < -0.5:
    # Poor opening → try swapping first two moves
    opening_sequence = [opening[1], opening[0]] + opening[2:]
```

### Parameter Persistence

```python
# Auto-save after each game
params_path = ~/.tangled/petersen_params.json
strategy.save_params(params_path)

# Load on startup
if os.path.exists(params_path):
    strategy.load_params(params_path)
```

---

## Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     tangled-game.com                             │
│  ┌─────────┐    ┌──────────┐    ┌─────────┐                     │
│  │   SVG   │───→│  Lines   │───→│ Colors  │                     │
│  │  Graph  │    │ (edges)  │    │ G/P/-   │                     │
│  └─────────┘    └──────────┘    └─────────┘                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │ Playwright
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      WebPlayer                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Vertex     │───→│    Edge      │───→│    State     │      │
│  │  Discovery   │    │   Mapping    │    │   String     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                       │                │
│         │            ┌──────────────┐          │                │
│         └───────────→│    Score     │←─────────┘                │
│                      │   Reading    │                           │
│                      └──────────────┘                           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PetersenStrategy                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Edge      │───→│    Color     │───→│    Move      │      │
│  │   Scoring    │    │  Selection   │    │   Output     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         ▲                                       │                │
│         │            ┌──────────────┐          │                │
│         └────────────│   Learning   │←─────────┘                │
│                      │    Update    │                           │
│                      └──────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

### State Representation

```
State String: "---G-----GGP---"
              |||||||||||||||
              E0............E14

'-' = Grey (available)
'G' = Green (Ferromagnetic)
'P' = Purple (Antiferromagnetic)
'?' = Unknown (opponent's move, color not visible)
```

### Score History Format

```python
score_history = [
    (edge_index, color, score_after_move),
    (9, 'G', 1.008),   # Move 1: E9 Green, score became 1.008
    (10, 'G', 0.956),  # Move 2: E10 Green, score became 0.956
    ...
]
```

---

## Usage

### Running the Web Bot

```bash
# Play 5 games against MCTS Melissa
python play_tangled.py

# Play 10 games against Random Randy
python play_tangled.py --opponent randy --games 10

# Keep browser open 10 seconds after games
python play_tangled.py --keep-open 10

# Enable debug logging
python play_tangled.py --debug
```

### Environment Setup

Create `.env` file:
```
TANGLED_USERNAME=your_username
TANGLED_PASSWORD=your_password
```

### Running Local Tournament

```bash
poetry run python snowdrop_tangled_agents/playing_games/run_local_parallel_tournament.py
```

---

## Future Improvements

1. **Better Opening Book**: Analyze successful openings across many games
2. **Monte Carlo Tree Search**: Implement MCTS for deeper lookahead
3. **Neural Network Policy**: Train a neural net on game outcomes
4. **Multi-Graph Support**: Extend strategy to other X-Prize graphs
5. **Opponent Modeling**: Build models of specific opponents' tendencies
