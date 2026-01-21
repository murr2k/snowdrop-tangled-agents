# snowdrop-tangled-agents

A framework for building and testing agents for the Tangled quantum game. Includes strategy engines, web automation for playing on tangled-game.com, and statistical analysis tools.

## Quick Start

### Prerequisites

- Python 3.11 - 3.13
- Git

### Installation

**Automated Setup (Recommended)**

```bash
# Clone the repository
git clone https://github.com/snowdropquantum/snowdrop-tangled-agents.git
cd snowdrop-tangled-agents

# Run the setup script
# On Windows:
setup.bat

# On macOS/Linux:
chmod +x setup.sh
./setup.sh
```

**Manual Setup**

```bash
# Using Poetry (recommended)
poetry install
poetry run playwright install chromium

# Using pip
pip install -e .
pip install snowdrop-tangled-game-engine snowdrop-adjudicators
python -m playwright install chromium
```

### Configuration

Copy `.env.example` to `.env` and add your tangled-game.com credentials:

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Play a Game

```bash
# Play against MCTS Melissa on tangled-game.com
python play_tangled.py --games 1

# Use different strategies
python play_tangled.py --strategy mcts --mcts-time 5 --games 3
python play_tangled.py --strategy hybrid --games 3

# View game statistics
python play_tangled.py --stats

# View adjudicator calibration
python play_tangled.py --calibration
```

---

## How to build your own agent
This repo includes a random agent (see `RandomRandyAgent` in `random_agent.py`). To build your
own agent, follow the instructions and pattern for that agent. The process is straightforward - your
agent gets a Game instance as input/observable, which is a snapshot of the current game being played 
which includes the current state of the game, and you output an action/move. The move is just which 
edge to pick, and what color to apply to it. That's all! Obviously the underlying process you use to 
select this move can be arbitrarily complex but the actual agent interface is simple.

## Testing agents
A test script is included in `run_local_parallel_tournament.py`. Out of the box it runs 100,000 games
for each of the X-Prize graphs (not including the mutant C60 graph) where the adjudicator used is
simulated annealing, and the two agents in the competition are both random agents.

## Suggested next steps
If you want to build your own agent, I would start by building your own adjudicator (look at both
the adjudicators in the `snowdrop-adjudicators` repo). The better you
can spoof the D-Wave hardware, the better quality reward your agent will have to learn from. To do this,
I would recommend you build a look-up table adjudicator, by first enumerating all the terminal states
for a small game graph, and then adjudicating all of them using the schrodinger equation adjudicator (this will
be limited to small game graphs, but it's still a great warm-up for doing something more ambitious). Once
you have your lookup table adjudicator, you can then build an agent that uses it to give you the correct
answer every time. You can use `run_local_parallel_tournament.py` - add your adjudicator and new agents to
and see how well they do against the random agent and any other agents you might have built!

---

## Development Progress

This section documents the actual development steps taken to build a competitive Tangled agent.

### 1. MCTS Strategy Engine

Implemented a Monte Carlo Tree Search (MCTS) strategy with the following features:

- **UCB1 exploration** with configurable exploration constant
- **Simulated Annealing adjudicator** for terminal state evaluation
- **Opening book** with pre-computed strong opening moves for the Petersen graph
- **Time and iteration limits** for move calculation
- **Statistics collection** for game analysis

```bash
# Play with MCTS strategy
python play_tangled.py --strategy mcts --mcts-time 5 --games 3
```

### 2. Web Automation Bot

Built a Playwright-based bot to play on tangled-game.com:

- **Automatic login** with credentials from `.env`
- **Game state parsing** from the web interface
- **Move execution** via browser automation
- **Score tracking** and game result detection
- **Keep-alive mode** for continuous play sessions

```bash
# Play games on tangled-game.com
python play_tangled.py --strategy hybrid --games 10 --keep-open
```

### 3. Statistics Collection System

Created a SQLite-based statistics system:

- **Game history** tracking (opponent, result, scores, timestamps)
- **Move-by-move analysis** with state snapshots
- **Adjudicator calibration** data collection
- **Win rate analysis** by opponent and strategy

```bash
# View statistics
python play_tangled.py --stats
python play_tangled.py --calibration
```

### 4. MATLAB Toolbox Integration

Extended the system with MATLAB integration for advanced ML capabilities:

| Toolbox | Purpose |
|---------|---------|
| **Deep Learning Toolbox** | Neural network training for position evaluation |
| **Statistics and ML Toolbox** | Opponent clustering and style classification |
| **Database Toolbox** | Direct MATLAB-SQLite access for training pipelines |
| **MATLAB Compiler SDK** | Package MATLAB code as Python-callable modules |

#### Neural Network Architecture

**Value Network** (position evaluation):
- Input: 50-element feature vector (board state, edge categories, game phase)
- Hidden: FC(128) → ReLU → Dropout → FC(64) → ReLU → FC(32) → ReLU
- Output: Tanh → value ∈ [-1, +1]

**Opponent Model** (style classification):
- 20-element opponent feature vector (edge preferences, aggression metrics)
- K-means clustering into play styles (aggressive, defensive, hub-focused)
- Adaptive prior adjustment based on opponent tendencies

### 5. Compiled Package Deployment

Built standalone Python packages using MATLAB Compiler SDK:

| Package | Functions | Purpose |
|---------|-----------|---------|
| `tangled_value_network` | `evaluate_position_nn` | Neural network inference |
| `tangled_opponent_model` | `classify_opponent`, `adapt_to_opponent` | Opponent modeling |
| `tangled_training` | `train_value_network`, `cluster_opponents` | Model training |

**For users without MATLAB:**
1. Download MATLAB Runtime R2026a (free) from [MathWorks](https://www.mathworks.com/products/compiler/matlab-runtime.html)
2. Install compiled packages: `pip install tangled_value_network tangled_opponent_model`

**Important:** Runtime version must exactly match the compile version (R2026a).

### 6. Backend Fallback Chain

The system automatically selects the best available backend:

```
1. Compiled Packages (fastest, MATLAB Runtime only)
       ↓ if unavailable
2. MATLAB Engine API (full functionality, requires license)
       ↓ if unavailable
3. Pure Python Heuristics (always available)
```

```bash
# Check what's available
python play_tangled.py --training-status

# Play with MATLAB enhancement
python play_tangled.py --strategy matlab --use-nn --adapt-opponent --games 5
```

### Documentation

- `docs/THEORY_OF_OPERATION.md` - Comprehensive system documentation
- `docs/MATLAB_INTEGRATION.md` - Complete MATLAB integration guide
- `docs/TEST_SUITE.md` - Regression test documentation
- `docs/THE_MATHEMATICS_OF_TANGLED_GAME.md` - Game theory analysis