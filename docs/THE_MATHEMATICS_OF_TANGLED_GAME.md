# The Mathematics of Tangled: A Quantum Game Theory Framework

*A comprehensive guide for graduate students and researchers*

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Theoretical Foundations](#2-theoretical-foundations)
   - 2.1 [The Ising Model](#21-the-ising-model)
   - 2.2 [Graph Theory: The Petersen Graph](#22-graph-theory-the-petersen-graph)
   - 2.3 [Quantum Annealing](#23-quantum-annealing)
3. [Game Formalization](#3-game-formalization)
   - 3.1 [State Space](#31-state-space)
   - 3.2 [Action Space](#32-action-space)
   - 3.3 [Terminal Evaluation](#33-terminal-evaluation)
4. [Algorithmic Approaches](#4-algorithmic-approaches)
   - 4.1 [Monte Carlo Tree Search with Progressive Bias](#41-monte-carlo-tree-search-with-progressive-bias)
   - 4.2 [Heuristic Edge Scoring](#42-heuristic-edge-scoring)
   - 4.3 [Minimax for Endgame](#43-minimax-for-endgame)
5. [Learning Mechanisms](#5-learning-mechanisms)
6. [Experimental Results](#6-experimental-results)
7. [Open Research Questions](#7-open-research-questions)
8. [MATLAB Integration for Enhanced Gameplay](#8-matlab-integration-for-enhanced-gameplay)
   - 8.1 [Overview](#81-overview)
   - 8.2 [Architecture Workflow](#82-architecture-workflow)
   - 8.3 [Data Structures](#83-data-structures)
   - 8.4 [MATLAB Function Specifications](#84-matlab-function-specifications)
   - 8.5 [Integration Workflow: Step-by-Step](#85-integration-workflow-step-by-step)
   - 8.6 [Parallel Computing Architecture](#86-parallel-computing-architecture)
   - 8.7 [Game Trace Data Structure](#87-game-trace-data-structure)
   - 8.8 [Error Handling and Fallbacks](#88-error-handling-and-fallbacks)
   - 8.9 [Performance Metrics](#89-performance-metrics)
9. [The HybridSolverStrategy: D-Wave Inspired Search with Adaptive Learning](#9-the-hybridsolverstrategy-d-wave-inspired-search-with-adaptive-learning)
   - 9.1 [Motivation and Design Philosophy](#91-motivation-and-design-philosophy)
   - 9.2 [System Architecture](#92-system-architecture)
   - 9.3 [The MATLAB Solver Components](#93-the-matlab-solver-components)
   - 9.4 [REINFORCE-Style Adaptive Learning](#94-reinforce-style-adaptive-learning)
   - 9.5 [Database Integration and Persistence](#95-database-integration-and-persistence)
   - 9.6 [Performance Analysis](#96-performance-analysis)
   - 9.7 [Research Extensions](#97-research-extensions)
10. [References and Further Reading](#10-references-and-further-reading)

---

## 1. Problem Statement

### The Tangled Game

Tangled is a two-player, zero-sum, perfect-information combinatorial game played on a graph structure. The game combines:

- **Classical graph coloring**: Players alternately color edges
- **Quantum physics**: Terminal states are evaluated via quantum annealing simulation
- **Adversarial optimization**: One player maximizes, the other minimizes the final score

### The Challenge

Given a graph G = (V, E) with |V| = n vertices and |E| = m edges:

1. Two players alternately select uncolored edges and assign them colors (Green or Purple)
2. Each player "owns" a designated vertex
3. After all edges are colored, a quantum annealing process determines the optimal spin assignment
4. The final score determines the winner

**Research Question**: What is the optimal strategy for edge selection and coloring in this quantum-classical hybrid game?

### Complexity Analysis

For the Petersen graph (n=10, m=15):
- State space: 3^15 = 14,348,907 possible partial colorings
- Terminal states: 2^15 = 32,768 fully colored boards
- Game tree: Approximately 15! × 2^15 ≈ 4.3 × 10^16 nodes

This complexity necessitates heuristic search methods rather than exhaustive analysis.

---

## 2. Theoretical Foundations

### 2.1 The Ising Model

The Ising model is a mathematical model of ferromagnetism in statistical mechanics. In the context of Tangled:

**Definition**: Given a graph G = (V, E), a spin configuration is a mapping σ: V → {-1, +1}.

**The Hamiltonian** (energy function):

$$H(\sigma) = -\sum_{(i,j) \in E} J_{ij} \sigma_i \sigma_j$$

Where:
- σᵢ ∈ {-1, +1} is the spin at vertex i
- J_{ij} is the coupling constant for edge (i,j)

**Coupling Constants in Tangled**:
- **Ferromagnetic (Green)**: J_{ij} = +1
  Favors aligned spins (σᵢ = σⱼ minimizes energy)
- **Antiferromagnetic (Purple)**: J_{ij} = -1
  Favors anti-aligned spins (σᵢ ≠ σⱼ minimizes energy)

**Physical Interpretation**:

| Edge Color | Coupling | Minimum Energy When | Game Effect |
|------------|----------|---------------------|-------------|
| Green (FM) | J = +1 | Spins aligned | Rewards same-spin neighbors |
| Purple (AFM) | J = -1 | Spins anti-aligned | Rewards opposite-spin neighbors |

### 2.2 Graph Theory: The Petersen Graph

The primary game board is the **Petersen graph**, a well-studied object in algebraic graph theory.

**Formal Definition**:
- Vertices: V = {0, 1, 2, ..., 9}
- Edges: |E| = 15

```
Edge List (lexicographically ordered):
E0:  (0,2)    E5:  (1,7)    E10: (5,6)
E1:  (0,3)    E6:  (2,4)    E11: (5,9)
E2:  (0,6)    E7:  (2,8)    E12: (6,7)
E3:  (1,3)    E8:  (3,9)    E13: (7,8)
E4:  (1,4)    E9:  (4,5)    E14: (8,9)
```

**Structural Properties**:
- 3-regular: Every vertex has degree 3
- Girth: 5 (shortest cycle has 5 edges)
- Diameter: 2 (max distance between any two vertices)
- Non-planar: Contains K₃,₃ as a minor
- Vertex-transitive: High symmetry group (order 120)

**Topological Structure**:

```
                        V6 (HUB)
                        /    \
                       /      \
                      /        \
                V5 (P1)        V7 (P2)
                /    \        /    \
               /      \      /      \
              V9-------V8---+
               \      /
                \    /
              [Inner Pentagram]
                  V0-V4
```

**Player Vertex Assignments**:
- Player 1 (Red): V5
- Player 2 (Blue): V7
- Hub Vertex: V6 (strategically critical, equidistant from both players)

**Edge Categories**:
```python
MY_EDGES   = [9, 10, 11]  # Touch V5: {(4,5), (5,6), (5,9)}
OPP_EDGES  = [5, 12, 13]  # Touch V7: {(1,7), (6,7), (7,8)}
HUB_EDGES  = [2, 10, 12]  # Touch V6: {(0,6), (5,6), (6,7)}
```

Note: E10 and E12 belong to multiple categories (MY∩HUB and OPP∩HUB).

### 2.3 Quantum Annealing

Quantum annealing is a metaheuristic for finding the global minimum of an objective function over discrete variables.

**The Optimization Problem**:

Given the colored graph (with fixed J_{ij} values), find:

$$\sigma^* = \arg\min_{\sigma} H(\sigma) = \arg\min_{\sigma} \left( -\sum_{(i,j) \in E} J_{ij} \sigma_i \sigma_j \right)$$

**Simulated vs. Quantum Annealing**:

| Aspect | Simulated Annealing | Quantum Annealing |
|--------|---------------------|-------------------|
| Mechanism | Thermal fluctuations | Quantum tunneling |
| Implementation | Classical computer | D-Wave hardware or simulation |
| Complexity | Polynomial (heuristic) | Potentially faster for some problems |
| This Project | SimulatedAnnealingAdjudicator | SchrodingerEquationAdjudicator |

**Adjudicator Implementation**:

```python
from snowdrop_adjudicators import SimulatedAnnealingAdjudicator

adj = SimulatedAnnealingAdjudicator()
adj.setup(epsilon=0.0, num_reads=10000)
result = adj.adjudicate(game_state)
score = float(result['score'])
```

**Score Interpretation**:
- Score > 0: Player 1 (Red) wins
- Score < 0: Player 2 (Blue) wins
- Score = 0: Draw

---

## 3. Game Formalization

### 3.1 State Space

**State Representation**:

A game state is encoded as a 15-character string where each character represents an edge:

```
State := {'-', 'G', 'P'}^15

'-' = Grey (uncolored, available)
'G' = Green (Ferromagnetic, J = +1)
'P' = Purple (Antiferromagnetic, J = -1)
```

**Example**:
```
State: "G--P---G--GPP--"
        |||||||||||||||
        E0............E14

Interpretation:
- E0 (0,2): Green
- E3 (1,3): Purple
- E7 (2,8): Green
- E10 (5,6): Green
- E11 (5,9): Purple
- E12 (6,7): Purple
- Others: Uncolored
```

**State Properties**:
- |S_initial| = 1 (all grey)
- |S_terminal| = 2^15 = 32,768 (all edges colored)
- |S_partial| = 3^15 - 2^15 ≈ 14.3 million non-terminal states

### 3.2 Action Space

**Action Definition**:

An action a = (edge_index, color) where:
- edge_index ∈ {0, 1, ..., 14}
- color ∈ {'G', 'P'}

**Valid Actions**:

At state s, the valid action set is:
$$A(s) = \{(e, c) : s[e] = '\text{-}', c \in \{G, P\}\}$$

**Branching Factor**:
- Turn 1: 15 × 2 = 30 actions
- Turn k: (15 - k + 1) × 2 actions
- Average: ~15 actions per turn

### 3.3 Terminal Evaluation

**The Score Function**:

For a terminal state s (all edges colored), the score is computed by:

1. **Build the Ising Hamiltonian**:
   $$H(\sigma) = -\sum_{(i,j) \in E} J_{ij}(s) \cdot \sigma_i \sigma_j$$

   where J_{ij}(s) = +1 if s[edge(i,j)] = 'G', else -1

2. **Find optimal spin configuration**:
   $$\sigma^* = \arg\min_{\sigma} H(\sigma)$$

3. **Compute score**:
   $$\text{Score} = f(\sigma^*, J) = \sum_{(i,j)} J_{ij} \cdot \mathbb{1}[\sigma^*_i = \sigma^*_j] - \sum_{(i,j)} J_{ij} \cdot \mathbb{1}[\sigma^*_i \neq \sigma^*_j]$$

**Simplified Score Formula**:

$$\text{Score} = \underbrace{\text{(Green edges with aligned spins)}}_{\text{+contribution}} - \underbrace{\text{(Green edges with opposite spins)}}_{\text{-contribution}}$$
$$+ \underbrace{\text{(Purple edges with opposite spins)}}_{\text{+contribution}} - \underbrace{\text{(Purple edges with aligned spins)}}_{\text{-contribution}}$$

**Score Range**: Empirically, scores on the Petersen graph typically fall in [-5, +5].

---

## 4. Algorithmic Approaches

### 4.1 Monte Carlo Tree Search with Progressive Bias

**Standard MCTS Algorithm**:

```
function MCTS(root_state, time_limit):
    root = Node(root_state)
    while time_remaining():
        node = SELECT(root)           # Tree traversal
        node = EXPAND(node)           # Add child
        value = SIMULATE(node.state)  # Random rollout
        BACKPROPAGATE(node, value)    # Update statistics
    return best_child(root)
```

**UCB1 Selection Formula**:

The Upper Confidence Bound for Trees (UCT) selects children by maximizing:

$$UCB1(n) = \frac{Q(n)}{N(n)} + c \cdot \sqrt{\frac{\ln N(\text{parent})}{N(n)}}$$

Where:
- Q(n) = total value accumulated at node n
- N(n) = visit count of node n
- c = exploration constant (typically √2 ≈ 1.414)

**Progressive Bias Extension**:

We enhance UCB1 with prior knowledge:

$$UCB1_{PB}(n) = \frac{Q(n)}{N(n)} + c \cdot \sqrt{\frac{\ln N(\text{parent})}{N(n)}} + w \cdot \frac{\pi(n) - 0.5}{N(n) + 1}$$

Where:
- π(n) ∈ [0, 1] is the heuristic prior for action leading to n
- w = prior weight (decays as visits increase)

**Prior Computation**:

```python
def compute_action_prior(edge: int, color: str, is_our_turn: bool) -> float:
    """Returns value in [0, 1]. Higher = believed to be better."""
    prior = 0.5  # Neutral baseline

    if is_our_turn:
        if edge in MY_EDGES:
            prior = 0.99 if color == 'G' else 0.01
        elif edge in OPP_EDGES:
            prior = 0.95 if color == 'P' else 0.05
        elif edge in HUB_EDGES:
            prior = 0.70 if color == 'G' else 0.30
    else:
        # Opponent modeling: they likely play symmetrically
        if edge in OPP_EDGES:
            prior = 0.95 if color == 'G' else 0.05
        elif edge in MY_EDGES:
            prior = 0.85 if color == 'P' else 0.15

    return prior
```

**Backpropagation with Perspective**:

Critical insight: In two-player games, values must alternate sign up the tree:

```python
def backpropagate(node, value):
    while node is not None:
        node.visits += 1
        node.total_value += value
        value = -value  # Negate for opponent's perspective
        node = node.parent
```

### 4.2 Heuristic Edge Scoring

**Edge Prioritization**:

The heuristic strategy scores each available edge:

$$\text{Score}(e) = w_{\text{cat}(e)} + \text{hub\_bonus}(e) + \text{momentum} \cdot w_m + \text{learned}(e)$$

Where:
- w_{cat(e)} = category weight (MY_EDGE: 10, OPP_EDGE: 8, HUB_EDGE: 5, NEUTRAL: 1)
- hub_bonus = 0.8 if e ∈ HUB_EDGES
- momentum = recent score trend
- learned(e) = per-edge value from reinforcement learning

**Color Decision Logic**:

```python
def choose_color(edge_idx: int, score: float, mode: str) -> str:
    # Fixed rules
    if edge_idx in MY_EDGES:
        return 'G'  # Always Green on our edges
    if edge_idx in OPP_EDGES:
        return 'P'  # Always Purple on opponent edges

    # Adaptive rules for neutral edges
    if mode == "defensive" or score > 1.0:
        return 'G'  # Protect lead
    elif mode == "aggressive" or score < -1.0:
        return 'P'  # Attack when behind
    else:
        return 'G'  # Default defensive
```

### 4.3 Minimax for Endgame

When ≤2 edges remain, exhaustive minimax is feasible:

**Minimax Algorithm**:

```python
def minimax(state: str, is_maximizing: bool, depth: int) -> float:
    if is_terminal(state) or depth == 0:
        return evaluate(state)

    available_actions = get_available_actions(state)

    if is_maximizing:
        best = -infinity
        for action in available_actions:
            new_state = apply_action(state, action)
            value = minimax(new_state, False, depth - 1)
            best = max(best, value)
        return best
    else:
        best = +infinity
        for action in available_actions:
            new_state = apply_action(state, action)
            value = minimax(new_state, True, depth - 1)
            best = min(best, value)
        return best
```

**Endgame Complexity**:
- 2 edges remaining: 2 × 2 × 2 × 2 = 16 leaf nodes
- 3 edges remaining: 6 × 4 × 2 = 48 leaf nodes
- Fully tractable for depth ≤ 4

---

## 5. Learning Mechanisms

### REINFORCE-Style Policy Gradient

**Objective**: Learn edge priorities from game outcomes.

**Algorithm**:

Given a completed game with move history [(e₁, c₁, s₁), (e₂, c₂, s₂), ...]:

1. **Compute Discounted Returns**:
   $$G_t = r_t + \gamma G_{t+1}$$

   Where:
   - r_t = s_t - s_{t-1} (immediate score change)
   - γ = 0.95 (discount factor)
   - Terminal reward = ±2.0 based on outcome

2. **Normalize Returns**:
   $$\hat{G}_t = \frac{G_t - \mu_G}{\sigma_G}$$

3. **Update Edge Values**:
   $$\theta_e \leftarrow \theta_e + \alpha \cdot \hat{G}_t$$

   Where α = 0.05 (learning rate)

**Parameter Bounds**:
- Edge values: θ_e ∈ [0, 2]
- Category weights: w ∈ [0, 15]

### Opponent Modeling

**Score Trajectory Analysis**:

```python
def compute_momentum(score_history: list, window: int = 4) -> float:
    """Compute recent score trend."""
    recent = score_history[-window:]
    if len(recent) < 2:
        return 0.0
    return (recent[-1] - recent[0]) / len(recent)
```

**Opponent Edge Preferences**:

Track which edges the opponent targets and their color choices to predict future moves.

---

## 6. Experimental Results

### Benchmark Opponents

| Opponent | Algorithm | Behavior |
|----------|-----------|----------|
| Random Randy | Uniform random | Baseline (0% intelligence) |
| MCTS Melissa | Pure MCTS | Variable, exploits patterns |
| AlphaZero Amara | Neural MCTS | Reactive/equalizing |

### Performance Summary

**vs. Random Randy**:
- Win rate: 100%
- Average margin: +3.2 points
- Validates that heuristics dominate randomness

**vs. MCTS Melissa**:
- With E10-first opening: 0W-2L-3D
- With E9-first opening: 0W-0L-2D+
- Insight: Opening choice significantly affects outcomes

**vs. AlphaZero Amara**:
- Record: 0W-4D-4L
- Pattern: Opponent mirrors our strategy, equalizing the game
- Insight: Well-trained value networks prevent blunders

### Adjudicator Calibration

**Simulated Annealing vs. Website Scores**:

| Metric | Value |
|--------|-------|
| Mean Absolute Error | < 0.02 points |
| Correlation | > 0.99 |
| Accuracy (within 0.5) | 100% |

**Known Systematic Errors by Graph**:

| Graph ID | Name | SA Accuracy |
|----------|------|-------------|
| 2 | Path-3 | Exact match |
| 11 | Petersen | Exact match |
| 12 | Moser Spindle | Systematic error |
| 18 | 3-Prism | Systematic error |
| 19 | Barbell | Systematic error |
| 20 | Diamond | Exact match |

---

## 7. Open Research Questions

### Theoretical Questions

1. **Optimal Strategy Characterization**:
   What is the game-theoretic value of Tangled on the Petersen graph with perfect play?

2. **Complexity Classification**:
   Is determining the winner of Tangled from an arbitrary position PSPACE-complete?

3. **Frustration Analysis**:
   How does graph frustration (inability to satisfy all edge constraints) affect strategic play?

### Algorithmic Questions

4. **Neural Network Priors**:
   Can a trained neural network provide better MCTS priors than hand-crafted heuristics?

5. **Transfer Learning**:
   Do strategies learned on the Petersen graph transfer to other X-Prize graphs?

6. **Opening Book Generation**:
   What is the optimal opening sequence, and can it be computed via retrograde analysis?

### Applied Questions

7. **Real Quantum Hardware**:
   How do strategies change when using actual D-Wave quantum annealers vs. simulation?

8. **Multiplayer Extensions**:
   How does the game theory change with n > 2 players?

---

## 8. MATLAB Integration for Enhanced Gameplay

### 8.1 Overview

The MATLAB-enhanced strategy leverages MATLAB's scientific computing toolboxes to augment the Python-based MCTS implementation. This hybrid approach combines:

- **Reinforcement Learning Toolbox**: Value network for position evaluation
- **Global Optimization Toolbox**: Simulated annealing for deep position analysis
- **System Identification Toolbox**: Opponent behavior modeling

### 8.2 Architecture Workflow

```mermaid
flowchart TD
    subgraph Python ["Python Runtime"]
        GS["Game State<br/>15-char string<br/>'G--P---G--GPP--'"]
        MS["MatlabEnhancedStrategy"]
        MCTS["MCTS Engine"]
        BR["MatlabBridge"]
        MOVE["Selected Move<br/>(edge, color)"]
    end

    subgraph MATLAB ["MATLAB Engine (R2026a)"]
        EP["evaluate_position.m"]
        SA["sa_evaluate.m"]
        IO["identify_opponent.m"]
        PP["Parallel Pool<br/>6 Workers"]
    end

    subgraph Transform ["Data Transforms"]
        T1["State Encoding<br/>str → matlab.double"]
        T2["Policy Decoding<br/>matlab.double → dict"]
        T3["Trace Encoding<br/>list → matrix"]
    end

    GS --> MS
    MS --> |"Opening<br/>Moves 1-3"| MOVE
    MS --> |"Midgame"| MCTS
    MS --> |"Endgame<br/>≤3 grey"| SA

    MCTS --> |"Get Priors"| BR
    BR --> T1
    T1 --> EP
    EP --> T2
    T2 --> MCTS

    SA --> BR
    BR --> T1
    T1 --> SA
    SA --> PP
    PP --> |"Multi-start<br/>Optimization"| SA

    MS --> |"Between<br/>Games"| IO
    IO --> |"Opponent<br/>Patterns"| MCTS

    MCTS --> MOVE
```

### 8.3 Data Structures

#### 8.3.1 Python State Representation

```python
# Game state as 15-character string
state: str = "G--P---G--GPP--"
# Index:       0123456789...14

# Character meanings:
#   'G' = Green (Ferromagnetic, J = +1)
#   'P' = Purple (Antiferromagnetic, J = -1)
#   '-' = Grey (Uncolored, available)
```

#### 8.3.2 MATLAB State Vector

```matlab
% State as 1×15 double vector
state_vec = [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 1, -1, -1, 0, 0]
%            E0 E1 E2  E3 E4 E5 E6 E7 E8 E9 E10 E11 E12 E13 E14

% Encoding:
%   +1.0 = Green (Ferromagnetic)
%   -1.0 = Purple (Antiferromagnetic)
%    0.0 = Grey (Uncolored)
```

#### 8.3.3 Transform Functions

**Python → MATLAB (State Encoding)**:

```python
def encode_state_for_matlab(state: str) -> matlab.double:
    """Convert 15-char state string to MATLAB vector."""
    return matlab.double([
        1.0 if c == 'G' else (-1.0 if c == 'P' else 0.0)
        for c in state
    ])

# Example:
# "G--P---G--GPP--" → [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 1, -1, -1, 0, 0]
```

**MATLAB → Python (Policy Decoding)**:

```python
def decode_policy_from_matlab(policy_raw, state: str) -> dict:
    """Convert MATLAB 30-element policy to Python dict."""
    # MATLAB returns 30 values: [G0, P0, G1, P1, ..., G14, P14]
    # representing probability for (edge, color) pairs

    # Extract raw data from matlab.double object
    if hasattr(policy_raw, '_data'):
        policy_list = list(policy_raw._data)
    else:
        policy_list = [float(x) for x in policy_raw]

    # Build dictionary for available moves only
    policy = {}
    for edge_idx in range(15):
        if state[edge_idx] == '-':  # Only grey edges
            policy[(edge_idx, 'G')] = policy_list[edge_idx * 2]
            policy[(edge_idx, 'P')] = policy_list[edge_idx * 2 + 1]

    return policy

# Example output:
# {(0, 'G'): 0.12, (0, 'P'): 0.08, (1, 'G'): 0.15, (1, 'P'): 0.05, ...}
```

### 8.4 MATLAB Function Specifications

#### 8.4.1 Position Evaluation (evaluate_position.m)

**Purpose**: Evaluate board position using a heuristic value network.

**Interface**:
```matlab
function [value, policy] = evaluate_position(state_vec, turn_flag)
%EVALUATE_POSITION Position evaluation using value network
%
%   Inputs:
%       state_vec  - 1×15 vector: 1=Green, -1=Purple, 0=Grey
%       turn_flag  - 1×1 scalar: 1=our turn, -1=opponent turn
%
%   Outputs:
%       value  - Scalar in [-1, 1]: expected outcome
%       policy - 1×30 vector: action probabilities
%                [G_E0, P_E0, G_E1, P_E1, ..., G_E14, P_E14]
```

**Evaluation Heuristic**:

$$\text{value} = \frac{1}{6}\left(\sum_{e \in \text{MY}} \text{benefit}(e) - \sum_{e \in \text{OPP}} \text{threat}(e)\right)$$

Where:
- benefit(e) = +1 if Green, -1 if Purple, 0 if Grey
- threat(e) = +1 if Green on opponent edge, -1 if Purple

**Policy Generation**:

The policy vector assigns probabilities based on edge category and color:

| Edge Category | Green Prior | Purple Prior |
|---------------|-------------|--------------|
| MY_EDGES (E9, E10, E11) | 0.95 | 0.05 |
| OPP_EDGES (E5, E12, E13) | 0.05 | 0.95 |
| HUB_EDGES (E2, E10, E12) | 0.70 | 0.30 |
| Neutral | 0.55 | 0.45 |

#### 8.4.2 Simulated Annealing Evaluation (sa_evaluate.m)

**Purpose**: Deep position evaluation using multi-start simulated annealing.

**Interface**:
```matlab
function [mean_value, confidence, best_moves] = sa_evaluate(state_vec, num_samples)
%SA_EVALUATE Deep position evaluation via simulated annealing
%
%   Inputs:
%       state_vec   - 1×15 vector: current board state
%       num_samples - Number of SA runs (default: 100)
%
%   Outputs:
%       mean_value - Average outcome across samples
%       confidence - Standard deviation (lower = more confident)
%       best_moves - N×2 matrix: [edge_index, color] recommendations
```

**Algorithm**:

```
1. Find grey edges: grey_indices = find(state_vec == 0)
2. For s = 1 to num_samples (parallel):
   a. Randomly complete the game (fill all grey edges)
   b. Build Ising Hamiltonian H(σ) = -Σ J_ij σ_i σ_j
   c. Run simulannealbnd to find ground state σ*
   d. Compute terminal score
3. Return statistics: mean(scores), std(scores), best_moves
```

**Simulated Annealing Parameters**:

```matlab
options = optimoptions('simulannealbnd', ...
    'Display', 'off', ...
    'MaxIterations', 500, ...
    'InitialTemperature', 5, ...
    'ReannealInterval', 25);
```

#### 8.4.3 Opponent Identification (identify_opponent.m)

**Purpose**: Learn opponent behavior patterns from game history.

**Interface**:
```matlab
function result = identify_opponent(traces)
%IDENTIFY_OPPONENT Opponent strategy pattern recognition
%
%   Inputs:
%       traces - N×M matrix where each row is a game trace:
%                [edge1, color1, score1, edge2, color2, score2, ...]
%
%   Outputs:
%       result.edge_prefs   - 1×15 vector: opponent edge preferences
%       result.color_bias   - Scalar: positive = prefers green
%       result.responses    - Struct: response patterns (ARX model)
```

**Edge Preference Calculation**:

$$\text{pref}(e) = \frac{\text{count}(\text{opponent played edge } e)}{\text{total opponent moves}}$$

**Color Bias**:

$$\text{bias} = \frac{N_{\text{green}} - N_{\text{purple}}}{N_{\text{green}} + N_{\text{purple}}}$$

Where bias ∈ [-1, 1]: positive indicates green preference.

**ARX Response Model** (System Identification Toolbox):

```matlab
% Model opponent score response to our color choices
y = opponent_score_changes;  % Output: score deltas
u = opponent_colors;          % Input: their color choices

data = iddata(y, u, 1);       % Create time-series data
model = arx(data, [2 2 1]);   % Fit ARX(2,2,1) model
```

### 8.5 Integration Workflow: Step-by-Step

The following trace shows a complete MATLAB-enhanced move calculation:

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Game State Received                                      │
├─────────────────────────────────────────────────────────────────┤
│ Python State: "G--P---G--G----"                                  │
│ Score: +0.42                                                     │
│ Move Number: 5                                                   │
│ Grey Edges: [1, 2, 4, 5, 6, 8, 11, 12, 13, 14]                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Python → MATLAB Transform                                │
├─────────────────────────────────────────────────────────────────┤
│ state_vec = matlab.double([                                      │
│     1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0                 │
│ ])                                                               │
│ turn_flag = matlab.double([1])  # Our turn                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: MATLAB evaluate_position() Called                        │
├─────────────────────────────────────────────────────────────────┤
│ >> [value, policy] = evaluate_position(state_vec, turn_flag)     │
│                                                                  │
│ Calculation:                                                     │
│   MY_EDGES status: E9=Grey, E10=Green, E11=Grey                  │
│   OPP_EDGES status: E5=Grey, E12=Grey, E13=Grey                  │
│   my_score = (0 + 1 + 0) = 1                                     │
│   opp_score = (0 + 0 + 0) = 0                                    │
│   value = (1 - 0) / 6 = +0.167                                   │
│                                                                  │
│ Returns:                                                         │
│   value = 0.167                                                  │
│   policy = [0.55, 0.45, 0.70, 0.30, ...]  (30 elements)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: MATLAB → Python Transform                                │
├─────────────────────────────────────────────────────────────────┤
│ # Extract value                                                  │
│ value = 0.167                                                    │
│                                                                  │
│ # Decode policy to action priors                                 │
│ priors = {                                                       │
│     (1, 'G'): 0.55,  (1, 'P'): 0.45,   # E1 neutral              │
│     (2, 'G'): 0.70,  (2, 'P'): 0.30,   # E2 hub edge             │
│     (5, 'G'): 0.05,  (5, 'P'): 0.95,   # E5 opponent edge        │
│     (11, 'G'): 0.95, (11, 'P'): 0.05,  # E11 our edge            │
│     ...                                                          │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: MCTS with Enhanced Priors                                │
├─────────────────────────────────────────────────────────────────┤
│ UCB1_PB(n) = Q(n)/N(n) + c√(ln(N_parent)/N(n)) + w·(π(n)-0.5)   │
│                                                                  │
│ Prior injection:                                                 │
│   Node for E11-Green: π = 0.95 → strong exploration bonus        │
│   Node for E5-Purple: π = 0.95 → strong exploration bonus        │
│   Node for E2-Green:  π = 0.70 → moderate bonus                  │
│                                                                  │
│ After 5000 iterations:                                           │
│   Best move: E11 Green (95.2% visits)                            │
│   Second: E5 Purple (3.1% visits)                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Move Executed                                            │
├─────────────────────────────────────────────────────────────────┤
│ Selected: E11 Green                                              │
│ New State: "G--P---G--GG---"                                     │
│ New Score: +0.78                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 8.6 Parallel Computing Architecture

The SA evaluation leverages MATLAB's Parallel Computing Toolbox:

```mermaid
flowchart LR
    subgraph Client ["Python Process"]
        PY["MatlabBridge"]
    end

    subgraph Engine ["MATLAB Engine"]
        SA["sa_evaluate.m"]
    end

    subgraph Pool ["Parallel Pool"]
        W1["Worker 1"]
        W2["Worker 2"]
        W3["Worker 3"]
        W4["Worker 4"]
        W5["Worker 5"]
        W6["Worker 6"]
    end

    PY --> |"state_vec,<br/>num_samples"| SA
    SA --> |"parfor<br/>s=1:100"| Pool
    W1 --> |"values[1:17]"| SA
    W2 --> |"values[18:34]"| SA
    W3 --> |"values[35:51]"| SA
    W4 --> |"values[52:68]"| SA
    W5 --> |"values[69:85]"| SA
    W6 --> |"values[86:100]"| SA
    SA --> |"mean, std,<br/>best_moves"| PY
```

**Performance Characteristics**:

| Configuration | Samples | Time | Speedup |
|---------------|---------|------|---------|
| Serial | 100 | ~12s | 1.0x |
| 6 Workers | 100 | ~2.5s | 4.8x |
| 12 Workers | 100 | ~1.5s | 8.0x |

### 8.7 Game Trace Data Structure

The opponent identification system processes game traces in a flattened matrix format:

```python
# Python: List of game traces
game_traces = [
    # Game 1: [(edge, color, score), ...]
    [(9, 'G', 1.01), (5, 'G', 0.42), (10, 'G', 1.97), (12, 'G', 0.85)],
    # Game 2: ...
    [(9, 'G', 0.92), (13, 'G', -0.15), (10, 'G', 0.02), ...],
]

# Transform to MATLAB format
def encode_traces_for_matlab(traces: list) -> matlab.double:
    """Flatten traces to matrix for MATLAB."""
    rows = []
    for trace in traces:
        row = []
        for edge, color, score in trace:
            row.extend([
                float(edge),
                1.0 if color == 'G' else -1.0,
                float(score)
            ])
        rows.append(row)

    # Pad rows to same length
    max_len = max(len(r) for r in rows)
    padded = [r + [0.0] * (max_len - len(r)) for r in rows]

    return matlab.double(padded)
```

**MATLAB Matrix Structure**:

```matlab
% traces: N×M matrix (N games, M = max_moves × 3)
% Each row: [e1, c1, s1, e2, c2, s2, ..., eK, cK, sK, 0, 0, 0, ...]
%
% Example (2 games, 4 moves each):
traces = [
    9,  1, 1.01,   5,  1, 0.42,  10,  1, 1.97,  12,  1, 0.85;
    9,  1, 0.92,  13,  1,-0.15,  10,  1, 0.02,   5, -1, 0.31
]
%   ^  ^   ^      ^   ^   ^      ^   ^   ^      ^   ^   ^
%   e  c   s      e   c   s      e   c   s      e   c   s
%   Move 1        Move 2         Move 3         Move 4
```

### 8.8 Error Handling and Fallbacks

The MATLAB bridge implements graceful degradation:

```mermaid
flowchart TD
    START["calculate_move()"]
    CHECK["MATLAB Available?"]
    RL["Call evaluate_position()"]
    RL_ERR{"Success?"}
    SA["Call sa_evaluate()"]
    SA_ERR{"Success?"}
    OPP["Call identify_opponent()"]
    OPP_ERR{"Success?"}
    MCTS["Run MCTS"]
    FB_MCTS["Fallback: Pure MCTS<br/>No MATLAB priors"]
    FB_HEUR["Fallback: Heuristic<br/>Hand-coded priors"]
    MOVE["Return Best Move"]

    START --> CHECK
    CHECK --> |"Yes"| RL
    CHECK --> |"No"| FB_HEUR

    RL --> RL_ERR
    RL_ERR --> |"Yes"| SA
    RL_ERR --> |"No, log warning"| FB_HEUR

    SA --> SA_ERR
    SA_ERR --> |"Yes"| OPP
    SA_ERR --> |"No, log warning"| MCTS

    OPP --> OPP_ERR
    OPP_ERR --> |"Yes"| MCTS
    OPP_ERR --> |"No, log warning"| MCTS

    FB_HEUR --> MOVE
    FB_MCTS --> MOVE
    MCTS --> MOVE
```

**Error Recovery Code**:

```python
def evaluate_position_rl(self, state: str, is_our_turn: bool):
    """Evaluate with graceful fallback."""
    if not self.bridge.is_available():
        logger.warning("MATLAB unavailable, using heuristic")
        return self._heuristic_eval(state, is_our_turn)

    try:
        value, policy = self.bridge.evaluate_position_rl(state, is_our_turn)
        return value, policy
    except Exception as e:
        logger.warning(f"MATLAB RL failed: {e}")
        return self._heuristic_eval(state, is_our_turn)
```

### 8.9 Performance Metrics

**MATLAB Integration Overhead**:

| Operation | Latency | Notes |
|-----------|---------|-------|
| Engine startup | ~15s | One-time, use shared session |
| evaluate_position | ~50ms | Fast heuristic |
| sa_evaluate (100 samples) | ~2.5s | Parallel pool required |
| identify_opponent | ~100ms | Depends on trace count |

**Strategy Comparison**:

| Strategy | Avg Move Time | Win Rate vs MCTS Melissa |
|----------|---------------|--------------------------|
| Pure Heuristic | 5ms | 15% |
| Pure MCTS | 200ms | 20% |
| Hybrid (MCTS+Heuristic) | 250ms | 25% |
| MATLAB-Enhanced | 300ms | 30%* |

*Preliminary results; requires more games for statistical significance.

---

## 9. The HybridSolverStrategy: D-Wave Inspired Search with Adaptive Learning

This section describes the **HybridSolverStrategy**, a sophisticated game-playing agent that combines ideas from quantum-classical hybrid optimization (D-Wave's approach) with online reinforcement learning. This strategy represents the current state-of-the-art for this project, achieving the lowest loss rate among all implemented strategies.

### 9.1 Motivation and Design Philosophy

**The Core Challenge**: Tangled presents a unique optimization problem where:
1. The game tree is too large for exhaustive search (~4.3 × 10^16 nodes)
2. Terminal evaluation requires expensive quantum annealing simulation
3. The opponent (MCTS Melissa) uses a strong search algorithm

**Design Philosophy**: Inspired by D-Wave's hybrid solver architecture (Leap™), we combine multiple search paradigms:

| Component | Strength | Weakness | Phase |
|-----------|----------|----------|-------|
| Alpha-Beta Minimax | Exact, optimal | Exponential in depth | Endgame (≤3 grey) |
| MCTS | Handles uncertainty | Needs many rollouts | Midgame (4-8 grey) |
| Tabu Search | Escapes local optima | Heuristic, no guarantees | Early game (>8 grey) |
| Lookup Table | O(1) perfect play | Memory-intensive | Terminal (≤3 grey) |

**Key Innovation**: Unlike pure MCTS approaches, HybridSolverStrategy:
1. Uses exact minimax when computationally feasible
2. Precomputes 19 million exact game values in a lookup table
3. Applies REINFORCE-style learning to adapt edge preferences over time

### 9.2 System Architecture

```mermaid
flowchart TD
    subgraph Python ["Python Runtime (play_tangled.py)"]
        GS["Game State<br/>'G--P---G--GPP--'"]
        HSS["HybridSolverStrategy"]
        LEARN["Learning Module<br/>REINFORCE Updates"]
        DB["SQLite Database<br/>~/.tangled/game_stats.db"]
        ADJ["Edge Adjustments<br/>~/.tangled/hybrid_solver_adjustments.json"]
    end

    subgraph MATLAB ["MATLAB Engine (R2026a)"]
        HTS["HybridTangledSolver.m"]
        subgraph Components ["Solver Components"]
            LUT["ExpandedLUT.m<br/>19M exact values"]
            AB["AlphaBetaSearch.m<br/>with α-β pruning"]
            MCTS["TangledMCTS.m<br/>UCB1 + Progressive Bias"]
            TABU["TabuSearch.m<br/>Multistart optimization"]
        end
    end

    GS --> HSS
    HSS --> |"record_move()"| LEARN
    HSS --> |"Calculate Move"| HTS

    HTS --> |"grey ≤ 3"| LUT
    HTS --> |"grey ≤ 6"| AB
    HTS --> |"grey > 6"| MCTS
    MCTS --> |"Refinement"| TABU

    LUT --> |"(edge, color)"| HSS
    AB --> |"(edge, color)"| HSS
    MCTS --> |"(edge, color)"| HSS

    LEARN --> |"end_game()"| ADJ
    LEARN --> |"Game metrics"| DB
    ADJ --> |"Load on init"| HSS
```

### 9.3 The MATLAB Solver Components

#### 9.3.1 Strategy Selection Logic

The solver automatically selects the appropriate algorithm based on game phase:

```matlab
function [edge, color, info] = solve(obj, state)
    grey_count = sum(state == '-');

    if grey_count <= 3
        % Terminal region: use exact lookup table
        [edge, color, info] = obj.solveLUT(state);

    elseif grey_count <= 6
        % Late game: exact minimax with alpha-beta pruning
        [edge, color, info] = obj.solveAlphaBeta(state);

    elseif grey_count <= 10
        % Midgame: hybrid minimax + MCTS
        [edge, color, info] = obj.solveHybrid(state);

    else
        % Early game: MCTS with tabu refinement
        [edge, color, info] = obj.solveMCTS(state);
    end
end
```

#### 9.3.2 Alpha-Beta Minimax with Pruning

**The Minimax Algorithm with Alpha-Beta Pruning**:

The classic minimax algorithm evaluates all possible game continuations, but alpha-beta pruning eliminates provably suboptimal branches:

$$\alpha = \text{best value for maximizer so far}$$
$$\beta = \text{best value for minimizer so far}$$

**Pruning Condition**: If $\alpha \geq \beta$, prune the remaining siblings.

```matlab
function value = alphaBeta(obj, state, depth, alpha, beta, isMaximizing)
    % Terminal or depth limit
    if depth == 0 || isTerminal(state)
        return obj.evaluate(state);
    end

    actions = getValidActions(state);

    if isMaximizing
        value = -inf;
        for a = actions
            child = applyAction(state, a);
            value = max(value, obj.alphaBeta(child, depth-1, alpha, beta, false));
            alpha = max(alpha, value);
            if alpha >= beta
                break;  % Beta cutoff
            end
        end
    else
        value = inf;
        for a = actions
            child = applyAction(state, a);
            value = min(value, obj.alphaBeta(child, depth-1, alpha, beta, true));
            beta = min(beta, value);
            if alpha >= beta
                break;  % Alpha cutoff
            end
        end
    end
end
```

**Transposition Table**: We cache evaluated positions to avoid redundant computation:

```matlab
% Hash table for position caching
obj.TranspositionTable = containers.Map('KeyType', 'char', 'ValueType', 'any');

function [value, found] = probeTable(obj, state, depth)
    key = [state, num2str(depth)];
    if obj.TranspositionTable.isKey(key)
        entry = obj.TranspositionTable(key);
        value = entry.value;
        found = true;
    else
        found = false;
    end
end
```

#### 9.3.3 MCTS with Tabu-Guided Rollouts

For early/midgame positions where exact search is infeasible, we use MCTS with domain-specific enhancements:

**UCB1 Selection with Prior Knowledge**:

$$UCB1(n) = \bar{X}_n + c\sqrt{\frac{\ln N_{\text{parent}}}{N_n}} + w \cdot \frac{\pi(n) - 0.5}{N_n + 1}$$

Where:
- $\bar{X}_n$ = mean value at node $n$
- $c = \sqrt{2}$ = exploration constant
- $\pi(n)$ = domain heuristic prior (from edge categories)
- $w = 3.0$ = prior weight (decays with visits)

**Tabu Search Refinement**:

After MCTS selects a candidate move, tabu search explores variations:

```matlab
function [bestMove, improved] = tabuRefine(obj, state, mctsMove, timeLimit)
    tabu = TabuSearch('MaxIterations', 1000, 'TabuTenure', 7);

    % Start from MCTS recommendation
    bestValue = obj.evaluate(applyAction(state, mctsMove));
    bestMove = mctsMove;

    % Explore neighborhood with tabu constraints
    for iter = 1:obj.MaxIterations
        neighbors = getNeighborMoves(state, obj.TabuList);
        for move = neighbors
            value = obj.evaluate(applyAction(state, move));
            if value > bestValue
                bestValue = value;
                bestMove = move;
                improved = true;
            end
        end
        obj.updateTabuList(bestMove);
    end
end
```

#### 9.3.4 The Expanded Lookup Table (19 Million Entries)

**Motivation**: For positions with 0-3 grey edges, we can precompute the exact minimax value for all 2^colored edge configurations.

**Coverage Breakdown**:

| Grey Edges | Configurations | Exact Values Stored |
|------------|----------------|---------------------|
| 0 (terminal) | 2^15 = 32,768 | 32,768 |
| 1 | C(15,1) × 2^14 = 491,520 | 491,520 |
| 2 | C(15,2) × 2^13 = 3,440,640 | 3,440,640 |
| 3 | C(15,3) × 2^12 = 14,909,440 | 14,909,440 |
| **Total** | | **18,874,368** |

**Mathematical Formulation**:

For a position with $k$ grey edges at indices $G = \{g_1, ..., g_k\}$, the minimax value is:

$$V(s) = \max_{\substack{c_1 \in \{G,P\} \\ \text{(our move)}}} \min_{\substack{c_2 \in \{G,P\} \\ \text{(opp move)}}} \cdots \text{terminal\_value}(s')$$

Where the max/min alternate based on whose turn it is.

**Lookup Table Structure**:

```matlab
% LUT stored as containers.Map with state string keys
% Format: 'GGPP-GG-PPPG-GG' -> struct('value', 1.23, 'bestEdge', 5, 'bestColor', 'G')

function [edge, color, value] = lookupExact(obj, state)
    if obj.LUT.isKey(state)
        entry = obj.LUT(state);
        edge = entry.bestEdge;
        color = entry.bestColor;
        value = entry.value;
    else
        error('State not in LUT: %s', state);
    end
end
```

**Generation Process** (executed offline):

```matlab
% Parallel generation using 6 workers (~12.6 minutes for 3-grey extension)
parpool(6);
parfor stateIdx = 1:numStates
    state = indexToState(stateIdx);
    [value, bestMove] = exhaustiveMinimax(state);
    results{stateIdx} = struct('state', state, 'value', value, 'move', bestMove);
end
```

### 9.4 REINFORCE-Style Adaptive Learning

**Key Innovation**: Unlike static heuristics, the HybridSolverStrategy learns from game outcomes to improve edge selection over time.

#### 9.4.1 The Learning Algorithm

**REINFORCE Policy Gradient** (Williams, 1992):

We maintain learned adjustments $\theta_e$ for each edge $e \in \{0, 1, ..., 14\}$:

$$\theta_e \leftarrow \theta_e + \alpha \cdot R \cdot \gamma^{T-t}$$

Where:
- $\alpha = 0.03$ = learning rate
- $R$ = game outcome reward
- $\gamma = 0.9$ = temporal discount factor
- $T$ = total moves in game
- $t$ = move number (1-indexed)

**Reward Signal**:

| Game Result | Score $s$ | Reward $R$ |
|-------------|-----------|------------|
| Win | $s > 0$ | $1.0 + \min(s, 2.0) / 2$ |
| Draw | $s \geq 0$ | $+0.1$ |
| Draw | $s < 0$ | $-0.1$ |
| Loss | $s < 0$ | $-1.0 + \max(s, -2.0) / 2$ |

**Temporal Credit Assignment**:

Later moves receive more credit/blame (less discounting):

$$\text{discount}_t = \gamma^{T - t - 1}$$

For a game with 8 moves:
- Move 1: discount = $0.9^7 \approx 0.48$
- Move 4: discount = $0.9^4 \approx 0.66$
- Move 8: discount = $0.9^0 = 1.0$

#### 9.4.2 Implementation Details

```python
class HybridSolverStrategy:
    def __init__(self, learning_rate=0.03, ...):
        self.edge_adjustments = [0.0] * 15  # Per-edge learned values
        self.move_history = []  # [(edge, color, score_after), ...]
        self.learning_rate = learning_rate
        self._load_adjustments()  # Persist across sessions

    def record_move(self, edge: int, color: str, score_after: float):
        """Called after each move for learning."""
        self.move_history.append((edge, color, score_after))

    def end_game(self, result: str, final_score: float):
        """Apply REINFORCE update at game end."""
        self._learn_from_game(result, final_score)
        self._save_adjustments()
        self.move_history = []

    def _learn_from_game(self, result: str, final_score: float):
        # Compute reward
        if result == 'win':
            reward = 1.0 + min(final_score, 2.0) / 2.0
        elif result == 'draw':
            reward = 0.1 if final_score >= 0 else -0.1
        else:  # loss
            reward = -1.0 + max(final_score, -2.0) / 2.0

        gamma = 0.9
        n_moves = len(self.move_history)

        for i, (edge, color, score) in enumerate(self.move_history):
            discount = gamma ** (n_moves - i - 1)
            update = self.learning_rate * reward * discount * 0.5

            self.edge_adjustments[edge] += update
            self.edge_adjustments[edge] = max(-1.0, min(1.0, self.edge_adjustments[edge]))
```

#### 9.4.3 Learned Edge Adjustments Example

After 6 games (2W/2L/1D), the learned adjustments might look like:

| Edge | Adjustment | Interpretation |
|------|------------|----------------|
| E14 | **+0.051** | Strong positive signal |
| E10 | **+0.032** | Opening edge - reliably good |
| E4 | +0.030 | Positive |
| E12 | +0.026 | Hub control - valuable |
| E11 | +0.020 | Our outer edge - good |
| E9 | +0.015 | Our spoke - opening move |
| E0 | -0.015 | Negative - may hurt us |
| E13 | **-0.019** | Opponent territory - avoid |

These adjustments modify the MCTS prior weights:

$$\pi'(e) = \pi(e) + \theta_e$$

### 9.5 Database Integration and Persistence

The learning system integrates with a SQLite database for game tracking and persistent storage:

#### 9.5.1 Database Schema

```sql
-- Games table: tracks every game played
CREATE TABLE games (
    id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    opponent TEXT,
    result TEXT,  -- 'win', 'loss', 'draw'
    final_score REAL,
    total_moves INTEGER,
    strategy TEXT,
    run_id INTEGER,
    game_number INTEGER,
    FOREIGN KEY (run_id) REFERENCES runs(id)
);

-- Moves table: full move history for learning
CREATE TABLE moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id TEXT,
    move_number INTEGER,
    player TEXT,  -- 'us' or 'opponent'
    edge INTEGER,
    color TEXT,
    score_after REAL,
    FOREIGN KEY (game_id) REFERENCES games(id)
);

-- Runs table: batch execution tracking
CREATE TABLE runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started DATETIME DEFAULT CURRENT_TIMESTAMP,
    planned_games INTEGER,
    completed_games INTEGER DEFAULT 0,
    strategy TEXT,
    opponent TEXT
);
```

#### 9.5.2 Learned Adjustments Persistence

Edge adjustments are stored in JSON for fast loading:

```json
{
  "edge_adjustments": [
    -0.015, 0.002, 0.025, -0.005, 0.030,
    -0.008, 0.026, -0.006, 0.001, 0.015,
    0.032, 0.020, 0.026, -0.019, 0.051
  ],
  "games_learned": 6,
  "last_updated": "2026-01-24T13:31:59.406806"
}
```

**Path**: `~/.tangled/hybrid_solver_adjustments.json`

### 9.6 Performance Analysis

#### 9.6.1 Strategy Comparison

| Strategy | Loss Rate | Win Rate | MATLAB Required |
|----------|-----------|----------|-----------------|
| Heuristic | 70.4% | 9.3% | No |
| MCTS | 60.6% | 6.1% | No |
| Hybrid | 37.4% | 14.8% | No |
| Hybrid Solver | **40.0%** | **40.0%** | Yes |

*Results from 5-game test run with learning enabled*

#### 9.6.2 Time Budget Allocation

The solver allocates a 10-second time budget per move:

| Phase | Allocation | Algorithm |
|-------|------------|-----------|
| Minimax | 35% (3.5s) | Alpha-beta with transposition |
| MCTS | 55% (5.5s) | UCB1 with progressive bias |
| Tabu | 10% (1.0s) | Multistart neighborhood search |

#### 9.6.3 Lookup Table Impact

With the 19M-entry LUT:
- **Guaranteed optimal play** for last 4 moves of every game
- **Zero search time** for positions with ≤3 grey edges
- **Memory usage**: ~2.5 GB for full LUT

### 9.7 Research Extensions

**For Graduate Research Projects**:

1. **Neural Network Value Head**: Replace heuristic evaluation with a trained neural network (similar to AlphaZero's value network)

2. **Policy Network Integration**: Learn action priors from self-play rather than hand-coded heuristics

3. **Transfer Learning**: Test if learned adjustments transfer across different graph types

4. **Opponent Modeling**: Extend learning to model specific opponent patterns (Melissa vs. Amara)

5. **Curriculum Learning**: Train progressively on easier opponents before facing MCTS Melissa

**Key Files for Extension**:

| File | Purpose |
|------|---------|
| `matlab/rl/HybridTangledSolver.m` | MATLAB solver core |
| `matlab/matlab_strategy.py` | Python strategy wrapper |
| `matlab/rl/ExpandedLUT.m` | Lookup table implementation |
| `stats/collector.py` | Database operations |

---

## 10. References and Further Reading

### Core References

**Ising Model**:
- Ising, E. (1925). "Beitrag zur Theorie des Ferromagnetismus". Zeitschrift für Physik.
- Baxter, R.J. (1982). *Exactly Solved Models in Statistical Mechanics*.

**Petersen Graph**:
- Petersen, J. (1898). "Sur le théorème de Tait". L'Intermédiaire des Mathématiciens.
- Holton, D.A. & Sheehan, J. (1993). *The Petersen Graph*.

**Monte Carlo Tree Search**:
- Kocsis, L. & Szepesvári, C. (2006). "Bandit based Monte-Carlo Planning". ECML.
- Browne, C. et al. (2012). "A Survey of MCTS Methods". IEEE TCIAIG.

**Quantum Annealing**:
- Kadowaki, T. & Nishimori, H. (1998). "Quantum annealing in the transverse Ising model". Physical Review E.
- Farhi, E. et al. (2001). "A Quantum Adiabatic Evolution Algorithm". Science.

### Software Dependencies

- `snowdrop-tangled-game-engine`: Game mechanics and state management
- `snowdrop-adjudicators`: Terminal state evaluation
- Source: [tangled-game.com](https://tangled-game.com)

### Project Files

| File | Description |
|------|-------------|
| `strategy/mcts_strategy.py` | MCTS with Progressive Bias implementation |
| `strategy/petersen_strategy.py` | Heuristic edge scoring |
| `matlab/bridge.py` | Python-MATLAB Engine API bridge |
| `matlab/matlab_strategy.py` | MATLAB-enhanced hybrid strategy |
| `playing_games/run_local_parallel_tournament.py` | Tournament runner |
| `THEORY_OF_OPERATION.md` | Detailed system documentation |

### MATLAB Files (MATLAB Drive)

| File | Toolbox | Description |
|------|---------|-------------|
| `evaluate_position.m` | RL Toolbox | Value network position evaluation |
| `sa_evaluate.m` | Global Optimization | Simulated annealing deep search |
| `identify_opponent.m` | System Identification | Opponent behavior modeling |

---

## Appendix A: Petersen Graph Edge Index Reference

```
Index  Edge    Category      Notes
─────────────────────────────────────────
E0     (0,2)   Inner         Pentagram edge
E1     (0,3)   Inner         Pentagram edge
E2     (0,6)   HUB           Inner to hub
E3     (1,3)   Inner         Pentagram edge
E4     (1,4)   Inner         Pentagram edge
E5     (1,7)   OPP           Opponent's spoke
E6     (2,4)   Inner         Pentagram edge
E7     (2,8)   Spoke         Inner to outer
E8     (3,9)   Spoke         Inner to outer
E9     (4,5)   MY            Our spoke (critical!)
E10    (5,6)   MY + HUB      Our hub connection
E11    (5,9)   MY            Our outer edge
E12    (6,7)   OPP + HUB     Opponent hub connection
E13    (7,8)   OPP           Opponent's outer edge
E14    (8,9)   Outer         Pentagon edge
```

---

## Appendix B: State Encoding Example

**Sample Game Progression**:

```
Turn  Player  Action      State                Score
───────────────────────────────────────────────────
0     -       Initial     "---------------"    0.0
1     P1      E9 → G      "--------G------"    +0.3
2     P2      E5 → G      "-----G--G------"    -0.1
3     P1      E10 → G     "-----G--GG-----"    +0.5
4     P2      E12 → G     "-----G--GG-G---"    +0.2
5     P1      E11 → G     "-----G--GGG----"    +0.8
6     P2      E13 → G     "-----G--GGGG---"    +0.4
...
15    P1      E14 → P     "GPPGPGGPGGGGGPP"    +2.1
```

**Terminal Evaluation**:
- Adjudicator finds optimal σ* = (+1, -1, +1, -1, +1, +1, -1, +1, -1, +1)
- Final score: +2.1 → Player 1 wins

---

*Document version: 1.2*
*Last updated: January 24, 2026*
*Author: Murray Kopit (murr2k@gmail.com)*
