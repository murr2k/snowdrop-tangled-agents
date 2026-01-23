# MCTS Depth Enhancement Research

## Problem Statement

Our MATLAB MCTS implementation achieves **depth 5** with 5000 iterations against a branching factor of **24 actions** (12 edges × 2 colors). At move 3-4, critical score collapses occur that the shallow search fails to predict.

| Current State | Value |
|---------------|-------|
| Iterations | 5000 |
| Time limit | 20s |
| Achieved depth | 5 |
| Branching factor | 24 |
| Tree size at depth 5 | 24^5 = 7.9M paths |

**Goal**: Increase effective search depth from 5 to 8-10 to better predict mid-game score dynamics.

---

## Technique Index

### Category A: Branching Factor Control

#### A1. Progressive Widening (PW)

**Core Idea**: Limit children to k = C × N^α where N = visit count, gradually expanding as confidence grows.

**Formula**:
```
k(N) = ceil(C × N^α)
```
- Typical values: C = 2, α = 0.5
- At N=1: k=2 children
- At N=4: k=4 children
- At N=25: k=10 children

**Expected Impact**: Forces depth over breadth; should increase depth from 5 to 8-10.

**Complexity**: Low - modify expansion logic in MCTSNode

**Key Papers**:
- [Couëtoux et al. - Continuous RAVE](http://proceedings.mlr.press/v20/couetoux11/couetoux11.pdf)
- [Chaslot et al. - pMCTS](https://dke.maastrichtuniversity.nl/m.winands/documents/pMCTS.pdf)

**Status**: [ ] Not started

**Notes**:
- Need to decide expansion order (random vs. policy-guided)
- Consider combining with Progressive Bias for expansion ordering

---

#### A2. Progressive Unpruning

**Core Idea**: Start with limited move set, gradually reintroduce pruned moves as evidence accumulates.

**Mechanism**:
1. Initially consider only top-k moves by prior
2. As visits increase, "unprune" additional moves
3. Prevents early commitment to suboptimal paths

**Expected Impact**: Complementary to PW; prevents missing good moves that have weak priors.

**Complexity**: Low

**Key Papers**:
- [Chaslot et al. - pMCTS](https://dke.maastrichtuniversity.nl/m.winands/documents/pMCTS.pdf)

**Status**: [ ] Not started

**Notes**:
- Requires good move ordering heuristic
- Our LUT can provide terminal-aware move ordering

---

#### A3. Beam Search MCTS (BMCTS)

**Core Idea**: At each depth level, keep only top-W nodes by value; permanently prune rest.

**Parameters**:
- W: beam width (nodes to keep per level)
- d: maximum depth

**Expected Impact**: Guarantees minimum depth; trades breadth for depth deterministically.

**Complexity**: Low

**Key Papers**:
- Baier & Winands 2012 (referenced in [Springer Review](https://link.springer.com/article/10.1007/s10462-022-10228-y))

**Status**: [ ] Not started

**Notes**:
- Risk: may prune optimal path if early estimates wrong
- Best combined with good value estimates (our LUT helps)

---

#### A4. Heuristic Move Pruning

**Core Idea**: Use domain knowledge to filter obviously bad moves before expansion.

**Our Domain Knowledge**:
- MY_EDGES (E9, E10, E11): Green strongly preferred
- OPP_EDGES (E5, E12, E13): Purple strongly preferred
- HUB_EDGES (E2, E10, E12): Context-dependent

**Expected Impact**: Reduces effective branching from 24 to ~12-16.

**Complexity**: Low (we already have priors)

**Key Papers**:
- [Low 2014 - Heuristic Move Pruning](http://orangehelicopter.com/academic/papers/cig2014-low-heuristics.pdf)

**Status**: [ ] Not started

**Notes**:
- Already have `computeRolloutPrior()` - could use for hard pruning
- Risk: pruning optimal move if prior is wrong

---

### Category B: Value Estimation Acceleration

#### B1. RAVE (Rapid Action Value Estimation)

**Core Idea**: Share move statistics across tree branches using AMAF (All-Moves-As-First) heuristic.

**Assumption**: A move's value is similar regardless of when it's played in a sequence.

**Formula** (UCB-RAVE):
```
UCB_RAVE(s,a) = (1-β) × Q(s,a) + β × Q_AMAF(s,a) + C × sqrt(ln(N(s))/N(s,a))

β = sqrt(k / (3×N(s) + k))  where k ≈ 300-700
```

**Expected Impact**: 50-60% win rate vs 24% without RAVE (Go experiments).

**Complexity**: Medium - need AMAF statistics tracking

**Key Papers**:
- [Gelly & Silver - MC-RAVE](https://www.cs.utexas.edu/~pstone/Courses/394Rspring13/resources/mcrave.pdf)
- [Maastricht RAVE Comparison](https://dke.maastrichtuniversity.nl/m.winands/documents/CIG2016_RAVE.pdf)

**Status**: [ ] Not started

**Notes**:
- RAVE works well in Go because moves have persistent value
- In Tangled, edge colors are permanent - RAVE assumption may hold well
- Need to track AMAF stats per action across all nodes

---

#### B2. GRAVE (Generalized RAVE)

**Core Idea**: Use ancestor node's AMAF statistics when local statistics insufficient.

**Mechanism**:
- If node has < threshold visits, use parent's AMAF stats
- Propagates knowledge down the tree faster

**Expected Impact**: Faster convergence than standard RAVE.

**Complexity**: Medium

**Key Papers**:
- [Cazenave - GRAVE](https://hal.science/hal-01436522/document)

**Status**: [ ] Not started

**Notes**:
- Extension of RAVE; implement RAVE first

---

#### B3. Progressive Bias

**Core Idea**: Add heuristic bias term to UCB that decays with visit count.

**Formula**:
```
UCB_PB(s,a) = Q(s,a) + C × sqrt(ln(N(s))/N(s,a)) + H(s,a) / (N(s,a) + 1)
```
Where H(s,a) is heuristic value for action a in state s.

**Expected Impact**: Guides early exploration toward good moves; effect diminishes as tree matures.

**Complexity**: Low - we already have priors

**Key Papers**:
- [Chaslot et al. - pMCTS](https://dke.maastrichtuniversity.nl/m.winands/documents/pMCTS.pdf)

**Status**: [~] Partially implemented (priors exist, not in UCB formula)

**Notes**:
- Our `PriorWeight` parameter exists but may not be used optimally
- Could use LUT-based lookahead for better H(s,a)

---

### Category C: Neural Network Guided

#### C1. PUCT (AlphaZero Style)

**Core Idea**: Neural network provides policy prior P(s,a) and value estimate V(s).

**Formula**:
```
PUCT(s,a) = Q(s,a) + c_puct × P(s,a) × sqrt(N(s)) / (1 + N(s,a))
```

**Expected Impact**: State-of-the-art in many games.

**Complexity**: High - requires training neural network

**Key Papers**:
- [Silver et al. - AlphaZero](https://joshvarty.github.io/AlphaZero/)
- [OpenSpiel AlphaZero](https://openspiel.readthedocs.io/en/stable/alpha_zero.html)

**Status**: [ ] Not started

**Notes**:
- Would need to train policy/value network on self-play data
- Our existing RL training infrastructure could be adapted
- High effort but potentially highest reward

---

#### C2. Value Network Replacing Rollouts

**Core Idea**: Instead of random rollouts, use neural network to estimate leaf value directly.

**Expected Impact**: Faster iterations, more accurate leaf evaluation.

**Complexity**: High

**Status**: [ ] Not started

**Notes**:
- Our LUT already provides perfect terminal evaluation
- Could train NN for non-terminal state evaluation

---

### Category D: Hybrid Approaches

#### D1. MCTS-Minimax Hybrid

**Core Idea**: Use minimax with αβ pruning at shallow depths, MCTS at deeper levels.

**Mechanism**:
- Depth 0-3: Minimax with αβ (complete search)
- Depth 4+: MCTS (sampling-based)

**Expected Impact**: Guaranteed accuracy at critical early depths.

**Complexity**: Medium

**Key Papers**:
- [Truong 2023 - MCTS-Minimax Hybrid](https://ml-research.github.io/papers/truong2023monte.pdf)

**Status**: [ ] Not started

**Notes**:
- At depth 3, 24^3 = 13,824 nodes - feasible for complete search
- Would guarantee no blunders in first 3 plies

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Implementation effort | 25% | Time to implement and test |
| Expected depth gain | 30% | Increase in effective search depth |
| Risk of missing good moves | 20% | Does pruning risk optimal play? |
| Compatibility | 15% | Works with existing LUT evaluation |
| Parallelization | 10% | Works with parallel rollouts |

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
- [ ] **A1 - Progressive Widening**: Highest impact, low effort
- [ ] **B3 - Progressive Bias**: Already have priors, just integrate into UCB

### Phase 2: Medium Effort (3-5 days)
- [ ] **B1 - RAVE**: Proven technique, moderate implementation
- [ ] **A4 - Heuristic Pruning**: Use existing priors for hard cutoffs

### Phase 3: Advanced (1-2 weeks)
- [ ] **D1 - MCTS-Minimax Hybrid**: Guarantee shallow depth accuracy
- [ ] **C1 - PUCT**: Train policy network

---

## Experiment Log

| Date | Technique | Config | Depth | Win Rate | Notes |
|------|-----------|--------|-------|----------|-------|
| 2026-01-22 | Baseline | 5000 iter, 20s | 5 | ~10% | vs Melissa |
| | | | | | |

---

## Current Implementation Analysis

### MCTSNode.m Structure

```
MCTSNode
├── State (15-char string)
├── IsOurTurn (bool)
├── Parent/Children (tree structure)
├── Action {edge, color}
├── Prior (0-1 heuristic value)
├── Visits, TotalValue (statistics)
├── UntriedActions (priority queue)
└── ActionPriors (cached priors)
```

### Current Expansion Logic (MCTSNode.m:212-242)

```matlab
function child = expand(this)
    % Pop first action (highest priority) - SORTED BY PRIOR
    action = this.UntriedActions{1};
    this.UntriedActions(1) = [];
    ...
    child = MCTSNode(newState, ~this.IsOurTurn, this, action, prior);
    this.Children(key) = child;
end
```

**Key Observation**: Expansion is **priority-ordered** but **not limited**. All 24 actions will eventually be expanded before going deeper. This is our bottleneck.

### Current UCB1 Formula (MCTSNode.m:163-190)

```matlab
exploitation = this.TotalValue / double(this.Visits);
explorationTerm = exploration * sqrt(log(double(this.Parent.Visits)) / double(this.Visits));
priorBonus = priorWeight * (this.Prior - 0.5) / (double(this.Visits) + 1);

value = exploitation + explorationTerm + priorBonus;  % (negated for opponent)
```

**Current Features**:
- [x] UCB1 exploration/exploitation
- [x] Progressive Bias (prior bonus decays with visits)
- [ ] Progressive Widening (not implemented)
- [ ] RAVE/AMAF statistics (not tracked)

### isFullyExpanded() Check (MCTSNode.m:158-161)

```matlab
function tf = isFullyExpanded(this)
    tf = isempty(this.UntriedActions);  % TRUE when all 24 tried
end
```

**This is the key modification point for Progressive Widening.**

### Existing Priors (MCTSNode.m:98-151)

Strong priors already defined:
| Edge Type | Our Turn | Opponent Turn |
|-----------|----------|---------------|
| MY_EDGES Green | 0.99 | 0.15 |
| MY_EDGES Purple | 0.01 | 0.85 |
| OPP_EDGES Purple | 0.95 | 0.05 |
| OPP_EDGES Green | 0.05 | 0.95 |
| HUB_EDGES | 0.70 G / 0.30 P | 0.65 G / 0.35 P |

These priors can guide Progressive Widening expansion order.

---

## Key Reference Files

### Our Implementation
- `matlab/rl/TangledMCTS.m:229-263` - Main search loop
- `matlab/rl/MCTSNode.m:158-161` - `isFullyExpanded()` - **modify for PW**
- `matlab/rl/MCTSNode.m:212-242` - `expand()` - **modify for PW**
- `matlab/rl/MCRollout.m` - Rollout engine with LUT

### Research Papers (Local Cache)
- TODO: Download key PDFs to `docs/papers/`

---

---

## MATLAB-Specific Literature & Resources

### Existing MATLAB MCTS Implementations

#### 1. MCTS for Behavior Planning (Autonomous Driving)
**Repository**: [zhongshun/MCTS_for_Behavior_Planning](https://github.com/zhongshun/MCTS_for_Behavior_Planning)
**Paper**: "Monte-Carlo Tree Search for Behavior Planning in Autonomous Driving" (IEEE SSRR 2024)
**Requirements**: MATLAB 2023a+, Automated Driving Toolbox 3.7+

**Key Implementation Details**:
- Entry point: `mctsPlanning.m`
- Uses UCB selection: `UCB(vi) = -C(v')/n(v') + const√(2ln N/n(v'))`
- Cost-minimization framing (not reward maximization)
- Rollout uses only longitudinal actions (simplified)
- Achieved 64.33% near-optimal solutions at 3000 iterations

**Relevance**: This is the closest existing MATLAB MCTS to our use case. Worth studying their tree structure and UCB implementation.

#### 2. Tinevez matlab-tree Library
**Repository**: [tinevez/matlab-tree](http://tinevez.github.io/matlab-tree/)
**File Exchange**: [Tree data structure as a MATLAB class](https://www.mathworks.com/matlabcentral/fileexchange/35623-tree-data-structure-as-a-matlab-class)

**Design Philosophy**:
- Simple, lightweight - no Node class, plain MATLAB arrays/cell arrays
- **Value class** (not handle) - copies on assignment
- "Small memory overhead, by construction"
- "Not designed for speed performance" - recursion and array copying

**Relevance**: Our MCTSNode uses handle class, which is different. Their array-based approach may be faster for large trees.

#### 3. UCB/Bandit Implementations
**Resource**: [Bandit Algorithms Package](https://www.math.univ-toulouse.fr/~agarivie/Telecom/bandits/) (Python & MATLAB)
**File Exchange**: [CMA_MOMAB](https://www.mathworks.com/matlabcentral/fileexchange/69867-cma_momab) - Multi-objective UCB

---

### MATLAB Performance Considerations

#### containers.Map Performance
**Problem**: containers.Map lookups are often the bottleneck in MATLAB code.

**Solutions**:
| Approach | Performance | MATLAB Version |
|----------|-------------|----------------|
| `containers.Map` | Baseline (slow) | All |
| `dictionary` | **2-10x faster** | R2022b+ |
| Struct arrays | Fast for scalar data | All |
| Java HashMap | Similar to containers.Map | All |

**Recommendation**: If using R2022b+, switch from `containers.Map` to `dictionary` for child node storage.

**Source**: [MathWorks Answers](https://www.mathworks.com/matlabcentral/answers/354799-faster-alternative-to-containers-map)

#### Handle Class Memory Issues
**Problem**: Large arrays of handle objects have significant overhead.

| Scenario | Memory Usage |
|----------|--------------|
| 1.8M handle objects initialized | 1.3 GB |
| All 19 properties reassigned | 7.3 GB |
| Handle "pointer" shows as | 8 bytes (misleading) |

**Source**: [MathWorks Answers](https://www.mathworks.com/matlabcentral/answers/262904-initializing-an-array-of-handle-objects-memory-usage-inefficient)

**Implications for MCTS**:
- Our MCTSNode is a handle class
- With 5000+ nodes, memory could become significant
- Consider struct-based alternative for large trees

#### Parallel Computing Limitations
**Critical**: `parfor` does not work nested.

```matlab
parfor i = 1:N           % Outer parallel loop
    parfor j = 1:M       % RUNS SERIAL, not parallel!
        ...
    end
end
```

**Implications**:
- Cannot parallelize both tree traversal AND rollouts
- Must choose: parallel at tree level OR parallel rollouts
- Current implementation: parallel rollouts only

**Source**: [MathWorks Documentation](https://www.mathworks.com/help/parallel-computing/quick-start-parallel-computing-in-matlab.html)

#### OOP Performance Overhead
**Problem**: "There is a performance penalty for using objects in MATLAB."

**Specific Issues**:
- Repeated property access is slow
- Method calls have overhead vs. functions
- Array operations on objects are slower than structs

**Source**: [MathWorks File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/41349-performance-in-object-oriented-matlab-code)

---

### Cross-Reference: Techniques vs. MATLAB Support

| Technique | MATLAB Implementation Exists? | Notes |
|-----------|------------------------------|-------|
| **A1 Progressive Widening** | No | Must implement from scratch |
| **A2 Progressive Unpruning** | No | Must implement |
| **A3 Beam Search** | No | Straightforward to implement |
| **A4 Heuristic Pruning** | Partial | Already have priors |
| **B1 RAVE** | No | Complex, needs AMAF tracking |
| **B2 GRAVE** | No | Extension of RAVE |
| **B3 Progressive Bias** | **Yes** | Already in MCTSNode.ucb1Value() |
| **C1 PUCT** | No | Need neural network |
| **D1 MCTS-Minimax** | Partial | Minimax easy, integration complex |

### Key Academic References (with MATLAB relevance)

| Paper | Year | Key Contribution | MATLAB? |
|-------|------|------------------|---------|
| [MCTS for Behavior Planning](https://arxiv.org/abs/2310.12075) | 2024 | Full MATLAB impl for autonomous driving | **Yes** |
| [Progressive Strategies (pMCTS)](https://dke.maastrichtuniversity.nl/m.winands/documents/pMCTS.pdf) | 2008 | Original PW/Unpruning | No |
| [Continuous RAVE](http://proceedings.mlr.press/v20/couetoux11/couetoux11.pdf) | 2011 | PW + RAVE combination | No |
| [Extensions of MCTS to Continuous Spaces](https://aaltodoc.aalto.fi/bitstream/handle/123456789/115245/master_Koverola_Risto_2022.pdf) | 2022 | "PW is best general-purpose extension" | No |
| [BOKR-MCTS](https://www.sciencedirect.com/science/article/pii/S2405896325020105) | 2025 | Bayesian-optimized PW | No |
| [Double Progressive Widening](http://juliapomdp.github.io/MCTS.jl/latest/dpw/) | - | Julia impl, well-documented | No (Julia) |

---

## Discussion Notes

### 2026-01-22: Initial Research

**Findings**:
1. Progressive Widening is the most commonly recommended solution for high branching factor
2. RAVE has strong empirical results in Go (50-60% vs 24% win rate)
3. Our LUT provides a unique advantage - perfect terminal evaluation enables better:
   - Move ordering for Progressive Unpruning
   - Heuristic values for Progressive Bias
   - Leaf evaluation without rollouts

**Open Questions**:
1. Does RAVE assumption hold for Tangled? (move value consistent across positions)
2. What's the optimal Progressive Widening formula for 24-action space?
3. Should we combine multiple techniques?

### 2026-01-22: MATLAB Literature Review

**Key Finding**: Only ONE existing MATLAB MCTS implementation found (zhongshun/MCTS_for_Behavior_Planning).

**MATLAB-Specific Recommendations**:

1. **Data Structure Choice**:
   - Current: `containers.Map` for children - consider switching to `dictionary` (R2022b+)
   - Current: Handle class nodes - acceptable but watch memory with large trees
   - Alternative: Struct-array based tree (like tinevez/matlab-tree) may be faster

2. **Progressive Widening is Viable**:
   - No existing MATLAB implementation → we build from scratch
   - Aalto thesis concludes "PW is best general-purpose extension"
   - Formula: `k = ceil(C × N^α)` with C=2, α=0.5 is standard starting point

3. **Parallelization Strategy**:
   - Current: Parallel rollouts via parfor ✓
   - Cannot add parallel tree traversal (nested parfor limitation)
   - Keep current approach, focus on depth improvements

4. **RAVE Complexity Assessment**:
   - No MATLAB implementation exists
   - Requires tracking AMAF statistics per action across all nodes
   - Medium complexity, but high potential reward
   - Consider after Progressive Widening proves effective

**Implementation Priority (Revised)**:

| Priority | Technique | Rationale |
|----------|-----------|-----------|
| **1** | Progressive Widening | Best general-purpose, no MATLAB impl exists, directly addresses our problem |
| **2** | Dictionary swap | Easy win if R2022b+, improves all tree operations |
| **3** | RAVE | High reward if AMAF assumption holds for Tangled |
| **4** | Struct-based tree | Only if memory becomes issue |

---

## D-Wave / Quantum Annealing Cross-Reference

### Relevance to Tangled

Our game uses `SimulatedAnnealingAdjudicator` for terminal state evaluation, making D-Wave's research on optimization and search algorithms directly relevant.

### Key D-Wave Publications on Optimization

| Paper | Authors | Year | Relevance |
|-------|---------|------|-----------|
| [Discrete optimization using quantum annealing on sparse Ising models](https://www.frontiersin.org/articles/10.3389/fphy.2014.00056/full) | Bian, Chudak, Israel, Lackey, Macready, Roy | 2014 | Core optimization methodology |
| [Training a Binary Classifier with Quantum Adiabatic Algorithm](https://arxiv.org/abs/0811.0416) | Neven, Denchev, **Rose**, Macready | 2008 | Geordie Rose co-author; tabu search comparison |
| [Investigating Performance of Adiabatic Quantum Optimization](https://link.springer.com/article/10.1007/s11128-011-0235-0) | Karimi, Dickson, Hamze, Amin, Drew-Brook, Chudak, Bunyk, Macready, **Rose** | 2012 | 4-6 orders magnitude faster than classical solvers |
| [Scaling advantage over path-integral Monte Carlo](https://www.nature.com/articles/s41467-021-20901-5) | King, Raymond, Lanting et al. | 2021 | Million-fold speedup over PIMC |

### D-Wave's Hybrid Classical-Quantum Approach

D-Wave's **qbsolv** is highly relevant to our MCTS problem:

**Architecture**:
```
Large QUBO Problem
       ↓
   [Decomposition]
       ↓
┌──────────────────────────────────────┐
│  Sub-problems solved by:             │
│  - D-Wave quantum annealer, OR       │
│  - Classical tabu search (default)   │
└──────────────────────────────────────┘
       ↓
   [Tabu post-processing]
       ↓
   Combined solution
```

**Key Insight**: D-Wave uses **tabu search** as their primary classical heuristic, not simulated annealing or MCTS. This suggests tabu search may have properties well-suited to QUBO/Ising optimization that we could leverage.

**Source**: [D-Wave qbsolv GitHub](https://github.com/dwavesystems/qbsolv)

### Hybrid Branch-and-Bound (2024)

Recent research combines classical branch-and-bound with quantum annealing:

> "The quantum annealer is still a developing technology, currently affected by noise that makes it unable to find the global solution even to simple problems. To overcome this limitation, we propose using a classical-hybrid protocol where the quantum hardware serves as a subroutine for a variant of the classical branch-and-bound algorithm."

**Relevance**: This hybrid decomposition approach mirrors how we might combine MCTS (tree search) with our LUT (exact evaluation).

**Source**: [Hybrid Classical-Quantum Branch-and-Bound](https://www.mdpi.com/1099-4300/26/4/345)

### QuantumZero: MCTS for Quantum Annealing

Directly relevant research combining MCTS with quantum annealing:

> "Inspired by DeepMind's AlphaZero, we propose a Monte Carlo tree search (MCTS) algorithm—named QuantumZero (QZero)—to automate the design of annealing schedules in a hybrid quantum-classical framework."

**Key Finding**: MCTS is being used to optimize quantum annealing schedules, creating a feedback loop where:
- MCTS guides the quantum optimization
- Quantum evaluation improves MCTS decisions

**Source**: [Nature Machine Intelligence - QuantumZero](https://www.nature.com/articles/s42256-022-00446-y)

### Quantum Game Theory Applications

Research on quantum approaches to game trees:

| Application | Status | Notes |
|-------------|--------|-------|
| Nash equilibria via quantum annealing | Active research | [Q-Nash on D-Wave 2000Q](https://pmc.ncbi.nlm.nih.gov/articles/PMC7304779/) |
| Quantum minimax evaluation | Emerging | Superposition-based parallel branch evaluation |
| QUBO formulation of games | Established | Maps game decisions to optimization |

### Potential Novel Approaches

Based on D-Wave research, unexplored directions for Tangled MCTS:

1. **Tabu-Enhanced MCTS**: Replace random rollouts with tabu search
   - D-Wave uses tabu as primary classical solver
   - Could improve rollout quality without LUT

2. **QUBO Formulation of Tangled**: Map game state evaluation to QUBO
   - 15 binary variables (edges) maps naturally
   - Could use simulated annealing for mid-game evaluation

3. **Hybrid Decomposition**:
   - Use MCTS for high-level move selection
   - Use annealing-based solver for sub-tree evaluation
   - Similar to D-Wave's qbsolv decomposition

4. **Schedule Optimization for SA Adjudicator**:
   - Our LUT uses fixed SA parameters
   - QuantumZero approach could optimize annealing schedule per state

---

## Progressive Widening Implementation Sketch

### Proposed Changes to MCTSNode.m

**1. New Property**:
```matlab
properties
    MaxChildren int32 = 0  % Current max children (0 = unlimited)
end
```

**2. Modified isFullyExpanded()**:
```matlab
function tf = isFullyExpanded(this, pwConstant, pwExponent)
    if nargin < 2, pwConstant = 2.0; end
    if nargin < 3, pwExponent = 0.5; end

    % Progressive Widening: k = ceil(C * N^alpha)
    maxChildren = ceil(pwConstant * double(this.Visits)^pwExponent);

    % Fully expanded if we've expanded maxChildren OR no more actions
    tf = (this.Children.Count >= maxChildren) || isempty(this.UntriedActions);
end
```

**3. Modified bestChild() to consider expansion**:
```matlab
function child = bestChild(this, exploration, priorWeight, pwConstant, pwExponent)
    % If under PW limit and untried actions exist, might want to expand instead
    maxChildren = ceil(pwConstant * double(this.Visits)^pwExponent);

    if this.Children.Count < maxChildren && ~isempty(this.UntriedActions)
        % Could expand - but UCB1 decides
        % Add virtual "expand" option with prior-based value?
    end

    % ... rest of selection logic
end
```

### Expected Depth Improvement

| Visits | Max Children (C=2, α=0.5) | Branching Factor |
|--------|---------------------------|------------------|
| 1 | 2 | 2 |
| 4 | 4 | 4 |
| 9 | 6 | 6 |
| 16 | 8 | 8 |
| 25 | 10 | 10 |
| 100 | 20 | 20 |

With 5000 iterations and branching capped at ~2-6 early:
- Depth 5 with BF=2: 2^5 = 32 nodes
- Depth 8 with BF=2-4: ~500-1000 nodes
- **Expected new depth: 7-9**

### Tuning Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| C (pwConstant) | 1.0 - 3.0 | Higher = more breadth |
| α (pwExponent) | 0.3 - 0.7 | Higher = faster widening |

Start with C=2.0, α=0.5 (standard values from literature).

---

## Next Steps

1. [x] Read MCTSNode.m to understand current expansion logic
2. [ ] Decide on first technique to implement (recommend **A1 - Progressive Widening**)
3. [ ] Design experiment protocol for measuring depth improvement
4. [ ] Implement Progressive Widening in MCTSNode.m
5. [ ] Test depth improvement with fixed iterations
6. [ ] Run games against Melissa to measure win rate change
