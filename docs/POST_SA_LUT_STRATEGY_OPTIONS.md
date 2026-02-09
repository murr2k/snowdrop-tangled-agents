# Strategy Options Beyond the SA LUT

**Context:** The SA LUT exhibits polarity inversion against AlphaQ (r = -0.396) and cannot be calibrated. What alternatives exist for strategy development?

## Current Assets

### 1. Empirical Data
- **2,436 terminal states** with website scores (7.43% coverage)
- **Opponent history database** with AlphaQ move patterns
- **120 high-quality terminal states** from recent campaigns (diverse openings)

### 2. Working Mechanisms
- **MCTS implementation** (currently used as fallback)
- **Opponent model** (entropy tracking, prediction accuracy)
- **Oracle routing system** (proven to work mechanically)
- **Website score capture** (calibration table)

### 3. Known Constraints
- **AlphaQ zero-loss equilibrium:** Max observed score +0.861 across 120 states
- **Terminal state basin:** AlphaQ restricts play to [-8.8, +0.9] range
- **Deterministic opponent:** AlphaQ is 100% consistent on well-observed paths (min_obs > 50)

## Strategy Alternatives

### Option 1: Pure MCTS (No Terminal Evaluation)

**Approach:** Use MCTS without any terminal state scoring, relying purely on simulation rollouts.

**Advantages:**
- Already implemented and working
- No bias from incorrect terminal evaluation
- Can adapt to any opponent

**Disadvantages:**
- Computationally expensive (~4 min/game with 50k iterations)
- No strategic guidance for long-term planning
- Still achieves 0% win rate against AlphaQ

**Feasibility:** ★★★★☆ (already working, just remove SA LUT dependency)

**Implementation:**
```python
# Current: MCTS uses SA LUT for terminal evaluation
# New: Remove terminal evaluation entirely, use pure rollouts

class MCTSStrategy:
    def __init__(self, time_limit, max_iterations, use_terminal_eval=False):
        self.use_terminal_eval = use_terminal_eval  # Set to False
```

---

### Option 2: Empirical Website LUT

**Approach:** Build a LUT purely from observed website scores, use interpolation for unobserved states.

**Advantages:**
- Uses ground truth (website scores) directly
- 2,436 observations provide reasonable coverage for common states
- Can be continuously improved with more games

**Disadvantages:**
- Only 7.43% coverage means 92% of states need interpolation
- Interpolation quality is uncertain (R² = 0.01 for simple features)
- Still won't escape AlphaQ's attractor basin

**Feasibility:** ★★★☆☆ (tools exist, but interpolation is unreliable)

**Implementation:**
```bash
# Already have this tool
poetry run python snowdrop_tangled_agents/tools/build_website_lut.py --opponent all --output website_lut.bin

# Use in oracle solver
cd oracle-solver
cargo run --release -- --lut-path ../snowdrop_tangled_agents/data/website_lut.bin
```

**Critical question:** What interpolation method for the 92% of unobserved states?
- Nearest neighbor (most conservative)
- Regression model (R² = 0.01, unreliable)
- Default to zero (neutral assumption)
- NaN (refuse to evaluate unobserved states)

---

### Option 3: Reinforcement Learning from Website Outcomes

**Approach:** Train a neural network to predict website scores based on game state features, using actual game outcomes as training data.

**Advantages:**
- Learns from ground truth outcomes
- Can discover complex patterns beyond linear features
- Continuously improves with more games

**Disadvantages:**
- Requires thousands of games for training
- Needs feature engineering (what state features predict website scores?)
- May still be limited by AlphaQ's attractor basin

**Feasibility:** ★★☆☆☆ (requires significant ML infrastructure)

**Implementation outline:**
1. Extract features: edge colors, graph structure, move history
2. Train model: `website_score = NN(game_state_features)`
3. Use model for move evaluation in MCTS or minimax
4. Retrain periodically with new game data

**Required dependencies:**
- PyTorch or TensorFlow
- Feature extraction pipeline
- Training loop with validation

---

### Option 4: Opponent Modeling (Predict AlphaQ, Not Terminal Scores)

**Approach:** Focus on predicting AlphaQ's next move rather than evaluating terminal states. Build a policy model of AlphaQ's decision-making.

**Advantages:**
- AlphaQ is deterministic (easier to model than stochastic opponents)
- We have rich opponent history data
- Can enable adversarial play (force AlphaQ into unfavorable positions)

**Disadvantages:**
- Still need to know what "unfavorable" means (back to terminal evaluation)
- Prediction accuracy plateaus around state space boundaries
- May just learn to avoid losses, not find wins

**Feasibility:** ★★★★☆ (we already have opponent modeling infrastructure)

**Current implementation:**
```python
# Already exists in stats/collector.py
class StatsCollector:
    def build_opponent_model(self, opponent_name):
        # Builds transition probability matrix
        # Computes entropy, prediction accuracy, top-k hit rate
```

**Enhancement opportunities:**
1. Use opponent model to **predict full game tree** (not just next move)
2. Identify **low-confidence states** where AlphaQ's policy is uncertain
3. **Adversarial search:** Find move sequences that maximize AlphaQ's entropy
4. **Policy exploitation:** Detect and exploit suboptimal AlphaQ moves

---

### Option 5: Minimax with Website Oracle Queries

**Approach:** Use minimax/alpha-beta search, but evaluate positions by **actually playing them out against AlphaQ** to get website scores.

**Advantages:**
- Ground truth evaluation (no prediction error)
- Works even with zero coverage of terminal states
- Guarantees optimal play within search depth

**Disadvantages:**
- Requires playing many games per move (exponential branching)
- Too slow for real-time play (would need distributed infrastructure)
- AlphaQ is an online opponent, can't query offline

**Feasibility:** ★☆☆☆☆ (theoretically interesting, practically infeasible)

**Why it fails:** Can't query website adjudicator offline. Would need to play full games to terminal for each evaluation.

---

### Option 6: Adversarial Opening Library

**Approach:** Build a curated library of opening sequences designed to **force AlphaQ out of its comfort zone**, rather than optimizing for terminal scores.

**Advantages:**
- Exploits AlphaQ's determinism
- Can target low-observation states (where policy may be weaker)
- Doesn't require terminal evaluation
- Low computational cost

**Disadvantages:**
- Requires understanding what states are "uncomfortable" for AlphaQ
- May not exist - AlphaQ may have uniform strength across state space
- Still limited by zero-loss equilibrium

**Feasibility:** ★★★★☆ (feasible, worth exploring)

**Implementation strategy:**
1. **Identify sparse states:** Query opponent_history for states with min_obs < 10
2. **Construct routes to sparse states:** Find move sequences that lead there
3. **Measure AlphaQ uncertainty:** Track entropy and prediction accuracy
4. **Adversarial selection:** Choose openings that maximize AlphaQ's policy entropy

```python
class AdversarialOpeningStrategy:
    def __init__(self, opponent_model, sparsity_threshold=10):
        self.opponent_model = opponent_model
        self.sparsity_threshold = sparsity_threshold

    def select_opening(self):
        """Choose opening that leads to sparse AlphaQ states."""
        candidate_openings = self.enumerate_openings()

        # For each opening, simulate AlphaQ response
        # Score by: (1) low observation count, (2) high entropy
        scored_openings = []
        for opening in candidate_openings:
            state_after = self.apply_opening(opening)
            obs_count = self.opponent_model.get_obs_count(state_after)
            entropy = self.opponent_model.get_entropy(state_after)
            score = entropy / (1 + obs_count)  # Prefer high entropy + low obs
            scored_openings.append((score, opening))

        return max(scored_openings)[1]
```

---

### Option 7: Graph-Theoretic Heuristics

**Approach:** Design heuristics based on graph structure, connectivity, and combinatorial properties instead of physics-based terminal evaluation.

**Advantages:**
- Fast to compute
- Interpretable (unlike neural networks)
- Domain-specific to Petersen graph

**Disadvantages:**
- Unclear what heuristics correlate with website wins
- May be as unreliable as SA LUT
- Requires domain expertise in graph theory

**Feasibility:** ★★☆☆☆ (possible but high effort, uncertain payoff)

**Potential heuristics:**
- **Edge color balance:** Minimize |green - purple|
- **Vertex frustration:** Count frustrated vertices in implied spin configuration
- **Connectivity patterns:** Favor/avoid specific subgraph colorings
- **Symmetry breaking:** Choose moves that maximize asymmetry

**Challenge:** We don't know which heuristics correlate with website wins. Would need empirical validation.

---

### Option 8: Hybrid Ensemble Strategy

**Approach:** Combine multiple weak strategies (MCTS, opponent model, empirical LUT, heuristics) into an ensemble that votes on moves.

**Advantages:**
- Diversifies risk across multiple approaches
- Can leverage strengths of each component
- More robust than any single strategy

**Disadvantages:**
- Complex to tune (how to weight each component?)
- Debugging is difficult (which component caused failure?)
- May not exceed best individual component

**Feasibility:** ★★★☆☆ (implementable, but coordination overhead)

**Implementation:**
```python
class EnsembleStrategy:
    def __init__(self, strategies, weights):
        self.strategies = strategies  # [MCTS, OpponentModel, Heuristic, ...]
        self.weights = weights        # [0.5, 0.3, 0.2, ...]

    def calculate_move(self, state, score, score_history):
        votes = []
        for strategy, weight in zip(self.strategies, self.weights):
            move = strategy.calculate_move(state, score, score_history)
            votes.append((move, weight))

        # Weighted voting
        move_scores = defaultdict(float)
        for move, weight in votes:
            move_scores[move] += weight

        return max(move_scores.items(), key=lambda x: x[1])[0]
```

---

## Recommended Path Forward

### Tier 1: High-Value, Low-Effort
1. **Pure MCTS** (remove SA LUT dependency)
   - Already working, just needs SA LUT removal
   - Establishes baseline performance

2. **Adversarial Opening Library**
   - Leverages AlphaQ's determinism
   - Targets low-observation states
   - Can be implemented with existing tools

### Tier 2: Medium-Term Development
3. **Enhanced Opponent Modeling**
   - Predict full game trees, not just next move
   - Identify and exploit low-confidence states
   - Build adversarial search on top of predictions

4. **Empirical Website LUT with Conservative Interpolation**
   - Use observed scores where available (7.43% coverage)
   - Default to zero or nearest-neighbor for unobserved states
   - Continuously improve with more game data

### Tier 3: Research Projects
5. **Reinforcement Learning**
   - Requires infrastructure build-out
   - Long training time
   - Uncertain payoff given AlphaQ equilibrium

6. **Graph-Theoretic Heuristics**
   - Needs domain expertise
   - Uncertain correlation with website wins
   - High research risk

## Critical Insight

**The SA LUT was never the problem.** The problem is **AlphaQ's zero-loss equilibrium.**

Across 120 diverse terminal states reached via oracle routes and systematic exploration, the maximum website score is +0.861 - less than half the +2 win threshold. This suggests:

1. **AlphaQ's policy is strong:** It restricts play to a narrow attractor basin
2. **Winning terminal states may not be reachable:** Against AlphaQ's optimal responses
3. **Terminal evaluation is not the bottleneck:** Better evaluation won't help if wins are unreachable

### Alternative Hypothesis

Perhaps we need to **change the game**, not the strategy:

- **Test against weaker opponents** (melissa, amara) where wins are possible
- **Validate strategies on winnable positions** first
- **Return to AlphaQ** only after demonstrating wins elsewhere

This would tell us whether our strategies are fundamentally flawed, or whether AlphaQ is simply unbeatable with current game mechanics.

## Next Steps

**Immediate (1-2 days):**
1. Implement pure MCTS without SA LUT
2. Run 50 games vs AlphaQ to establish baseline
3. Compare to previous 0% win rate

**Short-term (1 week):**
4. Build adversarial opening library targeting sparse AlphaQ states
5. Test if forced exploration breaks the [-8.8, +0.9] basin
6. Analyze if new terminal states have higher scores

**If still no wins against AlphaQ:**
7. Pivot to melissa/amara to validate strategy mechanisms
8. Document AlphaQ as solved position (zero-loss equilibrium)
9. Publish findings on evaluator non-equivalence and opponent convergence

The SA LUT was a red herring. The real question is: **Can AlphaQ be beaten at all?**
