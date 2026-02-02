# AlphaQ Strategy: Run Analysis & Improvements

## Introduction

This document tracks the performance of the AlphaQ Explorer strategy against the AlphaQ Up opponent. The strategy combines:
- **Ground-truth Schrödinger equation terminal evaluation** (eliminating SA bias)
- **Thompson Sampling opening selection** (Beta distribution-based exploration/exploitation)
- **REINFORCE learning with edge bias** (adaptive mid-game optimization)
- **TangledMCTS solver** (MCTS + minimax hybrid with LUT acceleration)

Each run section documents observed patterns, statistical analysis, and proposed improvements. The goal is to achieve consistent wins against AlphaQ Up, which represents the strongest available opponent.

---

## Run 60 Analysis

**Date:** 2026-02-01 to 2026-02-02
**Duration:** 9 hours 5 minutes
**Games:** 500
**Opponent:** AlphaQ Up
**Strategy:** alphaq_explorer (Thompson Sampling + ground-truth LUTs)

### Results Summary

| Metric | Value |
|--------|-------|
| Wins | 0 (0.0%) |
| Draws | 464 (92.8%) |
| Losses | 36 (7.2%) |
| Avg Score | +0.232 |
| Best Score | +0.864 |
| Worst Score | -8.810 |

### Opening Selection (Thompson Sampling)

Thompson Sampling successfully avoided catastrophic openings identified in previous runs:

| Opening | Games | Win Rate | Outcome |
|---------|-------|----------|---------|
| E10G | 65 | 0% | Safe - avoid catastrophic E9G/E11G |
| E1G | 64 | 0% | Safe |
| E12P | 60 | 0% | Safe |
| E5P | 59 | 0% | Safe |
| E9G | 1 | 0% | **Avoided** (was 89% loss rate) |
| E11G | 0 | 0% | **Avoided** (was 100% loss rate) |

**Thompson Sampling Status:** ✓ Working as designed - eliminating proven losers

### Critical Discovery: E7G Dominance

**All top 10 scoring games used E7G opening:**

| Game ID | Score | Moves | Opening |
|---------|-------|-------|---------|
| f99d0efe | +0.864 | 8 | E7G |
| 63861f0f | +0.832 | 8 | E7G |
| 3a6a883b | +0.822 | 8 | E7G |
| a3bca702 | +0.816 | 8 | E7G |
| bde56f62 | +0.813 | 8 | E7G |
| e91663e4 | +0.808 | 8 | E7G |
| cabd74d0 | +0.807 | 8 | E7G |
| 2a3d4adf | +0.806 | 8 | E7G |
| 6f47f4fb | +0.803 | 8 | E7G |
| ee93d0b9 | +0.799 | 8 | E7G |

### Opening Performance Comparison

| Opening | Games | Wins | Draws | Losses | Avg Score | Max Score |
|---------|-------|------|-------|--------|-----------|-----------|
| **E7G** | 49 | 0 | 49 | 0 | **+0.779** | **+0.864** |
| E13P | 54 | 0 | 53 | 1 | +0.524 | +0.587 |
| E8G | 51 | 0 | 51 | 0 | +0.308 | +0.336 |
| E10G | 64 | 0 | 64 | 0 | +0.232 | +0.276 |
| E1G | 64 | 0 | 64 | 0 | +0.034 | +0.067 |
| E8P | 11 | 0 | 0 | 11 | **-0.553** | -0.528 |
| E0G | 5 | 0 | 0 | 5 | -1.673 | - |
| E6P | 2 | 0 | 0 | 2 | **-8.806** | - |

**Key Insight:** E7G outperforms second-best opening (E13P) by **+0.255 average score** (48% improvement).

### Winning Threshold Analysis

Distribution of draw scores:

| Score Range | Count | Percentage |
|-------------|-------|------------|
| Very close (>0.85) | 1 | 0.2% |
| Close (0.5-0.85) | 100 | 21.6% |
| Positive (0-0.5) | 362 | 78.0% |
| Negative/Zero | 1 | 0.2% |

**Conclusion:** We reached near-winning territory (0.864) once, and scored >0.5 in 101 games (21.6%). The strategy is close to winning but needs refinement to convert high scores into actual wins.

### Best Game Deep Dive: f99d0efe (E7G, +0.864)

Move-by-move analysis:

| Move# | Player | Move | Score | Delta | Notes |
|-------|--------|------|-------|-------|-------|
| 1 | us | 7G | +0.050 | +0.050 | Strong opening |
| 1 | opp | 0P | -0.001 | -0.051 | Opponent weakens |
| 2 | us | 1G | +0.058 | +0.008 | Maintain lead |
| 2 | opp | 2G | -0.075 | -0.133 | Opponent blunders |
| 3 | us | 3G | +0.072 | +0.014 | Building advantage |
| 3 | opp | 4G | +0.128 | +0.056 | Opponent recovers slightly |
| 4 | us | 5P | **+4.513** | **+4.441** | **Huge breakthrough!** |
| 5 | us | 10G | +2.684 | -1.829 | Giving back advantage |
| 5 | opp | 12G | +0.092 | -2.592 | Opponent punishes |
| 6 | us | 13P | +0.313 | +0.221 | Recover slightly |
| 6 | opp | 6P | +0.499 | +0.186 | Opponent helps us |
| 7 | us | 14P | +0.150 | -0.349 | Retreating |
| 8 | us | 11G | +0.864 | +0.714 | Strong finish |

**Critical Observation:** We achieved **+4.513** on move 4 (a massive winning position), but retreated to +0.864 by endgame. This suggests:
1. MCTS is finding winning positions
2. Conservative play in late-game is giving back advantages
3. Need more aggressive strategy when ahead

### Worst Game Deep Dive: aeb8d813 (E6P, -8.810)

Move-by-move analysis:

| Move# | Player | Move | Score | Delta | Notes |
|-------|--------|------|-------|-------|-------|
| 1 | us | 6P | -0.011 | -0.011 | Poor opening |
| 1 | opp | 0P | -0.012 | -0.001 | Both weak |
| 2 | us | 1G | +0.030 | +0.041 | Attempted recovery |
| 2 | opp | 2P | +0.056 | +0.026 | Opponent improves |
| 3 | us | 3G | -0.018 | -0.074 | Losing ground |
| 3 | opp | 5G | **-2.846** | **-2.828** | **Devastating blow** |
| 4 | us | 4G | -2.865 | -0.019 | Can't recover |
| 4 | opp | 8G | -3.403 | -0.538 | Spiral continues |
| 5 | us | 9G | -0.885 | +2.518 | Partial recovery |
| 5 | opp | 11P | -3.138 | -2.253 | Back down |
| 6 | us | 10P | -4.059 | -0.921 | Collapsing |
| 6 | opp | 7P | -3.980 | +0.079 | - |
| 7 | us | 12G | -7.664 | -3.684 | Catastrophic |
| 8 | us | 14G | -8.810 | -1.146 | Final loss |

**Critical Observation:** E6P opening led to immediate disadvantage. Opponent's move 3 (E5G) caused -2.828 delta, and we never recovered. This validates Thompson Sampling's avoidance of E6P.

### Loss Analysis by Opening

Openings that led to losses:

| Opening | Losses | Avg Loss Score | Status |
|---------|--------|----------------|--------|
| E8P | 11 | -0.553 | **Catastrophic** (100% loss rate) |
| E0G | 5 | -1.673 | High risk |
| E6P | 2 | -8.806 | **Catastrophic** |
| E7P | 2 | -0.049 | Marginal |
| E3G | 2 | -0.496 | Marginal |
| E13G | 2 | +0.017 | Edge case |

**Thompson Sampling should eliminate these over time.**

---

## Improvement Approaches

Based on Run 60 analysis, the following improvement approaches are proposed:

### Approach A: Force E7G Opening (High Priority)

**Rationale:**
- E7G has 0% loss rate and +0.779 average score (vastly superior to alternatives)
- All top 10 games used E7G
- Need more E7G data to understand optimal continuation patterns

**Implementation:**
1. Temporarily override Thompson Sampling to always select E7G
2. Run 100-200 games to gather E7G-specific move sequences
3. Analyze mid-game patterns after E7G opening
4. Build E7G-specific opening book with proven continuations

**Expected Outcome:** Better understanding of E7G endgame, identify why we retreat from winning positions

### Approach B: Increase MCTS Aggressiveness (High Priority)

**Rationale:**
- Best game reached +4.513 but retreated to +0.864
- Conservative play in winning positions gives back advantages
- Need to "push for the win" when ahead

**Implementation:**
1. Increase UCT exploration constant when score > 0.5
2. Add "winning mode" that prefers high-variance moves when ahead
3. Adjust MCTS rollout policy to favor aggressive continuations
4. Reduce risk aversion in terminal evaluation when winning

**Expected Outcome:** Convert 0.5+ scores into actual wins

### Approach C: Thompson Sampling Adjustment (Medium Priority)

**Rationale:**
- E7G is under-sampled (49 games) compared to E10G (65 games) and E1G (64 games)
- Thompson Sampling is exploring too much, not exploiting E7G enough

**Implementation:**
1. Increase Beta parameters for E7G based on observed performance
2. Manually seed E7G with bonus "virtual wins" to bias selection
3. Add score-based reward (not just W/D/L) to Thompson Sampling
4. Adjust half-draw credit to 0.6-0.7 for high-scoring draws (0.5+ score)

**Expected Outcome:** More E7G games in future runs, faster convergence to optimal opening

### Approach D: Opponent Response Analysis (Low Priority)

**Rationale:**
- Opponent's responses to E7G are predictable
- Can build counterstrategy for specific opponent patterns

**Implementation:**
1. Query opponent_history table for E7G response patterns
2. Identify opponent's most common responses (move 2)
3. Pre-compute optimal counter-moves for each response
4. Add opponent-conditional opening book

**Expected Outcome:** Optimized move 2-3 selections after E7G

### Approach E: Add Winning-Push Heuristic (Medium Priority)

**Rationale:**
- MCTS doesn't understand "we're winning, stay aggressive"
- Need explicit logic to preserve/extend leads

**Implementation:**
1. Track score history during game
2. When score increases significantly (>+0.5 delta), enter "preserve lead" mode
3. In preserve lead mode:
   - Prefer moves that maintain or increase score
   - Avoid "safe but regressive" moves
   - Increase search depth on critical decisions
4. Add score momentum tracking to MCTS evaluation

**Expected Outcome:** Maintain +4.5 positions instead of retreating to +0.8

---

## Recommended Action Plan

**Phase 1: Immediate (Next Run)**
1. Implement Approach A: Force E7G opening for 100-200 games
2. Implement Approach B: Increase MCTS exploration constant to 1.8-2.0 (from default ~1.41)
3. Goal: Achieve first win by converting high-scoring E7G games

**Phase 2: Short-term (Following Run)**
1. Analyze E7G game data from Phase 1
2. Implement Approach E: Add winning-push heuristic based on Phase 1 patterns
3. Re-enable Thompson Sampling with E7G bias (Approach C)
4. Goal: 5-10% win rate

**Phase 3: Medium-term**
1. Implement Approach D: Opponent response counterstrategy
2. Fine-tune MCTS parameters based on win/loss analysis
3. Goal: 20%+ win rate, consistently convert 0.8+ scores to wins

---

## Run 64 Analysis (Phase 1 Complete)

**Date:** 2026-02-02
**Duration:** ~1.8 hours
**Games:** 100
**Opponent:** AlphaQ Up
**Strategy:** alphaq_explorer (Forced E7G + Aggressive MCTS)
**Approaches Tested:** A (Force E7G) + B (Exploration 1.8)

### Results Summary

| Metric | Value | Change from Run 60 |
|--------|-------|-------------------|
| Wins | 0 (0.0%) | 0 (same) |
| Draws | 100 (100.0%) | +7.2% (was 92.8%) |
| Losses | 0 (0.0%) | **-7.2% (was 7.2%)** ✓ |
| Avg Score | +0.779 | +0.547 (+236%) |
| Best Score | +0.877 | +0.013 |
| Worst Score | +0.712 | -0.152 (improved floor) |

### Key Achievements

**✓ Eliminated All Losses**
- 100% draw rate achieved
- No losses in 100 games (vs 36 losses in Run 60's 500 games)
- Minimum score +0.712 (never dropped below 0.7)

**✓ Validated E7G Dominance**
- Average score +0.779 exactly matches Run 60's E7G average
- Consistent performance: 0.712-0.877 range (0.165 spread)
- 27% of games scored >0.8 (very high territory)

**✓ Proven Strategy Stability**
- Zero catastrophic openings (E9G, E11G, E6P, E0G all avoided)
- All 100 games with E7G = 100% safe outcomes
- Approach A validated: E7G is the optimal opening

### Score Distribution

| Score Range | Count | Percentage | Category |
|-------------|-------|------------|----------|
| >0.85 | 1 | 1% | Very close to winning |
| 0.8-0.85 | 26 | 26% | Very high |
| 0.7-0.8 | 73 | 73% | High |
| 0.5-0.7 | 0 | 0% | Medium |
| <0.5 | 0 | 0% | Low |

**Observation:** Scores cluster tightly in 0.7-0.85 range, suggesting E7G games converge to predictable high-scoring draws.

### Top 5 Games

| Game ID | Score | Moves | Notes |
|---------|-------|-------|-------|
| 4bc36edd | +0.877 | 8 | Best Run 64 score |
| a6d0a5a3 | +0.846 | 8 | |
| 10e04907 | +0.839 | 8 | |
| 42c35e6e | +0.837 | 8 | |
| 95f07259 | +0.832 | 8 | |

### Approach Testing Results

**Approach A: Force E7G Opening** ✅ **SUCCESS**
- **Goal:** Gather focused E7G data and validate dominance
- **Result:** Perfect 100/100 draw rate, +0.779 average
- **Conclusion:** E7G is definitively the best opening (0% loss rate, highest avg score)
- **Next:** Can re-enable Thompson Sampling with heavy E7G bias

**Approach B: Increase MCTS Aggressiveness** ❌ **NO IMPACT**
- **Goal:** Convert high scores to wins by holding advantages
- **Implementation:** Increased exploration constant 1.414 → 1.8
- **Result:** Scores identical to Run 60's E7G games (+0.779 avg)
- **Conclusion:** Problem is not mid-game conservatism but strategic convergence
- **Insight:** E7G games naturally converge to 0.7-0.8 draws against strong opponents

### Strategic Insights

**1. E7G Creates Stable, High-Scoring Draws**
- The tight score distribution (0.712-0.877) indicates E7G follows consistent patterns
- Unlike Run 60's best game (+4.513 spike then retreat to +0.864), Run 64 never exceeded +0.877
- Suggests the +4.513 was an opponent blunder, not a repeatable E7G property

**2. AlphaQ Up Adapts Well to E7G**
- Opponent has seen E7G 149 times now (49 Run 60 + 100 Run 64)
- Consistent opponent responses keep scores in 0.7-0.8 range
- Need to find exploitable patterns in opponent's E7G responses

**3. Winning Requires Different Strategy**
- More exploration doesn't help - need tactical changes
- Potential paths:
  - Exploit specific opponent response patterns
  - Add "winning push" logic when score >0.8
  - Diversify openings to prevent opponent adaptation

### Comparison: Run 60 E7G Games vs Run 64

| Metric | Run 60 (E7G subset) | Run 64 (All E7G) | Change |
|--------|---------------------|------------------|--------|
| Games | 49 | 100 | +51 |
| Avg Score | +0.779 | +0.779 | 0.000 (identical!) |
| Max Score | +0.864 | +0.877 | +0.013 |
| Min Score | +0.717 | +0.712 | -0.005 |
| Loss Rate | 0% | 0% | Same |

**Conclusion:** Forcing E7G doesn't change outcomes - the opening itself determines score range.

### Phase 1 Summary

**Proven Facts:**
1. E7G is the optimal opening (0% loss rate, +0.779 avg across 149 games)
2. Thompson Sampling successfully avoids bad openings (E9G, E11G eliminated)
3. Ground-truth Schrödinger LUTs work correctly (stable, predictable scores)
4. Aggressive MCTS exploration doesn't convert draws to wins

**Remaining Challenge:**
- How to convert 0.8+ scores into actual wins?
- AlphaQ Up appears to have strong defensive responses to E7G
- Need tactical innovations, not just parameter tuning

---

## Phase 2: Winning-Push Heuristic + Thompson Sampling with E7G Bias

### Approach C: Re-enable Thompson Sampling with E7G Bias

**Rationale:**
E7G dominance is proven. Now diversify openings while maintaining E7G preference to:
1. Gather data on other safe openings (E13P, E8G showed promise)
2. Prevent opponent from over-adapting to E7G
3. Find alternative winning paths

**Implementation:**
- Remove `force_opening='E7G'` from play_tangled.py
- Seed E7G with bonus virtual games to bias Thompson Sampling
- Adjust draw credit from 0.5 to 0.6 for high-scoring draws (>0.7)

### Approach E: Add Winning-Push Heuristic

**Rationale:**
Games consistently reach 0.8+ but don't convert to wins. Need explicit logic to:
1. Detect when we're in winning position (score >0.75)
2. Maintain pressure instead of regressing to safe moves
3. Avoid the pattern: +0.8 → safe play → +0.75 → opponent recovers

**Implementation:**
1. Track score momentum during game
2. When score >0.75, enter "preserve lead" mode:
   - Prefer moves that maintain/increase score
   - Increase MCTS iterations by 50% for critical moves
   - Add penalty for moves that decrease score
3. Add score trend analysis to move selection

**Goal:** Convert 10-20% of 0.8+ games into wins (targeting 5-10% win rate in next 100 games)

---

## Next Run: [To be filled]

*Future run analyses will be appended here.*
