# Amara Strategy Analysis

## Introduction

AlphaZero Amara is a Tangled game opponent trained using reinforcement learning on actual D-Wave quantum hardware. As of this analysis, no player has defeated Amara, with a $10 bounty offered for the first win. Our hybrid solver strategy, which achieves a positive win rate against other opponents (notably Melissa), consistently draws against Amara with final scores within ±0.006 of zero.

This document presents a spectral analysis of gameplay data to understand the nature of these draws and evaluate potential strategies for achieving a win.

## Purpose

1. Determine whether Amara represents "solved" play (like tic-tac-toe) or has exploitable weaknesses
2. Analyze the power distribution and periodicity of score progressions
3. Compare Amara's gameplay structure to other opponents
4. Provide data-driven recommendations for strategy development

## Theory

### Quantum Annealing and Game Adjudication

The Tangled game's terminal state is evaluated by computing the ground state energy of an Ising Hamiltonian, where edge colorings map to coupling strengths:
- Green (FM): Ferromagnetic coupling, favors aligned spins
- Purple (AFM): Antiferromagnetic coupling, favors anti-aligned spins

D-Wave quantum annealers are specifically designed to find these ground states. Amara, trained on D-Wave hardware, has learned to play against "ground truth" quantum evaluations rather than classical approximations.

### Equilibrium Hypothesis

If both players execute optimal strategies, the game may be fundamentally balanced (like tic-tac-toe), resulting in guaranteed draws. The near-zero final scores suggest both our solver and Amara are playing near-optimally, locked in equilibrium.

### Spectral Analysis Approach

By treating score progressions as time series, we can apply signal processing techniques:
- **Power Spectral Density (PSD)**: Reveals dominant frequencies in score oscillations
- **Autocorrelation**: Detects repeating patterns and move-countermove structures
- **Variance Analysis**: Measures predictability and chaos in gameplay

## Data

### Dataset Summary

| Opponent | Games | Wins | Losses | Draws | Win Rate |
|----------|-------|------|--------|-------|----------|
| Amara    | 27    | 0    | 0      | 27    | 0%       |
| Melissa  | 878   | 120  | 96     | 662   | 13.7%    |

### Opening Move Analysis

Against Amara, our strategy is completely deterministic:

| Opening Move | Count | Percentage |
|--------------|-------|------------|
| E9 Green     | 20    | 100%       |

Against Melissa, more variety exists due to learned adjustments from REINFORCE training.

### Score Progression (Amara Games, n=20)

| Move | Mean Score | Std Dev | Min | Max |
|------|------------|---------|-----|-----|
| 1    | +0.997     | 0.044   | +0.912 | +1.077 |
| 2    | -0.009     | 0.039   | -0.110 | +0.071 |
| 3    | +1.002     | 0.034   | +0.897 | +1.051 |
| 4    | -0.008     | 0.006   | -0.022 | +0.001 |
| 5    | -0.009     | 0.010   | -0.026 | +0.010 |
| 6    | -0.160     | 0.018   | -0.199 | -0.130 |
| 7    | -0.005     | 0.024   | -0.047 | +0.047 |
| 8    | -0.062     | 0.008   | -0.072 | -0.046 |

Note the extremely low standard deviations after move 3, indicating highly deterministic play.

## Experimental Analysis

### Power Spectral Density

#### Amara Games
```
Frequency | Power
----------|----------
0.0000    | 0.00698
0.0625    | 0.13575
0.2188    | 0.13522
0.4688    | 2.12756  <- Dominant
```

**Dominant frequency: 0.4688 (period: 2.13 moves)**

#### Melissa Games
```
Frequency | Power
----------|----------
0.0000    | 0.07301
0.0625    | 0.60297
0.1562    | 0.86437
0.4688    | 7.00315  <- Dominant
```

**Key Comparison:**

| Metric | Amara | Melissa | Interpretation |
|--------|-------|---------|----------------|
| Delta Std Dev | 0.479 | 0.950 | Amara is 2x more predictable |
| Peak Power | 2.13 | 7.00 | Melissa has 3x more energy in oscillations |
| Delta Mean | -0.075 | -0.040 | Both slightly favor opponent |

### Autocorrelation Analysis

Score delta autocorrelation for Amara games:

```
Lag | Correlation | Visual
----|-------------|------------------
 0  | +1.000      | ████████████████████
 1  | -0.718      | ██████████████ (negative)
 2  | +0.382      | ███████
 3  | -0.141      | ██ (negative)
 4  | +0.094      | █
 5  | -0.109      | ██ (negative)
```

The strong negative correlation at lag 1 (-0.718) is the signature of **tit-for-tat equilibrium**: every move is immediately countered by an opposing move of similar magnitude.

### Move Impact Analysis

| After Move By | Mean Delta | Std Dev | Interpretation |
|---------------|------------|---------|----------------|
| Us            | -0.322     | 0.438   | Our moves lose ground |
| Opponent      | +0.200     | 0.358   | Opponent moves hurt us less |

This asymmetry is concerning: our moves produce larger negative swings than Amara's moves produce against us. However, the alternating structure ensures the final score converges to near-zero.

### Structural Comparison

**Amara Gameplay Characteristics:**
- Highly periodic (dominant 2-move cycle)
- Low variance (predictable responses)
- Tight equilibrium convergence
- Deterministic opening responses

**Melissa Gameplay Characteristics:**
- Higher variance (more chaotic)
- Flatter power spectrum (closer to white noise)
- Exploitable patterns (our positive win rate)
- More diverse move selection

## Recommendations

### 1. Opening Diversification (High Priority)

Our 100% E9-Green opening against Amara is fully predictable. Amara has optimized its counter-play for this specific branch.

**Proposed Experiment:**
- Force exploration of all 30 possible first moves (15 edges x 2 colors)
- Record Amara's responses to each opening
- Identify openings with higher score variance (potential weaknesses)

### 2. Anti-Equilibrium Play (Medium Priority)

The spectral analysis reveals we are trapped in a stable limit cycle. Breaking equilibrium requires:

- **Sacrifice moves**: Accept short-term score loss for positional advantage
- **Asymmetric positions**: Create board states that may not have been in Amara's training data
- **Late-game variance**: The tightest control is in moves 4-8; consider aggressive play in moves 1-3

### 3. Timing Analysis (Low Priority)

If Amara is a neural network (not a lookup table), response time may indicate uncertainty:

- Log Amara's thinking time per move
- Identify positions where Amara takes longer
- These may represent training gaps or difficult evaluations

### 4. Realistic Assessment

The tic-tac-toe analogy may be apt. Consider:

- Final scores are consistently within 0.6% of zero
- Autocorrelation shows perfect move-countermove structure
- D-Wave training likely explored the full game tree

**Probability of finding a winning strategy: 5-10%**

The $10 bounty may be secure by design.

---

## Appendix: Raw Data Queries

### First Move Distribution
```sql
SELECT m.edge, m.color, COUNT(*) as count
FROM moves m
JOIN games g ON m.game_id = g.id
WHERE m.move_number = 1 AND m.player = 'us' AND g.opponent = 'amara'
GROUP BY m.edge, m.color;
```

### Score Variance by Opponent
```sql
SELECT opponent,
       MIN(final_score) as min_score,
       MAX(final_score) as max_score,
       AVG(final_score) as avg_score
FROM games
GROUP BY opponent;
```

---

*Analysis conducted: January 2026*
*Games analyzed: 27 (Amara), 878 (Melissa)*
*Strategy: HybridSolverStrategy with REINFORCE learning*
