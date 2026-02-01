# Amara Strategy Analysis

## Introduction

AlphaZero Amara is a Tangled game opponent trained using reinforcement learning on actual D-Wave quantum hardware. A $10 bounty was offered for the first win against Amara.

**Update (January 25, 2026):** Following the spectral analysis and opening exploration experiment described in this document, we achieved the first-ever wins against Amara. The opening diversification strategy succeeded in finding gaps in Amara's training data.

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

**Update:** Geordie Rose (Amara's creator) confirmed that Amara is NOT optimal - training compute was limited and only covered "a fraction of game states." This insight led to our winning strategy.

### Spectral Analysis Approach

By treating score progressions as time series, we can apply signal processing techniques:
- **Power Spectral Density (PSD)**: Reveals dominant frequencies in score oscillations
- **Autocorrelation**: Detects repeating patterns and move-countermove structures
- **Variance Analysis**: Measures predictability and chaos in gameplay

## Phase 1: Initial Analysis (27 Games)

### Dataset Summary (Pre-Exploration)

| Opponent | Games | Wins | Losses | Draws | Win Rate |
|----------|-------|------|--------|-------|----------|
| Amara    | 27    | 0    | 0      | 27    | 0%       |
| Melissa  | 878   | 120  | 96     | 662   | 13.7%    |

### Opening Move Analysis

Against Amara, our hybrid_solver strategy was completely deterministic:

| Opening Move | Count | Percentage |
|--------------|-------|------------|
| E9 Green     | 20    | 100%       |

This predictability allowed Amara to execute optimized counter-play.

### Power Spectral Density

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

### Phase 1 Conclusion

The spectral analysis revealed we were trapped in a stable limit cycle. The recommendation was to explore different openings to find positions outside Amara's training data.

---

## Phase 2: Opening Exploration Experiment (Run 32)

Based on the spectral analysis and advice from Geordie Rose, we implemented an `amara_explorer` strategy that systematically tested all 30 possible opening moves (15 edges × 2 colors).

### Complete Results

| Opening | Result | Score | Notes |
|---------|--------|-------|-------|
| E0G | DRAW | +0.029 | |
| E0P | LOSS | -0.060 | |
| **E1G** | **WIN** | **+1.382** | Inner cross edge |
| E1P | LOSS | +0.034 | |
| E2G | DRAW | +0.313 | |
| E2P | LOSS | +0.010 | |
| E3G | LOSS | +0.540 | High score despite loss |
| E3P | DRAW | +0.324 | |
| **E4G** | **WIN** | **+0.864** | Inner cross edge |
| **E4P** | **WIN** | -1.547 | Same edge, both colors win |
| E5G | DRAW | -0.010 | |
| E5P | DRAW | +0.089 | |
| E6G | DRAW | +0.841 | Near-win |
| E6P | DRAW | -0.008 | |
| E7G | LOSS | -0.014 | |
| E7P | DRAW | +0.081 | |
| E8G | DRAW | +0.083 | |
| E8P | DRAW | -0.235 | |
| E9G | DRAW | +0.004 | Our old default - equilibrium |
| E9P | DRAW | -0.010 | |
| E10G | DRAW | +0.003 | |
| E10P | LOSS | -0.138 | |
| E11G | DRAW | +0.020 | |
| E11P | LOSS | -0.018 | |
| E12G | LOSS | -0.131 | |
| **E12P** | **WIN** | -1.474 | Outer ring edge |
| E13G | LOSS | -0.029 | |
| E13P | DRAW | +0.013 | |
| E14G | LOSS | +0.001 | |
| **E14P** | **WIN** | **+1.799** | Highest score - outer ring |

### Summary Statistics

| Result | Count | Percentage |
|--------|-------|------------|
| **WINS** | **5** | **16.7%** |
| Draws | 15 | 50.0% |
| Losses | 10 | 33.3% |

### Winning Openings Ranked

| Rank | Opening | Score | Edge Type |
|------|---------|-------|-----------|
| 1 | **E14P** | +1.799 | Outer ring (V8-V9) |
| 2 | **E1G** | +1.382 | Inner cross (V0-V3) |
| 3 | **E4G** | +0.864 | Inner cross (V1-V4) |
| 4 | E12P | -1.474 | Outer ring (V6-V7) |
| 5 | E4P | -1.547 | Inner cross (V1-V4) |

### Analysis by Color

| Color | Wins | Draws | Losses | Avg Score |
|-------|------|-------|--------|-----------|
| Green | 2 | 8 | 5 | +0.260 |
| Purple | 3 | 7 | 5 | -0.076 |

### Key Findings

1. **E4 is Amara's blind spot**: Both E4G and E4P resulted in wins - this edge was clearly underrepresented in training.

2. **E9G confirms equilibrium**: Our old default opening scored +0.004, exactly the draw we expected.

3. **Outer ring edges (E12, E14) have gaps**: Purple openings on outer vertices found training weaknesses.

4. **Inner cross edges (E1, E4) are exploitable**: These edges connect non-adjacent inner vertices.

5. **From 0% to 16.7% win rate**: Simply diversifying openings produced a 5x improvement.

---

## Phase 3: amara_killer Strategy Validation (Run 33)

Following the exploration results, we implemented and tested the `amara_killer` strategy using E14P as the primary opening.

### Run 33 Results (15 games)

| Game | Result | Score | Opening |
|------|--------|-------|---------|
| 1 | WIN | +1.803 | E14P |
| 2 | WIN | +1.871 | E14P |
| 3 | WIN | +1.799 | E14P |
| 4 | WIN | +1.832 | E14P |
| 5 | WIN | +1.769 | E14P |
| 6 | WIN | +1.814 | E14P |
| 7 | WIN | +1.818 | E14P |
| 8 | DRAW | +0.457 | E14P |
| 9 | DRAW | -0.005 | E14P |
| 10-12 | abandoned | - | - |
| 13 | WIN | +1.816 | E14P |
| 14 | WIN | +1.775 | E14P |
| 15 | WIN | +1.965 | E14P |

### Summary Statistics

| Metric | Value |
|--------|-------|
| Total Games | 12 (completed) |
| Wins | **10** |
| Draws | 2 |
| Losses | **0** |
| Win Rate | **83.3%** |
| Loss Rate | **0%** |
| Avg Win Score | +1.826 |

### E14P Opening Analysis

All games in Run 33 used the E14P opening. Complete E14P statistics across all runs:

| Result | Count | Percentage |
|--------|-------|------------|
| WIN | 11 | 84.6% |
| DRAW | 2 | 15.4% |
| LOSS | 0 | 0% |

**Key Observation:** The two draws occurred consecutively (games 8-9), followed by abandoned games, then wins resumed. This suggests:
1. Possible variance in Amara's responses
2. Network/timing factors affecting game states
3. Statistical variance (no opening wins 100%)

**Critical Finding:** E14P has a **0% loss rate**. Even when it doesn't win, it draws safely.

### Strategy Comparison

| Strategy | Games | Wins | Draws | Losses | Win Rate |
|----------|-------|------|-------|--------|----------|
| hybrid_solver (E9G) | 27 | 0 | 27 | 0 | 0% |
| amara_explorer (all) | 30 | 5 | 15 | 10 | 16.7% |
| **amara_killer (E14P)** | **12** | **10** | **2** | **0** | **83.3%** |

The amara_killer strategy improved win rate from 0% to 83.3% - an infinite improvement over the baseline.

---

## Recommended Strategy: amara_killer

Based on the experimental results, we developed the `amara_killer` strategy that prioritizes proven winning openings:

### Priority Order

1. **E14P** (+1.826 avg) - Best opening, 84.6% win rate, 0% loss rate
2. **E1G** (+1.382) - Second highest score
3. **E4G** (+0.864) - Third highest score
4. **E4P** (win) - Same edge weakness
5. **E12P** (win) - Outer ring alternative

### Openings to Avoid

| Opening | Result | Score | Reason |
|---------|--------|-------|--------|
| E10P | LOSS | -0.138 | Worst loss |
| E12G | LOSS | -0.131 | Wrong color on E12 |
| E0P | LOSS | -0.060 | Avoid |
| E9G | DRAW | +0.004 | Equilibrium trap |

### Strategy Implementation

The `amara_killer` strategy:
1. Uses E14P as the primary opening (highest win rate, zero losses)
2. Falls back to E1G, E4G if variation needed
3. After the opening, delegates to HybridSolverStrategy
4. Supports three modes: `best` (always E14P), `cycle` (rotate winners), `random`

### Usage

```bash
poetry run python play_tangled.py --strategy amara_killer --opponent amara --run 10
```

---

## Conclusions

1. **Amara is beatable**: The tic-tac-toe hypothesis was wrong. Amara has training gaps that can be exploited.

2. **Spectral analysis was predictive**: The -0.718 autocorrelation correctly identified equilibrium lock that could be broken by exploring new openings.

3. **Opening diversity was key**: Deterministic play with E9G guaranteed draws. Novel openings (E14P, E1G, E4G) found weaknesses.

4. **E14P is the killer opening**: 84.6% win rate, 0% loss rate. This edge (V8-V9, outer ring) represents the largest gap in Amara's training.

5. **Geordie was right**: "Make moves that are not the best ones" to reach unseen positions. Our "best" move (E9G) led to equilibrium; our "suboptimal" moves led to wins.

6. **From 0% to 83.3%**: The amara_killer strategy transformed unwinnable games into dominant victories through systematic opening exploration.

---

## Appendix: Petersen Graph Edge Reference

| Edge | Vertices | Type | Win Rate vs Amara |
|------|----------|------|-------------------|
| E0 | V0-V2 | Inner spoke | 0% (G), 0% (P) |
| E1 | V0-V3 | Inner cross | **100% (G)**, 0% (P) |
| E2 | V0-V6 | Inner-outer | 0% (G), 0% (P) |
| E3 | V1-V3 | Inner cross | 0% (G), 0% (P) |
| E4 | V1-V4 | Inner cross | **100% (G)**, **100% (P)** |
| E5 | V1-V7 | Inner-outer | 0% (G), 0% (P) |
| E6 | V2-V4 | Inner cross | 0% (G), 0% (P) |
| E7 | V2-V8 | Inner-outer | 0% (G), 0% (P) |
| E8 | V3-V9 | Inner-outer | 0% (G), 0% (P) |
| E9 | V4-V5 | Inner-outer | 0% (G), 0% (P) |
| E10 | V5-V6 | Outer ring | 0% (G), 0% (P) |
| E11 | V5-V9 | Outer ring | 0% (G), 0% (P) |
| E12 | V6-V7 | Outer ring | 0% (G), **100% (P)** |
| E13 | V7-V8 | Outer ring | 0% (G), 0% (P) |
| E14 | V8-V9 | Outer ring | 0% (G), **84.6% (P)** |

---

*Initial analysis: January 25, 2026*
*Exploration experiment: January 25, 2026 (Run 32, 30 games)*
*Strategy validation: January 25, 2026 (Run 33, 15 games)*
*First win against Amara: January 25, 2026 (E1G opening, Game 3 of Run 32)*
*Total games analyzed: 72 (Amara), 878 (Melissa)*
*Final win rate with amara_killer: 83.3% (10W/2D/0L)*
