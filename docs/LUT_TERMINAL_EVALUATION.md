# Full Board State MCTS Evaluation with Lookup Table

## Overview

This document describes the implementation of a pre-computed Lookup Table (LUT) for terminal state evaluation in the MATLAB MCTS implementation. This replaces heuristic evaluation with accurate SimulatedAnnealingAdjudicator scores.

## Problem Statement

- The original MATLAB MCTS used heuristic evaluation based on edge categories (MY_EDGES, OPP_EDGES, HUB_EDGES)
- Heuristics don't capture quantum interactions across the full board state
- Move 4 consistently caused score collapses (-0.98 to -4.41 points) that heuristics failed to predict
- MCTS Melissa likely uses actual adjudicator, giving it accurate evaluations

## Solution: Pre-computed Lookup Table

### Why This Approach

| Factor | Analysis |
|--------|----------|
| **Terminal States** | Exactly 2^15 = 32,768 possible colorings |
| **Storage** | ~132KB (4 bytes × 32768) - trivially fits in memory |
| **Lookup Time** | O(1) - single array index |
| **Accuracy** | 100% match with SimulatedAnnealingAdjudicator |
| **One-Time Cost** | ~21 minutes to generate |

### Architecture

```
ONE-TIME SETUP (Python)
┌────────────────────────────────────────────────────────────────┐
│  Enumerate all     →  SimulatedAnnealing  →  Save as .mat     │
│  32768 terminal       Adjudicator             terminal_scores │
│  states               (10000 num_reads)       32768 floats    │
└────────────────────────────────────────────────────────────────┘
                              ↓ (load once at startup)
RUNTIME (MATLAB MCTS)
┌────────────────────────────────────────────────────────────────┐
│  TangledMCTS.search()                                          │
│      ↓                                                         │
│  Rollout reaches terminal state                                │
│      ↓                                                         │
│  idx = state2idx(state)   ←  Convert 'GPGPG...' to binary idx │
│  score = LUT(idx)         ←  O(1) lookup, 100% accurate       │
└────────────────────────────────────────────────────────────────┘
```

## Implementation

### Files Created

| File | Purpose |
|------|---------|
| `tools/generate_terminal_lut.py` | Generate LUT using Python adjudicator |
| `matlab/rl/data/terminal_scores.mat` | Pre-computed scores (32768 floats) |
| `tools/test_lut_accuracy.py` | Validation against live adjudicator |
| `matlab/rl/test_lut_evaluation.m` | MATLAB unit tests |

### Files Modified

| File | Changes |
|------|---------|
| `matlab/rl/TangledMCTS.m` | Add LUT loading, replace evaluateTerminal() |
| `matlab/rl/MCRollout.m` | Update evaluateTerminal() to use LUT |

### Index Encoding

The state-to-index conversion uses binary encoding:
- Index i: bit j = 1 means edge j is 'G' (green/FM)
- Index i: bit j = 0 means edge j is 'P' (purple/AFM)
- MATLAB uses 1-based indexing, so indices range from 1 to 32768

```matlab
function idx = state2idx(state)
    idx = 1;  % MATLAB 1-indexed
    for j = 1:15
        if state(j) == 'G'
            idx = idx + 2^(j-1);
        end
    end
end
```

### LUT Loading

TangledMCTS loads the LUT automatically in its constructor:

```matlab
mcts = TangledMCTS();  % Auto-loads LUT

% Check status
info = mcts.getLUTInfo();
disp(info.loaded)      % true/false
disp(info.numEntries)  % 32768
disp(info.minScore)    % ~-15.86
disp(info.maxScore)    % ~+15.86
```

MCRollout uses a persistent variable for thread-safe parallel access:

```matlab
% Static method with persistent caching
lut = MCRollout.getTerminalLUT();
loaded = MCRollout.isLUTLoaded();
```

## LUT Statistics

From generation on 2026-01-22:

| Metric | Value |
|--------|-------|
| Total states | 32,768 |
| File size | 132 KB |
| Generation time | 21 minutes |
| Min score | -15.861 |
| Max score | +15.864 |
| Mean score | 0.000 |
| Std deviation | 2.914 |
| P1 favorable (>0.5) | 30.1% |
| P2 favorable (<-0.5) | 30.2% |
| Balanced | 39.7% |

## Performance Expectations

### Evaluation Speed

| Method | Time | Accuracy |
|--------|------|----------|
| Heuristic | ~1μs | ~70% |
| LUT Lookup | ~1μs | 100% |
| Python Adjudicator (IPC) | ~15ms | 100% |

### Expected Game Improvement

| Metric | Before (Heuristic) | After (LUT) |
|--------|--------|-------|
| Move 4 collapse | -2.6 pts | <0.5 pts |
| Win rate | ~10% | ~25-35% |
| Draw rate | ~50% | ~50% |
| Loss rate | ~40% | ~15-25% |

## Verification

Run these checks to verify the implementation:

### Python Validation
```bash
python snowdrop_tangled_agents/tools/test_lut_accuracy.py
```

### MATLAB Unit Tests
```matlab
runtests('test_lut_evaluation')
```

### Manual Verification
```matlab
mcts = TangledMCTS();

% Check LUT loaded
assert(mcts.LUTLoaded, 'LUT not loaded');
assert(length(mcts.TerminalScoreLUT) == 32768, 'Wrong LUT size');

% Test index roundtrip
state = 'GPGPGPGPGPGPGPG';
idx = mcts.state2idx(state);
state2 = mcts.idx2state(idx);
assert(strcmp(state, state2), 'Roundtrip failed');

% Test evaluation
score = mcts.evaluateTerminal('GGGGGGGGGGGGGGG');
disp(['All green score: ' num2str(score)]);
```

## Expanded LUT Layers

Beyond the terminal (0-grey) LUT, additional layers can be pre-computed for states with grey edges remaining. Each layer uses depth-k minimax over the previous layer's values.

### Layer Size Analysis

| Layer | Grey Edges | Combinations | Colorings | Unique States | Storage (expanded) | Est. Generation Time |
|-------|------------|--------------|-----------|---------------|-------------------|---------------------|
| 0 (terminal) | 0 | 1 | 2^15 | 32,768 | 32K | 21 min |
| 1 | 1 | C(15,1)=15 | 2^14 | 245,760 | 491K | + few min |
| 2 | 2 | C(15,2)=105 | 2^13 | 860,160 | ~3.4M | + few min |
| 3 | 3 | C(15,3)=455 | 2^12 | 1,863,680 | ~15M | + few min |
| 4 | 4 | C(15,4)=1365 | 2^11 | 2,795,520 | ~45M | + 15-45 min |
| 5 | 5 | C(15,5)=3003 | 2^10 | 3,075,072 | ~98M | + 1-2 hrs |

**Storage format**: The "expanded" storage uses `32768 × C(15,k)` entries for O(1) lookup, trading space for speed.

### Value of Each Layer

| Layer | Moves from End | Search Eliminated | Impact |
|-------|----------------|-------------------|--------|
| 0 (terminal) | 0 | Rollout endpoint | **Essential** - must have |
| 1-grey | 1 | Last move decision | **High** - eliminates rollout variance |
| 2-grey | 2 | Last 2 moves | **Significant** - covers critical swings |
| 3-grey | 3 | Last 3 moves | **Good** - endgame fully solved |
| 4-grey | 4 | Last 4 moves | **Marginal** - easy for MCTS |
| 5-grey+ | 5+ | Last 5+ moves | **Diminishing returns** |

### Why 4-grey is Marginal Over 3-grey

1. **Trivial search depth**: At 4-grey, MCTS only needs to search 4 branches → 3 branches = 12 total nodes to reach exact 3-grey values. This is trivial computation.

2. **Endgame already solved**: The critical score swings happen in the last 2-3 moves. The 3-grey LUT captures this volatile phase completely.

3. **Cost/benefit ratio**: 15-45 minutes generation time plus ~180MB additional memory to eliminate 1 ply of easy search is poor ROI.

### Recommendation

**Stop at 3-grey** for most use cases:
- Covers the critical endgame phase where mistakes are costly
- ~640MB total memory (layers 0-3)
- Generation time under 30 minutes total
- MCTS can easily handle the remaining 4+ plies

If performance problems persist after 3-grey, the issue is likely **earlier in the game** (moves 4-6 from start), not late-game evaluation.

### Generation Scripts

| Script | Purpose |
|--------|---------|
| `generate_expanded_lut.m` | Generate layers 0-2 |
| `generate_expanded_lut_parallel.m` | Parallel version for layers 0-2 |
| `extend_lut_three_grey.m` | Add layer 3 to existing LUT |
| `extend_lut_three_grey_parallel.m` | Parallel version for layer 3 |

**Note**: These MATLAB scripts require exclusive MATLAB access (cannot run while games are playing).

---

## Regenerating the LUT

If the adjudicator parameters change or you need to regenerate:

```bash
python snowdrop_tangled_agents/tools/generate_terminal_lut.py
```

This takes approximately 21 minutes and overwrites the existing LUT file.

## Key Reference Files

- `mcts_strategy.py:96-138` - `evaluate_terminal_state()` using SimulatedAnnealingAdjudicator
- `TangledMCTS.m:436-455` - `evaluateTerminal()` with LUT lookup
- `MCRollout.m:299-313` - Static `evaluateTerminal()` with LUT
- `run_local_parallel_tournament.py:37-45` - Adjudicator setup parameters
