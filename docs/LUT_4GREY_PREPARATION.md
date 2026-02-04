# 4-Grey LUT Generation - Preparation Guide

## Overview

You're about to calculate the 4-grey layer of the lookup table, extending beyond the current 3-grey implementation. This will add **~45 million entries** to your LUT.

## Current LUT Status

| Layer | States | Storage | Status |
|-------|--------|---------|--------|
| 0-grey (terminal) | 32,768 | 131 KB | ✓ Complete |
| 1-grey | 491,520 | 1.9 MB | ✓ Complete |
| 2-grey | 3,440,640 | 13.1 MB | ✓ Complete |
| 3-grey | 14,909,440 | 56.8 MB | ✓ Complete |
| **4-grey** | **44,748,800** | **170 MB** | **← Next** |

**Total after 4-grey:** ~242 MB (from current ~72 MB)

## Is 4-Grey Worth It?

### ⚠️ Cost-Benefit Analysis

According to your own documentation (LUT_TERMINAL_EVALUATION.md):

> **Why 4-grey is Marginal Over 3-grey**
>
> 1. At 4-grey, MCTS only needs to search 4×3 = 12 nodes to reach exact 3-grey values
> 2. The critical score swings happen in the last 2-3 moves (already covered by 3-grey)
> 3. 15-45 minutes generation time + 180MB memory to eliminate 1 ply of easy search is poor ROI

### When to Generate 4-Grey

Only proceed if:
- ✓ You've confirmed 3-grey isn't sufficient
- ✓ You have profiling evidence that 4-ply from endgame is a bottleneck
- ✓ Your MCTS search time budget is extremely limited (<100 iterations)
- ✓ You have spare memory (≥1GB RAM available)

### Alternative: Deeper MCTS Instead

Consider increasing MCTS iterations by 2-3x instead. This:
- Costs zero storage
- Improves the entire game tree, not just the last 4 moves
- Takes no generation time

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB free | 8 GB free |
| CPU Cores | 4 | 8+ |
| Disk Space | 300 MB | 1 GB |
| GPU | Not used | Not used |

### Software

- MATLAB R2022a or later
- Existing LUT with 3-grey data (`expanded_lut.mat`)
- Sufficient MATLAB license for long-running computations

## Pre-Generation Checklist

### 1. Verify Current LUT Status

```matlab
cd snowdrop_tangled_agents/matlab/rl
lut = ExpandedLUT();
info = lut.getInfo();

% Check what you have
assert(info.hasThreeGreyData, '3-grey data required!');
disp(['Current version: ' info.version]);
disp(['Total entries: ' num2str(info.totalEntries)]);
```

Expected output:
```
Current version: 2.0
Total entries: 18874368
```

### 2. Check Available Memory

```matlab
memInfo = memory;
availableGB = memInfo.MemAvailableAllArrays / 1024^3;
requiredGB = 0.17;  % 170 MB for 4-grey scores

fprintf('Available: %.2f GB\n', availableGB);
fprintf('Required:  %.2f GB\n', requiredGB);
fprintf('Safety factor: %.1fx\n', availableGB / requiredGB);
```

You want safety factor **>= 3x** (at least 0.5 GB free).

### 3. Close Other MATLAB Sessions

The generation script needs exclusive access to MATLAB:

```bash
# Check for running MATLAB processes
tasklist | findstr MATLAB

# If any are running (besides your current session), close them
```

### 4. Back Up Current LUT

```matlab
cd snowdrop_tangled_agents/matlab/rl/data
copyfile('expanded_lut.mat', 'expanded_lut_backup_3grey.mat');
```

### 5. Check Disk Space

```bash
# Windows PowerShell
Get-PSDrive C | Select-Object Used,Free

# Should show at least 500 MB free
```

## Generation Process

### Step 1: Start Generation

```matlab
cd snowdrop_tangled_agents/matlab/rl
extend_lut_four_grey_fast();
```

### Step 2: Monitor Progress

The script will display progress every 5 seconds:

```
╔════════════════════════════════════════════════════════════╗
║   FAST 4-GREY LUT EXTENSION (Vectorized)                  ║
╚════════════════════════════════════════════════════════════╝

[1/4] Loading existing LUT...
    Loaded 32768 terminal scores
    Loaded 14909440 3-grey scores

[2/4] Building 3-grey index map...
    Built index for 455 triples

[3/4] Generating 4-grey states...
    Target: 44748800 states (44.7 million)
    Quads: 1365
    Expected time: 15-45 minutes

    Memory required: 170.3 MB
    Memory available: 4096.0 MB
    Processing in 16 chunks of 2048 base states...

    Progress: 2048/32768 (6.2%) - 15234 states/sec - ETA: 45 min
    Progress: 4096/32768 (12.5%) - 15876 states/sec - ETA: 42 min
    ...
```

### Step 3: Estimated Timeline

| Hardware | Time Estimate |
|----------|---------------|
| 8-core CPU, 16GB RAM | 15-20 minutes |
| 4-core CPU, 8GB RAM | 30-40 minutes |
| 2-core CPU, 4GB RAM | 60-90 minutes |

**Processing rate:** ~10,000-20,000 states/second

### Step 4: Verify Completion

After completion, you'll see:

```
    Completed 44748800 states in 18.3 minutes
    Rate: 40731 states/sec

[4/4] Verifying and saving...
    4-grey statistics:
      Min score:  -15.8640
      Max score:  +15.8640
      Mean score: +0.0012
      Std dev:    2.9138
    Saving extended LUT...
    Saved to: C:\...\expanded_lut.mat
    Total entries: 63623168 (63.6 million)
    File size: 242.3 MB

╔════════════════════════════════════════════════════════════╗
║   4-GREY LUT GENERATION COMPLETE                           ║
╚════════════════════════════════════════════════════════════╝
```

## Post-Generation Verification

### 1. Test LUT Loading

```matlab
clear all
lut = ExpandedLUT();
info = lut.getInfo();

% Verify 4-grey loaded
assert(info.hasFourGreyData, '4-grey data not loaded!');
assert(info.fourGreyCount == 44748800, 'Wrong 4-grey count!');
disp('✓ 4-grey LUT loaded successfully');
```

### 2. Test Lookups

```matlab
% Test 4-grey lookup
state = 'GGGGGGGGGGG----';  % 4 grey edges
score = lut.evaluate(state);
fprintf('4-grey test score: %.4f\n', score);

% Should return instantly (O(1) lookup)
tic;
for i = 1:1000
    score = lut.evaluate(state);
end
elapsed = toc;
fprintf('1000 lookups: %.2f ms (%.2f us each)\n', elapsed*1000, elapsed*1e6/1000);
```

Expected: <1 microsecond per lookup

### 3. Compare to 3-Grey Evaluation

```matlab
% Generate test states with 4-5 grey edges
testStates = {
    'GGGGGGGGGGG----',  % 4 grey
    'GGGGGGGGGG-----',  % 5 grey
    'GGGGGGGGG------',  % 6 grey
};

for i = 1:length(testStates)
    state = testStates{i};
    numGrey = sum(state == '-');

    score = lut.evaluate(state);
    fprintf('State %d (%d grey): %.4f\n', i, numGrey, score);
end
```

States with 4 grey should return different scores than 5+grey (which use heuristics).

## Troubleshooting

### Out of Memory Error

```matlab
Error using zeros
Requested array exceeds maximum array size preference.
```

**Solution:**
1. Close other programs
2. Increase MATLAB memory limit: Preferences → General → Java Heap Memory
3. Run on a machine with more RAM
4. Use 32-bit chunking in the script (edit `chunkSize = 1024`)

### Slow Performance (<5000 states/sec)

**Causes:**
- CPU thermal throttling
- Background processes
- Insufficient RAM (swapping to disk)

**Solutions:**
- Close browser, IDE, other heavy applications
- Run overnight when system is idle
- Check Task Manager for CPU/RAM usage

### Incorrect Scores

```matlab
assert(abs(mean(fourGreyScores)) < 0.1, 'Mean should be near zero!');
```

If this fails, regenerate from scratch:

```matlab
copyfile('expanded_lut_backup_3grey.mat', 'expanded_lut.mat');
extend_lut_four_grey_fast();
```

## Performance Impact

### MCTS Speedup

Expected MCTS performance improvements:

| Scenario | Before 4-Grey | After 4-Grey | Improvement |
|----------|--------------|--------------|-------------|
| Endgame (4 moves left) | 1000 rollouts | 1000 rollouts | None (MCTS already fast) |
| Mid-game (8 moves left) | 1000 rollouts | 1000 rollouts | <1% (minimal) |
| Opening (12+ moves left) | 1000 rollouts | 1000 rollouts | <0.1% (negligible) |

### Memory Usage

| Component | Before | After | Delta |
|-----------|--------|-------|-------|
| MATLAB process | ~500 MB | ~700 MB | +200 MB |
| LUT file size | 72 MB | 242 MB | +170 MB |

### When You'll See Benefits

4-grey LUT helps in scenarios where:
- **Shallow MCTS**: Using <500 iterations per move
- **Time pressure**: <1 second per move
- **Batch evaluation**: Evaluating many 4-grey positions

For most use cases, **the 3-grey LUT is sufficient**.

## Alternative: Generate 5-Grey Instead?

**DON'T.**

5-grey would be:
- **Size:** 98M entries, ~373 MB
- **Time:** 2-3 hours
- **Benefit:** Essentially zero over 4-grey
- **Cost:** Memory waste

The 4-grey → 5-grey benefit is even smaller than 3-grey → 4-grey.

## Summary

**Before you proceed:**

1. Confirm you actually need 4-grey (profile your MCTS first)
2. Check you have ≥500 MB free memory
3. Back up your current LUT
4. Allow 15-45 minutes for generation
5. Verify completion with test suite

**Recommendation:** Try optimizing your MCTS parameters before generating 4-grey. The 3-grey LUT should be sufficient for most competitive play.

---

## Quick Start Commands

```matlab
% Check current status
cd snowdrop_tangled_agents/matlab/rl
lut = ExpandedLUT();
info = lut.getInfo()

% Back up
cd data
copyfile('expanded_lut.mat', 'expanded_lut_backup_3grey.mat');

% Generate 4-grey
cd ..
extend_lut_four_grey_fast();

% Verify
lut = ExpandedLUT();
assert(lut.HasFourGreyData, '4-grey not loaded!');
disp('SUCCESS: 4-grey LUT ready');
```

Ready to proceed? Run `extend_lut_four_grey_fast()` in MATLAB.
