# AlphaQ Replica Plan

Status: **stopped at stage 1** (2026-09-26): both variants failed the gate.
Owner notes live in `logs/goal_ledger.md`.

Stage 1 result (win/draw/loss leaf values, 1000 simulations, 962 held-out
middlegame decisions):

| Measure | Clone | Replica |
|---|---|---|
| Top-1 agreement | 0.63 | 0.55 |
| Top-3 agreement | 0.83 | 0.81 |
| Result-class agreement | | 0.99 |
| False alarms | | 0 of 1331 |

The gate failed. Flat values inside the draw class let the search drift away from
AlphaQ's choices, most at move 2 (clone 0.80, replica 0.42).

Variant 1b (leaf value = expected result, i.e. the learned value divided by
TIE_SCALE and clipped to [-1, 1]) also failed:

| Measure | Clone | Replica 1b |
|---|---|---|
| Top-1 agreement | 0.63 | 0.56 |
| Top-3 agreement | 0.83 | 0.82 |

Its class-agreement and false-alarm figures from that run were a metric bug:
soft mode compared floats instead of result classes. The bug is fixed.

**Conclusion.** Search over our best priors and values does not predict AlphaQ
better than the surface-feature clone, at any move number. Per the gate, stage 2
is not worth the compute.

**What remains.** Snowdrop's own information: whether AlphaQ was audited against
exact table minimax, and its weights and search settings.

## Why

By 2026-09-25, 1389 ranked games had produced no win against AlphaQ Up (917 draws,
472 losses). AlphaQ held its draw at move 14 everywhere and at moves 12 and 10 in
every position we fully probed (386 and 25 positions). It showed no error any of
our models could see in 2621 recorded decisions. A win needs an AlphaQ error at a
position we can steer to. Errors, if any, are most likely in the middlegame
(moves 2-9), where 1000 MCTS rollouts cannot reach the end of the game and its
value network decides.

Live probing costs about 35 s per game and 70-100 games per move-10 position. An
offline replica would let us search for middlegame errors with CPU time instead,
and spend live games only on confirming candidates. This follows the
adversarial-policy literature: Wang et al. 2023 (KataGo) found exploits against a
replica of the victim that transferred to the real one.

## What AlphaQ is (site bundle, 2026-09-25)

AlphaZero-style network: 9 residual blocks, 128 channels, 6.06M parameters, one
input channel, policy and value heads. It runs 1000 MCTS rollouts per move and is
deterministic (same position gives the same move). It was trained against the
site's lookup table with a draw band of 0.0005.

## The fidelity problem

A replica is useful only if it errs where AlphaQ errs. We have never observed an
AlphaQ error, so this cannot be measured directly. Fidelity is validated
indirectly, in layers, with explicit go/no-go gates. Every candidate a replica
produces is checked live before anything is claimed, so a poor replica costs CPU
time and a few games, never a false result.

## Validation data

| Data | Size (2026-09-25) | Tests |
|---|---|---|
| AlphaQ decisions (position -> move) | 2621 | Move agreement, held out by line family (position after move 4), so near-duplicates never straddle folds |
| AlphaQ decisions proven table-safe | Move 14 everywhere; move 12 in 386 positions; move 10 in 25 | False alarms: a replica move in a worse result class than AlphaQ's move at a position where AlphaQ is known to hold |
| Pre-registered live games | About 7 AlphaQ decisions per game | Out-of-sample agreement: predict AlphaQ's full reply sequence for a planned line before playing it |

## Metrics (per move number)

1. Top-1 and top-3 agreement with AlphaQ's move.
   - Baselines: best learned-table value 0.29; gradient-boosting clone
     (`tools/alphaq_clone.py`) 0.60 overall, 0.67-0.86 at moves 2-6.
2. Result-class agreement: the chosen move is in AlphaQ's win/draw/loss class.
   AlphaQ's own moves are in the learned table's best class 98% of the time.
3. Calibration: confidence (visit share) against accuracy.
4. False-alarm rate on the proven-safe decisions.
5. Whole-line accuracy: the share of games whose full AlphaQ reply sequence is
   predicted. Errors compound, so this is the honest planning measure.

## Stages and gates

### Stage 1: search-level replica (hours of work, 0 games)

- **Search:** PUCT MCTS with 1000 simulations per move and argmax visits,
  deterministic. Ties go to the higher prior.
- **Priors:** the behaviour clone, trained only on the training folds.
- **Values:** win/draw/loss class (+1/0/-1) of the learned table's minimax value
  (`tools/learned_table.py`), because AlphaQ optimises results, not raw score.
  Terminal boards use recorded table values where known.
- **Scope:** it tests whether AlphaZero-style search over our best prior and
  value reproduces AlphaQ's choices. It cannot reproduce value-network errors,
  because its values are near-exact under our table.
- **Variant 1b:** replace the exact leaf values with a fitted value approximator,
  so the replica makes approximation errors of its own. Their overlap with
  AlphaQ's behaviour is the first real signal on error-structure fidelity.
- **Gate:**
  - held-out top-1 at moves 2-9 at least equal to the clone's, on the same folds;
  - held-out top-3 at moves 2-9 of 0.85 or better;
  - false-alarm rate of 2% or less.

  If it fails, search over our models does not explain AlphaQ, and stage 2 is
  unlikely to either. Stop.

### Stage 2: network replica (days of CPU, 0 games until the gate)

- **Build:** a network of AlphaQ's shape (a reduced 3-5 block version first),
  trained by self-play on the learned table and distilled toward AlphaQ's recorded
  moves.
- **Gate:** the stage 1 gates, plus 10 pre-registered live games whose
  per-decision accuracy stays within 5 points of the held-out figure.

### Stage 3: adversary (CPU, then 1 live game per candidate)

- **Search:** against the stage-2 replica, look for our move sequences that lead
  the replica to a move that leaves the draw set under the learned table, at moves
  2-9.
- **Check each candidate live:**
  1. One game to see whether AlphaQ plays the predicted move.
  2. If it does, finish the game with endgame probing, since the table decides.
- **Stop rule:** no confirmed AlphaQ deviation in 50 candidates.

## Possible outcomes

- **A replica passes its gate and its adversary finds nothing:** fair evidence
  that AlphaQ has no exploitable middlegame blind spot.
- **Stage 1 fails:** no model we can build predicts AlphaQ well enough to aim an
  attack. What remains is Snowdrop's own information: whether AlphaQ was audited
  against exact table minimax, and where it disagrees.
