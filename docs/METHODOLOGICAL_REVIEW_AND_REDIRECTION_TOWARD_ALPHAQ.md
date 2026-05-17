# Methodological Review and Redirection Toward AlphaQ

**Murray Kopit**
Independent Researcher
2026-05-17

---

## Abstract

This document is a critical scientific review of the current investigation
program, conducted at the inflection point following completion of the
Oracle Revision Project and the launch of Investigation 4 (exhaustive
terminal state mapping vs Melissa). It argues that Investigation 4 is
collecting data orthogonal to the stated goal of defeating AlphaQ Up, that
the calibrated oracle has structural limitations that further data cannot
address, and that the adversary model embedded in the current solver is
mismatched to AlphaQ's actual behaviour. It recommends pausing or redirecting
Investigation 4 and prioritising three AlphaQ-targeted analyses that have not
been executed and that are the binary-decision inputs for whether classical
defeat of AlphaQ is achievable at all.

---

## What is working

The empirical work to date is methodologically sound and well documented.
Several specific contributions are intellectually substantive:

- The **polarity-inversion finding** ($r = -0.396$ between SA and quantum
  terminal scores on AlphaQ-reachable boards) is a real, statistically
  significant result with clean evidence. It is the most important scientific
  output of the program so far.
- The **empirical closure paper** is honest about its negative result and
  computes confidence bounds correctly. The 95% bound at $p < 2.5\%$ for
  reachable winning terminals is a defensible statement.
- The **infrastructure** (parallel sessions, resume mechanism, SQLite
  schema, retrograde DP pipeline, MATLAB integration) is high-quality
  engineering. The DB hygiene supports rigorous post-analysis.
- The **Investigation 1 finding** (LUT internal inconsistency: 0.957
  evaluation swing in a single round) correctly diagnosed a structural flaw
  in the SA oracle and motivated the calibration work.

These are not in question. The critique that follows concerns whether the
*next* increment of effort is being applied to the right experiment.

---

## The central methodological problem

The stated goal of the program is to win a game against AlphaQ Up. The
current active investigation (Investigation 4) is mapping terminal boards
reachable from games against Melissa. These are different state spaces.

The empirical closure paper explicitly documents that AlphaQ constrains
the reachable terminal basin to $[-8.806, +0.861]$. AlphaQ is an
adversarial agent. It does not visit the Melissa-reachable subspace; that
is precisely the property that makes it strong. Discovering 9,830 winning
terminals under Melissa play tells us nothing about whether any of those
boards can be reached when AlphaQ is the opponent, and the prior evidence
suggests they cannot.

The defensible rationale for Investigation 4, as written in
`INVESTIGATION_AVENUES.md` B2, is that additional terminal coverage will
improve the website-adjudicator calibration. This is a coherent argument
only if:

1. Improved calibration translates into AlphaQ-game wins, AND
1. The improvement is achievable from Melissa data.

Both premises are weak. The calibration fit is currently $R^{2} = 0.60$ on
947 Melissa boards. The functional form of the fit (single anneal-time
parameter, fixed model class) is not guaranteed to converge to $R^{2} = 1$
even with infinite data; if the model class does not match the website's
adjudicator, additional boards reduce variance without correcting bias.

More fundamentally, even a perfectly calibrated oracle does not by itself
produce a winning move sequence against AlphaQ. It produces a value
function over board states. To convert values into wins, the solver must
search over move sequences. The search algorithm currently used (alpha-beta
minimax with MCTS rollouts) assumes a best-response opponent. AlphaQ is not
a best-response opponent; it is a fixed AlphaZero policy. This is treated
in the next section.

The current configuration of 10 parallel sessions against Melissa is
motion. It is not, in the technical sense, information gain relative to
the stated objective.

---

## Calibrated oracle: three structural concerns

### Concern 1: $R^{2} = 0.60$ is insufficient for 14-level retrograde DP

The calibrated terminal LUT has a leaf-level fit of $R^{2} = 0.60$ against
website scores. Unexplained variance per leaf is on the order of 40%. The
retrograde minimax DP propagates these leaf values upward through 14 levels.
At each level the DP selects the best (or worst) child value, which
preferentially amplifies the extremes of leaf-level noise. After 14 levels
the intermediate-grey evaluations carry noise of the same order as the value
differences they are meant to resolve.

Investigation 1 already documented this exact failure mode on the SA oracle.
The calibrated oracle is quantitatively better but qualitatively similar.
The 0.957 evaluation swing observed in Investigation 1 is the symptom; the
underlying cause (DP amplification of leaf noise) is not eliminated by
recalibration, only attenuated.

### Concern 2: anneal-time = 1.85 ns may be a model-class artefact

The recovered website parameter (anneal-time = 1.85 ns) is 22 times shorter
than the local solver default (40 ns). Two interpretations are consistent
with the data:

1. The website genuinely uses very-short-time annealing. This is physically
   plausible: D-Wave hardware does run on nanosecond scales and the anneal
   schedule is operator-controlled.
1. The fitter is finding the wrong parameter inside the wrong functional
   class. If the website's actual adjudicator differs structurally from the
   split-operator Schrödinger model (different couplings, different schedule,
   different ε floor, hardware-specific decoherence), then the best
   single-parameter fit minimises MSE within the wrong family and produces
   a numerically misleading "recovered" value.

$R^{2} = 0.60$ is consistent with either interpretation. A one-parameter
fit cannot distinguish them. Without an independent check (e.g., fitting
a richer parameterisation and observing whether the additional parameters
absorb significant variance), the 1.85 ns figure should be treated as a
proxy parameter, not a recovered physical constant.

### Concern 3: calibration is opponent-conditional and AlphaQ data was not used

Section 11 of the project README documents that opponent-specific
calibration curves are necessary: Melissa-fitted calibration mispredicts
against opponents with different noise profiles. This is the project's own
established finding.

The calibration boards used in Investigation 3 came from Melissa games.
The database contains roughly 1,500 AlphaQ games with terminal-state and
website-score pairs. **These AlphaQ pairs were not used in the calibration
fit.** AlphaQ visits a different region of the terminal basin; the
appropriate calibration for AlphaQ-targeted play is fitted on
AlphaQ-reachable terminals.

This is not a critique that requires new data collection. The fit can be
re-run against the existing AlphaQ corpus in one day.

---

## The minimax-vs-actual-policy mismatch

The hybrid solver assumes the opponent plays best-response to the current
value function. This is the standard adversarial-search assumption. It is
not the correct assumption for AlphaQ.

AlphaQ is an AlphaZero-trained agent. Its move at any board state is a
function of its trained policy network, not a best-response computation
against our oracle. The two coincide only if (a) AlphaQ's training
converged to a true Nash policy AND (b) our oracle accurately represents
the value function under which AlphaQ was trained. Neither is established.

The consequence is that the solver discards moves that minimax considers
"too risky" — moves where minimax assumes AlphaQ would punish optimally —
even when AlphaQ's actual learned policy does not contain that punishment.
These discarded moves are precisely where exploits live, if any exist.

The right formulation, given the data we have, is:

$$
a^{\ast} = \arg\max_{a} \mathbb{E}_{a^{\prime} \sim \pi_{\mathrm{AlphaQ}}(\cdot \mid s)} \left[ V(s, a, a^{\prime}) \right]
$$

where $\pi_{\mathrm{AlphaQ}}(\cdot \mid s)$ is AlphaQ's predicted move
distribution conditional on board state $s$, estimated empirically from the
1,500-game corpus, and $V(s, a, a^{\prime})$ is the calibrated value of the
position after our move $a$ and AlphaQ's response $a^{\prime}$.

This replaces the worst-case adversary in minimax with the empirical
adversary in expectation. Against a best-response opponent, this would
underperform; against AlphaQ specifically, it should outperform because it
exploits the gap between AlphaQ's actual policy and a true minimaxer.

---

## Investigation 2 should precede Investigation 4

The project roadmap lists Investigation 2 (spectral and mutual-information
analysis of AlphaQ's policy) as priority 2, with an estimated cost of one
day. It has not been executed. Investigation 4 was launched without it.

Investigation 2 is the binary-decision input that determines whether
classical defeat of AlphaQ is plausible at all:

| Outcome of Investigation 2                                | Strategic implication                                           |
|-----------------------------------------------------------|-----------------------------------------------------------------|
| MI high, response entropy uniformly low                   | AlphaQ is at or near Nash. Pivot to tensor networks (Inv. 5).   |
| Pockets of low MI or high response entropy                | Exploit candidates exist. Build predictive model and target.    |
| Behaviour switches on low-observation-count states        | AlphaQ is locally optimal on familiar states only.              |

The same methodology, applied to Amara, correctly predicted exploitability
and produced an 83.3% win rate. The cost is one day of analysis script
writing. The information value, in either direction, is sufficient to
re-plan the program.

Running Investigation 4 for 9–17 days at 10× parallelism before running
Investigation 2 for one day is a resource-allocation inversion.

---

## Probability assessment

The prior on classical defeat of AlphaQ, conditional on the evidence to
date, is low. The empirical closure paper estimates $p < 2.5\%$ with 95%
confidence that winning terminals comprise more than that fraction of
AlphaQ's reachable basin. The polarity-inversion finding indicates that
classical surrogates are not merely unhelpful but actively misleading for
AlphaQ-game outcomes.

However, the prior is not zero. Two structural facts admit the possibility
of exploit:

1. **AlphaQ training is on a finite distribution.** AlphaZero policies have
   decision boundaries. Small board perturbations can flip moves. The
   1,500-game corpus underrepresents the reachable state space; AlphaQ has
   not been observed responding to most of its reachable positions.
1. **AlphaQ exhibits 100% move consistency only on well-observed states**
   (observation count > 50). The empirical closure paper notes that AlphaQ
   "was observed changing moves on states with $\leq$ 6 observations." This
   is direct evidence that AlphaQ's policy is not uniformly deterministic
   and that low-observation states are decision-boundary candidates.

The recommended program (Investigation 2 + predictive model +
expected-value solver) is the cheapest path to either finding an exploit
or providing strong evidence that no exploit exists within
classically-reachable states. If that program concludes negatively, the
remaining path is tensor-network simulation of the website adjudicator
(Investigation 5), which is 3–6 months of work but produces a true quantum
oracle and is independent of opponent modelling.

---

## Disposition of Investigation 4

Two acceptable outcomes for the currently-running 10 parallel sessions:

1. **Pause.** The marginal value of additional Melissa terminals is low
   until the AlphaQ-targeted program either finds an exploit candidate
   (in which case Melissa data is irrelevant) or rules out classical
   exploits (in which case the program pivots to tensor networks, for
   which Melissa data is also irrelevant).
1. **Redirect to AlphaQ.** Change the launcher's `--opponent melissa`
   to `--opponent alphaq` and continue running. This grows the
   AlphaQ-specific calibration corpus by ~10× without losing the
   parallelism investment. The games will be near-uniformly losses, but
   the move-by-move data feeds directly into the predictive opponent
   model proposed below.

Continuing as currently configured is the worst of the three options.

---

## Summary of recommendations

In strict priority order:

1. Execute Investigation 2 (MI and response-entropy analysis of AlphaQ
   policy from the existing 1,500-game corpus). One day of work. Binary
   decision input.
1. Build an AlphaQ predictive policy model
   ($\pi_{\mathrm{AlphaQ}}(\cdot \mid s)$) from the same corpus. Two to
   three days of work. Required input for the expected-value solver.
1. Re-run the website-adjudicator calibration fit using AlphaQ terminal
   pairs in addition to (or instead of) Melissa pairs. One day. May
   produce a materially better oracle in AlphaQ's basin.
1. Reformulate `HybridTangledSolver.m` to maximise expected value under
   the predicted AlphaQ policy rather than minimax. One week.
1. Run 50 games with the reformulated solver against AlphaQ. Decision
   gate: any non-zero win count justifies scaling; persistent zero
   triggers the long-term pivot.
1. If decision gate fails, commit to Investigation 5 (tensor-network
   simulation) as the only remaining path with $R^{2} \rightarrow 1$.

Implementation details are in
[`PROJECT_PLAN_ALPHAQ_TARGETED_INVESTIGATION.md`](PROJECT_PLAN_ALPHAQ_TARGETED_INVESTIGATION.md).

---

## What this review does not claim

This review does not claim:

- That the calibrated oracle is useless. It is the best stable oracle the
  project has produced. It is appropriate for the use cases it was
  designed for (P2 draws against AlphaQ; targeted exploration of terminal
  basins). It is not, on its own, a defeat mechanism for AlphaQ.
- That Investigation 4 was the wrong decision when planned. The plan
  predated this review and was internally consistent with the calibration
  pipeline. The argument here is about marginal value going forward, not
  about prior decisions.
- That AlphaQ is beatable. The evidence to date is that it is not, under
  classical strategies. The proposed program is the cheapest test of
  whether that evidence generalises to all classical strategies or only to
  the ones tried so far.

The review is, in summary, about cost-per-bit of additional experiment, and
the claim is that the cost-per-bit on the proposed program is materially
lower than on the current Investigation 4 configuration.
