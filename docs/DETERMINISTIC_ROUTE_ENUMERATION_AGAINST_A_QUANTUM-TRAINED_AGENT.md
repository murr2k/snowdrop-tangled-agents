# Deterministic Route Enumeration Against a Quantum-Trained Agent

### Evaluator–Policy Misalignment, Oracle Solvers, and the Limits of Exploitability in *Tangled*

**Murray Kopit**
February 2026

---

## Abstract

We investigate whether a deterministic winning strategy can be constructed against **AlphaQ**, a reinforcement-learning agent trained with access to quantum computation, in the *Tangled* game. AlphaQ is stationary at inference time and adjudicated using a quantum-derived terminal lookup table.

By combining historical opponent behavior into a high-confidence oracle and enumerating deterministic game routes under that oracle, we reduce the adversarial game to a routing problem. This enables the discovery of terminal states that appear winning under simulated annealing (SA) evaluation.

Live trials, however, reveal that these routes consistently terminate in draws under the platform's quantum adjudicator. We show that this outcome is not accidental but arises from **persistent evaluator–policy misalignment**, where both AlphaQ and the exploit strategy optimise against the same approximate evaluator. SA scores undergo nonlinear compression when mapped to the platform's adjudication—a 66× reduction on the tested terminal state—rendering ε-level advantages operationally meaningless. The result formalises a limit: **determinism alone is insufficient to guarantee exploitability when the planning signal is not semantically equivalent to the adjudicator**.

---

## 1. Introduction

The central question motivating this work is deceptively simple:

> *Can a classical agent, armed only with historical game data and terminal evaluations, force a win against a quantum-trained but stationary opponent?*

This question is not merely about beating an agent. It probes a deeper issue at the boundary of reinforcement learning and quantum computation: **what guarantees does quantum-assisted training actually provide at deployment time?**

AlphaQ, developed for the *Tangled* platform, is trained using reinforcement learning with access to quantum computation and evaluated against a quantum-derived terminal adjudicator. Importantly, AlphaQ’s policy is fixed at inference time—it does not adapt, retrain, or incorporate opponent modeling online.

From a classical adversary’s perspective, this creates an apparent opportunity. In a finite, perfect-information game with a deterministic opponent, the game tree collapses. Opponent turns no longer branch; only the adversary’s moves do. In principle, one should be able to enumerate routes through this reduced tree and select a terminal state that exceeds the draw threshold.

This work demonstrates why that intuition fails—and why the failure itself is instructive.

---

## 2. Background: Tangled and AlphaQ

*Tangled* is a two-player, turn-based game played on the Petersen graph. Players alternate coloring edges either:

* **Green** (ferromagnetic coupling, ( J=-1 )),
* **Purple** (antiferromagnetic coupling, ( J=+1 )), or
* **Grey** (uncoupled, ( J=0 )).

When all edges are colored, the configuration defines an Ising Hamiltonian. The terminal state is adjudicated by evaluating quantum correlations derived from that Hamiltonian. A score exceeding a small threshold ( \varepsilon ) constitutes a win; values near zero are draws.

AlphaQ was trained using reinforcement learning with access to quantum computation during training. At inference time:

* The policy is **fixed**.
* The response to a given board state is effectively deterministic.
* No online learning or opponent adaptation occurs.

These properties were explicitly confirmed by the platform’s creator.

---

## 3. Mental Model and Research Framing

The conceptual hierarchy underlying this work is:

```
Quantum Dynamics Fidelity
        ↓
Terminal State Labels
        ↓
Reward Signal
        ↓
Policy Gradient
        ↓
Agent Behavior
```

Most prior work—including much of the published literature—stops at the **first arrow**, assuming that higher-fidelity quantum evaluation propagates cleanly through training into optimal behavior.

This project explicitly traces **all five levels**, asking whether misalignment at any layer can persist into deployment and remain exploitable.

This shift—from assuming alignment to auditing it—is the core research contribution.

---

## 4. Determinism and the Oracle Hypothesis

Analysis of over 1,400 games against AlphaQ revealed:

* Over **98%** of observed board states elicited a single, deterministic response.
* The same small set of terminal states recurred across nearly all games.
* AlphaQ’s behavior showed no evidence of stochastic exploration at inference time.

These observations motivate the **oracle hypothesis**:

> If AlphaQ’s response function is deterministic and stationary, then a sufficiently accurate oracle can predict its moves, collapsing the game tree into a directed acyclic graph.

Under this hypothesis, winning becomes a routing problem: enumerate all routes consistent with the oracle and select those that terminate in winning states.

---

## 5. Oracle Solver Design

A dedicated solver was implemented to operationalize this idea:

1. **Oracle construction**
   Historical game data was used to map board states to AlphaQ’s most frequent response, with confidence estimates.

2. **Route enumeration**
   From each possible opening, the solver enumerated all paths where:

   * The classical player branches over all legal moves.
   * AlphaQ follows the oracle deterministically (or top-(k) responses when uncertain).

3. **Terminal evaluation**
   Terminal states were scored using a precomputed lookup table derived from simulated annealing (SA), treated initially as ground truth.

The solver identified dozens of terminal states predicted to exceed the draw threshold, some by large margins.

---

## 6. Live Trials and Falsification

When the highest-confidence routes were executed live against AlphaQ, two outcomes emerged:

1. **Route determinism was confirmed**
   For well-observed states, AlphaQ followed predicted responses with perfect consistency.

2. **Predicted wins did not materialize**
   Every terminal state classified as a win by the solver was adjudicated as a **draw** on the platform, often with scores near zero.

This falsified the central operational hypothesis: **enumeration alone was insufficient to produce a win**.

---

## 7. Evaluator Non-Equivalence

The failure mode was traced to a deeper issue: **the solver and AlphaQ optimised against the same approximate evaluator, which is not semantically equivalent to the platform's adjudicator**.

* AlphaQ was trained against an SA-derived evaluator.
* The solver used a lookup table built from the same SA approximation.
* Both systems therefore agreed on which states *appeared* winning—yet the platform's quantum adjudicator did not.

Critically, the mismatch is not a simple scaling error. SA scores undergo **nonlinear compression** when mapped to the platform's adjudication: a terminal state scoring +1.985 under SA was adjudicated at +0.03 on the platform—a 66× reduction. This compression renders the formal draw threshold (ε = 0.0005 in SA score space) operationally meaningless; the operational threshold for a win on the platform is approximately +2 in website-reported scores, separated from the formal threshold by roughly four orders of magnitude.

This constitutes a recursive failure:

> A strategy designed to exploit evaluator misalignment cannot succeed if it relies on the same misaligned evaluator.

We formalize this as **persistent exploitability under evaluator–policy misalignment**:
even with perfect opponent modeling and deterministic routing, no exploit exists unless the evaluation function used for planning is semantically equivalent to the adjudicator that determines outcomes.

---

## 8. Implications

### 8.1 For Quantum-Trained Agents

Quantum-assisted training does not guarantee robustness at inference time. However, robustness may arise not from quantum optimality, but from **structural avoidance of high-margin losing states** under the true evaluator.

AlphaQ appears to have converged to a policy that restricts play to a narrow band of terminal states whose true outcomes are draws, regardless of opponent behavior. Whether this is an intrinsic property of AlphaQ's quantum-trained policy or a contingent feature of the Petersen graph's evaluation landscape remains an open interpretive question.

### 8.2 For Adversarial RL Research

This work demonstrates that:

* Determinism alone does not imply exploitability.
* Opponent modeling must be paired with **evaluator equivalence**—the planning signal must be semantically equivalent to the adjudicator, not merely correlated with it.
* Outcome calibration is opponent-conditional: a calibration curve derived from games against one agent cannot be transferred to another whose policy-evaluation coupling differs.
* Negative results can decisively rule out entire classes of attack strategies.

---

## 9. Conclusion

We set out to construct a deterministic winning route against a quantum-trained, stationary opponent. Instead, we obtained a stronger result:

> **Under evaluator non-equivalence, deterministic route enumeration cannot produce a win—even against a fully predictable opponent.**

This result reframes the challenge of beating AlphaQ. The obstacle is not hidden randomness, nor insufficient search, nor lack of opponent modeling. It is the absence of a planning signal semantically equivalent to the adjudicator itself. The formal draw threshold (ε = 0.0005) and the operational win threshold (~+2 in platform scores) are separated by approximately four orders of magnitude; no amount of routing precision can bridge a gap in evaluation fidelity.

From a research standpoint, this is not a dead end. It precisely delineates where further progress must occur: **at the level of evaluation, not policy**. The oracle is solved; the evaluator is not.

---

## Acknowledgments

The author thanks **Geordie Rose** for clarification of AlphaQ’s evaluation pipeline and for creating *Tangled* as a rare experimental platform where questions about quantum advantage, reinforcement learning, and deployment-time alignment can be meaningfully tested.
