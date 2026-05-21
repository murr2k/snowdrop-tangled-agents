# Switchback Move Selection

## Introduction

The Tangled game is played on the Petersen graph: two players take turns coloring
edges green (ferromagnetic) or purple (antiferromagnetic). When all 15 edges are
colored, a quantum adjudicator — a D-Wave quantum annealer — evaluates the result.
The adjudicator rewards configurations that minimise frustration across the graph's
12 five-cycles: a cycle is *satisfied* if it has an even number of purple edges, and
*frustrated* if it has an odd number. Because frustrated cycles cannot be simultaneously
satisfied in a classical Ising model, the quantum annealer finds the global ground
state, which classical heuristics cannot reliably predict.

The standard approach — using a calibration oracle trained on past games — breaks
down on heavily frustrated boards. On those boards the oracle's gradient collapses
($R^2 \approx 0$), so following it is no better than guessing. The switchback
algorithm addresses this by abandoning the oracle entirely and selecting moves
based purely on the graph's structural geometry.

The key insight is that the quantum score is invariant under the graph's automorphism
group. This means structurally equivalent board states receive the same score. The
switchback algorithm exploits this by choosing moves that keep the board in a region
of high symmetry — spreading coloring evenly across the nine symmetry-equivalent
edge classes — while also avoiding locking five-cycles into frustrated states
prematurely. Rather than climbing directly toward a score target (which fails when
the gradient is broken), it moves *laterally* along the constraint surface, preserving
future options. The name comes from the mountain-trail technique of traversing
sideways across a steep slope to gain altitude safely rather than attempting a
direct ascent.

The algorithm requires no search tree, no rollouts, and no oracle. Each move is
chosen in a single greedy pass over the available edges: try every candidate move,
score the resulting board state using four structural features, pick the best.

---

## Mathematical Description

---

## 1. Setup

**Graph.** The Petersen graph $G = (V, E)$ has $|V| = 10$ vertices and $|E| = 15$ edges. Player nodes are fixed: $p_1 = 5$, $p_2 = 7$.

**State space.** A game state is a partial coloring $\sigma : E \to \{G,\, P,\, \emptyset\}$, where $G$ = ferromagnetic, $P$ = antiferromagnetic, $\emptyset$ = uncolored. The full state space is $\Sigma = \{G, P, \emptyset\}^{15}$.

**Five-cycles.** The Petersen graph has exactly 12 five-cycles, denoted $\mathcal{C} = \{C_1, \ldots, C_{12}\}$, each a set of 5 edges. These are fixed graph-theoretic constants.

**Edge orbits.** The stabilizer $\text{Stab}(p_1, p_2) \leq \text{Aut}(G)$ — the subgroup of automorphisms fixing the ordered pair $(5, 7)$ — has order 2 and partitions the 15 edges into 9 orbits:

$$\mathcal{O} = \{O_1, \ldots, O_9\}$$

Six orbits are swap pairs $|O_k| = 2$; three are fixed points $|O_k| = 1$.

---

## 2. Cycle Classification

For a cycle $C \in \mathcal{C}$ and state $\sigma$:

**Locked:** $C$ is *locked* under $\sigma$ iff every edge is colored:

$$\text{locked}(C, \sigma) \iff \forall\, e \in C:\; \sigma(e) \neq \emptyset$$

**Parity of a locked cycle:** The antiferromagnetic parity of $C$ is

$$\pi(C, \sigma) = \bigl|\{e \in C : \sigma(e) = P\}\bigr| \bmod 2$$

**Satisfied:** A locked cycle with even parity — all spins can be mutually satisfied in the Ising sense:

$$\text{sat}(C, \sigma) \iff \text{locked}(C, \sigma) \;\wedge\; \pi(C, \sigma) = 0$$

**Frustrated:** A locked cycle with odd parity — at least one frustrated bond is unavoidable:

$$\text{frust}(C, \sigma) \iff \text{locked}(C, \sigma) \;\wedge\; \pi(C, \sigma) = 1$$

**Undecided:** Not yet locked — still flexible:

$$\text{undec}(C, \sigma) \iff \exists\, e \in C:\; \sigma(e) = \emptyset$$

These three cases partition $\mathcal{C}$, so $\text{sat} + \text{frust} + \text{undec} = 12$ at every state.

---

## 3. Feature Functions

Four scalar functions on $\Sigma$:

$$f_1(\sigma) = \bigl|\{C \in \mathcal{C} : \text{sat}(C, \sigma)\}\bigr| \tag{satisfied count}$$

$$f_2(\sigma) = \bigl|\{C \in \mathcal{C} : \text{frust}(C, \sigma)\}\bigr| \tag{frustrated count}$$

$$f_3(\sigma) = \bigl|\{C \in \mathcal{C} : \text{undec}(C, \sigma)\}\bigr| = 12 - f_1(\sigma) - f_2(\sigma) \tag{undecided count}$$

For $f_4$, define per-orbit quantities for each $O_k \in \mathcal{O}$:

$$\text{cov}(k, \sigma) = \bigl|\{e \in O_k : \sigma(e) \neq \emptyset\}\bigr| \tag{colored edges in orbit $k$}$$

$$\text{net}(k, \sigma) = \bigl|\{e \in O_k : \sigma(e) = G\}\bigr| - \bigl|\{e \in O_k : \sigma(e) = P\}\bigr| \tag{G minus P in orbit $k$}$$

Then:

$$f_4(\sigma) = -\Bigl[\,\text{std}_{k}\bigl(\text{cov}(k, \sigma)\bigr) \;+\; \text{std}_{k}\bigl(|\text{net}(k, \sigma)|\bigr)\Bigr] \tag{orbit balance}$$

where $\text{std}_k$ denotes the standard deviation over $k = 1, \ldots, 9$. The negative sign makes $f_4$ a reward — less spread across orbits yields a higher score.

---

## 4. Switchback Score

The score of a state $\sigma$ is the linear form:

$$\boxed{S(\sigma) = w_1 f_1(\sigma) + w_2 f_2(\sigma) + w_3 f_3(\sigma) + w_4 f_4(\sigma)}$$

with weight vector $\mathbf{w} = (+1,\; -2,\; +0.3,\; +1)$.

---

## 5. Move Selection

Let $E_\emptyset(\sigma) = \{e \in E : \sigma(e) = \emptyset\}$ be the set of uncolored edges at state $\sigma$. The switchback move is:

$$(e^*, c^*) = \underset{e \in E_\emptyset(\sigma),\; c \in \{G, P\}}{\arg\max}\; S\bigl(\sigma[e \mapsto c]\bigr)$$

where $\sigma[e \mapsto c]$ denotes $\sigma$ with edge $e$ assigned color $c$. Ties are broken by edge index (lowest first), color $G$ before $P$. The search is a greedy 1-step lookahead — no rollouts, no search tree.

---

## 5.1 Forced Opening Override

The switchback score is invariant under the graph's symmetry group. On an empty board every candidate move produces an identical score (the board has zero covered edges, zero locked cycles, and perfect orbit balance regardless of which edge is chosen). Tie-breaking therefore always resolves to edge 0, color $G$ — an empirically poor opening.

To handle this, move selection is extended with a **forced opening override map**:

$$M : \mathbb{N} \to E \times \{G, P\}$$

a finite partial function from grey-count to forced move. The complete selection rule becomes:

$$(e^*, c^*) = \begin{cases} M\!\left(|E_\emptyset(\sigma)|\right) & \text{if } |E_\emptyset(\sigma)| \in \operatorname{dom}(M) \\ \displaystyle\underset{e \in E_\emptyset(\sigma),\; c \in \{G, P\}}{\arg\max}\; S\bigl(\sigma[e \mapsto c]\bigr) & \text{otherwise} \end{cases}$$

**Empirically validated configuration.** Playing against AlphaQ as Player 1, the map

$$M = \bigl\{15 \mapsto (7,\; G)\bigr\}$$

forces edge 7 green on the first move (all 15 edges grey) and lets switchback govern all subsequent moves. This yields approximately 96% draws. Without the override, the default tie-break selects edge 0 green, which yields approximately 92% losses. Every subsequent move is identical; the opening is the only difference.

---

## 6. Interpretation of the Weights

| Term | Weight | Rationale |
|---|---|---|
| Satisfied cycles $f_1$ | $+1$ | Locking a cycle in a satisfied state is good — it contributes positively to the D-Wave Ising ground state |
| Frustrated cycles $f_2$ | $-2$ | Locking a frustrated cycle is worse than missing a satisfied one; the doubled penalty reflects irreversibility |
| Undecided cycles $f_3$ | $+0.3$ | A small reward for keeping cycles flexible — the "switchback" character, preserving optionality rather than committing |
| Orbit balance $f_4$ | $+1$ | Encourages moves that spread coloring evenly across structural orbits and keep G/P balanced within each orbit; the symmetry-preservation term |

---

## 7. The Core Idea

The calibration oracle gradient is unreliable on frustrated boards — its $R^2$ collapses toward zero precisely on the configurations that matter most. Rather than following a broken gradient, the switchback score moves *laterally along the constraint surface*: it avoids premature frustration, keeps structural options open ($f_3$), and preserves the graph's automorphic symmetry ($f_4$).

The name "switchback" refers to the mountain-trail analogy — making lateral progress to maintain altitude rather than ascending a cliff face directly.

---

## 8. Pseudocode

The following pseudocode is language-agnostic. Types are annotated for clarity but
are not part of any specific language's syntax. `state` throughout is a sequence of
15 symbols, one per edge, each drawn from `{G, P, GREY}`.

### Constants

```
CYCLES : list of 12 lists, each containing 5 edge indices (integers 0–14)

    [ [0,1,3,4,6], [0,1,7,8,14], [0,2,6,9,10], [0,2,7,12,13],
      [1,2,3,5,12], [1,2,8,10,11], [3,4,8,9,11], [3,5,8,13,14],
      [4,5,6,7,13], [4,5,9,10,12], [6,7,9,11,14], [10,11,12,13,14] ]

ORBIT : list of 15 integers (orbit id for each edge, ids 0–8)

    [ 0,0, 1, 2,3, 4, 5, 2, 5, 6, 7, 6, 8, 4, 3 ]
      ^0,1  ^2  ^3,4  ^5  ^6  ^7  ^8  ^9  ^10 ^11 ^12 ^13 ^14

WEIGHTS : (w1=+1.0, w2=−2.0, w3=+0.3, w4=+1.0)
NUM_ORBITS : 9

MOVE_OVERRIDES : map from grey_count (integer) to (edge_index, color)
    // Empirically validated against AlphaQ as Player 1:
    { 15 → (7, G) }   // force E7G on empty board; let switchback govern all other moves
```

---

### Function: cycle_features(state) → (sat, frust, undec)

Classifies each of the 12 five-cycles as satisfied, frustrated, or undecided.

```
function cycle_features(state):
    sat   ← 0
    frust ← 0
    undec ← 0

    for each cycle in CYCLES:
        has_grey ← false
        p_count  ← 0

        for each edge_index in cycle:
            if state[edge_index] == GREY:
                has_grey ← true
                break
            if state[edge_index] == P:
                p_count ← p_count + 1

        if has_grey:
            undec ← undec + 1
        else if p_count mod 2 == 0:
            sat   ← sat + 1
        else:
            frust ← frust + 1

    return (sat, frust, undec)
```

---

### Function: orbit_balance(state) → real number

Penalises uneven distribution of colored edges across the nine orbit classes,
and uneven G/P balance within each orbit. Returns a non-positive number;
values closer to zero are better.

```
function orbit_balance(state):
    cov[0..8] ← all zeros    // colored-edge count per orbit
    net[0..8] ← all zeros    // (G count − P count) per orbit

    for edge_index from 0 to 14:
        color ← state[edge_index]
        orbit ← ORBIT[edge_index]

        if color == G:
            cov[orbit] ← cov[orbit] + 1
            net[orbit] ← net[orbit] + 1
        else if color == P:
            cov[orbit] ← cov[orbit] + 1
            net[orbit] ← net[orbit] − 1

    abs_net[0..8] ← absolute value of each entry in net

    std_cov ← standard_deviation(cov)        // std over 9 values
    std_net ← standard_deviation(abs_net)    // std over 9 values

    return −(std_cov + std_net)
```

---

### Function: score(state) → real number

Combines the four structural features into a single scalar.

```
function score(state):
    (sat, frust, undec) ← cycle_features(state)
    balance             ← orbit_balance(state)

    return  WEIGHTS.w1 * sat    +
            WEIGHTS.w2 * frust  +
            WEIGHTS.w3 * undec  +
            WEIGHTS.w4 * balance
```

---

### Function: best_move(state, overrides) → (edge_index, color)

Checks the override map first. If the current grey count has a forced move,
returns it immediately. Otherwise runs a greedy one-step lookahead: tries
every (uncolored edge, color) pair, scores the resulting state, and returns
the pair that scores highest. No search tree, no rollouts.

```
function best_move(state, overrides):
    grey_count ← count of GREY symbols in state

    // Forced opening (or any forced move at a specific grey count).
    if grey_count is a key in overrides:
        return overrides[grey_count]          // e.g. (7, G) on empty board

    // Greedy one-step lookahead.
    best_score ← −infinity
    best_edge  ← none
    best_color ← none

    for edge_index from 0 to 14:
        if state[edge_index] != GREY:
            continue                          // skip already-colored edges

        for color in [G, P]:
            candidate ← copy of state
            candidate[edge_index] ← color    // hypothetically apply the move

            s ← score(candidate)

            if s > best_score:
                best_score ← s
                best_edge  ← edge_index
                best_color ← color

            // Ties broken implicitly: lower edge index wins (loop order),
            // G wins over P (G is tried before P in the inner loop).

    return (best_edge, best_color)
```

---

### Complexity

| Step | Work per call |
|---|---|
| Override check | 1 hash lookup |
| `cycle_features` | 12 cycles × 5 edge lookups = 60 comparisons |
| `orbit_balance` | 15 edge lookups + 9-element std twice |
| `score` | 4 multiplications + 3 additions |
| `best_move` outer loop | at most 15 edges × 2 colors = 30 calls to `score` |
| **Total per move** | ~1,800 arithmetic operations (override: O(1)) |

There are no recursive calls, no heap allocations beyond the single candidate
copy, and no dependence on game history. Move selection is effectively
instantaneous on any modern processor.

---

## 9. Empirical Results and Theoretical Analysis

### 9.1 Observed Behavior

Running the switchback algorithm with the forced E7G opening against AlphaQ
(D-Wave hardware lookup table adjudicator) yields approximately **96% draws**
across 500 games. No wins have been observed. This replicates the result from
earlier MATLAB-based switchback runs and confirms that the draw-seeking
behavior is structural rather than implementation-specific.

The opening choice is decisive. Removing the E7G override and allowing
tie-breaking to select E0G drops the draw rate to approximately 8% (92%
losses). Every move from position 2 onward uses identical switchback scoring.
The opening alone determines which basin the game enters.

### 9.2 The Nash Equilibrium Hypothesis

In a finite, perfect-information, two-player zero-sum game, the Nash
equilibrium is computed by backward induction (minimax). For the Tangled game
on the Petersen graph, the minimax tree has depth 15 and at most
$2^{15} = 32{,}768$ leaf nodes — trivially exhaustible given a complete
terminal evaluator.

AlphaQ holds exactly such an evaluator: a lookup table built by running every
terminal state on D-Wave hardware. If AlphaQ performs minimax search over the
game tree using this LUT as leaf values, it plays a Nash equilibrium strategy
and is, in principle, unbeatable.

Under Nash play in a zero-sum game, the root position has a unique value $V^*$:

| $V^*$ | Prediction |
|---|---|
| $> 0$ | P1 has a forcing win; no P2 strategy can prevent it |
| $= 0$ | Optimal play by both sides draws; P1 deviations can be punished with losses |
| $< 0$ | AlphaQ wins regardless of P1's strategy |

The empirical data is consistent with $V^* = 0$ (draw is the Nash value):

- Switchback + E7G achieves the Nash value (draw) by keeping P1 on the
  equilibrium path.
- E0G falls below the equilibrium path; AlphaQ punishes the deviation with wins.
- The ceiling is exactly draw — no win has ever been observed — consistent
  with a hard game-theoretic barrier rather than unexplored strategy space.

If this interpretation is correct, the $10{,}000 AlphaQ Up bounty is
**mathematically unachievable**: P1 wins do not exist in the minimax game
tree under quantum adjudication. AlphaQ is not playing aggressively or
"trying to win" — it is simply guaranteeing the Nash value from every
position. When P1 deviates (e.g., by opening E0G), the Nash value of that
subposition may already be negative for P1, and AlphaQ collects it without
effort.

### 9.3 The Minimum-Energy Equilibrium Conjecture

The switchback score function is a direct proxy for minimizing Ising cycle
frustration. It explicitly penalises frustrated locked cycles ($f_2$), rewards
satisfied locked cycles ($f_1$), and preserves flexibility ($f_3$). These are
exactly the properties that minimize the energy of the final configuration
evaluated by the D-Wave annealer.

This suggests a deeper connection.

**Conjecture (Minimum-Energy Equilibrium).** *For two-player edge-coloring
games on a graph $G$ where the payoff function is the ground-state energy of
the resulting Ising model under quantum annealing, the Nash equilibrium
terminal state is the minimum-frustration configuration reachable under
optimal play by both sides. The switchback algorithm, by directly optimizing
structural frustration along the game path, converges on this equilibrium
from the P1 side.*

Unpacked:

1. **The quantum annealer finds the minimum-energy spin configuration** given
   the edge-coloring constraints. This is the physical ground state of the
   Ising model defined by the board.

2. **The Nash equilibrium of the game is where the two players' frustration-
   minimization objectives meet symmetrically.** On the Petersen graph, under
   quantum adjudication, the ground state energy under optimal play distributes
   frustration equally between both player nodes — neither player's spin is
   preferred — producing a draw.

3. **Switchback is not merely a heuristic.** By directly targeting the
   structural properties (orbit balance, cycle frustration) that define the
   minimum-energy configuration, switchback approximates the Nash equilibrium
   move sequence without performing minimax search. The algorithm was derived
   from graph symmetry, not from game theory — yet it arrives at the same place.

If the conjecture holds, the result generalises beyond the Petersen graph: for
any graph in this family (X-Prize graphs 12, 18, 19, 20), switchback-like
frustration-minimizing play should converge toward the Nash equilibrium value.
Testing this prediction is a natural next step.

### 9.4 Why E7G?

Under the Nash equilibrium hypothesis, E7G is not special to AlphaQ — it is
the move that keeps P1 on the equilibrium path from the initial position.

Edge 7 connects $V_2$ (inner pentagon) to $V_8$ (outer vertex). $V_8$ is a
direct neighbor of the P2 player node ($p_2 = 7$) via edge $E_{13}$. Playing
$E_7$ green on move 1 establishes a ferromagnetic constraint ($V_2$ and $V_8$
must align) in P2's local neighborhood before P2 has moved. Under optimal
subsequent play, this constraint propagates through the graph's five-cycle
structure in a way that keeps the Ising ground-state energy balanced between
both player nodes.

Edge 0 ($E_0$, $V_0$–$V_2$) is an inner-pentagon edge in a different orbit
class. From the board state after $E_0 G$, the minimax value may already be
negative for P1 — not because of anything AlphaQ-specific, but because the
cycle-frustration structure of that position gives P2 an inherent advantage
under quantum adjudication.

**A testable prediction:** E7 and E3 are structurally identical under
$\text{Stab}(p_1, p_2)$ — they are in the same orbit. If the Nash hypothesis
is correct, opening $E_3 G$ should yield the same 96% draw rate as $E_7 G$.
This can be tested by changing `MOVE_OVERRIDES` to $\{15 \mapsto (3, G)\}$.

### 9.5 What This May Mean

If the minimum-energy equilibrium conjecture is correct, the project has found
something stronger than a draw strategy for one opponent:

- A **structural characterisation** of the Nash equilibrium for
  quantum-adjudicated edge-coloring games on the Petersen graph. The
  equilibrium is not an artifact of AlphaQ's implementation — it is a
  consequence of the graph's geometry and the physics of quantum annealing.

- Evidence that **cycle-frustration minimization** is the natural
  game-theoretic objective for this class of problem. The quantum annealer
  computes the minimum-energy configuration; the Nash equilibrium is where
  both players' frustration-minimization objectives meet in a symmetric draw.
  The two problems are, in this sense, the same problem.

- A **practical algorithm** (switchback) that approximates Nash equilibrium
  play without minimax search, by directly targeting the structural properties
  that characterise it. The algorithm runs in ~1,800 arithmetic operations per
  move and requires no training data, no search tree, and no oracle.

The open question is whether $V^* = 0$ (draw) is the correct Nash value, or
whether the game tree admits P1 wins that no strategy has yet reached. The
Track 3 full LUT harvest (all 32,768 terminal states scored on D-Wave
hardware) will answer this definitively: if any terminal state scores
$\geq +2$ along a reachable game path, P1 wins are achievable. If not, the
closure result is confirmed and the Nash value is draw — and the switchback
algorithm, derived from symmetry alone, has found the equilibrium.
