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

### Function: best_move(state) → (edge_index, color)

Greedy one-step lookahead. Tries every (uncolored edge, color) pair, scores
the resulting state, and returns the pair that scores highest. No search tree,
no rollouts.

```
function best_move(state):
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
| `cycle_features` | 12 cycles × 5 edge lookups = 60 comparisons |
| `orbit_balance` | 15 edge lookups + 9-element std twice |
| `score` | 4 multiplications + 3 additions |
| `best_move` outer loop | at most 15 edges × 2 colors = 30 calls to `score` |
| **Total per move** | ~1,800 arithmetic operations |

There are no recursive calls, no heap allocations beyond the single candidate
copy, and no dependence on game history. Move selection is effectively
instantaneous on any modern processor.
