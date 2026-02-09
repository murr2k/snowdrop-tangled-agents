## Empirical Closure of the Petersen Graph Under Quantum Adjudication

**Murray Kopit**
Independent Researcher
February 2026

---

### Abstract

We report an empirical negative result concerning the Tangled game [1] played on the Petersen graph [2] under quantum ground-truth adjudication. Despite systematic exploration spanning 2,436 terminal state observations, 120 diverse terminal configurations reached via independent strategies, and surrogate evaluation methods, no winning terminal state against the fixed AlphaQ agent was observed once evaluated using quantum-derived ground-state expectations. The maximum quantum score observed was +0.861, less than half the +2 win threshold. Apparent wins produced by simulated annealing (SA) [3] or lookup-table (LUT)-based evaluators were found to exhibit polarity inversion under quantum adjudication: terminal states scored as strong classical wins (+5 to +10 SA) consistently resolved as quantum losses (-1.3 to -2.5), with an overall SA-to-quantum anti-correlation of r = -0.396. We argue that this inversion is explained by order-by-disorder effects [4] and non-classical weighting of degenerate ground states, and we conclude that, under current conditions, the Petersen graph appears empirically closed. This result provides concrete evidence of a superclassical gap between classical surrogate optimization and quantum-evaluated game outcomes.

---

## Context and Motivation

Tangled [1] is a two-player, perfect-information game in which players alternately color edges of a fixed graph as grey, green (ferromagnetic), or purple (anti-ferromagnetic), inducing an Ising Hamiltonian evaluated at the terminal state. The winner is determined by which player achieves greater vertex influence -- measured by correlation alignment of the terminal quantum state -- after quantum annealing [5, 6]. The game is played online at tangled-game.com, where terminal states are adjudicated using D-Wave quantum hardware.

The game was designed by Geordie Rose at Snowdrop Quantum Applications Corporation [29] specifically so that evaluating terminal states requires solving a problem where quantum computational advantages have been demonstrated [7, 8]. As Rose describes in the Tangled development blog [29], three adjudicators are available: a numerical Schrodinger equation solver (exact but exponentially expensive), simulated thermal annealing (classical approximation), and D-Wave Advantage hardware (quantum ground truth). While all three methods achieve 100% agreement on small graphs (K_3, K_4), divergences emerge on larger frustrated graphs -- the barbell graph being the smallest case with approximately 6% discrepancy between SA and quantum annealing [30, 31].

The Petersen graph [2], with its high symmetry and frustration structure [9], has served as a canonical testbed for Tangled. Rose selected snarks -- cubic graphs whose edges cannot be 3-colored -- as particularly suitable game boards because their local homogeneity, non-planarity, and high frustration with antiferromagnetic couplings create spin-glass phases even at small scales [32]. The Petersen graph, as the smallest snark [10] (10 vertices, 15 edges), exhibits rich frustration properties when mapped to an Ising model [11, 12] while remaining computationally tractable for partition function calculation. While classical solvers and reinforcement-learning agents can be trained against surrogate evaluators, the Tangled platform uses quantum ground-truth adjudication derived from D-Wave hardware [13]. Rose has shown that agents trained on quantum annealing results dramatically outperform SA-trained counterparts when adjudicated by quantum ground truth [33], establishing that the choice of evaluator fundamentally shapes learnable strategy structure.

The motivating question for this appendix is narrow but fundamental:

> **Does a deterministic winning route exist against AlphaQ on the Petersen graph when outcomes are evaluated using quantum ground truth?**

This question has practical stakes: Rose has offered a $10,000 bounty [34] for any bot that can defeat AlphaQ Up, an AlphaZero-trained agent, on the Tangled platform.

---

## Experimental Setting

### Opponent

* **AlphaQ** (also referred to as "AlphaQ Up" [34]) is a fixed-policy agent trained offline using AlphaZero-style reinforcement learning [29] with access to quantum evaluation during training.
* No online learning or adaptation occurs during play.
* Terminal evaluations during AlphaQ's training were drawn from quantum annealing on D-Wave hardware [13], giving it access to ground-truth reward signals unavailable to classically trained agents.
* AlphaQ exhibits 100% move consistency on well-observed game states (observation count > 50), confirming deterministic policy execution during our experiments.

### Adjudication

* Outcomes are determined by expectation values of vertex-vertex correlations in the terminal quantum state, computed via quantum annealing on D-Wave Advantage processors [5, 7, 31].
* The adjudicator computes correlation matrices from quantum samples, derives influence vectors, and produces a scalar score [31].
* A draw is declared when the absolute score is below epsilon = 0.0005.
* Approximately 20% of all terminal states on the Petersen graph fall within this draw band.
* A score of +2 or above is required for a decisive win.

### Classical Surrogates

During analysis, two classical evaluators were used as surrogates:

* **Simulated Annealing (SA)** [3] terminal evaluation
* **SA-derived lookup tables (LUTs)** used by an oracle routing solver

The SA implementation derives from D-Wave's own simulated annealing code, using a specially tuned thermal annealing variant on the classical s=1 limit of the Hamiltonian [30]. These evaluators were explicitly not used for official adjudication, but only for hypothesis generation and route planning. The known divergences between SA and quantum annealing on frustrated systems [14, 15] motivated careful comparison.

### Terminal State Space

The Petersen graph has 15 edges, each colored Green (ferromagnetic) or Purple (anti-ferromagnetic) at terminal. This yields 2^15 = 32,768 possible terminal configurations. Each configuration is encoded as a 15-bit index (bit j = 1 for Green, 0 for Purple) and mapped to a quantum-evaluated score via the D-Wave adjudicator.

The Petersen graph's automorphism group is isomorphic to S_5 (order 120) [16, 17], acting transitively on both vertices and edges. By the Burnside-Cauchy-Frobenius lemma [18] and the Polya enumeration theorem [19], the number of symmetry-distinct terminal configurations under this group action is determined by the cycle index of S_5 acting on the 15-edge set. Configurations related by graph automorphism induce isomorphic Ising Hamiltonians and therefore share identical quantum scores. The exact number of equivalence classes depends on the fixed-point structure of each element of S_5 acting on edges; a naive upper bound of 32,768 / 120 = 273 significantly undercounts because most group elements fix few or no edge colorings. The true count of distinct orbits is likely on the order of a few hundred.

### Exploration Campaigns

To test empirical closure, multiple systematic campaigns were conducted:

**Campaign 1: Oracle Route Sweep (48 games)**

* Cycled through all 48 SA-predicted winning routes via round-robin
* Each route: deterministic path to terminal state with SA score > +0.5
* AlphaQ response: 100% deterministic on high-confidence states (obs > 50)
* Result: 0 wins, 27 losses, 21 draws
* New unique terminals discovered: +24

**Campaign 2: Terminal Explorer (17 games)**

* Systematic round-robin through 30 possible opening moves (15 edges x 2 colors)
* MCTS fallback (50,000 iterations) for mid-game decisions
* Result: 0 wins, max quantum score +0.763
* New unique terminals discovered: +17

**Historical baseline:**

* 1,403 games prior to campaigns, yielding 79 unique terminal states
* 0 wins across all historical play against AlphaQ

**Combined statistics:**

* Total unique terminals observed: 120 (from 79 baseline, +52%)
* Total game observations: 1,468
* Maximum quantum score: +0.861
* Minimum quantum score: -8.806
* Mean quantum score: -0.466
* SA-to-quantum correlation: r = -0.396 (p < 0.0001)

---

## Observed Discrepancy Between Classical and Quantum Evaluation

A consistent phenomenon was observed across experiments:

1. Certain deterministic routes produced **positive terminal scores** under SA or LUT evaluation.
1. The same terminal states, when evaluated using the website's quantum ground truth, yielded **scores of opposite sign or near-zero magnitude**.
1. These values fell well below the +2 winning threshold and were adjudicated as draws or losses.

The relationship between SA and quantum scores is not merely compressive but exhibits **polarity inversion**. Across 120 terminal states observed against AlphaQ:

| SA Score Range | N | Avg Quantum Score | Relationship |
|----------------|---|-------------------|--------------|
| [+0.5, +2.0) | 21 | -0.257 | Inverted |
| [+2.0, +5.0) | 17 | -1.286 | Strongly inverted |
| [+5.0, +10.0) | 5 | -2.476 | Strongly inverted |

Terminal states that SA scores as strong wins consistently produce quantum losses. The overall Pearson correlation of r = -0.396 (p < 0.0001) confirms systematic anti-correlation, not merely compression or noise.

This finding is consistent with known SA-QA divergences on frustrated systems. Ronnow et al. [14] established rigorous frameworks for detecting quantum speedup and showed that classical and quantum approaches can produce qualitatively different solution distributions. Albash et al. [15] demonstrated scaling advantages for quantum annealing over SA on frustrated loop problems, and Boixo et al. [20] showed that multiqubit quantum tunneling in D-Wave processors accesses solution pathways unavailable to classical thermal processes. The polarity inversion we observe represents an extreme case of this divergence: not merely different solution quality, but opposite-sign evaluation of the same terminal configurations.

In one representative early case, an SA/LUT terminal score of approximately +1.99 mapped to a quantum website score of approximately +0.03, which initially suggested a compression factor of roughly 60x. Subsequent systematic analysis revealed that this single-point observation understated the discrepancy: the mapping is not monotonic compression but polarity inversion across the score range.

---

## Interpretation: Order by Disorder and Expectation Weighting

This discrepancy is not a numerical artifact but reflects a physical distinction between classical sampling and quantum evolution.

Classical SA [3] estimates ground-state energy by sampling low-energy configurations. In frustrated systems [9], this sampling can preferentially weight certain symmetry-breaking configurations. Quantum annealing [5, 6], by contrast, evolves a wavefunction whose terminal occupation probabilities depend on the full adiabatic path, including interference and tunneling effects [20].

This distinction is commonly described as **order by disorder** [4]:

* Classical samplers may favor ordered configurations because they are entropically accessible.
* Quantum evolution weights degenerate ground states according to dynamical structure, not count alone.

The phenomenon was first identified by Villain et al. [4], who showed that in frustrated Ising models, thermal fluctuations can lift accidental ground-state degeneracy and restore order that the bare Hamiltonian does not select. Moessner and Sondhi [12] extended this analysis to quantum frustrated systems, demonstrating that transverse-field dynamics on frustrated lattices (triangular, kagome, fully frustrated square) produce qualitatively different ground-state selection than classical thermal dynamics. The Petersen graph, with its snark structure and high frustration [10], provides a concrete finite-graph instance of these phenomena.

As a result, classical solvers can report nonzero mean values even when the true quantum expectation is arbitrarily close to zero. More critically, the sign of the classical estimate can be opposite to the quantum expectation, as the polarity inversion data demonstrate. Increasing the number of SA samples reduces variance but does not correct the underlying weighting mismatch or sign error.

---

## Empirical Closure Result

Across multiple experimental phases, including:

* exhaustive opening re-exploration (30 unique first moves),
* oracle-guided deterministic routing (48 SA-predicted winning routes),
* cycling through all candidate oracle routes via round-robin, and
* targeted exploration for novel terminal states via MCTS fallback,

no terminal state was observed with a quantum-evaluated score exceeding +1, let alone the +2 threshold required for a decisive win. The maximum observed quantum score across all 120 unique terminal configurations is +0.861.

The results support the following empirical proposition:

> **Proposition A.1 (Empirical Closure).**
> Given fixed AlphaQ policy and quantum ground-truth adjudication, no terminal state yielding a decisive win (quantum score >= +2) has been observed across 120 diverse terminal configurations reached via multiple independent strategies on the Petersen graph. The maximum observed quantum score is +0.861, less than half the win threshold.

This is an empirical statement based on finite sampling, not a formal proof. However, the consistency of the null result across diverse exploration strategies, combined with the systematic polarity inversion of classical surrogates, provides convergent evidence that winning terminal states are either non-existent within AlphaQ's policy-constrained state space or occur with very low probability.

---

## Statistical Confidence and Sample Coverage

### Sample Characteristics

The empirical closure claim rests on analysis of:

* **2,436 total terminal observations** across all historical games
* **120 distinct terminal configurations** from systematic exploration campaigns
* **Coverage:** 0.37% of the total state space (32,768 possible terminal configurations)
* **Score distribution:** bounded within [-8.806, +0.861], with mean -0.466

Note: The 2,436 observations include repeated visits to the same terminal configurations. For statistical independence, confidence bounds are computed over the 120 distinct configurations, since repeated observations of the same state do not constitute independent samples of the state space.

### Confidence Bounds

Under the assumption that winning terminals (score >= +2) are uniformly distributed among reachable states, the probability of observing zero wins in N independent samples, given that winning terminals comprise a fraction p of the reachable space, is:

P(0 wins | N, p) = (1 - p)^N

For N = 120 diverse terminal states:

* If p >= 2.5%, then P(0 wins) < 0.05 (95% confidence of detection)
* If p >= 3.8%, then P(0 wins) < 0.01 (99% confidence of detection)

**Interpretation:** If winning terminals comprise more than 2.5% of the reachable state space, we would have found at least one with 95% confidence. The null result is therefore strong evidence against the hypothesis that winning terminals are common. It does not exclude the possibility that winning terminals are very rare (< 2.5% of reachable space) or that they exist but are unreachable under AlphaQ's policy.

**Sampling bias caveat:** The uniform distribution assumption may not hold. Terminal states reached via classical strategies (oracle routing, MCTS, systematic openings) are biased toward classically accessible regions of the state space. If winning terminals are reachable only through strategies that classical reasoning cannot generate -- a possibility consistent with the superclassical gap thesis [21, 22] -- the confidence bounds above overstate our coverage of the relevant state space. This sampling bias is inherent to any empirical approach that uses classical strategies for exploration, and represents a fundamental limitation: classical tools may be unable to reach the very states needed to refute classical insufficiency.

### Strengthening the Bound

A planned extension campaign targeting 10,000 unique terminal configurations (30% coverage via 50,000 parallel games) would tighten these bounds:

* If p >= 0.03%, then P(0 wins | 10000) < 0.05
* This would provide 95% confidence that winning terminals, if they exist, comprise less than 0.03% of the reachable space

However, the sampling bias caveat applies with equal force to larger campaigns conducted with classical strategies.

### What Formal Proof Would Require

Strict mathematical proof of non-existence is not attainable through empirical sampling alone. A formal proof would require one of:

1. **Complete enumeration:** Quantum evaluation of all 32,768 terminal states via D-Wave, independent of reachability constraints. This would establish ground truth for the full state space but does not address whether winning states are reachable under AlphaQ's policy.

2. **Game tree analysis:** Complete enumeration of the game tree under AlphaQ's fixed policy, proving that no sequence of moves by the opposing player leads to a terminal state with score >= +2. This requires knowledge of AlphaQ's policy function, which is not publicly available.

3. **Symmetry reduction:** Exploitation of the Petersen graph's automorphism group (S_5, order 120) [16, 17] to reduce the state space via Burnside's lemma [18] and Polya enumeration [19]. Terminal configurations related by graph automorphism share identical quantum scores. Schmidt [17] proved that the quantum automorphism group of the Petersen graph equals its classical automorphism group, meaning no additional quantum symmetries exist to further reduce the search space. Combined with reachability analysis, symmetry reduction could make exhaustive verification tractable.

None of these approaches has been completed.

### Falsifiability

The empirical closure claim is falsifiable: discovery of a single terminal state with quantum score >= +2 against AlphaQ would refute it. The claim is strengthened, not proven, by each additional null observation. This is the standard epistemological status of empirical negative results in physics and game theory.

---

## Implications for Strategy and Learning

This closure result has several implications:

1. **Classical routing is insufficient.**
   Deterministic traversal of the game tree guided by classical evaluators does not translate into quantum advantage. All 48 oracle routes optimized against SA scores failed to produce wins under quantum adjudication. This is consistent with the broader finding that classical optimization heuristics can be structurally misaligned with quantum objectives on frustrated systems [14, 15].

1. **Surrogate reward signals are worse than uninformative.**
   SA-based rewards are not merely noisy but anti-correlated (r = -0.396) with quantum outcomes against AlphaQ. Reinforcement learning or search guided by SA-based rewards optimizes toward states that are actively penalized under quantum evaluation. This extends beyond the PLS complexity considerations of local search [23] to a qualitative sign inversion of the reward landscape.

1. **The Petersen graph is a saturation point.**
   Its symmetry and frustration structure appear sufficient to eliminate exploitable asymmetries under quantum adjudication. The AlphaQ policy constrains play to a terminal state basin bounded within [-8.806, +0.861]. The Petersen graph's properties as the smallest snark [10] may be fundamental to this saturation.

1. **Superclassical gap is operational, not abstract.**
   The gap manifests concretely as polarity inversion: classical heuristics not only fail to predict quantum outcomes but assign opposite sign to terminal state values. This provides an operational instance of the quantum advantages demonstrated on D-Wave hardware for frustrated optimization problems [7, 8, 15].

---

## Relation to Prior Work

These findings directly reinforce the central claims of *Reinforcement Learning Agents With and Without Access to Quantum Computation*, namely that access to quantum evaluation changes not merely performance but the *structure* of learnable strategies.

Where many studies stop at comparing agent win rates under different evaluators, this work traces the full chain:

Quantum dynamics fidelity -> terminal state labeling -> reward signal -> policy optimization -> observed agent behavior

Most classical approaches implicitly assume equivalence between the first two steps. This appendix demonstrates a concrete case where that assumption fails, and further shows that the failure mode is not degradation but inversion.

Rose's own experiments with quantum-trained agents [33] provide direct evidence for this chain: agents trained using D-Wave quantum annealing results "dramatically outperformed" their SA-trained counterparts when adjudicated by quantum ground truth, even on the barbell graph where only 6% of terminal states show SA-QA disagreement [30]. Our findings explain why: SA-based training optimizes toward terminal states that are anti-correlated with quantum evaluation, meaning SA-trained agents are not merely suboptimal but actively misled.

The polarity inversion finding connects to several threads in the quantum computing literature. Eisert, Wilkens, and Lewenstein [21] showed that quantum strategies can fundamentally alter game-theoretic outcomes, eliminating classical dilemmas through entanglement. Brassard, Broadbent, and Tapp [22] demonstrated quantum pseudo-telepathy in graph coloring games, where shared entanglement enables outcomes impossible for any classical strategy. More recently, Zheng et al. [24] provided experimental demonstration of quantum advantage in the odd-cycle game, confirming that quantum-classical discrepancies in graph-based games are not merely theoretical. Our finding of polarity inversion provides a new mechanism by which quantum evaluation alters game outcomes: not through strategic entanglement between players, but through the evaluation function itself producing structurally different scores than its classical surrogate.

The use of quantum annealing for game evaluation also connects to work by Zanca and Zecchina [25], who applied quantum annealing to find Nash equilibria in graphical games, demonstrating that D-Wave hardware can effectively map game-theoretic problems to QUBO/Ising formulations.

---

## Limitations and Open Questions

### Scope Limitations

* **Graph-specific:** This result applies only to the Petersen graph. Larger or less symmetric graphs (e.g., mutated C60 variants) may exhibit different closure properties. The Petersen graph's exceptional symmetry [10, 16] and frustration structure [9, 12] may make it uniquely resistant to classical exploitation.
* **Opponent-specific:** AlphaQ's fixed policy defines the reachable terminal state space. Different opponents (e.g., Melissa, Amara) permit access to different terminal basins, and 244 winning terminal states have been observed against those opponents.
* **Sample size:** 120 diverse terminal configurations represent 0.37% of the total state space. While diverse in construction (oracle routes, systematic openings, MCTS exploration), the sample is not exhaustive.
* **Strategy class:** Only deterministic and MCTS-based strategies were tested. Stochastic, adversarial-adaptive, or reinforcement-learning-derived strategies remain unexplored. Strategies with quantum evaluation access [21, 22] might access fundamentally different regions of the terminal state space.
* **Sampling bias:** All exploration was conducted with classically derived strategies. If winning terminals are reachable only through non-classical play, classical exploration cannot detect them regardless of sample size.

### What Remains Unproven

* Whether winning terminal states (score >= +2) exist anywhere in the Petersen graph's full 32,768-state terminal space, reachable or not
* Whether AlphaQ's policy makes such states unreachable, or whether they simply do not exist under quantum adjudication
* Whether the polarity inversion effect is specific to the Petersen graph or generalizes to other frustrated topologies

### Open Questions

* Do winning terminals exist but remain unreachable under AlphaQ's policy?
* Would quantum-trained strategies (e.g., strategies trained with D-Wave evaluation access) reach different terminal basins?
* Does the polarity inversion generalize to other frustrated graphs, or is it specific to the Petersen graph's automorphism structure?
* Can order-by-disorder effects be predicted from graph topology alone, enabling a priori identification of empirically closed graphs?
* Is the classical sampling bias itself a manifestation of the superclassical gap -- that is, are the very tools needed to find winning states unavailable without quantum resources?

---

## Conclusion

The Petersen graph, when coupled with quantum ground-truth adjudication and a fixed AlphaQ policy, appears empirically closed to winning strategies derived from classical reasoning. Across 120 diverse terminal configurations reached via multiple independent exploration strategies, the maximum observed quantum score is +0.861, well below the +2 win threshold. Terminal states predicted as wins by SA-based evaluators consistently resolve as losses under quantum adjudication, exhibiting polarity inversion (r = -0.396) rather than the commonly assumed monotonic compression.

This negative result is empirical, not formal. It establishes with 95% confidence that winning terminals, if they exist within the classically reachable portion of AlphaQ's state space, comprise less than 2.5% of the accessible configurations. A planned 50,000-game extension campaign would tighten this bound to 0.03%, though the fundamental limitation of classical sampling bias applies to any such extension.

Rather than a failure, this negative result provides a clean, operational demonstration of the superclassical gap. It shows not only that quantum-trained agents can outperform classical ones, but that classical intuitions about "advantage" can become structurally inverted under quantum evaluation. The polarity inversion finding goes beyond prior observations of score compression, revealing that the classical-quantum mismatch is not merely quantitative but qualitative. In the language of Villain et al. [4], disorder (quantum fluctuations) creates an order (consistent evaluation) that is invisible to classical sampling, and in this case, opposite in sign.

---

## References

[1] G. Rose, Tangled, Snowdrop Quantum Applications Corporation. Available at: https://tangled-game.com. A two-player graph-coloring game where players alternately color edges as ferromagnetic (green) or anti-ferromagnetic (purple), inducing an Ising Hamiltonian adjudicated via D-Wave quantum annealing. See also: snowdrop-tangled-game-engine, PyPI.

[2] J. Petersen, "Die Theorie der regularen graphs," *Acta Mathematica*, vol. 15, pp. 193-220, 1891.

[3] S. Kirkpatrick, C. D. Gelatt, and M. P. Vecchi, "Optimization by Simulated Annealing," *Science*, vol. 220, no. 4598, pp. 671-680, 1983.

[4] J. Villain, R. Bidaux, J. P. Carton, and R. Conte, "Order as an effect of disorder," *Journal de Physique*, vol. 41, pp. 1263-1272, 1980.

[5] T. Kadowaki and H. Nishimori, "Quantum annealing in the transverse Ising model," *Physical Review E*, vol. 58, p. 5355, 1998. arXiv: cond-mat/9804280.

[6] E. Farhi, J. Goldstone, S. Gutmann, and M. Sipser, "Quantum Computation by Adiabatic Evolution," arXiv: quant-ph/0001106, 2000. See also: E. Farhi et al., "A Quantum Adiabatic Evolution Algorithm Applied to Random Instances of an NP-Complete Problem," *Science*, vol. 292, pp. 472-475, 2001.

[7] A. D. King et al., "Quantum critical dynamics in a 5,000-qubit programmable spin glass," *Nature*, 2023.

[8] A. D. King et al., "Beyond-classical computation in quantum simulation," *Science*, 2025.

[9] G. Toulouse, "Theory of the frustration effect in spin glasses: I," *Communications in Physics*, vol. 2, pp. 115-119, 1977.

[10] D. A. Holton and J. Sheehan, *The Petersen Graph*, Cambridge University Press, 1993.

[11] A. Lucas, "Ising formulations of many NP problems," *Frontiers in Physics*, vol. 2, art. 5, 2014. arXiv: 1302.5843.

[12] R. Moessner and S. L. Sondhi, "Ising models of quantum frustration," *Physical Review B*, vol. 63, p. 224401, 2001. arXiv: cond-mat/0011250.

[13] T. Lanting, A. J. Przybysz, A. Yu. Smirnov, F. M. Spedalieri, M. H. Amin, A. J. Berkley, R. Harris, ..., and G. Rose, "Entanglement in a Quantum Annealing Processor," *Physical Review X*, vol. 4, p. 021041, 2014.

[14] T. F. Ronnow, Z. Wang, J. Job, S. Boixo, S. V. Isakov, D. Wecker, J. M. Martinis, D. A. Lidar, and M. Troyer, "Defining and detecting quantum speedup," *Science*, vol. 345, no. 6195, pp. 420-424, 2014. arXiv: 1401.2910.

[15] T. Albash, T. F. Ronnow, M. Troyer, and D. A. Lidar, "Demonstration of a Scaling Advantage for a Quantum Annealer over Simulated Annealing," *Physical Review X*, vol. 8, p. 031016, 2018.

[16] M. E. Watkins, "The Automorphism Group of the Petersen Graph is Isomorphic to S_5," arXiv: 2012.02942, 2020.

[17] S. Schmidt, "The Petersen graph has no quantum symmetry," *Bulletin of the London Mathematical Society*, vol. 50, no. 3, pp. 481-490, 2018. arXiv: 1801.02942.

[18] W. Burnside, *Theory of Groups of Finite Order*, 2nd ed., Cambridge University Press, 1911. (The counting lemma is due to Cauchy and Frobenius.)

[19] G. Polya, "Kombinatorische Anzahlbestimmungen fur Gruppen, Graphen und chemische Verbindungen," *Acta Mathematica*, vol. 68, pp. 145-254, 1937.

[20] S. Boixo, V. N. Smelyanskiy, A. Shabani, S. V. Isakov, M. Dykman, V. S. Denchev, M. H. Amin, A. Yu. Smirnov, M. Mohseni, and H. Neven, "Computational multiqubit tunnelling in programmable quantum annealers," *Nature Communications*, vol. 7, p. 10327, 2016.

[21] J. Eisert, M. Wilkens, and M. Lewenstein, "Quantum Games and Quantum Strategies," *Physical Review Letters*, vol. 83, no. 15, pp. 3077-3080, 1999. arXiv: quant-ph/9806088.

[22] G. Brassard, A. Broadbent, and A. Tapp, "Quantum Pseudo-Telepathy," *Foundations of Physics*, vol. 35, no. 11, pp. 1877-1907, 2005.

[23] D. S. Johnson, C. H. Papadimitriou, and M. Yannakakis, "How easy is local search?" *Journal of Computer and System Sciences*, vol. 37, no. 1, pp. 79-100, 1988.

[24] Z. Zheng et al., "Experimental Quantum Advantage in the Odd-Cycle Game," *Physical Review Letters*, vol. 134, p. 070201, 2025.

[25] T. Zanca and R. Zecchina, "A Quantum Annealing Algorithm for Finding Pure Nash Equilibria in Graphical Games," *Computational Science -- ICCS 2020*, Springer LNCS 12142, pp. 488-501, 2020. arXiv: 1903.06454.

[26] T. Albash and D. A. Lidar, "Adiabatic quantum computation," *Reviews of Modern Physics*, vol. 90, p. 015002, 2018. arXiv: 1611.04471.

[27] S. Boixo, T. F. Ronnow, S. V. Isakov, Z. Wang, D. Wecker, D. A. Lidar, J. M. Martinis, and M. Troyer, "Evidence for quantum annealing with more than one hundred qubits," *Nature Physics*, vol. 10, pp. 218-224, 2014. arXiv: 1304.4595.

[28] M. Bauza, A. D. King, T. Lanting, et al., "Scaling Advantage in Approximate Optimization with Quantum Annealing," *Physical Review Letters*, vol. 134, p. 160601, 2025. arXiv: 2401.07184.

[29] G. Rose, "Tangled Blog," Snowdrop Quantum Applications Corporation, 2024-2026. Available at: https://snowdropquantum.com/tangledblog. Development blog documenting the design, implementation, and AI agent training for the Tangled quantum game. Includes the AlphaZero deconstruction series, adjudication methodology, and game design rationale.

[30] G. Rose, "Adjudicating Tangled Terminal States on Tiny Graphs With Three Different Approaches," Snowdrop Quantum Applications Corporation blog, November 2024. Available at: https://snowdropquantum.com/tangledblog/evaluating-tangled-terminal-states-with-simulated-annealing-and-a-d-wave-quantum-computer. Reports 100% agreement between Schrodinger solver, SA, and D-Wave on K_3 and K_4 graphs. Notes that SA equally samples all degenerate ground states, while quantum evolution preferentially weights states with more single-spin-flip connections.

[31] G. Rose, "How I Adjudicate Tangled Terminal States Using D-Wave Hardware," Snowdrop Quantum Applications Corporation blog, October 2024. Available at: https://snowdropquantum.com/tangledblog/how-i-solve-problems-using-d-wave-hardware. Details the Ising Hamiltonian formulation, embedding algorithm, and scoring methodology using correlation matrices and influence vectors on D-Wave Advantage processors.

[32] G. Rose, "Snarks!," Snowdrop Quantum Applications Corporation blog, February 2025. Available at: https://snowdropquantum.com/tangledblog/snarks. Describes why snarks (cubic graphs not 3-edge-colorable) are particularly suitable for Tangled due to local homogeneity, non-planarity, and high frustration producing spin-glass phases at small scales.

[33] G. Rose, "My First Quantum Agent," Snowdrop Quantum Applications Corporation blog, April 2025. Available at: https://snowdropquantum.com/tangledblog/first-quantum-agent-deployed. Reports that AlphaZero agents trained using D-Wave quantum annealing results dramatically outperform SA-trained counterparts when adjudicated by quantum ground truth on the barbell graph.

[34] G. Rose, "The $10,000 AlphaQ Up Challenge," Snowdrop Quantum Applications Corporation blog, January 2026. Available at: https://snowdropquantum.com/tangledblog/the-alphaq-up-challenge. Offers a $10,000 bounty for any bot that can defeat AlphaQ Up, an AlphaZero-trained agent, on the Tangled platform.

[35] G. Rose, "Analysis of Adjudication of Terminal States for Small Graphs," Snowdrop Quantum Applications Corporation blog, April 2025. Available at: https://snowdropquantum.com/tangledblog/analysis-of-adjudication-of-terminal-states-for-small-graphs. Systematic comparison of four solvers across five graphs. The barbell graph (6 vertices, 7 edges) is the smallest where SA-QA disagreement emerges, with 6.1% discrepancy across 2,187 terminal states.

---
