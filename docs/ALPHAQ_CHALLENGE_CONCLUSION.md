# Beating AlphaQ Up Classically: Conclusion

*September 2026*

## Context

[Tangled](https://tangled-game.com) is a two-player game played on a graph. Players
take turns colouring edges grey, green or purple. The finished board is scored from
a lookup table produced on quantum-annealing hardware, and a score within ±0.0005
is a draw.

AlphaQ Up is the site's strongest agent. The task was to beat it on the Petersen
graph, first in single games and then in the site's Best-of-5 challenge, under these
rules:

- the agent is purely classical;
- it learns only from games actually played through the website;
- it never queries the backend directly;
- it never touches how the page reports results.

## Conclusion

**AlphaQ Up could not be beaten classically, and the evidence says it plays at or
near perfect play everywhere we could test it.**

We played 1,389 ranked games against it with a purely classical agent: 917 draws,
472 losses, no wins.

Along the way we worked out how the site's lookup table behaves:

- It is stored as float16.
- It follows quantum-annealing physics: the hardware favours some ground states
  over others.
- Close boards are decided by tiny sampling residues. A classifier trained on our
  games predicts how they break with 83% accuracy.

Every model we built (classical, simulated anneal, quantum Boltzmann, learned) says
the game is a draw with best play from both seats. Many boards score exactly zero,
so either side can always hold the draw.

AlphaQ is an AlphaZero-style network with 1000 search rollouts per move. It never
left the draw anywhere we could check:

- at its last move (move 14), in every position probed;
- at move 12, in all 386 positions tested to completion;
- at move 10, in all 25 positions tested to completion;
- in all 2,621 recorded decisions, none of which is a mistake under any of our
  models.

We also tried the approaches that beat superhuman Go AIs:

- **Replicas.** A model imitating AlphaQ's moves, then an AlphaZero-style replica
  of its search built on that model. The search replica predicted AlphaQ *worse*
  than the imitation alone, so no offline attack could be aimed.
- **Unfamiliar positions.** Adversarial steering into positions AlphaQ rarely sees.
  It found nothing.

A win would need a middlegame mistake by AlphaQ at a position we can steer it into.
Our evidence says such a mistake probably doesn't exist. Only Snowdrop's own
information could prove it either way: an audit of AlphaQ against exact best play on
the table, or AlphaQ's weights and search settings.

## What the effort produced

- **A strong classical defence.** 61 draws in 64 games as the first player when
  playing the learned model.
- **A tested Best-of-5 challenge runner.**
- **A draw-boundary result.** If the draw band shrank to zero, our draws would break
  410 to 158 in our favour, with 224 boards still tied at exactly 0.0. AlphaQ is
  indifferent inside the band, and our solver collects the small positive leanings.
- **A complete, reproducible record of the effort.**

## Where to look in this repository

| Topic | File |
|---|---|
| Three-colour game model and exact solver | `snowdrop_tangled_agents/strategy/ternary_model.py`, `strategy/ternary_strategy.py`, `tools/solve_ternary_game.py` |
| Quantum-annealing terminal models (Schrödinger emulation, quantum Boltzmann) | `snowdrop_tangled_agents/strategy/quantum_model.py` |
| Learned table (physics plus tie classifier) | `snowdrop_tangled_agents/tools/learned_table.py` |
| Line planning, endgame and move-10 probers | `snowdrop_tangled_agents/strategy/line_planner.py` |
| AlphaQ behaviour clone and search replica | `snowdrop_tangled_agents/tools/alphaq_clone.py`, `strategy/alphaq_replica.py`, `tools/replica_eval.py` |
| Replica plan and results | [`docs/ALPHAQ_REPLICA_PLAN.md`](ALPHAQ_REPLICA_PLAN.md) |
| Challenge mode and local mock | `play_tangled.py --challenge`, `tools/mock_tangled_site.py` |
