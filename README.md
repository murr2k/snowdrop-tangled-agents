# snowdrop-tangled-agents
A place to build and test agents for playing Tangled!

## How to build your own agent
This repo includes a random agent (see `RandomRandyAgent` in `random_agent.py`). To build your
own agent, follow the instructions and pattern for that agent. The process is straightforward - your
agent gets a Game instance as input/observable, which is a snapshot of the current game being played 
which includes the current state of the game, and you output an action/move. The move is just which 
edge to pick, and what color to apply to it. That's all! Obviously the underlying process you use to 
select this move can be arbitrarily complex but the actual agent interface is simple.

## Testing agents
A test script is included in `run_local_parallel_tournament.py`. Out of the box it runs 100,000 games
for each of the X-Prize graphs (not including the mutant C60 graph) where the adjudicator used is
simulated annealing, and the two agents in the competition are both random agents.

## Suggested next steps
If you want to build your own agent, I would start by building your own adjudicator (look at both
the adjudicators in the `snowdrop-adjudicators` repo). The better you
can spoof the D-Wave hardware, the better quality reward your agent will have to learn from. To do this,
I would recommend you build a look-up table adjudicator, by first enumerating all the terminal states
for a small game graph, and then adjudicating all of them using the schrodinger equation adjudicator (this will
be limited to small game graphs, but it's still a great warm-up for doing something more ambitious). Once
you have your lookup table adjudicator, you can then build an agent that uses it to give you the correct
answer every time. You can use `run_local_parallel_tournament.py` - add your adjudicator and new agents to
and see how well they do against the random agent and any other agents you might have built!