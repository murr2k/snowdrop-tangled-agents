%% Test TangledMCTS Implementation
%
% This script tests the MCTS implementation to verify it works correctly.

fprintf('\n');
fprintf('========================================\n');
fprintf('  MCTS Implementation Test\n');
fprintf('========================================\n\n');

%% Test 1: Basic MCTS Search
fprintf('Test 1: Basic MCTS Search\n');
fprintf('------------------------\n');

mcts = TangledMCTS('Iterations', 500, 'TimeLimit', 2.0, 'UseParallel', false);

% Empty board
state = '---------------';
fprintf('State: %s (empty board)\n', state);

tic;
[edge, color, info] = mcts.search(state);
elapsed = toc;

fprintf('Result: E%d %s\n', edge, color);
fprintf('Iterations: %d in %.2fs (%.0f/s)\n', info.iterations, elapsed, info.iterationsPerSecond);
fprintf('\n');

%% Test 2: Mid-game position
fprintf('Test 2: Mid-game Position\n');
fprintf('-------------------------\n');

state = 'G-P--G----P--P-';  % Some edges colored
fprintf('State: %s\n', state);

tic;
[edge, color, info] = mcts.search(state);
elapsed = toc;

fprintf('Result: E%d %s\n', edge, color);
fprintf('Iterations: %d in %.2fs (%.0f/s)\n', info.iterations, elapsed, info.iterationsPerSecond);
fprintf('\n');

%% Test 3: Late-game position
fprintf('Test 3: Late-game Position\n');
fprintf('--------------------------\n');

state = 'GPGPPGPGGP-P-G-';  % Only 3 edges left
fprintf('State: %s\n', state);

tic;
[edge, color, info] = mcts.search(state);
elapsed = toc;

fprintf('Result: E%d %s\n', edge, color);
fprintf('Iterations: %d in %.2fs (%.0f/s)\n', info.iterations, elapsed, info.iterationsPerSecond);
fprintf('\n');

%% Test 4: SimulatedOpponent with real MCTS
fprintf('Test 4: SimulatedOpponent with MCTS\n');
fprintf('-----------------------------------\n');

opp = SimulatedOpponent('Style', 'mcts', 'Iterations', 300, 'TimeLimit', 1.0);

state = '---------------';
fprintf('State: %s\n', state);

tic;
move = opp.selectMove(state);
elapsed = toc;

fprintf('Result: E%d %s (%.2fs)\n', move.edge, move.color, elapsed);
fprintf('\n');

%% Test 5: Compare MCTS vs Fast MCTS vs Petersen
fprintf('Test 5: Strategy Comparison\n');
fprintf('---------------------------\n');

state = '---------------';

% Real MCTS
mcts_opp = SimulatedOpponent('Style', 'mcts', 'Iterations', 300, 'TimeLimit', 1.0);
tic;
mcts_move = mcts_opp.selectMove(state);
mcts_time = toc;

% Fast MCTS
fast_opp = SimulatedOpponent('Style', 'fast_mcts');
tic;
fast_move = fast_opp.selectMove(state);
fast_time = toc;

% Petersen
petersen_opp = SimulatedOpponent('Style', 'petersen');
tic;
petersen_move = petersen_opp.selectMove(state);
petersen_time = toc;

fprintf('Real MCTS:  E%d %s (%.3fs)\n', mcts_move.edge, mcts_move.color, mcts_time);
fprintf('Fast MCTS:  E%d %s (%.3fs)\n', fast_move.edge, fast_move.color, fast_time);
fprintf('Petersen:   E%d %s (%.3fs)\n', petersen_move.edge, petersen_move.color, petersen_time);
fprintf('\n');

%% Test 6: Self-play game
fprintf('Test 6: Self-play Game (MCTS vs Petersen)\n');
fprintf('-----------------------------------------\n');

mcts_player = TangledMCTS('Iterations', 200, 'TimeLimit', 0.5, 'UseParallel', false);
petersen_player = SimulatedOpponent('Style', 'petersen');

state = '---------------';
isPlayer1Turn = true;  % MCTS goes first
moveNum = 0;

fprintf('Playing game...\n');

while any(state == '-')
    moveNum = moveNum + 1;

    if isPlayer1Turn
        [edge, color] = mcts_player.search(state);
        player = 'MCTS';
    else
        move = petersen_player.selectMove(state);
        edge = move.edge;
        color = move.color;
        player = 'Petersen';
    end

    % Apply move (edge is 0-indexed)
    state(edge + 1) = color;

    fprintf('Move %2d: %s plays E%d %s -> %s\n', moveNum, player, edge, color, state);

    isPlayer1Turn = ~isPlayer1Turn;
end

% Evaluate final state
finalScore = mcts_player.evaluateTerminal(state);
fprintf('\nFinal state: %s\n', state);
fprintf('Final score: %.2f (positive = MCTS wins)\n', finalScore);

if finalScore > 0.5
    fprintf('Result: MCTS WINS\n');
elseif finalScore < -0.5
    fprintf('Result: PETERSEN WINS\n');
else
    fprintf('Result: DRAW\n');
end

fprintf('\n========================================\n');
fprintf('  All tests completed!\n');
fprintf('========================================\n');
