%% MCTS Tournament Test
%
% Run multiple games between strategies to evaluate MCTS strength.

fprintf('\n');
fprintf('================================================\n');
fprintf('  MCTS Tournament: MCTS vs Petersen vs Random\n');
fprintf('================================================\n\n');

numGames = 20;  % Per matchup

%% Helper function for self-play
function [winner, finalScore] = playGame(player1, player2)
    state = '---------------';
    isPlayer1Turn = true;

    while any(state == '-')
        if isPlayer1Turn
            if isa(player1, 'TangledMCTS')
                [edge, color] = player1.search(state);
            else
                move = player1.selectMove(state);
                edge = move.edge;
                color = move.color;
            end
        else
            if isa(player2, 'TangledMCTS')
                [edge, color] = player2.search(state);
            else
                move = player2.selectMove(state);
                edge = move.edge;
                color = move.color;
            end
        end

        state(edge + 1) = color;
        isPlayer1Turn = ~isPlayer1Turn;
    end

    % Evaluate using P1 perspective for consistent scoring
    mcts_eval = TangledMCTS('Player', 1);
    finalScore = mcts_eval.evaluateTerminal(state);

    if finalScore > 0.5
        winner = 1;  % P1 wins
    elseif finalScore < -0.5
        winner = 2;  % P2 wins
    else
        winner = 0;  % Draw
    end
end

%% Matchup 1: MCTS vs Petersen
fprintf('Matchup 1: MCTS vs Petersen (%d games each side)\n', numGames);
fprintf('------------------------------------------------\n');

mcts_wins = 0;
petersen_wins = 0;
draws = 0;

petersen = SimulatedOpponent('Style', 'petersen');

% MCTS as Player 1
mcts_p1 = TangledMCTS('Iterations', 300, 'TimeLimit', 0.5, 'UseParallel', false, 'Player', 1);

fprintf('MCTS as P1: ');
for i = 1:numGames
    [winner, ~] = playGame(mcts_p1, petersen);
    if winner == 1
        mcts_wins = mcts_wins + 1;
        fprintf('W');
    elseif winner == 2
        petersen_wins = petersen_wins + 1;
        fprintf('L');
    else
        draws = draws + 1;
        fprintf('D');
    end
end
fprintf('\n');

% MCTS as Player 2
mcts_p2 = TangledMCTS('Iterations', 300, 'TimeLimit', 0.5, 'UseParallel', false, 'Player', 2);

fprintf('MCTS as P2: ');
for i = 1:numGames
    [winner, ~] = playGame(petersen, mcts_p2);
    if winner == 2
        mcts_wins = mcts_wins + 1;
        fprintf('W');
    elseif winner == 1
        petersen_wins = petersen_wins + 1;
        fprintf('L');
    else
        draws = draws + 1;
        fprintf('D');
    end
end
fprintf('\n');

total = 2 * numGames;
fprintf('\nResults: MCTS %d/%d (%.1f%%), Petersen %d/%d (%.1f%%), Draws %d/%d (%.1f%%)\n', ...
    mcts_wins, total, 100*mcts_wins/total, ...
    petersen_wins, total, 100*petersen_wins/total, ...
    draws, total, 100*draws/total);
fprintf('\n');

%% Matchup 2: MCTS vs Random
fprintf('Matchup 2: MCTS vs Random (%d games each side)\n', numGames);
fprintf('----------------------------------------------\n');

mcts_wins_r = 0;
random_wins = 0;
draws_r = 0;

random_opp = SimulatedOpponent('Style', 'random');

% MCTS as Player 1
fprintf('MCTS as P1: ');
for i = 1:numGames
    [winner, ~] = playGame(mcts_p1, random_opp);
    if winner == 1
        mcts_wins_r = mcts_wins_r + 1;
        fprintf('W');
    elseif winner == 2
        random_wins = random_wins + 1;
        fprintf('L');
    else
        draws_r = draws_r + 1;
        fprintf('D');
    end
end
fprintf('\n');

% MCTS as Player 2
fprintf('MCTS as P2: ');
for i = 1:numGames
    [winner, ~] = playGame(random_opp, mcts_p2);
    if winner == 2
        mcts_wins_r = mcts_wins_r + 1;
        fprintf('W');
    elseif winner == 1
        random_wins = random_wins + 1;
        fprintf('L');
    else
        draws_r = draws_r + 1;
        fprintf('D');
    end
end
fprintf('\n');

fprintf('\nResults: MCTS %d/%d (%.1f%%), Random %d/%d (%.1f%%), Draws %d/%d (%.1f%%)\n', ...
    mcts_wins_r, total, 100*mcts_wins_r/total, ...
    random_wins, total, 100*random_wins/total, ...
    draws_r, total, 100*draws_r/total);
fprintf('\n');

%% Matchup 3: MCTS vs Fast MCTS
fprintf('Matchup 3: Real MCTS vs Fast MCTS (%d games each side)\n', numGames);
fprintf('------------------------------------------------------\n');

real_wins = 0;
fast_wins = 0;
draws_f = 0;

fast_mcts = SimulatedOpponent('Style', 'fast_mcts');

% Real MCTS as Player 1
fprintf('Real as P1: ');
for i = 1:numGames
    [winner, ~] = playGame(mcts_p1, fast_mcts);
    if winner == 1
        real_wins = real_wins + 1;
        fprintf('W');
    elseif winner == 2
        fast_wins = fast_wins + 1;
        fprintf('L');
    else
        draws_f = draws_f + 1;
        fprintf('D');
    end
end
fprintf('\n');

% Real MCTS as Player 2
fprintf('Real as P2: ');
for i = 1:numGames
    [winner, ~] = playGame(fast_mcts, mcts_p2);
    if winner == 2
        real_wins = real_wins + 1;
        fprintf('W');
    elseif winner == 1
        fast_wins = fast_wins + 1;
        fprintf('L');
    else
        draws_f = draws_f + 1;
        fprintf('D');
    end
end
fprintf('\n');

fprintf('\nResults: Real MCTS %d/%d (%.1f%%), Fast MCTS %d/%d (%.1f%%), Draws %d/%d (%.1f%%)\n', ...
    real_wins, total, 100*real_wins/total, ...
    fast_wins, total, 100*fast_wins/total, ...
    draws_f, total, 100*draws_f/total);

fprintf('\n================================================\n');
fprintf('  Tournament Complete!\n');
fprintf('================================================\n');
