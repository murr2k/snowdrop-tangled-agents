function results = test_pretrained_agent(agent, options)
%TEST_PRETRAINED_AGENT Validate pre-trained agent performance
%
%   results = test_pretrained_agent(agent)
%   results = test_pretrained_agent(agent, Name=Value)
%
%   Tests the pre-trained agent against various opponents and
%   validates that the supervised learning transfer was successful.
%
%   Inputs:
%       agent - PPO agent (pre-trained or fine-tuned)
%
%   Name-Value Arguments:
%       NumGames       - Number of test games per opponent (default: 20)
%       Opponents      - Cell array of opponent types (default: {'random', 'heuristic'})
%       Verbose        - Print progress (default: true)
%       OutputPath     - Path to save results (default: '')
%
%   Outputs:
%       results - Struct with test results
%
%   Example:
%       results = test_pretrained_agent(agent, NumGames=50);
%       fprintf('Win rate vs random: %.1f%%\n', results.vsRandom.winRate * 100);

    arguments
        agent
        options.NumGames (1,1) double = 20
        options.Opponents cell = {'random', 'heuristic'}
        options.Verbose logical = true
        options.OutputPath string = ""
    end

    log_print(options.Verbose, '\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  PRE-TRAINED AGENT VALIDATION\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Games per opponent: %d\n', options.NumGames);
    log_print(options.Verbose, '  Opponents: %s\n', strjoin(options.Opponents, ', '));
    log_print(options.Verbose, '================================================================\n\n');

    results = struct();
    results.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    results.numGamesPerOpponent = options.NumGames;

    %% Get actor network for inference
    actor = getActor(agent);
    actorNet = getModel(actor);

    %% Test against each opponent
    for o = 1:length(options.Opponents)
        oppType = options.Opponents{o};
        log_print(options.Verbose, '=== Testing vs %s ===\n', oppType);

        oppResults = struct();
        oppResults.wins = 0;
        oppResults.losses = 0;
        oppResults.draws = 0;
        oppResults.totalReward = 0;
        oppResults.gameResults = {};

        env = TangledEnvironment();
        env.Opponent = SimulatedOpponent('Style', oppType);

        for g = 1:options.NumGames
            gameResult = play_test_game(actorNet, env);
            oppResults.gameResults{end+1} = gameResult;
            oppResults.totalReward = oppResults.totalReward + gameResult.totalReward;

            switch gameResult.result
                case 'win'
                    oppResults.wins = oppResults.wins + 1;
                case 'loss'
                    oppResults.losses = oppResults.losses + 1;
                case 'draw'
                    oppResults.draws = oppResults.draws + 1;
            end

            if options.Verbose && mod(g, 5) == 0
                winRate = oppResults.wins / g;
                log_print(true, '  Game %2d/%d: %s (running WR: %.1f%%)\n', ...
                    g, options.NumGames, gameResult.result, winRate * 100);
            end
        end

        oppResults.winRate = oppResults.wins / options.NumGames;
        oppResults.avgReward = oppResults.totalReward / options.NumGames;

        % Store results
        fieldName = sprintf('vs%s', capitalize(oppType));
        results.(fieldName) = oppResults;

        log_print(options.Verbose, '\n  Results vs %s:\n', oppType);
        log_print(options.Verbose, '    Win rate:   %.1f%% (%d/%d)\n', ...
            oppResults.winRate * 100, oppResults.wins, options.NumGames);
        log_print(options.Verbose, '    Record:     %dW / %dL / %dD\n', ...
            oppResults.wins, oppResults.losses, oppResults.draws);
        log_print(options.Verbose, '    Avg reward: %.3f\n\n', oppResults.avgReward);

        delete(env);
    end

    %% Summary
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  VALIDATION SUMMARY\n');
    log_print(options.Verbose, '================================================================\n');

    passed = true;
    for o = 1:length(options.Opponents)
        oppType = options.Opponents{o};
        fieldName = sprintf('vs%s', capitalize(oppType));
        oppResults = results.(fieldName);

        % Define thresholds
        switch oppType
            case 'random'
                threshold = 0.30;  % Should beat random > 30%
            case 'heuristic'
                threshold = 0.20;  % Should compete with heuristic > 20%
            otherwise
                threshold = 0.10;
        end

        if oppResults.winRate >= threshold
            status = 'PASS';
        else
            status = 'FAIL';
            passed = false;
        end

        log_print(options.Verbose, '  vs %-10s: %.1f%% win rate [%s] (threshold: %.0f%%)\n', ...
            oppType, oppResults.winRate * 100, status, threshold * 100);
    end

    log_print(options.Verbose, '================================================================\n');

    if passed
        log_print(options.Verbose, '  OVERALL: PASSED\n');
        results.overallPassed = true;
    else
        log_print(options.Verbose, '  OVERALL: FAILED (some thresholds not met)\n');
        results.overallPassed = false;
    end

    log_print(options.Verbose, '================================================================\n\n');

    %% Save results if requested
    if options.OutputPath ~= ""
        save(options.OutputPath, 'results');
        log_print(options.Verbose, 'Results saved to: %s\n\n', options.OutputPath);
    end
end

function gameResult = play_test_game(actorNet, env)
%PLAY_TEST_GAME Play a single game and return results

    gameResult = struct();
    gameResult.moves = [];
    gameResult.rewards = [];
    gameResult.totalReward = 0;

    obs = reset(env);
    done = false;
    moveNum = 0;

    while ~done && moveNum < 20
        moveNum = moveNum + 1;

        % Get action mask
        mask = getActionMask(env.State);
        validActions = find(mask);

        if isempty(validActions)
            break;
        end

        % Get action from network
        probs = forward(actorNet, dlarray(obs, 'CB'));
        probs = extractdata(probs);

        % Apply mask and select action
        maskedProbs = probs .* mask(:);
        if sum(maskedProbs) > 0
            maskedProbs = maskedProbs / sum(maskedProbs);
            cumProbs = cumsum(maskedProbs);
            action = find(cumProbs >= rand(), 1);
        else
            action = validActions(randi(length(validActions)));
        end

        if isempty(action)
            action = validActions(randi(length(validActions)));
        end

        % Step environment
        [obs, reward, done, ~] = step(env, action);

        gameResult.moves(end+1) = action;
        gameResult.rewards(end+1) = reward;
        gameResult.totalReward = gameResult.totalReward + reward;
    end

    % Determine result
    if gameResult.totalReward > 0.5
        gameResult.result = 'win';
    elseif gameResult.totalReward < -0.5
        gameResult.result = 'loss';
    else
        gameResult.result = 'draw';
    end

    gameResult.numMoves = moveNum;
end

function s = capitalize(str)
    s = [upper(str(1)), str(2:end)];
end

function log_print(verbose, varargin)
    if verbose
        fprintf(varargin{:});
    end
end
