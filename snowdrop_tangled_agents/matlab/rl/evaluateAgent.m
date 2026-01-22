function results = evaluateAgent(agent, env, options)
%EVALUATEAGENT Evaluate trained agent performance
%
%   results = evaluateAgent(agent, env)
%   results = evaluateAgent(agent, env, Name=Value)
%
%   Evaluates a trained agent over multiple episodes without learning,
%   collecting detailed performance statistics.
%
%   Inputs:
%       agent - Trained RL agent
%       env   - TangledEnvironment instance
%
%   Name-Value Arguments:
%       NumEpisodes   - Number of evaluation episodes (default: 100)
%       Verbose       - Print progress (default: true)
%       Deterministic - Use greedy action selection (default: true)
%
%   Outputs:
%       results - Struct with fields:
%           .rewards      - [NumEpisodes x 1] episode rewards
%           .lengths      - [NumEpisodes x 1] episode lengths
%           .wins         - Number of wins
%           .losses       - Number of losses
%           .draws        - Number of draws
%           .winRate      - Win rate [0, 1]
%           .avgReward    - Average reward
%           .stdReward    - Reward standard deviation
%           .finalScores  - [NumEpisodes x 1] final game scores
%
%   Example:
%       % Load trained agent
%       data = load('ppo_final.mat');
%       agent = data.trainedAgent;
%
%       % Evaluate
%       env = TangledEnvironment();
%       results = evaluateAgent(agent, env, 'NumEpisodes', 200);
%       fprintf('Win Rate: %.1f%%\n', results.winRate * 100);

    arguments
        agent
        env
        options.NumEpisodes (1,1) double = 100
        options.Verbose logical = true
        options.Deterministic logical = true
    end

    %% Initialize results
    results = struct();
    results.rewards = zeros(options.NumEpisodes, 1);
    results.lengths = zeros(options.NumEpisodes, 1);
    results.finalScores = zeros(options.NumEpisodes, 1);
    results.wins = 0;
    results.losses = 0;
    results.draws = 0;

    if options.Verbose
        fprintf('\nEvaluating agent over %d episodes...\n', options.NumEpisodes);
    end

    %% Run evaluation episodes
    for episode = 1:options.NumEpisodes
        obs = reset(env);
        episodeReward = 0;
        stepCount = 0;
        done = false;

        while ~done && stepCount < 20
            % Get action mask
            mask = getActionMask(env.State);
            validActions = find(mask);

            if isempty(validActions)
                break;
            end

            % Get action (deterministic or stochastic)
            if options.Deterministic
                action = selectGreedyAction(agent, obs, mask);
            else
                action = selectStochasticAction(agent, obs, mask);
            end

            % Step
            [obs, reward, done, info] = step(env, action);
            episodeReward = episodeReward + reward;
            stepCount = stepCount + 1;
        end

        % Record results
        results.rewards(episode) = episodeReward;
        results.lengths(episode) = stepCount;

        if isfield(info, 'FinalScore')
            results.finalScores(episode) = info.FinalScore;
        end

        if isfield(info, 'Result')
            switch info.Result
                case 'win'
                    results.wins = results.wins + 1;
                case 'loss'
                    results.losses = results.losses + 1;
                otherwise
                    results.draws = results.draws + 1;
            end
        end

        % Progress
        if options.Verbose && mod(episode, 20) == 0
            currentWinRate = results.wins / episode;
            fprintf('  %d/%d episodes | Win Rate: %.1f%%\n', ...
                episode, options.NumEpisodes, currentWinRate * 100);
        end
    end

    %% Compute summary statistics
    results.winRate = results.wins / options.NumEpisodes;
    results.avgReward = mean(results.rewards);
    results.stdReward = std(results.rewards);
    results.avgLength = mean(results.lengths);
    results.avgFinalScore = mean(results.finalScores);

    %% Print summary
    if options.Verbose
        fprintf('\n=== Evaluation Results ===\n');
        fprintf('Episodes:     %d\n', options.NumEpisodes);
        fprintf('Win Rate:     %.1f%% (%d/%d)\n', ...
            results.winRate * 100, results.wins, options.NumEpisodes);
        fprintf('Record:       %dW / %dL / %dD\n', ...
            results.wins, results.losses, results.draws);
        fprintf('Avg Reward:   %.3f +/- %.3f\n', ...
            results.avgReward, results.stdReward);
        fprintf('Avg Score:    %.3f\n', results.avgFinalScore);
        fprintf('Avg Length:   %.1f steps\n', results.avgLength);
        fprintf('==========================\n');
    end
end

function action = selectGreedyAction(agent, obs, mask)
%SELECTGREEDYACTION Select highest probability valid action

    % Get action probabilities
    actorNet = getActor(agent);
    probs = predict(actorNet, dlarray(obs, 'CB'));
    probs = extractdata(probs);

    % Mask invalid actions
    probs(mask == 0) = -inf;

    % Select argmax
    [~, action] = max(probs);
end

function action = selectStochasticAction(agent, obs, mask)
%SELECTSTOCHASTICACTION Sample from masked action distribution

    % Get action probabilities
    actorNet = getActor(agent);
    probs = predict(actorNet, dlarray(obs, 'CB'));
    probs = extractdata(probs);

    % Mask and renormalize
    probs = probs .* mask;
    if sum(probs) > 0
        probs = probs / sum(probs);
    else
        probs = mask / sum(mask);
    end

    % Sample
    cumProbs = cumsum(probs);
    action = find(cumProbs >= rand(), 1);
end
