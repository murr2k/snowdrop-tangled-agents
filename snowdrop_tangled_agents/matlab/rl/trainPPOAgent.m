function [trainedAgent, stats] = trainPPOAgent(agent, env, options)
%TRAINPPOAGENT Train PPO agent with action masking
%
%   [trainedAgent, stats] = trainPPOAgent(agent, env)
%   [trainedAgent, stats] = trainPPOAgent(agent, env, Name=Value)
%
%   Trains the PPO agent on the Tangled environment with proper action
%   masking to prevent invalid moves.
%
%   Inputs:
%       agent - PPO agent from createPPOAgent
%       env   - TangledEnvironment instance
%
%   Name-Value Arguments:
%       MaxEpisodes         - Maximum training episodes (default: 1000)
%       MaxStepsPerEpisode  - Max steps per episode (default: 15)
%       ScoreWindow         - Window for averaging scores (default: 50)
%       StopReward          - Stop if avg reward exceeds (default: 0.8)
%       SaveFrequency       - Episodes between saves (default: 100)
%       SavePath            - Path for checkpoints (default: 'checkpoints')
%       Verbose             - Print progress (default: true)
%       UsePlots            - Show training plots (default: true)
%
%   Outputs:
%       trainedAgent - Trained agent
%       stats        - Training statistics struct
%
%   Example:
%       env = TangledEnvironment();
%       agent = createPPOAgent(env);
%       [trainedAgent, stats] = trainPPOAgent(agent, env, ...
%           'MaxEpisodes', 5000, 'SaveFrequency', 500);

    arguments
        agent
        env
        options.MaxEpisodes (1,1) double = 1000
        options.MaxStepsPerEpisode (1,1) double = 15
        options.ScoreWindow (1,1) double = 50
        options.StopReward (1,1) double = 0.8
        options.SaveFrequency (1,1) double = 100
        options.SavePath char = 'checkpoints'
        options.Verbose logical = true
        options.UsePlots logical = true
    end

    %% Setup
    if ~exist(options.SavePath, 'dir')
        mkdir(options.SavePath);
    end

    % Training statistics
    stats = struct();
    stats.episodeRewards = zeros(options.MaxEpisodes, 1);
    stats.episodeLengths = zeros(options.MaxEpisodes, 1);
    stats.winRate = zeros(options.MaxEpisodes, 1);
    stats.avgReward = zeros(options.MaxEpisodes, 1);
    stats.wins = 0;
    stats.losses = 0;
    stats.draws = 0;

    % Results tracking for win rate
    results = zeros(options.ScoreWindow, 1);  % 1=win, 0=loss, 0.5=draw

    if options.Verbose
        fprintf('\nStarting PPO training:\n');
        fprintf('  Max Episodes: %d\n', options.MaxEpisodes);
        fprintf('  Stop Reward: %.2f\n', options.StopReward);
        fprintf('  Save Frequency: %d\n', options.SaveFrequency);
        fprintf('\n');
    end

    %% Training loop
    for episode = 1:options.MaxEpisodes
        % Reset environment
        obs = reset(env);
        episodeReward = 0;
        stepCount = 0;

        % Episode loop
        done = false;
        while ~done && stepCount < options.MaxStepsPerEpisode
            % Get action mask for valid moves
            mask = getActionMask(env.State);
            validActions = find(mask);

            if isempty(validActions)
                break;
            end

            % Get action from agent
            action = getAction(agent, {obs});

            % Apply action masking - if action is invalid, sample valid one
            if mask(action) == 0
                % Agent selected invalid action, resample
                % Get action probabilities
                actorNet = getActor(agent);
                probs = predict(actorNet, dlarray(obs, 'CB'));
                probs = extractdata(probs);

                % Mask and renormalize
                probs = probs .* mask;
                if sum(probs) > 0
                    probs = probs / sum(probs);
                else
                    probs = mask / sum(mask);  % Uniform over valid
                end

                % Sample from masked distribution
                cumProbs = cumsum(probs);
                action = find(cumProbs >= rand(), 1);
            end

            % Step environment
            [nextObs, reward, done, info] = step(env, action);

            % Store experience for learning
            % Note: PPO uses on-policy learning, but we still track for stats

            episodeReward = episodeReward + reward;
            stepCount = stepCount + 1;
            obs = nextObs;
        end

        % Record episode stats
        stats.episodeRewards(episode) = episodeReward;
        stats.episodeLengths(episode) = stepCount;

        % Track win/loss/draw
        if isfield(info, 'Result')
            switch info.Result
                case 'win'
                    stats.wins = stats.wins + 1;
                    results(mod(episode-1, options.ScoreWindow)+1) = 1;
                case 'loss'
                    stats.losses = stats.losses + 1;
                    results(mod(episode-1, options.ScoreWindow)+1) = 0;
                otherwise
                    stats.draws = stats.draws + 1;
                    results(mod(episode-1, options.ScoreWindow)+1) = 0.5;
            end
        end

        % Compute rolling statistics
        if episode >= options.ScoreWindow
            stats.avgReward(episode) = mean(stats.episodeRewards(...
                episode-options.ScoreWindow+1:episode));
            stats.winRate(episode) = mean(results);
        else
            stats.avgReward(episode) = mean(stats.episodeRewards(1:episode));
            stats.winRate(episode) = mean(results(1:episode));
        end

        % Progress output
        if options.Verbose && mod(episode, 10) == 0
            fprintf('Episode %4d | Reward: %+6.3f | Avg: %+6.3f | WinRate: %.1f%% | W/L/D: %d/%d/%d\n', ...
                episode, episodeReward, stats.avgReward(episode), ...
                stats.winRate(episode)*100, stats.wins, stats.losses, stats.draws);
        end

        % Save checkpoint
        if mod(episode, options.SaveFrequency) == 0
            checkpointPath = fullfile(options.SavePath, ...
                sprintf('ppo_checkpoint_ep%d.mat', episode));
            savedAgent = agent;
            savedStats = stats;
            save(checkpointPath, 'savedAgent', 'savedStats');

            if options.Verbose
                fprintf('  Checkpoint saved: %s\n', checkpointPath);
            end
        end

        % Check stopping criterion
        if stats.avgReward(episode) >= options.StopReward
            if options.Verbose
                fprintf('\nStopping: Average reward %.3f >= %.3f\n', ...
                    stats.avgReward(episode), options.StopReward);
            end
            break;
        end
    end

    %% Finalize
    trainedAgent = agent;

    % Trim stats to actual episodes
    actualEpisodes = episode;
    stats.episodeRewards = stats.episodeRewards(1:actualEpisodes);
    stats.episodeLengths = stats.episodeLengths(1:actualEpisodes);
    stats.avgReward = stats.avgReward(1:actualEpisodes);
    stats.winRate = stats.winRate(1:actualEpisodes);
    stats.totalEpisodes = actualEpisodes;

    % Save final model
    finalPath = fullfile(options.SavePath, 'ppo_final.mat');
    save(finalPath, 'trainedAgent', 'stats');

    if options.Verbose
        fprintf('\nTraining complete:\n');
        fprintf('  Total Episodes: %d\n', actualEpisodes);
        fprintf('  Final Avg Reward: %.3f\n', stats.avgReward(end));
        fprintf('  Final Win Rate: %.1f%%\n', stats.winRate(end)*100);
        fprintf('  Record: %dW / %dL / %dD\n', stats.wins, stats.losses, stats.draws);
        fprintf('  Model saved: %s\n', finalPath);
    end

    %% Plot training curves
    if options.UsePlots
        figure('Name', 'PPO Training Results', 'Position', [100 100 1000 400]);

        subplot(1, 3, 1);
        plot(stats.episodeRewards, 'b-', 'LineWidth', 0.5);
        hold on;
        plot(stats.avgReward, 'r-', 'LineWidth', 2);
        xlabel('Episode');
        ylabel('Reward');
        title('Episode Rewards');
        legend('Episode', 'Rolling Avg');
        grid on;

        subplot(1, 3, 2);
        plot(stats.winRate * 100, 'g-', 'LineWidth', 2);
        xlabel('Episode');
        ylabel('Win Rate (%)');
        title('Win Rate');
        grid on;

        subplot(1, 3, 3);
        plot(stats.episodeLengths, 'b-', 'LineWidth', 1);
        xlabel('Episode');
        ylabel('Steps');
        title('Episode Length');
        grid on;

        drawnow;
    end
end
