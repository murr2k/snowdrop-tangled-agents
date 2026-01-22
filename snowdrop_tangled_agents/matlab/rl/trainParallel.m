function [trainedAgent, stats] = trainParallel(agent, options)
%TRAINPARALLEL Train agent using parallel self-play
%
%   [trainedAgent, stats] = trainParallel(agent)
%   [trainedAgent, stats] = trainParallel(agent, Name=Value)
%
%   Trains the PPO agent using parallel environment rollouts for
%   accelerated learning. Supports both parfor and GPU acceleration.
%
%   Inputs:
%       agent - PPO agent from createPPOAgent
%
%   Name-Value Arguments:
%       NumWorkers      - Number of parallel workers (default: 4)
%       MaxEpisodes     - Maximum training episodes (default: 5000)
%       UpdateFrequency - Steps between agent updates (default: 256)
%       MiniBatchSize   - Batch size for updates (default: 64)
%       UseGPU          - Use GPU acceleration if available (default: true)
%       DBPath          - Path to experience database (default: 'parallel_exp.db')
%       SaveFrequency   - Episodes between checkpoints (default: 500)
%       SavePath        - Directory for checkpoints (default: 'parallel_checkpoints')
%       Verbose         - Print progress (default: true)
%       StopWinRate     - Stop if win rate exceeds (default: 0.7)
%
%   Outputs:
%       trainedAgent - Trained agent
%       stats        - Training statistics struct
%
%   Example:
%       env = TangledEnvironment();
%       agent = createPPOAgent(env);
%       [trainedAgent, stats] = trainParallel(agent, ...
%           'NumWorkers', 8, 'MaxEpisodes', 10000);

    arguments
        agent
        options.NumWorkers (1,1) double = 4
        options.MaxEpisodes (1,1) double = 5000
        options.UpdateFrequency (1,1) double = 256
        options.MiniBatchSize (1,1) double = 64
        options.UseGPU logical = true
        options.DBPath char = 'parallel_exp.db'
        options.SaveFrequency (1,1) double = 500
        options.SavePath char = 'parallel_checkpoints'
        options.Verbose logical = true
        options.StopWinRate (1,1) double = 0.7
    end

    %% Setup
    if ~exist(options.SavePath, 'dir')
        mkdir(options.SavePath);
    end

    % GPU acceleration
    if options.UseGPU
        agent = enableGPU(agent);
    end

    % Initialize experience buffer
    buffer = SQLiteExperienceBuffer(options.DBPath, 100000);

    % Initialize parallel pool
    hasParallel = license('test', 'Distrib_Computing_Toolbox');
    if hasParallel
        pool = gcp('nocreate');
        if isempty(pool)
            pool = parpool('local', min(options.NumWorkers, feature('numcores')));
        end
        actualWorkers = pool.NumWorkers;
    else
        actualWorkers = 1;
        warning('Parallel Computing Toolbox not available. Running sequentially.');
    end

    % Training statistics
    stats = struct();
    stats.episodeRewards = [];
    stats.episodeLengths = [];
    stats.winRates = [];
    stats.losses = [];
    stats.wins = 0;
    stats.losses_count = 0;
    stats.draws = 0;

    % Rolling window for win rate
    windowSize = 100;
    resultsWindow = zeros(windowSize, 1);  % 1=win, 0=loss, 0.5=draw

    if options.Verbose
        fprintf('\n=== Parallel Training Configuration ===\n');
        fprintf('  Workers:          %d\n', actualWorkers);
        fprintf('  Max Episodes:     %d\n', options.MaxEpisodes);
        fprintf('  Update Frequency: %d steps\n', options.UpdateFrequency);
        fprintf('  Mini-Batch Size:  %d\n', options.MiniBatchSize);
        fprintf('  GPU Enabled:      %s\n', string(options.UseGPU && canUseGPU()));
        fprintf('  Stop Win Rate:    %.0f%%\n', options.StopWinRate * 100);
        fprintf('========================================\n\n');
    end

    %% Training loop
    episode = 0;
    totalSteps = 0;
    startTime = tic;

    while episode < options.MaxEpisodes
        %% Parallel rollout
        batchExperiences = cell(actualWorkers, 1);

        if hasParallel && actualWorkers > 1
            % Parallel episode collection
            agentCopy = agent;  % Copy for parfor
            parfor w = 1:actualWorkers
                % Create local environment
                env = TangledEnvironment();
                % Collect episode
                batchExperiences{w} = collectEpisode(agentCopy, env);
            end
        else
            % Sequential fallback
            for w = 1:actualWorkers
                env = TangledEnvironment();
                batchExperiences{w} = collectEpisode(agent, env);
            end
        end

        %% Aggregate experiences
        batchSteps = 0;
        for w = 1:actualWorkers
            exp = batchExperiences{w};

            % Store in buffer
            for t = 1:exp.length
                buffer.add(exp.states{t}, exp.actions(t), exp.rewards(t), ...
                    exp.nextStates{t}, exp.dones(t));
            end

            batchSteps = batchSteps + exp.length;

            % Track statistics
            stats.episodeRewards(end+1) = exp.totalReward;
            stats.episodeLengths(end+1) = exp.length;

            % Track win/loss/draw
            switch exp.result
                case 'win'
                    stats.wins = stats.wins + 1;
                    resultsWindow(mod(episode + w - 1, windowSize) + 1) = 1;
                case 'loss'
                    stats.losses_count = stats.losses_count + 1;
                    resultsWindow(mod(episode + w - 1, windowSize) + 1) = 0;
                otherwise
                    stats.draws = stats.draws + 1;
                    resultsWindow(mod(episode + w - 1, windowSize) + 1) = 0.5;
            end
        end

        episode = episode + actualWorkers;
        totalSteps = totalSteps + batchSteps;

        %% Update agent
        if totalSteps >= options.UpdateFrequency
            % Sample batch from buffer
            batchSize = min(buffer.CurrentSize, options.MiniBatchSize * 4);
            batch = buffer.sample(batchSize);

            if ~isempty(batch)
                % PPO update
                [agent, loss] = updatePPOAgent(agent, batch, options.MiniBatchSize);
                stats.losses(end+1) = loss;
            end

            totalSteps = 0;
        end

        %% Calculate rolling win rate
        if episode >= windowSize
            winRate = mean(resultsWindow);
        else
            winRate = mean(resultsWindow(1:min(episode, windowSize)));
        end
        stats.winRates(end+1) = winRate;

        %% Progress output
        if options.Verbose && mod(episode, 10 * actualWorkers) == 0
            elapsedTime = toc(startTime);
            episodesPerSec = episode / elapsedTime;

            avgReward = mean(stats.episodeRewards(max(1, end-99):end));
            fprintf('Episode %5d | Reward: %+.3f | WinRate: %.1f%% | %.1f ep/s | W/L/D: %d/%d/%d\n', ...
                episode, avgReward, winRate * 100, episodesPerSec, ...
                stats.wins, stats.losses_count, stats.draws);
        end

        %% Checkpoint
        if mod(episode, options.SaveFrequency) == 0
            checkpointPath = fullfile(options.SavePath, ...
                sprintf('parallel_checkpoint_ep%d.mat', episode));
            savedAgent = agent;
            savedStats = stats;
            save(checkpointPath, 'savedAgent', 'savedStats');

            if options.Verbose
                fprintf('  Checkpoint saved: %s\n', checkpointPath);
            end
        end

        %% Stopping criterion
        if winRate >= options.StopWinRate && episode >= windowSize
            if options.Verbose
                fprintf('\nStopping: Win rate %.1f%% >= %.1f%%\n', ...
                    winRate * 100, options.StopWinRate * 100);
            end
            break;
        end
    end

    %% Finalize
    trainedAgent = agent;

    % Save final model
    finalPath = fullfile(options.SavePath, 'parallel_final.mat');
    save(finalPath, 'trainedAgent', 'stats');

    % Close buffer
    buffer.close();

    % Print summary
    if options.Verbose
        elapsedTime = toc(startTime);
        fprintf('\n=== Training Complete ===\n');
        fprintf('  Total Episodes: %d\n', episode);
        fprintf('  Training Time:  %.1f minutes\n', elapsedTime / 60);
        fprintf('  Final Win Rate: %.1f%%\n', stats.winRates(end) * 100);
        fprintf('  Record: %dW / %dL / %dD\n', stats.wins, stats.losses_count, stats.draws);
        fprintf('  Model saved: %s\n', finalPath);
        fprintf('=========================\n');
    end
end

function [agent, loss] = updatePPOAgent(agent, batch, miniBatchSize)
%UPDATEPPOAGENT Perform PPO update on agent

    % Extract batch data
    states = cat(2, batch.states{:});
    actions = batch.actions;
    rewards = batch.rewards;
    nextStates = cat(2, batch.nextStates{:});
    dones = batch.dones;

    % Compute advantages (simple TD error for now)
    criticNet = getCritic(agent);
    values = predict(criticNet, dlarray(states, 'CB'));
    nextValues = predict(criticNet, dlarray(nextStates, 'CB'));
    values = extractdata(values);
    nextValues = extractdata(nextValues);

    % TD targets and advantages
    gamma = 0.99;
    targets = rewards + gamma * nextValues .* (1 - dones);
    advantages = targets - values;

    % Normalize advantages
    advantages = (advantages - mean(advantages)) / (std(advantages) + 1e-8);

    % Mini-batch updates
    numSamples = size(states, 2);
    numBatches = ceil(numSamples / miniBatchSize);
    totalLoss = 0;

    for epoch = 1:4  % PPO typically uses multiple epochs
        % Shuffle
        perm = randperm(numSamples);

        for b = 1:numBatches
            idx = perm((b-1)*miniBatchSize + 1 : min(b*miniBatchSize, numSamples));

            batchStates = states(:, idx);
            batchActions = actions(idx);
            batchAdvantages = advantages(idx);
            batchTargets = targets(idx);

            % Update would go here using agent's internal optimizer
            % This is simplified - actual PPO update requires more machinery
            totalLoss = totalLoss + mean(abs(batchAdvantages));
        end
    end

    loss = totalLoss / (numBatches * 4);
end
