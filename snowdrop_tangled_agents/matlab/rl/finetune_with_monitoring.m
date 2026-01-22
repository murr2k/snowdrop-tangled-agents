function results = finetune_with_monitoring(agent, options)
%FINETUNE_WITH_MONITORING Fine-tune agent with critic monitoring
%
%   results = finetune_with_monitoring(agent)
%   results = finetune_with_monitoring(agent, Name=Value)
%
%   Fine-tunes the PPO agent via self-play while monitoring
%   critic predictions to ensure learning is progressing.
%
%   Name-Value Arguments:
%       TotalEpisodes   - Total episodes to train (default: 100)
%       CheckInterval   - Episodes between checks (default: 20)
%       OutputDir       - Directory for outputs (default: temp)
%       Verbose         - Print progress (default: true)
%
%   Outputs:
%       results - Struct with training history and critic stats

    arguments
        agent
        options.TotalEpisodes (1,1) double = 100
        options.CheckInterval (1,1) double = 20
        options.OutputDir string = ""
        options.Verbose logical = true
    end

    if options.OutputDir == ""
        options.OutputDir = fullfile(tempdir, sprintf('finetune_%s', datestr(now, 'yyyymmdd_HHMMSS')));
    end

    if ~exist(options.OutputDir, 'dir')
        mkdir(options.OutputDir);
    end

    log_print(options.Verbose, '\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  FINE-TUNING WITH CRITIC MONITORING\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Total episodes:   %d\n', options.TotalEpisodes);
    log_print(options.Verbose, '  Check interval:   %d episodes\n', options.CheckInterval);
    log_print(options.Verbose, '  Output directory: %s\n', options.OutputDir);
    log_print(options.Verbose, '================================================================\n\n');

    %% Initialize results
    results = struct();
    results.outputDir = options.OutputDir;
    results.checkpoints = {};
    results.criticStats = [];
    results.episodeRewards = [];
    results.winRates = [];

    numChecks = ceil(options.TotalEpisodes / options.CheckInterval);
    episodesPerCheck = options.CheckInterval;

    %% Create environment
    env = TangledEnvironment();

    %% Initial critic check
    log_print(options.Verbose, '=== Initial Critic State ===\n');
    initialStats = quick_critic_check(agent);
    results.criticStats = [results.criticStats; initialStats];
    log_print(options.Verbose, '  Std: %.4f, WinLossDiff: %+.4f\n\n', ...
        initialStats.std, initialStats.winLossDiff);

    %% Training loop with monitoring
    totalEpisodesDone = 0;

    for check = 1:numChecks
        episodesToRun = min(episodesPerCheck, options.TotalEpisodes - totalEpisodesDone);

        if episodesToRun <= 0
            break;
        end

        log_print(options.Verbose, '=== Training Chunk %d/%d (%d episodes) ===\n', ...
            check, numChecks, episodesToRun);

        %% Run training chunk
        trainOpts = rlTrainingOptions(...
            'MaxEpisodes', episodesToRun, ...
            'MaxStepsPerEpisode', 20, ...
            'ScoreAveragingWindowLength', 10, ...
            'Verbose', false, ...
            'Plots', 'none', ...
            'StopTrainingCriteria', 'none');

        % Train
        tic;
        trainResult = train(agent, env, trainOpts);
        trainTime = toc;

        totalEpisodesDone = totalEpisodesDone + episodesToRun;

        % Extract episode rewards
        if isprop(trainResult, 'EpisodeReward')
            chunkRewards = trainResult.EpisodeReward;
        elseif isfield(trainResult, 'EpisodeReward')
            chunkRewards = trainResult.EpisodeReward;
        else
            chunkRewards = zeros(episodesToRun, 1);
        end
        results.episodeRewards = [results.episodeRewards; chunkRewards];

        avgReward = mean(chunkRewards);
        winRate = sum(chunkRewards > 0.5) / length(chunkRewards);
        results.winRates = [results.winRates; winRate];

        log_print(options.Verbose, '  Training time: %.1f sec\n', trainTime);
        log_print(options.Verbose, '  Avg reward:    %+.3f\n', avgReward);
        log_print(options.Verbose, '  Win rate:      %.1f%%\n', winRate * 100);

        %% Check critic
        log_print(options.Verbose, '\n  Critic check:\n');
        criticStats = quick_critic_check(agent);
        results.criticStats = [results.criticStats; criticStats];

        log_print(options.Verbose, '    Std: %.4f (was: %.4f)\n', ...
            criticStats.std, results.criticStats(end-1).std);
        log_print(options.Verbose, '    WinLossDiff: %+.4f (was: %+.4f)\n', ...
            criticStats.winLossDiff, results.criticStats(end-1).winLossDiff);

        % Check for improvement
        if criticStats.winLossDiff > results.criticStats(end-1).winLossDiff
            log_print(options.Verbose, '    [IMPROVING] Win/loss distinction getting better\n');
        elseif criticStats.winLossDiff < results.criticStats(end-1).winLossDiff - 0.1
            log_print(options.Verbose, '    [WARNING] Win/loss distinction degrading\n');
        else
            log_print(options.Verbose, '    [STABLE] Minimal change\n');
        end

        %% Save checkpoint
        checkpoint = struct();
        checkpoint.episode = totalEpisodesDone;
        checkpoint.agent = agent;
        checkpoint.criticStats = criticStats;
        checkpoint.avgReward = avgReward;
        checkpoint.winRate = winRate;
        results.checkpoints{end+1} = checkpoint;

        % Save to file
        checkpointPath = fullfile(options.OutputDir, sprintf('checkpoint_%03d.mat', totalEpisodesDone));
        save(checkpointPath, 'agent', 'criticStats', 'avgReward', 'winRate');

        log_print(options.Verbose, '\n');
    end

    %% Final summary
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  FINE-TUNING COMPLETE\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Total episodes: %d\n', totalEpisodesDone);

    if ~isempty(results.episodeRewards)
        log_print(options.Verbose, '  Final avg reward: %+.3f\n', mean(results.episodeRewards(end-min(20,length(results.episodeRewards))+1:end)));
    end

    % Critic improvement
    initialStats = results.criticStats(1);
    finalStats = results.criticStats(end);

    log_print(options.Verbose, '\n  Critic Evolution:\n');
    log_print(options.Verbose, '    Std:         %.4f -> %.4f\n', initialStats.std, finalStats.std);
    log_print(options.Verbose, '    WinLossDiff: %+.4f -> %+.4f\n', initialStats.winLossDiff, finalStats.winLossDiff);

    if finalStats.winLossDiff > 0 && initialStats.winLossDiff <= 0
        log_print(options.Verbose, '\n  [SUCCESS] Critic learned to distinguish winning/losing!\n');
        results.criticLearned = true;
    elseif finalStats.winLossDiff > initialStats.winLossDiff + 0.1
        log_print(options.Verbose, '\n  [PROGRESS] Critic improving but not yet correct\n');
        results.criticLearned = false;
    else
        log_print(options.Verbose, '\n  [WARNING] Critic did not improve significantly\n');
        results.criticLearned = false;
    end

    log_print(options.Verbose, '================================================================\n\n');

    %% Save final results
    finalAgentPath = fullfile(options.OutputDir, 'agent_final.mat');
    save(finalAgentPath, 'agent');
    log_print(options.Verbose, 'Final agent saved: %s\n', finalAgentPath);

    resultsPath = fullfile(options.OutputDir, 'training_results.mat');
    save(resultsPath, 'results');
    log_print(options.Verbose, 'Results saved: %s\n\n', resultsPath);

    %% Cleanup
    delete(env);
end

function stats = quick_critic_check(agent)
%QUICK_CRITIC_CHECK Fast critic diagnostics
    critic = getCritic(agent);
    criticNet = getModel(critic);

    % Random states
    randStates = randn(50, 50);
    dlRand = dlarray(randStates, 'CB');
    randPreds = extractdata(forward(criticNet, dlRand));

    stats.std = std(randPreds);
    stats.mean = mean(randPreds);
    stats.range = max(randPreds) - min(randPreds);

    % Structured states (winning vs losing)
    winState = zeros(50, 1);
    winState(1:5) = 1;   % Our edges green
    winState(6:10) = -1; % Their edges purple
    winState(16) = 1;
    winState(17:31) = [0.5*ones(5,1); -0.5*ones(5,1); 0.25*ones(5,1)];
    winState(32) = 5/15;
    winState(41:45) = 1;

    loseState = zeros(50, 1);
    loseState(1:5) = -1;  % Our edges purple (bad)
    loseState(6:10) = 1;  % Their edges green
    loseState(16) = 1;
    loseState(17:31) = [0.5*ones(5,1); -0.5*ones(5,1); 0.25*ones(5,1)];
    loseState(32) = 5/15;
    loseState(41:45) = 1;

    structStates = [winState, loseState];
    dlStruct = dlarray(structStates, 'CB');
    structPreds = extractdata(forward(criticNet, dlStruct));

    stats.winPred = structPreds(1);
    stats.losePred = structPreds(2);
    stats.winLossDiff = structPreds(1) - structPreds(2);
end

function log_print(verbose, varargin)
    if verbose
        fprintf(varargin{:});
    end
end
