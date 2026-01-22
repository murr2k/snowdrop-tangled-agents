function results = train_selfplay(agent, options)
%TRAIN_SELFPLAY Train agent via self-play
%
%   results = train_selfplay(agent)
%   results = train_selfplay(agent, Name=Value)
%
%   Trains the agent by playing against a copy of itself.
%   The opponent is periodically updated to match the current agent.
%
%   Name-Value Arguments:
%       TotalEpisodes   - Total episodes to train (default: 500)
%       SyncInterval    - Episodes between opponent sync (default: 50)
%       Epsilon         - Opponent exploration rate (default: 0.1)
%       OutputDir       - Output directory (default: temp)
%       Verbose         - Print progress (default: true)

    arguments
        agent
        options.TotalEpisodes (1,1) double = 500
        options.SyncInterval (1,1) double = 50
        options.Epsilon (1,1) double = 0.1
        options.OutputDir string = ""
        options.Verbose logical = true
    end

    if options.OutputDir == ""
        options.OutputDir = fullfile(tempdir, sprintf('selfplay_%s', datestr(now, 'yyyymmdd_HHMMSS')));
    end

    if ~exist(options.OutputDir, 'dir')
        mkdir(options.OutputDir);
    end

    log_print(options.Verbose, '\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  SELF-PLAY TRAINING\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Total episodes:  %d\n', options.TotalEpisodes);
    log_print(options.Verbose, '  Sync interval:   %d episodes\n', options.SyncInterval);
    log_print(options.Verbose, '  Opponent epsilon: %.2f\n', options.Epsilon);
    log_print(options.Verbose, '================================================================\n\n');

    results = struct();
    results.outputDir = options.OutputDir;
    results.episodes = [];
    results.rewards = [];
    results.criticStats = [];

    %% Create self-play opponent
    selfOpp = SelfPlayOpponent(agent, 'Epsilon', options.Epsilon);
    env = TangledEnvironment('Opponent', selfOpp);

    %% Initial critic check
    log_print(options.Verbose, 'Initial critic state:\n');
    initialStats = quick_critic_check(agent);
    log_print(options.Verbose, '  Std: %.4f, WinLossDiff: %+.4f\n\n', ...
        initialStats.std, initialStats.winLossDiff);
    results.criticStats = [results.criticStats; initialStats];

    %% Training loop
    numChunks = ceil(options.TotalEpisodes / options.SyncInterval);
    episodesDone = 0;

    for chunk = 1:numChunks
        episodesToRun = min(options.SyncInterval, options.TotalEpisodes - episodesDone);
        if episodesToRun <= 0
            break;
        end

        log_print(options.Verbose, '=== Chunk %d/%d (%d episodes) ===\n', ...
            chunk, numChunks, episodesToRun);

        %% Train chunk
        trainOpts = rlTrainingOptions(...
            'MaxEpisodes', episodesToRun, ...
            'MaxStepsPerEpisode', 20, ...
            'ScoreAveragingWindowLength', 10, ...
            'Verbose', false, ...
            'Plots', 'none', ...
            'StopTrainingCriteria', 'none');

        tic;
        trainResult = train(agent, env, trainOpts);
        trainTime = toc;

        episodesDone = episodesDone + episodesToRun;

        %% Extract rewards
        if isprop(trainResult, 'EpisodeReward')
            chunkRewards = trainResult.EpisodeReward;
        else
            chunkRewards = zeros(episodesToRun, 1);
        end
        results.rewards = [results.rewards; chunkRewards];

        avgReward = mean(chunkRewards);
        winRate = sum(chunkRewards > 0.5) / length(chunkRewards);
        drawRate = sum(abs(chunkRewards) < 0.1) / length(chunkRewards);

        log_print(options.Verbose, '  Time: %.1fs, Avg reward: %+.3f\n', trainTime, avgReward);
        log_print(options.Verbose, '  Win: %.0f%%, Draw: %.0f%%, Loss: %.0f%%\n', ...
            winRate*100, drawRate*100, (1-winRate-drawRate)*100);

        %% Check critic
        criticStats = quick_critic_check(agent);
        results.criticStats = [results.criticStats; criticStats];
        log_print(options.Verbose, '  Critic WinLossDiff: %+.4f\n', criticStats.winLossDiff);

        %% Sync opponent to current agent
        selfOpp.updateAgent(agent);
        log_print(options.Verbose, '  [SYNC] Opponent updated to current policy\n\n');

        %% Save checkpoint
        checkpointPath = fullfile(options.OutputDir, sprintf('agent_ep%04d.mat', episodesDone));
        save(checkpointPath, 'agent');
    end

    %% Final summary
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  SELF-PLAY COMPLETE\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Total episodes: %d\n', episodesDone);
    log_print(options.Verbose, '  Final avg reward: %+.3f\n', mean(results.rewards(end-min(50,length(results.rewards))+1:end)));

    % Critic evolution
    log_print(options.Verbose, '  Critic WinLossDiff: %+.4f -> %+.4f\n', ...
        results.criticStats(1).winLossDiff, results.criticStats(end).winLossDiff);

    log_print(options.Verbose, '================================================================\n\n');

    %% Save final
    finalPath = fullfile(options.OutputDir, 'agent_final.mat');
    save(finalPath, 'agent');
    resultsPath = fullfile(options.OutputDir, 'selfplay_results.mat');
    save(resultsPath, 'results');

    log_print(options.Verbose, 'Saved to: %s\n\n', options.OutputDir);

    %% Cleanup
    delete(env);
end

function stats = quick_critic_check(agent)
    critic = getCritic(agent);
    criticNet = getModel(critic);

    randStates = randn(50, 50);
    dlRand = dlarray(randStates, 'CB');
    randPreds = extractdata(forward(criticNet, dlRand));

    stats.std = std(randPreds);
    stats.mean = mean(randPreds);

    winState = zeros(50, 1);
    winState(1:5) = 1;
    winState(6:10) = -1;
    winState(16) = 1;
    winState(17:31) = [0.5*ones(5,1); -0.5*ones(5,1); 0.25*ones(5,1)];
    winState(32) = 5/15;
    winState(41:45) = 1;

    loseState = zeros(50, 1);
    loseState(1:5) = -1;
    loseState(6:10) = 1;
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
