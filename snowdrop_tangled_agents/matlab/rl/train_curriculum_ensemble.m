function results = train_curriculum_ensemble(options)
%TRAIN_CURRICULUM_ENSEMBLE Full curriculum training with MC-guided ensemble
%
%   results = train_curriculum_ensemble()
%   results = train_curriculum_ensemble(Name=Value)
%
%   Creates a fresh PPO agent and trains through curriculum:
%   1. Random opponent (learn basics)
%   2. Petersen opponent (learn strategy - MUST beat this)
%   3. Self-play with ensemble (refinement)
%
%   Uses reward shaping and MC-guided ensemble for improved learning.
%
%   Name-Value Arguments:
%       OutputDir       - Output directory (default: auto-generated)
%       NumWorkers      - Parallel workers for MC (default: 22)
%       RolloutsPerAction - MC rollouts per action (default: 50)
%       Verbose         - Print progress (default: true)
%
%   Curriculum Settings:
%       Level 1 (Random):   200 episodes, must reach 70% win rate
%       Level 2 (Petersen): 500 episodes, must reach 50% win rate
%       Level 3 (Self-play): 300 episodes with ensemble refinement

    arguments
        options.OutputDir string = ""
        options.NumWorkers (1,1) int32 = 22
        options.RolloutsPerAction (1,1) int32 = 50
        options.Verbose logical = true
    end

    if options.OutputDir == ""
        options.OutputDir = fullfile(pwd, sprintf('training_%s', datestr(now, 'yyyymmdd_HHMMSS')));
    end

    if ~exist(options.OutputDir, 'dir')
        mkdir(options.OutputDir);
    end

    %% Banner
    printBanner(options);

    results = struct();
    results.outputDir = options.OutputDir;
    results.startTime = datetime('now');
    results.levels = {};

    %% Initialize parallel pool
    % Auto-detect max workers if requested exceeds available
    cluster = parcluster('local');
    maxWorkers = cluster.NumWorkers;
    actualWorkers = min(options.NumWorkers, maxWorkers);

    if actualWorkers < options.NumWorkers
        log_print(options.Verbose, 'Note: Requested %d workers but max available is %d\n', options.NumWorkers, maxWorkers);
    end

    log_print(options.Verbose, 'Initializing parallel pool with %d workers...\n', actualWorkers);
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('local', actualWorkers);
    elseif pool.NumWorkers ~= actualWorkers
        delete(pool);
        parpool('local', actualWorkers);
    end
    options.NumWorkers = actualWorkers;  % Update for downstream use
    log_print(options.Verbose, 'Parallel pool ready.\n\n');

    %% Create fresh PPO agent
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  CREATING FRESH PPO AGENT\n');
    log_print(options.Verbose, '================================================================\n\n');

    agent = createFreshPPOAgent(options.Verbose);

    % Initial critic check
    criticStats = checkCritic(agent);
    log_print(options.Verbose, 'Initial critic: std=%.4f, range=[%.3f, %.3f]\n\n', ...
        criticStats.std, criticStats.min, criticStats.max);

    %% Save initial agent
    initialPath = fullfile(options.OutputDir, 'agent_initial.mat');
    save(initialPath, 'agent');
    log_print(options.Verbose, 'Saved initial agent: %s\n\n', initialPath);

    %% Level 1: Random Opponent
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  LEVEL 1: RANDOM OPPONENT\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Target: 70%% win rate\n');
    log_print(options.Verbose, '================================================================\n\n');

    level1Result = trainLevel(agent, 'random', struct(...
        'maxEpisodes', 200, ...
        'targetWinRate', 0.70, ...
        'checkInterval', 50, ...
        'outputDir', options.OutputDir, ...
        'levelName', 'level1_random', ...
        'verbose', options.Verbose));

    results.levels{1} = level1Result;
    agent = level1Result.agent;

    if level1Result.winRate < 0.60
        log_print(options.Verbose, '\n[WARNING] Failed to reach 60%% vs random. Continuing anyway.\n\n');
    end

    %% Level 2: Petersen Opponent (CRITICAL)
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  LEVEL 2: PETERSEN OPPONENT\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Target: 50%% win rate (critical milestone)\n');
    log_print(options.Verbose, '================================================================\n\n');

    level2Result = trainLevel(agent, 'petersen', struct(...
        'maxEpisodes', 500, ...
        'targetWinRate', 0.50, ...
        'checkInterval', 100, ...
        'outputDir', options.OutputDir, ...
        'levelName', 'level2_petersen', ...
        'verbose', options.Verbose));

    results.levels{2} = level2Result;
    agent = level2Result.agent;

    if level2Result.winRate < 0.40
        log_print(options.Verbose, '\n[WARNING] Win rate vs petersen below 40%%. Consider more training.\n\n');
    end

    %% Level 3: Self-Play with Ensemble Refinement
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  LEVEL 3: SELF-PLAY WITH ENSEMBLE\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Using MC-guided ensemble for move selection\n');
    log_print(options.Verbose, '================================================================\n\n');

    level3Result = trainSelfPlayEnsemble(agent, struct(...
        'maxEpisodes', 300, ...
        'syncInterval', 50, ...
        'numWorkers', options.NumWorkers, ...
        'rolloutsPerAction', options.RolloutsPerAction, ...
        'outputDir', options.OutputDir, ...
        'levelName', 'level3_selfplay', ...
        'verbose', options.Verbose));

    results.levels{3} = level3Result;
    agent = level3Result.agent;

    %% Final evaluation
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  FINAL EVALUATION\n');
    log_print(options.Verbose, '================================================================\n\n');

    evalResults = evaluateAgent(agent, options.Verbose);
    results.evaluation = evalResults;

    %% Summary
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  TRAINING COMPLETE\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Duration: %s\n', string(datetime('now') - results.startTime));
    log_print(options.Verbose, '  Output: %s\n', options.OutputDir);
    log_print(options.Verbose, '\n  Win Rates:\n');
    log_print(options.Verbose, '    vs Random:   %.1f%%\n', evalResults.vsRandom * 100);
    log_print(options.Verbose, '    vs Petersen: %.1f%%\n', evalResults.vsPetersen * 100);
    log_print(options.Verbose, '    vs MCTS:     %.1f%%\n', evalResults.vsMCTS * 100);
    log_print(options.Verbose, '================================================================\n\n');

    %% Save final
    finalPath = fullfile(options.OutputDir, 'agent_final.mat');
    save(finalPath, 'agent', 'results');
    log_print(options.Verbose, 'Saved final agent: %s\n\n', finalPath);

    % Copy to models directory
    modelsDir = fullfile(fileparts(mfilename('fullpath')), '..', 'models');
    if ~exist(modelsDir, 'dir')
        mkdir(modelsDir);
    end
    copyfile(finalPath, fullfile(modelsDir, 'agent_ensemble_v1.mat'));
    log_print(options.Verbose, 'Copied to models directory.\n');
end

%% Helper Functions

function agent = createFreshPPOAgent(verbose)
    %CREATEFRESHPPOAGENT Create a new PPO agent with good initialization

    % Create temporary environment to get specs
    opp = SimulatedOpponent('Style', 'random');
    env = TangledEnvironment('Opponent', opp, 'AutoCorrect', false);
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);

    % Actor network (policy)
    actorLayers = [
        featureInputLayer(50, 'Name', 'obs')
        fullyConnectedLayer(128, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(128, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(64, 'Name', 'fc3')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(30, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
    ];

    % Critic network (value function)
    criticLayers = [
        featureInputLayer(50, 'Name', 'obs')
        fullyConnectedLayer(128, 'Name', 'fc1')
        reluLayer('Name', 'relu1')
        fullyConnectedLayer(128, 'Name', 'fc2')
        reluLayer('Name', 'relu2')
        fullyConnectedLayer(64, 'Name', 'fc3')
        reluLayer('Name', 'relu3')
        fullyConnectedLayer(1, 'Name', 'value')
    ];

    actor = rlDiscreteCategoricalActor(actorLayers, obsInfo, actInfo);
    critic = rlValueFunction(criticLayers, obsInfo);

    % PPO options
    agentOpts = rlPPOAgentOptions(...
        'ExperienceHorizon', 128, ...
        'ClipFactor', 0.2, ...
        'EntropyLossWeight', 0.02, ...
        'MiniBatchSize', 64, ...
        'NumEpoch', 4, ...
        'AdvantageEstimateMethod', 'gae', ...
        'GAEFactor', 0.95, ...
        'SampleTime', 1, ...
        'DiscountFactor', 0.99);

    agentOpts.ActorOptimizerOptions = rlOptimizerOptions('LearnRate', 3e-4);
    agentOpts.CriticOptimizerOptions = rlOptimizerOptions('LearnRate', 1e-3);

    agent = rlPPOAgent(actor, critic, agentOpts);

    delete(env);

    if verbose
        fprintf('Created fresh PPO agent:\n');
        fprintf('  Actor:  50 -> 128 -> 128 -> 64 -> 30 (softmax)\n');
        fprintf('  Critic: 50 -> 128 -> 128 -> 64 -> 1\n');
        fprintf('  Entropy weight: 0.02\n');
        fprintf('  Clip factor: 0.2\n\n');
    end
end

function result = trainLevel(agent, oppStyle, opts)
    %TRAINLEVEL Train against a specific opponent style

    opp = SimulatedOpponent('Style', oppStyle);
    env = TangledEnvironment('Opponent', opp, 'AutoCorrect', true);

    result = struct();
    result.opponent = oppStyle;
    result.episodes = 0;
    result.rewards = [];
    result.winRates = [];

    numChunks = ceil(opts.maxEpisodes / opts.checkInterval);

    for chunk = 1:numChunks
        episodesToRun = min(opts.checkInterval, opts.maxEpisodes - result.episodes);
        if episodesToRun <= 0
            break;
        end

        log_print(opts.verbose, 'Chunk %d: Training %d episodes vs %s...\n', chunk, episodesToRun, oppStyle);

        trainOpts = rlTrainingOptions(...
            'MaxEpisodes', episodesToRun, ...
            'MaxStepsPerEpisode', 20, ...
            'ScoreAveragingWindowLength', 20, ...
            'Verbose', false, ...
            'Plots', 'none', ...
            'StopTrainingCriteria', 'none');

        tic;
        trainResult = train(agent, env, trainOpts);
        trainTime = toc;

        result.episodes = result.episodes + episodesToRun;

        % Extract rewards
        if isprop(trainResult, 'EpisodeReward')
            chunkRewards = trainResult.EpisodeReward;
        else
            chunkRewards = zeros(episodesToRun, 1);
        end
        result.rewards = [result.rewards; chunkRewards];

        % Calculate win rate
        wins = sum(chunkRewards > 0.3);
        winRate = wins / length(chunkRewards);
        result.winRates = [result.winRates; winRate];

        log_print(opts.verbose, '  Time: %.1fs, Avg reward: %+.3f, Win rate: %.1f%%\n', ...
            trainTime, mean(chunkRewards), winRate * 100);

        % Check critic
        criticStats = checkCritic(agent);
        log_print(opts.verbose, '  Critic std: %.4f, WinLossDiff: %+.4f\n', ...
            criticStats.std, criticStats.winLossDiff);

        % Early stop if target reached
        if winRate >= opts.targetWinRate
            log_print(opts.verbose, '\n  [SUCCESS] Target win rate %.0f%% reached!\n\n', opts.targetWinRate * 100);
            break;
        end
    end

    result.winRate = mean(result.winRates(max(1,end-2):end));  % Last 3 chunks
    result.agent = agent;

    % Save checkpoint
    checkpointPath = fullfile(opts.outputDir, sprintf('%s.mat', opts.levelName));
    save(checkpointPath, 'agent', 'result');
    log_print(opts.verbose, 'Saved checkpoint: %s\n\n', checkpointPath);

    delete(env);
end

function result = trainSelfPlayEnsemble(agent, opts)
    %TRAINSELFPLAYENSEMBLE Self-play training with ensemble guidance

    result = struct();
    result.episodes = 0;
    result.rewards = [];

    % Create ensemble policy for the opponent
    log_print(opts.verbose, 'Creating ensemble policy (TopK=5, Rollouts=%d)...\n', opts.rolloutsPerAction);
    ensemble = EnsemblePolicy(agent, ...
        'TopK', 5, ...
        'RolloutsPerAction', opts.rolloutsPerAction, ...
        'NumWorkers', opts.numWorkers);

    % Create self-play opponent using ensemble
    selfOpp = EnsembleSelfPlayOpponent(ensemble);
    env = TangledEnvironment('Opponent', selfOpp, 'AutoCorrect', true);

    numChunks = ceil(opts.maxEpisodes / opts.syncInterval);

    for chunk = 1:numChunks
        episodesToRun = min(opts.syncInterval, opts.maxEpisodes - result.episodes);
        if episodesToRun <= 0
            break;
        end

        log_print(opts.verbose, '\nChunk %d: Training %d episodes (self-play with ensemble)...\n', chunk, episodesToRun);

        trainOpts = rlTrainingOptions(...
            'MaxEpisodes', episodesToRun, ...
            'MaxStepsPerEpisode', 20, ...
            'ScoreAveragingWindowLength', 20, ...
            'Verbose', false, ...
            'Plots', 'none', ...
            'StopTrainingCriteria', 'none');

        tic;
        trainResult = train(agent, env, trainOpts);
        trainTime = toc;

        result.episodes = result.episodes + episodesToRun;

        if isprop(trainResult, 'EpisodeReward')
            chunkRewards = trainResult.EpisodeReward;
        else
            chunkRewards = zeros(episodesToRun, 1);
        end
        result.rewards = [result.rewards; chunkRewards];

        avgReward = mean(chunkRewards);
        winRate = sum(chunkRewards > 0.3) / length(chunkRewards);
        drawRate = sum(abs(chunkRewards) < 0.2) / length(chunkRewards);

        log_print(opts.verbose, '  Time: %.1fs, Avg: %+.3f, Win: %.0f%%, Draw: %.0f%%\n', ...
            trainTime, avgReward, winRate*100, drawRate*100);

        % Check critic
        criticStats = checkCritic(agent);
        log_print(opts.verbose, '  Critic std: %.4f, WinLossDiff: %+.4f\n', ...
            criticStats.std, criticStats.winLossDiff);

        % Sync opponent to current agent
        ensemble.updateAgent(agent);
        selfOpp.updateEnsemble(ensemble);
        log_print(opts.verbose, '  [SYNC] Opponent updated to current policy\n');
    end

    result.agent = agent;

    % Save checkpoint
    checkpointPath = fullfile(opts.outputDir, sprintf('%s.mat', opts.levelName));
    save(checkpointPath, 'agent', 'result');
    log_print(opts.verbose, '\nSaved checkpoint: %s\n\n', checkpointPath);

    delete(env);
end

function evalResults = evaluateAgent(agent, verbose)
    %EVALUATEAGENT Evaluate agent against multiple opponents

    evalResults = struct();
    opponents = {'random', 'petersen', 'mcts'};
    numGames = 50;

    for i = 1:length(opponents)
        oppStyle = opponents{i};
        log_print(verbose, 'Evaluating vs %s (%d games)...\n', oppStyle, numGames);

        opp = SimulatedOpponent('Style', oppStyle);
        env = TangledEnvironment('Opponent', opp, 'AutoCorrect', true);

        wins = 0;
        totalReward = 0;

        for g = 1:numGames
            obs = reset(env);
            isDone = false;
            episodeReward = 0;

            while ~isDone
                action = getAction(agent, obs);
                if iscell(action)
                    action = action{1};
                end
                [obs, reward, isDone, ~] = step(env, action);
                episodeReward = episodeReward + reward;
            end

            totalReward = totalReward + episodeReward;
            if episodeReward > 0.3
                wins = wins + 1;
            end
        end

        winRate = wins / numGames;
        avgReward = totalReward / numGames;

        log_print(verbose, '  Win rate: %.1f%%, Avg reward: %+.3f\n', winRate*100, avgReward);

        switch oppStyle
            case 'random'
                evalResults.vsRandom = winRate;
            case 'petersen'
                evalResults.vsPetersen = winRate;
            case 'mcts'
                evalResults.vsMCTS = winRate;
        end

        delete(env);
    end
end

function stats = checkCritic(agent)
    %CHECKCRITIC Check critic network predictions

    critic = getCritic(agent);
    criticNet = getModel(critic);

    % Random states
    randStates = randn(50, 50);
    dlRand = dlarray(randStates, 'CB');
    randPreds = extractdata(forward(criticNet, dlRand));

    stats.std = std(randPreds);
    stats.mean = mean(randPreds);
    stats.min = min(randPreds);
    stats.max = max(randPreds);

    % Win vs lose states
    winState = zeros(50, 1);
    winState(10:12) = 1;  % Our edges green
    winState(16) = 1;

    loseState = zeros(50, 1);
    loseState(10:12) = -1;  % Our edges purple
    loseState(16) = 1;

    structStates = [winState, loseState];
    dlStruct = dlarray(structStates, 'CB');
    structPreds = extractdata(forward(criticNet, dlStruct));

    stats.winPred = structPreds(1);
    stats.losePred = structPreds(2);
    stats.winLossDiff = structPreds(1) - structPreds(2);
end

function printBanner(options)
    fprintf('\n');
    fprintf('╔══════════════════════════════════════════════════════════════╗\n');
    fprintf('║     TANGLED RL TRAINING - CURRICULUM WITH ENSEMBLE          ║\n');
    fprintf('╠══════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Curriculum: Random -> Petersen -> Self-Play                 ║\n');
    fprintf('║  Ensemble:   RL Policy + MC Rollouts (%d workers)            ║\n', options.NumWorkers);
    fprintf('║  MC Rollouts per action: %d                                  ║\n', options.RolloutsPerAction);
    fprintf('╚══════════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
end

function log_print(verbose, varargin)
    if verbose
        fprintf(varargin{:});
    end
end
