function results = finetune_curriculum(agent, options)
%FINETUNE_CURRICULUM Curriculum learning with progressive opponent difficulty
%
%   results = finetune_curriculum(agent)
%   results = finetune_curriculum(agent, Name=Value)
%
%   Trains the agent using curriculum learning:
%   1. Random opponent (easy) - learn basics
%   2. Heuristic opponent (medium) - learn strategy
%   3. MCTS opponent (hard) - refine against strong play
%
%   Name-Value Arguments:
%       EpisodesPerLevel - Episodes at each difficulty (default: 50)
%       OutputDir        - Output directory (default: temp)
%       Verbose          - Print progress (default: true)

    arguments
        agent
        options.EpisodesPerLevel (1,1) double = 50
        options.OutputDir string = ""
        options.Verbose logical = true
    end

    if options.OutputDir == ""
        options.OutputDir = fullfile(tempdir, sprintf('curriculum_%s', datestr(now, 'yyyymmdd_HHMMSS')));
    end

    if ~exist(options.OutputDir, 'dir')
        mkdir(options.OutputDir);
    end

    log_print(options.Verbose, '\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  CURRICULUM FINE-TUNING\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Episodes per level: %d\n', options.EpisodesPerLevel);
    log_print(options.Verbose, '  Progression: random -> petersen -> mcts\n');
    log_print(options.Verbose, '================================================================\n\n');

    results = struct();
    results.outputDir = options.OutputDir;
    results.levels = {};

    opponents = {'random', 'petersen', 'mcts'};

    for level = 1:length(opponents)
        oppStyle = opponents{level};

        log_print(options.Verbose, '================================================================\n');
        log_print(options.Verbose, '  LEVEL %d: vs %s\n', level, upper(oppStyle));
        log_print(options.Verbose, '================================================================\n\n');

        %% Create environment with this opponent
        opp = SimulatedOpponent('Style', oppStyle);
        env = TangledEnvironment('Opponent', opp);

        %% Check critic before
        log_print(options.Verbose, 'Critic before:\n');
        criticBefore = quick_critic_check(agent);
        log_print(options.Verbose, '  Std: %.4f, WinLossDiff: %+.4f\n\n', ...
            criticBefore.std, criticBefore.winLossDiff);

        %% Train
        trainOpts = rlTrainingOptions(...
            'MaxEpisodes', options.EpisodesPerLevel, ...
            'MaxStepsPerEpisode', 20, ...
            'ScoreAveragingWindowLength', 10, ...
            'Verbose', false, ...
            'Plots', 'none', ...
            'StopTrainingCriteria', 'none');

        log_print(options.Verbose, 'Training %d episodes...\n', options.EpisodesPerLevel);
        tic;
        trainResult = train(agent, env, trainOpts);
        trainTime = toc;

        %% Extract results
        if isprop(trainResult, 'EpisodeReward')
            rewards = trainResult.EpisodeReward;
        else
            rewards = zeros(options.EpisodesPerLevel, 1);
        end

        avgReward = mean(rewards);
        winCount = sum(rewards > 0.5);
        winRate = winCount / length(rewards);

        log_print(options.Verbose, '\nResults:\n');
        log_print(options.Verbose, '  Training time: %.1f sec\n', trainTime);
        log_print(options.Verbose, '  Avg reward:    %+.3f\n', avgReward);
        log_print(options.Verbose, '  Win rate:      %.1f%% (%d/%d)\n', ...
            winRate * 100, winCount, length(rewards));

        %% Check critic after
        log_print(options.Verbose, '\nCritic after:\n');
        criticAfter = quick_critic_check(agent);
        log_print(options.Verbose, '  Std: %.4f (was: %.4f)\n', ...
            criticAfter.std, criticBefore.std);
        log_print(options.Verbose, '  WinLossDiff: %+.4f (was: %+.4f)\n', ...
            criticAfter.winLossDiff, criticBefore.winLossDiff);

        if criticAfter.winLossDiff > criticBefore.winLossDiff
            log_print(options.Verbose, '  [IMPROVING]\n');
        else
            log_print(options.Verbose, '  [NOT IMPROVING]\n');
        end

        %% Store level results
        levelResult = struct();
        levelResult.opponent = oppStyle;
        levelResult.episodes = length(rewards);
        levelResult.avgReward = avgReward;
        levelResult.winRate = winRate;
        levelResult.criticBefore = criticBefore;
        levelResult.criticAfter = criticAfter;
        levelResult.rewards = rewards;
        results.levels{end+1} = levelResult;

        %% Save checkpoint
        checkpointPath = fullfile(options.OutputDir, sprintf('agent_level%d_%s.mat', level, oppStyle));
        save(checkpointPath, 'agent', 'levelResult');
        log_print(options.Verbose, '\nCheckpoint: %s\n\n', checkpointPath);

        %% Cleanup
        delete(env);

        %% Check if we should continue
        if winRate < 0.1 && level > 1
            log_print(options.Verbose, '[WARNING] Win rate too low, might need more training at this level\n\n');
        end
    end

    %% Final summary
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  CURRICULUM TRAINING COMPLETE\n');
    log_print(options.Verbose, '================================================================\n');

    for i = 1:length(results.levels)
        lvl = results.levels{i};
        log_print(options.Verbose, '  Level %d (%s): WR=%.1f%%, Reward=%+.3f\n', ...
            i, lvl.opponent, lvl.winRate * 100, lvl.avgReward);
    end

    % Overall critic evolution
    firstCritic = results.levels{1}.criticBefore;
    lastCritic = results.levels{end}.criticAfter;
    log_print(options.Verbose, '\n  Critic evolution:\n');
    log_print(options.Verbose, '    WinLossDiff: %+.4f -> %+.4f\n', ...
        firstCritic.winLossDiff, lastCritic.winLossDiff);

    if lastCritic.winLossDiff > 0
        log_print(options.Verbose, '\n  [SUCCESS] Critic learned correct win/loss distinction\n');
        results.criticLearned = true;
    else
        log_print(options.Verbose, '\n  [INCOMPLETE] Critic still inverted, needs more training\n');
        results.criticLearned = false;
    end

    log_print(options.Verbose, '================================================================\n\n');

    %% Save final
    finalPath = fullfile(options.OutputDir, 'agent_final.mat');
    save(finalPath, 'agent');
    resultsPath = fullfile(options.OutputDir, 'curriculum_results.mat');
    save(resultsPath, 'results');
end

function stats = quick_critic_check(agent)
    critic = getCritic(agent);
    criticNet = getModel(critic);

    randStates = randn(50, 50);
    dlRand = dlarray(randStates, 'CB');
    randPreds = extractdata(forward(criticNet, dlRand));

    stats.std = std(randPreds);
    stats.mean = mean(randPreds);

    % Win vs lose states
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
