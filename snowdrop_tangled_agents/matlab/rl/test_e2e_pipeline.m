%% test_e2e_pipeline.m - End-to-End Pipeline Test
%
% This script tests the complete RL pipeline from agent creation to
% gameplay to deployment, saving all artifacts for post-analysis.
%
% Artifacts saved:
%   - game_log.json: Complete game history with states, actions, rewards
%   - agent_initial.mat: Agent before any training
%   - agent_trained.mat: Agent after mini-training session
%   - training_stats.json: Training statistics
%   - deployment_log.json: Deployment pipeline test results
%   - e2e_summary.json: Overall test summary
%
% Usage:
%   >> results = test_e2e_pipeline()
%   >> results = test_e2e_pipeline('OutputDir', 'my_artifacts')
%   >> results = test_e2e_pipeline('TrainingEpisodes', 20)

function results = test_e2e_pipeline(options)
    arguments
        options.OutputDir string = ""
        options.TrainingEpisodes (1,1) double = 10
        options.Verbose logical = true
    end

    %% Setup
    if options.OutputDir == ""
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        options.OutputDir = fullfile(tempdir, ['tangled_e2e_' timestamp]);
    end

    if ~exist(options.OutputDir, 'dir')
        mkdir(options.OutputDir);
    end

    results = struct();
    results.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    results.outputDir = options.OutputDir;
    results.phases = struct();
    results.passed = true;
    results.errors = {};

    log_print(options.Verbose, '\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  TANGLED RL - END-TO-END PIPELINE TEST\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Output directory: %s\n', options.OutputDir);
    log_print(options.Verbose, '  Training episodes: %d\n', options.TrainingEpisodes);
    log_print(options.Verbose, '================================================================\n\n');

    %% Phase 1: Environment Creation
    log_print(options.Verbose, '=== Phase 1: Environment Creation ===\n');
    try
        tic;
        env = TangledEnvironment();
        phase1_time = toc;

        % Verify environment properties
        obsInfo = getObservationInfo(env);
        actInfo = getActionInfo(env);

        results.phases.environment = struct();
        results.phases.environment.passed = true;
        results.phases.environment.time = phase1_time;
        results.phases.environment.obsSize = obsInfo.Dimension(1);
        results.phases.environment.actSize = numel(actInfo.Elements);

        log_print(options.Verbose, '  Environment created (%.2fs)\n', phase1_time);
        log_print(options.Verbose, '  Observation size: %d\n', obsInfo.Dimension(1));
        log_print(options.Verbose, '  Action size: %d\n', numel(actInfo.Elements));
        log_print(options.Verbose, '  PASSED\n\n');
    catch ME
        results.phases.environment = struct('passed', false, 'error', ME.message);
        results.passed = false;
        results.errors{end+1} = ['Phase 1: ' ME.message];
        log_print(options.Verbose, '  FAILED: %s\n\n', ME.message);
        save_results(results, options.OutputDir);
        return;
    end

    %% Phase 2: Agent Creation
    log_print(options.Verbose, '=== Phase 2: Agent Creation ===\n');
    try
        tic;
        agent = createPPOAgent(env);
        phase2_time = toc;

        % Save initial agent
        initialAgentPath = fullfile(options.OutputDir, 'agent_initial.mat');
        save(initialAgentPath, 'agent');

        % Get network info
        actor = getActor(agent);
        critic = getCritic(agent);
        actorNet = getModel(actor);
        criticNet = getModel(critic);

        results.phases.agent = struct();
        results.phases.agent.passed = true;
        results.phases.agent.time = phase2_time;
        results.phases.agent.actorParams = numel(actorNet.Learnables);
        results.phases.agent.criticParams = numel(criticNet.Learnables);
        results.phases.agent.initialAgentPath = initialAgentPath;

        log_print(options.Verbose, '  Agent created (%.2fs)\n', phase2_time);
        log_print(options.Verbose, '  Actor learnable layers: %d\n', numel(actorNet.Learnables));
        log_print(options.Verbose, '  Critic learnable layers: %d\n', numel(criticNet.Learnables));
        log_print(options.Verbose, '  Initial agent saved: %s\n', initialAgentPath);
        log_print(options.Verbose, '  PASSED\n\n');
    catch ME
        results.phases.agent = struct('passed', false, 'error', ME.message);
        results.passed = false;
        results.errors{end+1} = ['Phase 2: ' ME.message];
        log_print(options.Verbose, '  FAILED: %s\n\n', ME.message);
        save_results(results, options.OutputDir);
        return;
    end

    %% Phase 3: Single Game Playthrough
    log_print(options.Verbose, '=== Phase 3: Single Game Playthrough ===\n');
    try
        tic;
        gameLog = play_full_game(agent, env, options.Verbose);
        phase3_time = toc;

        % Save game log
        gameLogPath = fullfile(options.OutputDir, 'game_log.json');
        save_json(gameLog, gameLogPath);

        results.phases.gameplay = struct();
        results.phases.gameplay.passed = true;
        results.phases.gameplay.time = phase3_time;
        results.phases.gameplay.totalMoves = gameLog.totalMoves;
        results.phases.gameplay.finalReward = gameLog.totalReward;
        results.phases.gameplay.result = gameLog.result;
        results.phases.gameplay.gameLogPath = gameLogPath;

        log_print(options.Verbose, '  Game completed (%.2fs)\n', phase3_time);
        log_print(options.Verbose, '  Total moves: %d\n', gameLog.totalMoves);
        log_print(options.Verbose, '  Final reward: %.3f\n', gameLog.totalReward);
        log_print(options.Verbose, '  Result: %s\n', gameLog.result);
        log_print(options.Verbose, '  Game log saved: %s\n', gameLogPath);
        log_print(options.Verbose, '  PASSED\n\n');
    catch ME
        results.phases.gameplay = struct('passed', false, 'error', ME.message);
        results.passed = false;
        results.errors{end+1} = ['Phase 3: ' ME.message];
        log_print(options.Verbose, '  FAILED: %s\n\n', ME.message);
        save_results(results, options.OutputDir);
        return;
    end

    %% Phase 4: Mini Training Session
    log_print(options.Verbose, '=== Phase 4: Mini Training Session (%d episodes) ===\n', options.TrainingEpisodes);
    try
        tic;

        % Create isolated training environment
        trainDbPath = fullfile(options.OutputDir, 'training_buffer.db');

        [trainedAgent, stats] = trainParallel(agent, ...
            'MaxEpisodes', options.TrainingEpisodes, ...
            'NumWorkers', 1, ...
            'Verbose', false, ...
            'SavePath', options.OutputDir, ...
            'DBPath', trainDbPath);

        phase4_time = toc;

        % Save trained agent
        trainedAgentPath = fullfile(options.OutputDir, 'agent_trained.mat');
        save(trainedAgentPath, 'trainedAgent', 'stats');

        % Save training stats as JSON
        trainingStats = struct();
        trainingStats.episodes = options.TrainingEpisodes;
        trainingStats.wins = stats.wins;
        trainingStats.losses = stats.losses_count;
        trainingStats.draws = stats.draws;
        trainingStats.finalWinRate = stats.winRates(end);
        trainingStats.episodeRewards = stats.episodeRewards;
        trainingStats.winRates = stats.winRates;

        statsPath = fullfile(options.OutputDir, 'training_stats.json');
        save_json(trainingStats, statsPath);

        results.phases.training = struct();
        results.phases.training.passed = true;
        results.phases.training.time = phase4_time;
        results.phases.training.episodes = options.TrainingEpisodes;
        results.phases.training.wins = stats.wins;
        results.phases.training.losses = stats.losses_count;
        results.phases.training.draws = stats.draws;
        results.phases.training.finalWinRate = stats.winRates(end);
        results.phases.training.trainedAgentPath = trainedAgentPath;
        results.phases.training.statsPath = statsPath;

        log_print(options.Verbose, '  Training completed (%.2fs)\n', phase4_time);
        log_print(options.Verbose, '  Record: %dW / %dL / %dD\n', stats.wins, stats.losses_count, stats.draws);
        log_print(options.Verbose, '  Final win rate: %.1f%%\n', stats.winRates(end) * 100);
        log_print(options.Verbose, '  Trained agent saved: %s\n', trainedAgentPath);
        log_print(options.Verbose, '  PASSED\n\n');

        % Update agent for next phases
        agent = trainedAgent;
    catch ME
        results.phases.training = struct('passed', false, 'error', ME.message);
        results.passed = false;
        results.errors{end+1} = ['Phase 4: ' ME.message];
        log_print(options.Verbose, '  FAILED: %s\n\n', ME.message);
        save_results(results, options.OutputDir);
        return;
    end

    %% Phase 5: Deployment Pipeline
    log_print(options.Verbose, '=== Phase 5: Deployment Pipeline ===\n');
    try
        tic;

        deployLog = struct();
        deployLog.steps = {};

        % 5a. Create registry
        registryDbPath = fullfile(options.OutputDir, 'model_registry.db');
        modelDir = fullfile(options.OutputDir, 'models');
        registry = ModelRegistry(registryDbPath, modelDir);
        deployLog.steps{end+1} = struct('step', 'create_registry', 'passed', true);
        log_print(options.Verbose, '  [5a] Registry created\n');

        % 5b. Register model
        metrics = struct();
        metrics.episodes = options.TrainingEpisodes;
        metrics.avgReward = mean(stats.episodeRewards);
        metrics.winRate = stats.winRates(end);

        version = registry.registerModel(agent, metrics, 'E2E test model');
        deployLog.steps{end+1} = struct('step', 'register_model', 'passed', true, 'version', version);
        log_print(options.Verbose, '  [5b] Model registered: %s\n', version);

        % 5c. Deploy model
        registry.deployModel(version);
        deployLog.steps{end+1} = struct('step', 'deploy_model', 'passed', true);
        log_print(options.Verbose, '  [5c] Model deployed\n');

        % 5d. Load deployed model
        [loadedAgent, loadedVersion] = registry.loadDeployed();
        assert(strcmp(loadedVersion, version), 'Version mismatch');
        deployLog.steps{end+1} = struct('step', 'load_deployed', 'passed', true);
        log_print(options.Verbose, '  [5d] Deployed model loaded\n');

        % 5e. Test inference function
        testState = rand(50, 1) * 2 - 1;
        testMask = zeros(30, 1);
        testMask([1, 5, 10, 15, 20]) = 1;
        modelPath = fullfile(modelDir, 'deployed', 'current_model.mat');

        [action, value, probs] = tangled_agent_inference(testState, testMask, modelPath);
        assert(testMask(action) == 1, 'Invalid action selected');
        assert(abs(sum(probs) - 1) < 0.01, 'Probabilities do not sum to 1');
        deployLog.steps{end+1} = struct('step', 'inference', 'passed', true, ...
            'action', action, 'value', value);
        log_print(options.Verbose, '  [5e] Inference test passed (action=%d, value=%.3f)\n', action, value);

        % 5f. Test auto-deploy (should skip - no improvement)
        deployed = autoDeploy(agent, metrics, registry, 'Verbose', false);
        deployLog.steps{end+1} = struct('step', 'auto_deploy', 'passed', true, 'deployed', deployed);
        log_print(options.Verbose, '  [5f] Auto-deploy test passed (deployed=%d)\n', deployed);

        phase5_time = toc;

        % Save deployment log
        deployLogPath = fullfile(options.OutputDir, 'deployment_log.json');
        save_json(deployLog, deployLogPath);

        % Cleanup
        registry.close();

        results.phases.deployment = struct();
        results.phases.deployment.passed = true;
        results.phases.deployment.time = phase5_time;
        results.phases.deployment.version = version;
        results.phases.deployment.deployLogPath = deployLogPath;

        log_print(options.Verbose, '  Deployment pipeline completed (%.2fs)\n', phase5_time);
        log_print(options.Verbose, '  PASSED\n\n');
    catch ME
        results.phases.deployment = struct('passed', false, 'error', ME.message);
        results.passed = false;
        results.errors{end+1} = ['Phase 5: ' ME.message];
        log_print(options.Verbose, '  FAILED: %s\n\n', ME.message);
    end

    %% Phase 6: Post-Training Game
    log_print(options.Verbose, '=== Phase 6: Post-Training Game ===\n');
    try
        tic;

        % Reset environment
        env2 = TangledEnvironment();
        postGameLog = play_full_game(agent, env2, options.Verbose);
        phase6_time = toc;

        % Save post-training game log
        postGameLogPath = fullfile(options.OutputDir, 'game_log_post_training.json');
        save_json(postGameLog, postGameLogPath);

        results.phases.postGame = struct();
        results.phases.postGame.passed = true;
        results.phases.postGame.time = phase6_time;
        results.phases.postGame.totalMoves = postGameLog.totalMoves;
        results.phases.postGame.finalReward = postGameLog.totalReward;
        results.phases.postGame.result = postGameLog.result;
        results.phases.postGame.gameLogPath = postGameLogPath;

        log_print(options.Verbose, '  Post-training game completed (%.2fs)\n', phase6_time);
        log_print(options.Verbose, '  Total moves: %d\n', postGameLog.totalMoves);
        log_print(options.Verbose, '  Final reward: %.3f\n', postGameLog.totalReward);
        log_print(options.Verbose, '  Result: %s\n', postGameLog.result);
        log_print(options.Verbose, '  PASSED\n\n');

        delete(env2);
    catch ME
        results.phases.postGame = struct('passed', false, 'error', ME.message);
        results.passed = false;
        results.errors{end+1} = ['Phase 6: ' ME.message];
        log_print(options.Verbose, '  FAILED: %s\n\n', ME.message);
    end

    %% Summary
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  E2E TEST SUMMARY\n');
    log_print(options.Verbose, '================================================================\n');

    phaseNames = fieldnames(results.phases);
    passedCount = 0;
    totalTime = 0;

    for i = 1:length(phaseNames)
        phase = results.phases.(phaseNames{i});
        if phase.passed
            passedCount = passedCount + 1;
            status = 'PASSED';
        else
            status = 'FAILED';
        end
        if isfield(phase, 'time')
            totalTime = totalTime + phase.time;
        end
        log_print(options.Verbose, '  Phase %d (%s): %s\n', i, phaseNames{i}, status);
    end

    log_print(options.Verbose, '----------------------------------------------------------------\n');
    log_print(options.Verbose, '  Total: %d/%d phases passed\n', passedCount, length(phaseNames));
    log_print(options.Verbose, '  Total time: %.2f seconds\n', totalTime);
    log_print(options.Verbose, '  Output directory: %s\n', options.OutputDir);
    log_print(options.Verbose, '================================================================\n\n');

    if results.passed
        log_print(options.Verbose, 'SUCCESS: End-to-end pipeline test passed!\n\n');
    else
        log_print(options.Verbose, 'FAILED: %d error(s) occurred.\n', length(results.errors));
        for i = 1:length(results.errors)
            log_print(options.Verbose, '  - %s\n', results.errors{i});
        end
        log_print(options.Verbose, '\n');
    end

    % Save final results
    results.totalTime = totalTime;
    results.passedPhases = passedCount;
    results.totalPhases = length(phaseNames);
    save_results(results, options.OutputDir);

    log_print(options.Verbose, 'Artifacts saved to: %s\n', options.OutputDir);
    log_print(options.Verbose, '  - e2e_summary.json\n');
    log_print(options.Verbose, '  - agent_initial.mat\n');
    log_print(options.Verbose, '  - agent_trained.mat\n');
    log_print(options.Verbose, '  - game_log.json\n');
    log_print(options.Verbose, '  - game_log_post_training.json\n');
    log_print(options.Verbose, '  - training_stats.json\n');
    log_print(options.Verbose, '  - deployment_log.json\n');
    log_print(options.Verbose, '  - model_registry.db\n');
    log_print(options.Verbose, '  - models/\n\n');

    % Cleanup
    delete(env);
end

%% Helper Functions

function gameLog = play_full_game(agent, env, verbose)
    %PLAY_FULL_GAME Play a complete game and log all details

    gameLog = struct();
    gameLog.moves = {};
    gameLog.states = {};
    gameLog.actions = [];
    gameLog.rewards = [];
    gameLog.actionProbs = {};
    gameLog.values = [];
    gameLog.totalReward = 0;
    gameLog.totalMoves = 0;

    % Get networks for detailed logging
    actor = getActor(agent);
    critic = getCritic(agent);
    actorNet = getModel(actor);
    criticNet = getModel(critic);

    % Reset environment
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

        % Get action probabilities and value
        probs = forward(actorNet, dlarray(obs, 'CB'));
        probs = extractdata(probs);
        value = forward(criticNet, dlarray(obs, 'CB'));
        value = extractdata(value);

        % Apply mask and sample
        maskedProbs = probs .* mask(:);
        maskedProbs = maskedProbs / sum(maskedProbs);

        cumProbs = cumsum(maskedProbs);
        action = find(cumProbs >= rand(), 1);
        if isempty(action)
            action = validActions(randi(length(validActions)));
        end

        % Step environment
        [nextObs, reward, done, info] = step(env, action);

        % Log move
        moveLog = struct();
        moveLog.moveNum = moveNum;
        moveLog.state = env.State;
        moveLog.action = action;
        moveLog.edge = ceil(action / 2) - 1;  % 0-indexed edge
        moveLog.color = mod(action - 1, 2);   % 0=green, 1=purple
        moveLog.reward = reward;
        moveLog.value = value;
        moveLog.done = done;

        gameLog.moves{end+1} = moveLog;
        gameLog.states{end+1} = env.State;
        gameLog.actions(end+1) = action;
        gameLog.rewards(end+1) = reward;
        gameLog.actionProbs{end+1} = maskedProbs';
        gameLog.values(end+1) = value;
        gameLog.totalReward = gameLog.totalReward + reward;

        if verbose
            colorStr = {'Green', 'Purple'};
            log_print(true, '    Move %2d: Edge %2d %s -> reward %.3f\n', ...
                moveNum, moveLog.edge, colorStr{moveLog.color + 1}, reward);
        end

        obs = nextObs;
    end

    gameLog.totalMoves = moveNum;
    gameLog.finalState = env.State;

    % Determine result
    if gameLog.totalReward > 0.5
        gameLog.result = 'win';
    elseif gameLog.totalReward < -0.5
        gameLog.result = 'loss';
    else
        gameLog.result = 'draw';
    end
end

function save_json(data, filepath)
    %SAVE_JSON Save struct as JSON file
    jsonStr = jsonencode(data, 'PrettyPrint', true);
    fid = fopen(filepath, 'w');
    fprintf(fid, '%s', jsonStr);
    fclose(fid);
end

function save_results(results, outputDir)
    %SAVE_RESULTS Save results summary as JSON
    summaryPath = fullfile(outputDir, 'e2e_summary.json');
    save_json(results, summaryPath);
end

function log_print(verbose, varargin)
    %LOG_PRINT Print if verbose mode is enabled
    if verbose
        fprintf(varargin{:});
    end
end
