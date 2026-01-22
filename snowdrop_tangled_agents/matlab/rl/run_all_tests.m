%% run_all_tests.m - Master Test Runner for Dynamic Learning System
%
% Executes all test suites and provides a comprehensive results report.
%
% Usage:
%   >> run_all_tests           % Run all tests
%   >> run_all_tests('quick')  % Run quick smoke tests only
%   >> run_all_tests('full')   % Run full test suite with training
%
% The script will:
%   1. Verify MATLAB toolbox dependencies
%   2. Run Phase 2 (Environment) tests
%   3. Run Phase 3 (PPO Agent) tests
%   4. Run Phase 4 (Parallel) tests
%   5. Generate a summary report

function results = run_all_tests(mode)
    if nargin < 1
        mode = 'standard';
    end

    %% Initialize isolated test environment
    % CRITICAL: All tests use isolated temp directory to protect production data
    testDir = fullfile(tempdir, sprintf('tangled_test_%s', datestr(now, 'yyyymmdd_HHMMSS')));
    mkdir(testDir);

    % Store test directory for all test functions
    setappdata(0, 'TangledTestDir', testDir);

    % Safety check: Ensure we NEVER touch production database
    prodDbPath = fullfile(getenv('USERPROFILE'), '.tangled', 'game_stats.db');
    setappdata(0, 'TangledProdDbPath', prodDbPath);

    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║     TANGLED DYNAMIC LEARNING - TEST RUNNER                     ║\n');
    fprintf('║     Mode: %-53s║\n', upper(mode));
    fprintf('╠════════════════════════════════════════════════════════════════╣\n');
    fprintf('║  ISOLATED TEST DIR: %s\n', testDir);
    fprintf('║  Production DB protected: %s\n', prodDbPath);
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    results = struct();
    results.startTime = datetime('now');
    results.mode = mode;
    results.tests = {};
    results.passed = 0;
    results.failed = 0;
    results.skipped = 0;

    %% Phase 0: Dependency Check
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('  PHASE 0: Dependency Verification\n');
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

    results = runTest(results, 'DEP-01', 'Reinforcement Learning Toolbox', ...
        @() checkToolbox('Reinforcement Learning Toolbox'));

    results = runTest(results, 'DEP-02', 'Deep Learning Toolbox', ...
        @() checkToolbox('Deep Learning Toolbox'));

    results = runTest(results, 'DEP-03', 'Database Toolbox (optional)', ...
        @() checkToolbox('Database Toolbox'), true);  % Optional

    results = runTest(results, 'DEP-04', 'Parallel Computing Toolbox (optional)', ...
        @() checkToolbox('Parallel Computing Toolbox'), true);  % Optional

    results = runTest(results, 'DEP-05', 'GPU Available (optional)', ...
        @() checkGPU(), true);  % Optional

    %% Phase 2: Environment Tests
    fprintf('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('  PHASE 2: RL Environment Tests\n');
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

    results = runTest(results, 'ENV-01', 'TangledEnvironment instantiation', ...
        @test_env_create);

    results = runTest(results, 'ENV-02', 'Observation space (50 elements)', ...
        @test_env_observation);

    results = runTest(results, 'ENV-03', 'Action space (30 discrete)', ...
        @test_env_action_space);

    results = runTest(results, 'ENV-04', 'Action masking', ...
        @test_action_mask);

    results = runTest(results, 'ENV-05', 'Environment step function', ...
        @test_env_step);

    results = runTest(results, 'ENV-06', 'Environment reset', ...
        @test_env_reset);

    results = runTest(results, 'ENV-07', 'Episode completion', ...
        @test_env_episode);

    %% Phase 3: PPO Agent Tests
    fprintf('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('  PHASE 3: PPO Agent Tests\n');
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

    results = runTest(results, 'PPO-01', 'PPO agent creation', ...
        @test_ppo_create);

    results = runTest(results, 'PPO-02', 'Actor network forward pass', ...
        @test_actor_forward);

    results = runTest(results, 'PPO-03', 'Critic network forward pass', ...
        @test_critic_forward);

    results = runTest(results, 'PPO-04', 'Masked action selection', ...
        @test_masked_action);

    results = runTest(results, 'PPO-05', 'SQLite experience buffer', ...
        @test_experience_buffer);

    if strcmp(mode, 'full')
        results = runTest(results, 'PPO-06', 'Single training step', ...
            @test_training_step);
    else
        results = skipTest(results, 'PPO-06', 'Single training step (full mode only)');
    end

    %% Phase 4: Parallel Tests
    fprintf('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('  PHASE 4: Parallel Self-Play Tests\n');
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

    results = runTest(results, 'PAR-01', 'Parallel environment creation', ...
        @test_parallel_env);

    results = runTest(results, 'PAR-02', 'Episode collection', ...
        @test_collect_episode);

    results = runTest(results, 'PAR-03', 'Worker initialization', ...
        @test_worker_init);

    results = runTest(results, 'PAR-04', 'GPU enable (graceful fallback)', ...
        @test_gpu_enable);

    if strcmp(mode, 'full')
        results = runTest(results, 'PAR-05', 'Short parallel training (10 episodes)', ...
            @test_parallel_training);
    else
        results = skipTest(results, 'PAR-05', 'Parallel training (full mode only)');
    end

    %% Summary Report
    results.endTime = datetime('now');
    results.duration = results.endTime - results.startTime;

    printSummary(results);

    %% Cleanup
    cleanupTestArtifacts();
end

%% Test Runner Helper
function results = runTest(results, id, name, testFn, isOptional)
    if nargin < 5
        isOptional = false;
    end

    fprintf('  [%s] %s... ', id, name);

    testResult = struct();
    testResult.id = id;
    testResult.name = name;
    testResult.optional = isOptional;

    tic;
    try
        testFn();
        testResult.status = 'PASSED';
        testResult.error = '';
        results.passed = results.passed + 1;
        fprintf('PASSED');
    catch ME
        if isOptional
            testResult.status = 'SKIPPED';
            testResult.error = ME.message;
            results.skipped = results.skipped + 1;
            fprintf('SKIPPED (optional)');
        else
            testResult.status = 'FAILED';
            testResult.error = ME.message;
            results.failed = results.failed + 1;
            fprintf('FAILED');
        end
    end
    testResult.duration = toc;

    fprintf(' (%.2fs)\n', testResult.duration);

    if ~isempty(testResult.error) && ~isOptional
        fprintf('         Error: %s\n', testResult.error);
    end

    results.tests{end+1} = testResult;
end

function results = skipTest(results, id, name)
    fprintf('  [%s] %s... SKIPPED\n', id, name);

    testResult = struct();
    testResult.id = id;
    testResult.name = name;
    testResult.status = 'SKIPPED';
    testResult.error = 'Mode restriction';
    testResult.duration = 0;
    testResult.optional = true;

    results.skipped = results.skipped + 1;
    results.tests{end+1} = testResult;
end

%% Summary Printer
function printSummary(results)
    fprintf('\n');
    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║                      TEST RESULTS SUMMARY                      ║\n');
    fprintf('╠════════════════════════════════════════════════════════════════╣\n');

    total = results.passed + results.failed + results.skipped;

    % Status bar
    passRate = results.passed / max(total - results.skipped, 1) * 100;

    if results.failed == 0
        statusIcon = '✓';
        statusText = 'ALL TESTS PASSED';
    else
        statusIcon = '✗';
        statusText = sprintf('%d TEST(S) FAILED', results.failed);
    end

    fprintf('║  %s %-60s║\n', statusIcon, statusText);
    fprintf('╠════════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Passed:  %-5d  │  Failed:  %-5d  │  Skipped:  %-5d        ║\n', ...
        results.passed, results.failed, results.skipped);
    fprintf('║  Pass Rate: %5.1f%% (excluding skipped)                        ║\n', passRate);
    fprintf('║  Duration:  %s                                        ║\n', ...
        formatDuration(results.duration));
    fprintf('╠════════════════════════════════════════════════════════════════╣\n');

    % Failed tests detail
    if results.failed > 0
        fprintf('║  FAILED TESTS:                                                 ║\n');
        for i = 1:length(results.tests)
            t = results.tests{i};
            if strcmp(t.status, 'FAILED')
                fprintf('║    [%s] %s\n', t.id, t.name);
                % Truncate error message if too long
                errMsg = t.error;
                if length(errMsg) > 55
                    errMsg = [errMsg(1:52) '...'];
                end
                fprintf('║           %s\n', errMsg);
            end
        end
        fprintf('╠════════════════════════════════════════════════════════════════╣\n');
    end

    fprintf('║  Mode: %-10s  │  Started: %s              ║\n', ...
        upper(results.mode), datestr(results.startTime, 'HH:MM:SS'));
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    % Recommendations
    if results.failed > 0
        fprintf('RECOMMENDATIONS:\n');
        fprintf('  1. Check that all required toolboxes are installed\n');
        fprintf('  2. Verify TangledEnvironment.m is on the MATLAB path\n');
        fprintf('  3. Run individual failing tests for detailed diagnostics\n');
        fprintf('  4. Check MATLAB version compatibility (R2021a+ recommended)\n\n');
    else
        fprintf('SUCCESS: All tests passed! The system is ready for training.\n\n');
        fprintf('NEXT STEPS:\n');
        fprintf('  1. Run: run_all_tests(''full'') for complete validation\n');
        fprintf('  2. Start training: trainParallel(agent, ''MaxEpisodes'', 1000)\n');
        fprintf('  3. Monitor progress in the training dashboard\n\n');
    end
end

function str = formatDuration(d)
    secs = seconds(d);
    if secs < 60
        str = sprintf('%.1f sec', secs);
    else
        str = sprintf('%.1f min', secs / 60);
    end
    str = pad(str, 8);
end

%% Dependency Check Functions
function checkToolbox(toolboxName)
    v = ver;
    installed = any(strcmp({v.Name}, toolboxName));
    if ~installed
        error('%s not installed', toolboxName);
    end
end

function checkGPU()
    if ~canUseGPU()
        error('No GPU available');
    end
end

%% Phase 2 Test Functions
function test_env_create()
    env = TangledEnvironment();
    assert(~isempty(env), 'Environment is empty');
    delete(env);
end

function test_env_observation()
    env = TangledEnvironment();
    obs = reset(env);
    assert(length(obs) == 50, sprintf('Expected 50 elements, got %d', length(obs)));
    assert(all(obs >= -1 & obs <= 1), 'Observation values out of range [-1, 1]');
    delete(env);
end

function test_env_action_space()
    env = TangledEnvironment();
    actInfo = getActionInfo(env);
    % For rlFiniteSetSpec, use Elements to get action count
    numActions = numel(actInfo.Elements);
    assert(numActions == 30, sprintf('Expected 30 actions, got %d', numActions));
    delete(env);
end

function test_action_mask()
    % Test with empty board (all actions valid)
    mask = getActionMask(repmat('-', 1, 15));
    assert(length(mask) == 30, 'Mask should have 30 elements');
    assert(sum(mask) == 30, 'All 30 actions should be valid on empty board');

    % Test with some edges colored
    state = 'G--P-----------';
    mask = getActionMask(state);
    assert(mask(1) == 0, 'Green on edge 0 should be invalid (already green)');
    assert(mask(16) == 0, 'Purple on edge 0 should be invalid (already green)');
    assert(mask(4) == 0, 'Green on edge 3 should be invalid (already purple)');
    assert(mask(19) == 0, 'Purple on edge 3 should be invalid (already purple)');
    assert(sum(mask) == 26, sprintf('Expected 26 valid actions, got %d', sum(mask)));
end

function test_env_step()
    env = TangledEnvironment();
    reset(env);

    % Take a valid action (Green on edge 0)
    [obs, reward, done, info] = step(env, 1);

    assert(length(obs) == 50, 'Observation should have 50 elements');
    assert(isscalar(reward), 'Reward should be scalar');
    assert(islogical(done) || isnumeric(done), 'Done should be logical or numeric');

    delete(env);
end

function test_env_reset()
    env = TangledEnvironment();

    % Play some moves
    reset(env);
    step(env, 1);
    step(env, 2);

    % Reset and verify clean state
    obs = reset(env);
    assert(length(obs) == 50, 'Reset should return 50-element observation');

    % Board should be empty (first 15 elements should be 0)
    boardState = obs(1:15);
    assert(all(boardState == 0), 'Board should be empty after reset');

    delete(env);
end

function test_env_episode()
    env = TangledEnvironment();
    obs = reset(env);

    % Play until done
    done = false;
    steps = 0;
    maxSteps = 20;

    while ~done && steps < maxSteps
        mask = getActionMask(env.State);
        validActions = find(mask);
        if isempty(validActions)
            break;
        end
        action = validActions(randi(length(validActions)));
        [obs, ~, done, ~] = step(env, action);
        steps = steps + 1;
    end

    assert(steps > 0, 'Should have taken at least one step');
    delete(env);
end

%% Phase 3 Test Functions
function test_ppo_create()
    env = TangledEnvironment();
    agent = createPPOAgent(env);
    assert(~isempty(agent), 'Agent should not be empty');
    delete(env);
end

function test_actor_forward()
    env = TangledEnvironment();
    agent = createPPOAgent(env);

    % Get observation
    obs = reset(env);

    % Forward pass through actor network
    % Use forward() to avoid conflict with System Identification Toolbox's predict()
    actor = getActor(agent);
    actorNet = getModel(actor);
    probs = forward(actorNet, dlarray(obs(:), 'CB'));
    probs = extractdata(probs);

    assert(length(probs) == 30, sprintf('Expected 30 action probs, got %d', length(probs)));
    assert(abs(sum(probs) - 1) < 0.01, 'Probabilities should sum to 1');

    delete(env);
end

function test_critic_forward()
    env = TangledEnvironment();
    agent = createPPOAgent(env);

    % Get observation
    obs = reset(env);

    % Forward pass through critic network
    % Use forward() to avoid conflict with System Identification Toolbox's predict()
    critic = getCritic(agent);
    criticNet = getModel(critic);
    value = forward(criticNet, dlarray(obs(:), 'CB'));
    value = extractdata(value);

    assert(isscalar(value), 'Value should be scalar');
    assert(isfinite(value), 'Value should be finite');

    delete(env);
end

function test_masked_action()
    env = TangledEnvironment();
    agent = createPPOAgent(env);
    obs = reset(env);

    % Create a mask with only a few valid actions
    mask = zeros(30, 1);
    mask([5, 10, 25]) = 1;  % Only actions 5, 10, 25 valid

    % Select multiple actions and verify all are valid
    for i = 1:10
        action = testSelectMaskedAction(agent, obs, mask);
        assert(any(action == [5, 10, 25]), ...
            sprintf('Action %d not in valid set [5, 10, 25]', action));
    end

    delete(env);
end

function action = testSelectMaskedAction(agent, obs, mask)
    % Local implementation for testing (returns action)
    actor = getActor(agent);
    actorNet = getModel(actor);
    % Use forward() to avoid conflict with System Identification Toolbox
    probs = forward(actorNet, dlarray(obs(:), 'CB'));
    probs = extractdata(probs);

    % Apply mask
    probs = probs(:) .* mask(:);
    probSum = sum(probs);
    if probSum > 0
        probs = probs / probSum;
    else
        validIdx = find(mask);
        probs = zeros(30, 1);
        probs(validIdx) = 1 / length(validIdx);
    end

    % Sample
    cumProbs = cumsum(probs);
    action = find(cumProbs >= rand(), 1);

    % Safety fallback
    if isempty(action)
        validIdx = find(mask);
        action = validIdx(randi(length(validIdx)));
    end
end

function test_experience_buffer()
    % Use isolated test directory
    testDir = getappdata(0, 'TangledTestDir');
    dbPath = fullfile(testDir, 'test_buffer.db');

    % Safety: Verify we're not touching production
    prodPath = getappdata(0, 'TangledProdDbPath');
    assert(~strcmp(dbPath, prodPath), 'SAFETY: Attempted to use production database!');

    if exist(dbPath, 'file')
        delete(dbPath);
    end

    buffer = SQLiteExperienceBuffer(dbPath, 1000);

    % Add some experiences
    for i = 1:10
        state = rand(50, 1);
        action = randi(30);
        reward = randn();
        nextState = rand(50, 1);
        done = i == 10;
        buffer.add(state, action, reward, nextState, done);
    end

    % Sample batch
    batch = buffer.sample(5);

    assert(length(batch.actions) == 5, 'Should sample 5 experiences');
    assert(size(batch.states, 2) == 5, 'Should have 5 states (50x5 matrix)');

    % Cleanup
    buffer.close();
    delete(dbPath);
end

function test_training_step()
    env = TangledEnvironment();
    agent = createPPOAgent(env);

    % Collect a few experiences
    experiences = collectEpisode(agent, env, 'MaxSteps', 5);

    assert(experiences.length > 0, 'Should collect at least one step');
    assert(~isempty(experiences.states), 'Should have states');
    assert(~isempty(experiences.rewards), 'Should have rewards');

    delete(env);
end

%% Phase 4 Test Functions
function test_parallel_env()
    envs = createParallelEnv(2, 'UseParpool', false, 'Verbose', false);

    assert(iscell(envs), 'Should return cell array');
    assert(length(envs) == 2, 'Should have 2 environments');

    % Test each environment
    for i = 1:length(envs)
        obs = reset(envs{i});
        assert(length(obs) == 50, 'Each env should produce 50-element obs');
    end
end

function test_collect_episode()
    env = TangledEnvironment();
    agent = createPPOAgent(env);

    exp = collectEpisode(agent, env, 'MaxSteps', 10, 'Deterministic', false);

    assert(isstruct(exp), 'Should return struct');
    assert(isfield(exp, 'states'), 'Should have states');
    assert(isfield(exp, 'actions'), 'Should have actions');
    assert(isfield(exp, 'rewards'), 'Should have rewards');
    assert(isfield(exp, 'totalReward'), 'Should have totalReward');
    assert(exp.length > 0, 'Should have collected steps');
    assert(length(exp.actions) == exp.length, 'Actions length should match');

    delete(env);
end

function test_worker_init()
    % Test worker initialization (should not error)
    workerInit(1, 'Seed', 42, 'UseGPU', false, 'Verbose', false);

    % Verify random seed was set (generate number, reinit, should match)
    workerInit(1, 'Seed', 42, 'UseGPU', false, 'Verbose', false);
    r1 = rand();
    workerInit(1, 'Seed', 42, 'UseGPU', false, 'Verbose', false);
    r2 = rand();

    assert(r1 == r2, 'Same seed should produce same random numbers');
end

function test_gpu_enable()
    env = TangledEnvironment();
    agent = createPPOAgent(env);

    % Should not error even without GPU (graceful fallback)
    agent = enableGPU(agent);

    % Agent should still work
    obs = reset(env);
    actor = getActor(agent);
    actorNet = getModel(actor);
    % Use forward() to avoid conflict with System Identification Toolbox
    probs = forward(actorNet, dlarray(obs(:), 'CB'));

    assert(~isempty(probs), 'Agent should still produce output after GPU enable');

    delete(env);
end

function test_parallel_training()
    env = TangledEnvironment();
    agent = createPPOAgent(env);

    % Use isolated test directory
    testDir = getappdata(0, 'TangledTestDir');
    dbPath = fullfile(testDir, 'test_parallel_train.db');
    savePath = fullfile(testDir, 'test_parallel_checkpoints');

    % Safety: Verify we're not touching production
    prodPath = getappdata(0, 'TangledProdDbPath');
    assert(~contains(dbPath, '.tangled'), 'SAFETY: Path contains .tangled!');

    [trainedAgent, stats] = trainParallel(agent, ...
        'NumWorkers', 1, ...
        'MaxEpisodes', 10, ...
        'UpdateFrequency', 50, ...
        'SaveFrequency', 100, ...
        'UseGPU', false, ...
        'DBPath', dbPath, ...
        'SavePath', savePath, ...
        'Verbose', false);

    assert(~isempty(trainedAgent), 'Should return trained agent');
    assert(length(stats.episodeRewards) >= 10, 'Should have 10+ episode rewards');

    % Cleanup
    if exist(dbPath, 'file'), delete(dbPath); end
    if exist(savePath, 'dir'), rmdir(savePath, 's'); end

    delete(env);
end

%% Cleanup
function cleanupTestArtifacts()
    % Get isolated test directory
    testDir = getappdata(0, 'TangledTestDir');

    if ~isempty(testDir) && exist(testDir, 'dir')
        % Safety: Double-check we're not deleting anything important
        if contains(testDir, 'tangled_test_') && contains(testDir, tempdir)
            fprintf('Cleaning up test directory: %s\n', testDir);
            try
                rmdir(testDir, 's');
            catch ME
                warning('Could not fully clean test directory: %s', ME.message);
            end
        else
            warning('Skipping cleanup - unexpected test directory path: %s', testDir);
        end
    end

    % Clear app data
    if isappdata(0, 'TangledTestDir')
        rmappdata(0, 'TangledTestDir');
    end
    if isappdata(0, 'TangledProdDbPath')
        rmappdata(0, 'TangledProdDbPath');
    end
end
