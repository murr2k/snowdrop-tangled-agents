%% test_parallel.m - Test Phase 4 Parallel Self-Play Components
%
% This script tests all Phase 4 components for parallel training:
%   1. createParallelEnv - Parallel environment creation
%   2. collectEpisode - Episode collection
%   3. trainParallel - Parallel training (short test)
%   4. enableGPU - GPU acceleration
%   5. workerInit - Worker initialization
%
% Run from MATLAB command window:
%   >> test_parallel

fprintf('\n');
fprintf('============================================\n');
fprintf('  PHASE 4: Parallel Self-Play Tests\n');
fprintf('============================================\n\n');

%% Test 1: Create Parallel Environments
fprintf('=== Test 1: createParallelEnv ===\n');

try
    % Create 2 parallel environments (sequential mode)
    envs = createParallelEnv(2, 'UseParpool', false, 'Verbose', true);

    assert(iscell(envs), 'Expected cell array of environments');
    assert(length(envs) == 2, 'Expected 2 environments');

    % Test each environment
    for i = 1:length(envs)
        obs = reset(envs{i});
        assert(length(obs) == 50, sprintf('Env %d: Expected 50-element observation', i));
    end

    fprintf('PASSED: createParallelEnv\n\n');
catch ME
    fprintf('FAILED: createParallelEnv\n');
    fprintf('  Error: %s\n\n', ME.message);
end

%% Test 2: Collect Episode
fprintf('=== Test 2: collectEpisode ===\n');

try
    % Create environment and agent
    env = TangledEnvironment();
    agent = createPPOAgent(env);

    % Collect one episode
    exp = collectEpisode(agent, env, 'MaxSteps', 15, 'Deterministic', false);

    % Verify experience structure
    assert(isstruct(exp), 'Expected struct');
    assert(isfield(exp, 'states'), 'Missing states field');
    assert(isfield(exp, 'actions'), 'Missing actions field');
    assert(isfield(exp, 'rewards'), 'Missing rewards field');
    assert(isfield(exp, 'totalReward'), 'Missing totalReward field');
    assert(isfield(exp, 'result'), 'Missing result field');

    assert(length(exp.states) == exp.length, 'States length mismatch');
    assert(length(exp.actions) == exp.length, 'Actions length mismatch');

    fprintf('  Episode length: %d\n', exp.length);
    fprintf('  Total reward: %.3f\n', exp.totalReward);
    fprintf('  Result: %s\n', exp.result);

    fprintf('PASSED: collectEpisode\n\n');
catch ME
    fprintf('FAILED: collectEpisode\n');
    fprintf('  Error: %s\n\n', ME.message);
end

%% Test 3: Worker Initialization
fprintf('=== Test 3: workerInit ===\n');

try
    % Initialize worker 1
    workerInit(1, 'Seed', 42, 'UseGPU', false, 'Verbose', true);

    fprintf('PASSED: workerInit\n\n');
catch ME
    fprintf('FAILED: workerInit\n');
    fprintf('  Error: %s\n\n', ME.message);
end

%% Test 4: GPU Check
fprintf('=== Test 4: enableGPU ===\n');

try
    env = TangledEnvironment();
    agent = createPPOAgent(env);

    % Try to enable GPU (will warn if not available)
    agent = enableGPU(agent);

    fprintf('PASSED: enableGPU (may have used CPU fallback)\n\n');
catch ME
    fprintf('FAILED: enableGPU\n');
    fprintf('  Error: %s\n\n', ME.message);
end

%% Test 5: Short Training Run
fprintf('=== Test 5: trainParallel (short run) ===\n');

try
    % Create fresh agent
    env = TangledEnvironment();
    agent = createPPOAgent(env);

    % Run very short training (10 episodes, no parallel pool)
    [trainedAgent, stats] = trainParallel(agent, ...
        'NumWorkers', 1, ...
        'MaxEpisodes', 10, ...
        'UpdateFrequency', 50, ...
        'SaveFrequency', 100, ...
        'UseGPU', false, ...
        'DBPath', 'test_parallel_exp.db', ...
        'SavePath', 'test_parallel_checkpoints', ...
        'Verbose', true);

    % Verify stats
    assert(length(stats.episodeRewards) >= 10, 'Not enough episodes');
    assert(stats.wins + stats.losses_count + stats.draws == length(stats.episodeRewards), ...
        'Win/loss/draw count mismatch');

    fprintf('  Final record: %dW / %dL / %dD\n', ...
        stats.wins, stats.losses_count, stats.draws);

    fprintf('PASSED: trainParallel (short run)\n\n');

    % Cleanup test files
    if exist('test_parallel_exp.db', 'file')
        delete('test_parallel_exp.db');
    end
    if exist('test_parallel_checkpoints', 'dir')
        rmdir('test_parallel_checkpoints', 's');
    end

catch ME
    fprintf('FAILED: trainParallel\n');
    fprintf('  Error: %s\n\n', ME.message);
end

%% Summary
fprintf('============================================\n');
fprintf('  Phase 4 Tests Complete\n');
fprintf('============================================\n\n');

fprintf('To run full parallel training:\n');
fprintf('  >> env = TangledEnvironment();\n');
fprintf('  >> agent = createPPOAgent(env);\n');
fprintf('  >> [trainedAgent, stats] = trainParallel(agent, ...\n');
fprintf('       ''NumWorkers'', 4, ''MaxEpisodes'', 5000);\n\n');
