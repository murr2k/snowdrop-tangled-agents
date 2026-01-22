%% test_deployment.m - Test Phase 5 Deployment Pipeline Components
%
% This script tests all Phase 5 components for model deployment:
%   1. ModelRegistry - Version management
%   2. tangled_agent_inference - Inference function
%   3. autoDeploy - Automatic deployment
%
% Run from MATLAB command window:
%   >> test_deployment

fprintf('\n');
fprintf('============================================\n');
fprintf('  PHASE 5: Deployment Pipeline Tests\n');
fprintf('============================================\n\n');

%% Setup isolated test environment
testDir = fullfile(tempdir, sprintf('tangled_deploy_test_%s', datestr(now, 'yyyymmdd_HHMMSS')));
mkdir(testDir);
fprintf('Test directory: %s\n\n', testDir);

% Track test results
passed = 0;
failed = 0;

%% Test 1: ModelRegistry Creation
fprintf('=== Test 1: ModelRegistry Creation ===\n');

try
    dbPath = fullfile(testDir, 'test_models.db');
    modelDir = fullfile(testDir, 'models');

    registry = ModelRegistry(dbPath, modelDir);

    assert(exist(dbPath, 'file') == 2, 'Database should be created');
    assert(exist(modelDir, 'dir') == 7, 'Model directory should be created');
    assert(exist(fullfile(modelDir, 'deployed'), 'dir') == 7, 'Deploy directory should be created');

    fprintf('PASSED: ModelRegistry creation\n\n');
    passed = passed + 1;
catch ME
    fprintf('FAILED: ModelRegistry creation\n');
    fprintf('  Error: %s\n\n', ME.message);
    failed = failed + 1;
end

%% Test 2: Model Registration
fprintf('=== Test 2: Model Registration ===\n');

try
    % Create a simple agent for testing
    env = TangledEnvironment();
    agent = createPPOAgent(env);

    % Create test metrics
    metrics = struct();
    metrics.episodes = 100;
    metrics.avgReward = 0.5;
    metrics.winRate = 0.6;

    % Register model
    version = registry.registerModel(agent, metrics, 'Test model');

    assert(~isempty(version), 'Version should be returned');
    assert(startsWith(version, 'v'), 'Version should start with v');

    % Verify file was created
    expectedFile = fullfile(modelDir, [version '.mat']);
    assert(exist(expectedFile, 'file') == 2, 'Model file should be created');

    fprintf('  Registered version: %s\n', version);
    fprintf('PASSED: Model registration\n\n');
    passed = passed + 1;

    % Save version for later tests
    testVersion = version;
    delete(env);
catch ME
    fprintf('FAILED: Model registration\n');
    fprintf('  Error: %s\n\n', ME.message);
    failed = failed + 1;
    testVersion = '';
end

%% Test 3: Model Deployment
fprintf('=== Test 3: Model Deployment ===\n');

try
    if isempty(testVersion)
        error('No version to deploy (registration failed)');
    end

    % Deploy the registered model
    registry.deployModel(testVersion);

    % Verify deployment
    deployedFile = fullfile(modelDir, 'deployed', 'current_model.mat');
    assert(exist(deployedFile, 'file') == 2, 'Deployed model should exist');

    versionFile = fullfile(modelDir, 'deployed', 'current_version.txt');
    assert(exist(versionFile, 'file') == 2, 'Version file should exist');

    % Check version info
    info = registry.getDeployedInfo();
    assert(~isempty(info), 'Deployed info should be returned');
    assert(strcmp(info.version, testVersion), 'Version should match');

    fprintf('  Deployed: %s\n', testVersion);
    fprintf('PASSED: Model deployment\n\n');
    passed = passed + 1;
catch ME
    fprintf('FAILED: Model deployment\n');
    fprintf('  Error: %s\n\n', ME.message);
    failed = failed + 1;
end

%% Test 4: Load Deployed Model
fprintf('=== Test 4: Load Deployed Model ===\n');

try
    [loadedAgent, loadedVersion] = registry.loadDeployed();

    assert(~isempty(loadedAgent), 'Agent should be loaded');
    assert(strcmp(loadedVersion, testVersion), 'Version should match');

    % Verify agent works
    env = TangledEnvironment();
    obs = reset(env);
    actor = getActor(loadedAgent);
    actorNet = getModel(actor);
    probs = forward(actorNet, dlarray(obs(:), 'CB'));
    assert(length(extractdata(probs)) == 30, 'Should produce 30 action probs');

    fprintf('  Loaded version: %s\n', loadedVersion);
    fprintf('PASSED: Load deployed model\n\n');
    passed = passed + 1;
    delete(env);
catch ME
    fprintf('FAILED: Load deployed model\n');
    fprintf('  Error: %s\n\n', ME.message);
    failed = failed + 1;
end

%% Test 5: List Versions
fprintf('=== Test 5: List Versions ===\n');

try
    versions = registry.listVersions();

    assert(~isempty(versions), 'Should have at least one version');
    assert(height(versions) >= 1, 'Should list registered version');

    fprintf('  Found %d version(s)\n', height(versions));
    fprintf('PASSED: List versions\n\n');
    passed = passed + 1;
catch ME
    fprintf('FAILED: List versions\n');
    fprintf('  Error: %s\n\n', ME.message);
    failed = failed + 1;
end

%% Test 6: Inference Function
fprintf('=== Test 6: tangled_agent_inference ===\n');

try
    % Create test inputs
    stateVec = rand(50, 1) * 2 - 1;  % Random features in [-1, 1]
    actionMask = zeros(30, 1);
    actionMask([1, 5, 10, 20, 25]) = 1;  % Some valid actions

    % Set model path to test deployment
    modelPath = fullfile(modelDir, 'deployed', 'current_model.mat');

    % Run inference
    [action, value, probs] = tangled_agent_inference(stateVec, actionMask, modelPath);

    assert(isscalar(action), 'Action should be scalar');
    assert(action >= 1 && action <= 30, 'Action should be in range 1-30');
    assert(actionMask(action) == 1, 'Selected action should be valid');
    assert(isscalar(value), 'Value should be scalar');
    assert(length(probs) == 30, 'Should return 30 probabilities');
    assert(abs(sum(probs) - 1) < 0.01, 'Probabilities should sum to 1');

    fprintf('  Action: %d, Value: %.3f\n', action, value);
    fprintf('PASSED: Inference function\n\n');
    passed = passed + 1;
catch ME
    fprintf('FAILED: Inference function\n');
    fprintf('  Error: %s\n\n', ME.message);
    failed = failed + 1;
end

%% Test 7: Auto-Deploy (No Previous Model)
fprintf('=== Test 7: autoDeploy (New Deployment) ===\n');

try
    % Create new registry for this test
    dbPath2 = fullfile(testDir, 'test_autodeploy.db');
    modelDir2 = fullfile(testDir, 'models_autodeploy');
    registry2 = ModelRegistry(dbPath2, modelDir2);

    % Create agent and metrics
    env = TangledEnvironment();
    agent = createPPOAgent(env);
    metrics = struct('episodes', 200, 'avgReward', 0.4, 'winRate', 0.55);

    % Auto-deploy (should deploy since no model exists)
    deployed = autoDeploy(agent, metrics, registry2, 'Verbose', false);

    assert(deployed, 'Should deploy when no model exists');

    info = registry2.getDeployedInfo();
    assert(~isempty(info), 'Should have deployed model');

    fprintf('  Deployed new model: %s\n', info.version);
    fprintf('PASSED: autoDeploy (new deployment)\n\n');
    passed = passed + 1;

    delete(env);
    registry2.close();
catch ME
    fprintf('FAILED: autoDeploy (new deployment)\n');
    fprintf('  Error: %s\n\n', ME.message);
    failed = failed + 1;
end

%% Test 8: Auto-Deploy (Insufficient Improvement)
fprintf('=== Test 8: autoDeploy (Skip - No Improvement) ===\n');

try
    % Create agent with slightly worse metrics
    env = TangledEnvironment();
    agent = createPPOAgent(env);
    metrics = struct('episodes', 200, 'avgReward', 0.35, 'winRate', 0.58);

    % Get current deployed version
    info1 = registry.getDeployedInfo();

    % Try auto-deploy (should NOT deploy - improvement < 2%)
    deployed = autoDeploy(agent, metrics, registry, ...
        'MinImprovement', 0.05, 'Verbose', false);

    assert(~deployed, 'Should NOT deploy with insufficient improvement');

    % Verify same version is still deployed
    info2 = registry.getDeployedInfo();
    assert(strcmp(info1.version, info2.version), 'Deployed version should not change');

    fprintf('  Correctly skipped (improvement < 5%%)\n');
    fprintf('PASSED: autoDeploy (skip - no improvement)\n\n');
    passed = passed + 1;
    delete(env);
catch ME
    fprintf('FAILED: autoDeploy (skip - no improvement)\n');
    fprintf('  Error: %s\n\n', ME.message);
    failed = failed + 1;
end

%% Test 9: Auto-Deploy (Significant Improvement)
fprintf('=== Test 9: autoDeploy (Deploy - Improved) ===\n');

try
    % Create agent with much better metrics
    env = TangledEnvironment();
    agent = createPPOAgent(env);
    metrics = struct('episodes', 500, 'avgReward', 0.7, 'winRate', 0.75);

    % Get current deployed version
    info1 = registry.getDeployedInfo();

    % Auto-deploy (should deploy - significant improvement)
    deployed = autoDeploy(agent, metrics, registry, ...
        'MinImprovement', 0.05, 'Verbose', false);

    assert(deployed, 'Should deploy with significant improvement');

    % Verify new version is deployed
    info2 = registry.getDeployedInfo();
    assert(~strcmp(info1.version, info2.version), 'Should have new deployed version');
    assert(info2.winRate > info1.winRate, 'New version should have higher win rate');

    fprintf('  Upgraded: %s -> %s\n', info1.version, info2.version);
    fprintf('  Win rate: %.1f%% -> %.1f%%\n', info1.winRate*100, info2.winRate*100);
    fprintf('PASSED: autoDeploy (deploy - improved)\n\n');
    passed = passed + 1;
    delete(env);
catch ME
    fprintf('FAILED: autoDeploy (deploy - improved)\n');
    fprintf('  Error: %s\n\n', ME.message);
    failed = failed + 1;
end

%% Cleanup
fprintf('=== Cleanup ===\n');
try
    registry.close();
    rmdir(testDir, 's');
    fprintf('Test directory cleaned up\n\n');
catch ME
    fprintf('Warning: Cleanup failed: %s\n\n', ME.message);
end

%% Summary
fprintf('============================================\n');
fprintf('  Phase 5 Tests Complete\n');
fprintf('============================================\n\n');
fprintf('  Passed: %d\n', passed);
fprintf('  Failed: %d\n', failed);
fprintf('  Total:  %d\n\n', passed + failed);

if failed == 0
    fprintf('SUCCESS: All tests passed!\n\n');
    fprintf('The deployment pipeline is ready for use.\n\n');
    fprintf('To use in training:\n');
    fprintf('  >> registry = ModelRegistry(''models.db'', ''models'');\n');
    fprintf('  >> [agent, stats] = trainParallel(agent, ''MaxEpisodes'', 1000);\n');
    fprintf('  >> metrics = struct(''episodes'', 1000, ''winRate'', stats.winRates(end));\n');
    fprintf('  >> autoDeploy(agent, metrics, registry);\n\n');
else
    fprintf('FAILED: %d test(s) failed. Review errors above.\n\n');
end
