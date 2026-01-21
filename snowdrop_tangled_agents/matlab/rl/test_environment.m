%% Test Suite for Tangled RL Environment
% Run with: runtests('test_environment')

%% Test 1: Environment Construction
function test_environment_construction(testCase)
    env = TangledEnvironment();

    % Check observation spec
    obsInfo = getObservationInfo(env);
    testCase.verifyEqual(obsInfo.Dimension, [50 1]);
    testCase.verifyEqual(obsInfo.LowerLimit, -1);
    testCase.verifyEqual(obsInfo.UpperLimit, 1);

    % Check action spec
    actInfo = getActionInfo(env);
    testCase.verifyEqual(length(actInfo.Elements), 30);
end

%% Test 2: Environment Reset
function test_environment_reset(testCase)
    env = TangledEnvironment();
    obs = reset(env);

    % Check observation shape
    testCase.verifySize(obs, [50 1]);

    % Check initial state
    testCase.verifyEqual(env.State, '---------------');
    testCase.verifyEqual(env.MoveCount, 0);
    testCase.verifyEqual(env.Score, 0);
end

%% Test 3: Valid Action Execution
function test_valid_action(testCase)
    env = TangledEnvironment();
    reset(env);

    % Action 1 = Green on edge 0
    [obs, reward, isDone, info] = step(env, 1);

    testCase.verifyFalse(info.InvalidAction);
    testCase.verifyEqual(env.State(1), 'G');
    testCase.verifyFalse(isDone);  % Game not over after 1 move
end

%% Test 4: Invalid Action Handling
function test_invalid_action(testCase)
    env = TangledEnvironment();
    reset(env);

    % Play edge 0
    step(env, 1);

    % Try to play edge 0 again (should be invalid)
    [~, reward, ~, info] = step(env, 1);

    testCase.verifyTrue(info.InvalidAction);
    testCase.verifyEqual(reward, env.InvalidActionPenalty);
end

%% Test 5: Action Mask
function test_action_mask(testCase)
    state = 'GP-------------';
    mask = getActionMask(state);

    % Edges 0 and 1 are colored, so actions 1,2,16,17 should be 0
    testCase.verifyEqual(mask(1), 0);   % Green on E0
    testCase.verifyEqual(mask(2), 0);   % Green on E1
    testCase.verifyEqual(mask(16), 0);  % Purple on E0
    testCase.verifyEqual(mask(17), 0);  % Purple on E1

    % Edges 2-14 are grey, so should be 1
    testCase.verifyEqual(mask(3), 1);   % Green on E2
    testCase.verifyEqual(mask(18), 1);  % Purple on E2
end

%% Test 6: Feature Builder - Empty Board
function test_features_empty_board(testCase)
    state = '---------------';
    features = buildRLFeatures(state, 1, 0);

    testCase.verifySize(features, [50 1]);

    % Board state should be all zeros
    testCase.verifyEqual(features(1:15), zeros(15, 1));

    % Turn indicator
    testCase.verifyEqual(features(16), 1);

    % Grey count should be 1.0 (all grey)
    testCase.verifyEqual(features(32), 1.0);
end

%% Test 7: Feature Builder - Partial Board
function test_features_partial_board(testCase)
    state = 'GP-------------';
    features = buildRLFeatures(state, -1, 0.5);

    % E0 = Green = +1
    testCase.verifyEqual(features(1), 1);

    % E1 = Purple = -1
    testCase.verifyEqual(features(2), -1);

    % E2 = Grey = 0
    testCase.verifyEqual(features(3), 0);

    % Turn indicator = -1 (opponent)
    testCase.verifyEqual(features(16), -1);

    % Grey count = 13/15
    testCase.verifyEqual(features(32), 13/15, 'AbsTol', 1e-10);
end

%% Test 8: Simulated Opponent - Random
function test_opponent_random(testCase)
    opp = SimulatedOpponent('Style', 'random');
    state = '---------------';

    move = opp.selectMove(state);

    testCase.verifyGreaterThanOrEqual(move.edge, 0);
    testCase.verifyLessThanOrEqual(move.edge, 14);
    testCase.verifyTrue(move.color == 'G' || move.color == 'P');
end

%% Test 9: Simulated Opponent - MCTS Style
function test_opponent_mcts(testCase)
    opp = SimulatedOpponent('Style', 'mcts');
    state = '---------------';

    move = opp.selectMove(state);

    testCase.verifyGreaterThanOrEqual(move.edge, 0);
    testCase.verifyLessThanOrEqual(move.edge, 14);
    testCase.verifyTrue(move.color == 'G' || move.color == 'P');
end

%% Test 10: Full Game Simulation
function test_full_game(testCase)
    env = TangledEnvironment();
    obs = reset(env);

    isDone = false;
    moveCount = 0;
    maxMoves = 20;  % Safety limit

    while ~isDone && moveCount < maxMoves
        % Get valid actions
        mask = env.getActionMask();
        validActions = find(mask);

        if isempty(validActions)
            break;
        end

        % Random valid action
        action = validActions(randi(length(validActions)));
        [obs, reward, isDone, info] = step(env, action);
        moveCount = moveCount + 1;
    end

    % Game should end
    testCase.verifyTrue(isDone || sum(env.State == '-') == 0);
end

%% Test 11: Reward Bounds
function test_reward_bounds(testCase)
    env = TangledEnvironment();

    % Run multiple episodes
    for ep = 1:5
        obs = reset(env);
        isDone = false;

        while ~isDone
            mask = env.getActionMask();
            validActions = find(mask);

            if isempty(validActions)
                break;
            end

            action = validActions(randi(length(validActions)));
            [~, reward, isDone, ~] = step(env, action);

            % Reward should be bounded
            testCase.verifyGreaterThanOrEqual(reward, -1.5);
            testCase.verifyLessThanOrEqual(reward, 1.5);
        end
    end
end

%% Test 12: Observation Consistency
function test_observation_consistency(testCase)
    env = TangledEnvironment();
    obs1 = reset(env);

    % Same state should give same observation
    obs2 = env.getObservation();
    testCase.verifyEqual(obs1, obs2);
end
