function stats = monitor_critic(agent, numSamples)
%MONITOR_CRITIC Check critic prediction distribution across varied states
%
%   stats = monitor_critic(agent)
%   stats = monitor_critic(agent, numSamples)
%
%   Tests the critic network on random and structured board states
%   to check if it produces varied predictions or is stuck at constant.
%
%   Inputs:
%       agent      - PPO agent with critic network
%       numSamples - Number of random states to test (default: 100)
%
%   Outputs:
%       stats - Struct with prediction statistics

    if nargin < 2
        numSamples = 100;
    end

    fprintf('\n=== Critic Prediction Analysis ===\n\n');

    %% Extract critic network
    critic = getCritic(agent);
    criticNet = getModel(critic);

    %% Generate test states
    % 1. Random states
    randomStates = generate_random_states(numSamples);

    % 2. Structured states (opening, mid, endgame)
    structuredStates = generate_structured_states();

    %% Evaluate random states
    fprintf('Random states (%d samples):\n', numSamples);
    randomPreds = evaluate_states(criticNet, randomStates);

    stats.random.mean = mean(randomPreds);
    stats.random.std = std(randomPreds);
    stats.random.min = min(randomPreds);
    stats.random.max = max(randomPreds);
    stats.random.range = stats.random.max - stats.random.min;

    fprintf('  Mean:  %+.4f\n', stats.random.mean);
    fprintf('  Std:   %.4f\n', stats.random.std);
    fprintf('  Range: [%.4f, %.4f] (span: %.4f)\n', ...
        stats.random.min, stats.random.max, stats.random.range);

    %% Evaluate structured states
    fprintf('\nStructured states:\n');
    structuredPreds = evaluate_states(criticNet, structuredStates.states);

    stats.structured.predictions = structuredPreds;
    stats.structured.labels = structuredStates.labels;

    for i = 1:length(structuredPreds)
        fprintf('  %-25s: %+.4f\n', structuredStates.labels{i}, structuredPreds(i));
    end

    %% Diagnosis
    fprintf('\n=== Diagnosis ===\n');

    if stats.random.std < 0.01
        fprintf('  [WARNING] Critic predictions nearly constant (std=%.4f)\n', stats.random.std);
        fprintf('            Critic has not learned to distinguish positions.\n');
        stats.diagnosis = 'FLAT';
        stats.recommendation = 'Consider re-initializing critic or increasing learning rate';
    elseif stats.random.std < 0.05
        fprintf('  [CAUTION] Critic predictions have low variance (std=%.4f)\n', stats.random.std);
        fprintf('            Some learning occurred but limited.\n');
        stats.diagnosis = 'LOW_VARIANCE';
        stats.recommendation = 'Monitor during fine-tuning';
    else
        fprintf('  [OK] Critic produces varied predictions (std=%.4f)\n', stats.random.std);
        stats.diagnosis = 'HEALTHY';
        stats.recommendation = 'Proceed with fine-tuning';
    end

    % Check if winning/losing states are distinguished
    winIdx = find(contains(structuredStates.labels, 'winning'));
    loseIdx = find(contains(structuredStates.labels, 'losing'));

    if ~isempty(winIdx) && ~isempty(loseIdx)
        winPred = mean(structuredPreds(winIdx));
        losePred = mean(structuredPreds(loseIdx));
        diff = winPred - losePred;

        fprintf('\n  Win vs Loss distinction:\n');
        fprintf('    Avg winning state value: %+.4f\n', winPred);
        fprintf('    Avg losing state value:  %+.4f\n', losePred);
        fprintf('    Difference: %.4f\n', diff);

        if diff > 0.1
            fprintf('    [OK] Critic distinguishes win/loss\n');
        elseif diff > 0
            fprintf('    [WEAK] Slight distinction but may improve\n');
        else
            fprintf('    [BAD] Critic predicts losing > winning (inverted!)\n');
        end

        stats.winLossDiff = diff;
    end

    fprintf('\n');
end

function states = generate_random_states(n)
%GENERATE_RANDOM_STATES Create random 50-dim state vectors
    states = zeros(50, n);

    for i = 1:n
        % Random board (15 edges: -1, 0, or 1)
        board = randi(3, 15, 1) - 2;  % [-1, 0, 1]
        states(1:15, i) = board;

        % Random turn
        states(16, i) = (randi(2) - 1) * 2 - 1;  % -1 or 1

        % Edge categories (fixed based on Petersen graph)
        states(17:31, i) = [0.5*ones(5,1); -0.5*ones(5,1); 0.25*ones(5,1)];

        % Grey count
        greyCount = sum(board == 0);
        states(32, i) = greyCount / 15;

        % Score momentum (random)
        states(33:35, i) = randn(3, 1) * 0.1;

        % Game phase one-hot
        states(36:50, i) = 0;
        if greyCount > 10
            states(36:40, i) = 1;
        elseif greyCount >= 5
            states(41:45, i) = 1;
        else
            states(46:50, i) = 1;
        end
    end
end

function data = generate_structured_states()
%GENERATE_STRUCTURED_STATES Create specific test positions
    labels = {};
    states = [];

    % Empty board (opening)
    s = zeros(50, 1);
    s(1:15) = 0;  % All grey
    s(16) = 1;    % Our turn
    s(17:31) = [0.5*ones(5,1); -0.5*ones(5,1); 0.25*ones(5,1)];
    s(32) = 1.0;  % All grey
    s(36:40) = 1; % Opening phase
    states = [states, s];
    labels{end+1} = 'empty_board';

    % We control all our edges (winning-ish)
    s = zeros(50, 1);
    s(1:5) = 1;   % Our edges are green
    s(6:10) = -1; % Their edges are purple
    s(11:15) = 0; % Hubs grey
    s(16) = 1;
    s(17:31) = [0.5*ones(5,1); -0.5*ones(5,1); 0.25*ones(5,1)];
    s(32) = 5/15;
    s(41:45) = 1; % Mid phase
    states = [states, s];
    labels{end+1} = 'winning_control_edges';

    % Opponent controls our edges (losing-ish)
    s = zeros(50, 1);
    s(1:5) = -1;  % Our edges are purple (bad)
    s(6:10) = 1;  % Their edges are green (bad for us)
    s(11:15) = 0;
    s(16) = 1;
    s(17:31) = [0.5*ones(5,1); -0.5*ones(5,1); 0.25*ones(5,1)];
    s(32) = 5/15;
    s(41:45) = 1;
    states = [states, s];
    labels{end+1} = 'losing_opp_controls';

    % All green (extreme winning)
    s = zeros(50, 1);
    s(1:15) = 1;  % All green
    s(16) = 1;
    s(17:31) = [0.5*ones(5,1); -0.5*ones(5,1); 0.25*ones(5,1)];
    s(32) = 0;
    s(46:50) = 1; % Endgame
    states = [states, s];
    labels{end+1} = 'winning_all_green';

    % All purple (extreme losing)
    s = zeros(50, 1);
    s(1:15) = -1; % All purple
    s(16) = 1;
    s(17:31) = [0.5*ones(5,1); -0.5*ones(5,1); 0.25*ones(5,1)];
    s(32) = 0;
    s(46:50) = 1;
    states = [states, s];
    labels{end+1} = 'losing_all_purple';

    % Mixed endgame
    s = zeros(50, 1);
    s(1:15) = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]';
    s(16) = 1;
    s(17:31) = [0.5*ones(5,1); -0.5*ones(5,1); 0.25*ones(5,1)];
    s(32) = 0;
    s(46:50) = 1;
    states = [states, s];
    labels{end+1} = 'mixed_endgame';

    data.states = states;
    data.labels = labels;
end

function preds = evaluate_states(criticNet, states)
%EVALUATE_STATES Get critic predictions for states
    dlStates = dlarray(states, 'CB');
    dlPreds = forward(criticNet, dlStates);
    preds = extractdata(dlPreds);
    preds = preds(:);
end
