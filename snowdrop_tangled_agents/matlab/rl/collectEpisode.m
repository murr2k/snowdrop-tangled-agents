function exp = collectEpisode(agent, env, options)
%COLLECTEPISODE Run one episode and collect experience tuples
%
%   exp = collectEpisode(agent, env)
%   exp = collectEpisode(agent, env, Name=Value)
%
%   Runs a complete episode in the environment using the agent's policy,
%   collecting (state, action, reward, next_state, done) tuples for training.
%
%   Inputs:
%       agent - RL agent with getAction method
%       env   - TangledEnvironment instance
%
%   Name-Value Arguments:
%       MaxSteps       - Maximum steps per episode (default: 20)
%       Deterministic  - Use greedy action selection (default: false)
%       RecordMasks    - Record action masks (default: true)
%
%   Outputs:
%       exp - Struct with fields:
%           .states      - Cell array of state observations
%           .actions     - Array of actions taken
%           .rewards     - Array of rewards received
%           .nextStates  - Cell array of next state observations
%           .dones       - Array of done flags
%           .masks       - Cell array of action masks (if RecordMasks=true)
%           .totalReward - Sum of all rewards
%           .length      - Number of steps
%           .result      - 'win', 'loss', or 'draw'
%
%   Example:
%       env = TangledEnvironment();
%       agent = createPPOAgent(env);
%       exp = collectEpisode(agent, env);
%       fprintf('Episode reward: %.3f\n', exp.totalReward);

    arguments
        agent
        env
        options.MaxSteps (1,1) double = 20
        options.Deterministic logical = false
        options.RecordMasks logical = true
    end

    %% Initialize experience struct
    exp = struct();
    exp.states = {};
    exp.actions = [];
    exp.rewards = [];
    exp.nextStates = {};
    exp.dones = [];
    if options.RecordMasks
        exp.masks = {};
    end
    exp.totalReward = 0;
    exp.length = 0;
    exp.result = 'unknown';

    %% Reset environment
    obs = reset(env);
    done = false;
    stepCount = 0;

    %% Episode loop
    while ~done && stepCount < options.MaxSteps
        % Get action mask for valid moves
        mask = getActionMask(env.State);
        validActions = find(mask);

        if isempty(validActions)
            % No valid moves available
            break;
        end

        % Get action from agent with masking
        if options.Deterministic
            action = selectGreedyMaskedAction(agent, obs, mask);
        else
            action = selectMaskedAction(agent, obs, mask);
        end

        % Ensure action is valid (fallback if agent selected invalid)
        if mask(action) == 0
            % Resample from valid actions uniformly
            action = validActions(randi(length(validActions)));
        end

        % Step environment
        [nextObs, reward, done, info] = step(env, action);

        % Store experience
        exp.states{end+1} = obs;
        exp.actions(end+1) = action;
        exp.rewards(end+1) = reward;
        exp.nextStates{end+1} = nextObs;
        exp.dones(end+1) = done;
        if options.RecordMasks
            exp.masks{end+1} = mask;
        end

        exp.totalReward = exp.totalReward + reward;
        stepCount = stepCount + 1;

        % Update for next iteration
        obs = nextObs;
    end

    %% Finalize
    exp.length = stepCount;

    % Determine result
    if isfield(info, 'Result')
        exp.result = info.Result;
    elseif exp.totalReward > 0.5
        exp.result = 'win';
    elseif exp.totalReward < -0.5
        exp.result = 'loss';
    else
        exp.result = 'draw';
    end
end

function action = selectMaskedAction(agent, obs, mask)
%SELECTMASKEDACTION Sample from masked policy distribution

    % Get action probabilities from agent
    try
        actorNet = getActor(agent);
        probs = predict(actorNet, dlarray(obs, 'CB'));
        probs = extractdata(probs);
    catch
        % Fallback: uniform over valid actions
        probs = ones(30, 1) / 30;
    end

    % Apply mask
    probs = probs .* mask(:);

    % Renormalize
    probSum = sum(probs);
    if probSum > 0
        probs = probs / probSum;
    else
        % All masked - uniform over valid
        validIdx = find(mask);
        probs = zeros(30, 1);
        probs(validIdx) = 1 / length(validIdx);
    end

    % Sample from distribution
    cumProbs = cumsum(probs);
    action = find(cumProbs >= rand(), 1);

    % Safety fallback
    if isempty(action)
        validIdx = find(mask);
        action = validIdx(randi(length(validIdx)));
    end
end

function action = selectGreedyMaskedAction(agent, obs, mask)
%SELECTGREEDYMASKEDACTION Select highest probability valid action

    % Get action probabilities from agent
    try
        actorNet = getActor(agent);
        probs = predict(actorNet, dlarray(obs, 'CB'));
        probs = extractdata(probs);
    catch
        % Fallback: uniform over valid actions
        probs = ones(30, 1) / 30;
    end

    % Mask invalid actions with -inf
    probs(mask == 0) = -inf;

    % Select argmax
    [~, action] = max(probs);

    % Safety fallback
    if isinf(probs(action))
        validIdx = find(mask);
        action = validIdx(1);
    end
end
