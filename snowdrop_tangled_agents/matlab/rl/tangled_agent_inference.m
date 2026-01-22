function [action, value, probs] = tangled_agent_inference(stateVec, actionMask, modelPath)
%TANGLED_AGENT_INFERENCE Neural network inference for deployed agent
%
%   [action, value, probs] = tangled_agent_inference(stateVec, actionMask, modelPath)
%
%   This function is designed for compilation with MATLAB Compiler SDK.
%   It loads the deployed model and performs inference to select actions.
%
%   Inputs:
%       stateVec   - 50x1 observation vector (board state features)
%       actionMask - 30x1 binary mask (1 = valid action, 0 = invalid)
%       modelPath  - Path to deployed model file (optional, uses default)
%
%   Outputs:
%       action - Selected action (1-30)
%       value  - State value estimate
%       probs  - 30x1 action probabilities (after masking)
%
%   Example:
%       % From Python via compiled package:
%       state = [0.0, 0.0, 1.0, ...];  % 50 features
%       mask = [1, 1, 0, 1, ...];       % 30 valid/invalid flags
%       [action, value, probs] = tangled_agent_inference(state, mask);

    arguments
        stateVec (:,1) double
        actionMask (:,1) double
        modelPath char = ''
    end

    % Persistent cache for loaded model (avoids reloading on every call)
    persistent cachedAgent cachedModelPath lastCheckTime

    %% Determine model path
    if isempty(modelPath)
        % Default: look in standard deployment location
        if isdeployed()
            % Running as compiled application
            modelPath = fullfile(ctfroot(), 'deployed', 'current_model.mat');
        else
            % Development mode
            modelPath = fullfile(fileparts(mfilename('fullpath')), ...
                'deployed', 'current_model.mat');
        end
    end

    %% Check for model updates (hot-reload support)
    checkInterval = 60;  % seconds
    currentTime = now * 86400;  % Convert to seconds

    needsReload = isempty(cachedAgent) || ...
                  ~strcmp(cachedModelPath, modelPath) || ...
                  isempty(lastCheckTime) || ...
                  (currentTime - lastCheckTime) > checkInterval;

    if needsReload
        if exist(modelPath, 'file')
            try
                data = load(modelPath, 'agent');
                cachedAgent = data.agent;
                cachedModelPath = modelPath;
                lastCheckTime = currentTime;
            catch ME
                warning('tangled_agent_inference:LoadFailed', ...
                    'Failed to load model: %s', ME.message);
            end
        end
    end

    %% Validate inputs
    if length(stateVec) ~= 50
        error('stateVec must be 50 elements, got %d', length(stateVec));
    end

    if length(actionMask) ~= 30
        error('actionMask must be 30 elements, got %d', length(actionMask));
    end

    %% Run inference
    if ~isempty(cachedAgent)
        [action, value, probs] = agentInference(cachedAgent, stateVec, actionMask);
    else
        % Fallback: uniform random over valid actions
        [action, value, probs] = fallbackInference(actionMask);
    end
end

function [action, value, probs] = agentInference(agent, stateVec, actionMask)
%AGENTINFERENCE Run inference using trained agent

    % Get actor network
    actor = getActor(agent);
    actorNet = getModel(actor);

    % Forward pass through actor
    obsData = dlarray(stateVec(:), 'CB');
    rawProbs = forward(actorNet, obsData);
    rawProbs = extractdata(rawProbs);

    % Apply action mask
    probs = rawProbs(:) .* actionMask(:);

    % Renormalize
    probSum = sum(probs);
    if probSum > 0
        probs = probs / probSum;
    else
        % All actions masked - uniform over valid
        validIdx = find(actionMask);
        probs = zeros(30, 1);
        if ~isempty(validIdx)
            probs(validIdx) = 1 / length(validIdx);
        end
    end

    % Sample action from distribution
    cumProbs = cumsum(probs);
    r = rand();
    action = find(cumProbs >= r, 1);

    % Safety fallback
    if isempty(action)
        validIdx = find(actionMask);
        if ~isempty(validIdx)
            action = validIdx(randi(length(validIdx)));
        else
            action = 1;  % No valid actions (shouldn't happen)
        end
    end

    % Get value estimate from critic
    try
        critic = getCritic(agent);
        criticNet = getModel(critic);
        valueData = forward(criticNet, obsData);
        value = double(extractdata(valueData));
    catch
        value = 0;
    end
end

function [action, value, probs] = fallbackInference(actionMask)
%FALLBACKINFERENCE Random fallback when no model available

    validIdx = find(actionMask);

    if isempty(validIdx)
        action = 1;
        probs = zeros(30, 1);
        probs(1) = 1;
    else
        action = validIdx(randi(length(validIdx)));
        probs = zeros(30, 1);
        probs(validIdx) = 1 / length(validIdx);
    end

    value = 0;
end
