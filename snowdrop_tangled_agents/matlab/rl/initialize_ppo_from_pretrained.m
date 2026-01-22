function agent = initialize_ppo_from_pretrained(env, valueNet, policyNet, options)
%INITIALIZE_PPO_FROM_PRETRAINED Create PPO agent with pre-trained weights
%
%   agent = initialize_ppo_from_pretrained(env, valueNet, policyNet)
%   agent = initialize_ppo_from_pretrained(env, valueNet, policyNet, Name=Value)
%
%   Creates a PPO agent and initializes its actor and critic networks
%   with weights from pre-trained value and policy networks.
%
%   Inputs:
%       env       - TangledEnvironment instance
%       valueNet  - Pre-trained value network (dlnetwork)
%       policyNet - Pre-trained policy network (dlnetwork)
%
%   Name-Value Arguments:
%       TransferMode - 'full' (copy all matching layers) or
%                      'partial' (copy only first N layers) (default: 'full')
%       NumLayers    - Number of layers to transfer in 'partial' mode (default: 2)
%       Verbose      - Print progress (default: true)
%
%   Outputs:
%       agent - PPO agent with pre-trained weights
%
%   Example:
%       env = TangledEnvironment();
%       [valueNet, ~] = train_value_network(data);
%       [policyNet, ~] = train_policy_network(data);
%       agent = initialize_ppo_from_pretrained(env, valueNet, policyNet);

    arguments
        env
        valueNet
        policyNet
        options.TransferMode string = "full"
        options.NumLayers (1,1) double = 2
        options.Verbose logical = true
    end

    log_print(options.Verbose, '\n=== Initializing PPO from Pre-trained Networks ===\n\n');

    %% Create base PPO agent with standard architecture
    agent = createPPOAgent(env);

    %% Extract actor and critic from agent
    actor = getActor(agent);
    critic = getCritic(agent);

    actorNet = getModel(actor);
    criticNet = getModel(critic);

    log_print(options.Verbose, 'PPO Agent networks:\n');
    log_print(options.Verbose, '  Actor layers:  %d\n', numel(actorNet.Learnables));
    log_print(options.Verbose, '  Critic layers: %d\n', numel(criticNet.Learnables));

    %% Transfer weights to actor (from policy network)
    log_print(options.Verbose, '\nTransferring policy network weights to actor...\n');
    actorNet = transfer_weights(actorNet, policyNet, options.TransferMode, ...
        options.NumLayers, options.Verbose);

    %% Transfer weights to critic (from value network)
    log_print(options.Verbose, '\nTransferring value network weights to critic...\n');
    criticNet = transfer_weights(criticNet, valueNet, options.TransferMode, ...
        options.NumLayers, options.Verbose);

    %% Update agent with new networks
    actor = setModel(actor, actorNet);
    critic = setModel(critic, criticNet);
    agent = setActor(agent, actor);
    agent = setCritic(agent, critic);

    %% Verify transfer
    log_print(options.Verbose, '\n=== Transfer Complete ===\n');

    % Test forward pass
    obsInfo = getObservationInfo(env);
    testObs = rand(obsInfo.Dimension);

    try
        testActorNet = getModel(getActor(agent));
        testCriticNet = getModel(getCritic(agent));

        actorOut = forward(testActorNet, dlarray(testObs, 'CB'));
        criticOut = forward(testCriticNet, dlarray(testObs, 'CB'));

        log_print(options.Verbose, '  Actor forward pass:  OK (output size: %d)\n', numel(extractdata(actorOut)));
        log_print(options.Verbose, '  Critic forward pass: OK (output size: %d)\n', numel(extractdata(criticOut)));
    catch ME
        warning('Forward pass test failed: %s', ME.message);
    end

    log_print(options.Verbose, '\nPPO agent initialized with pre-trained weights.\n\n');
end

function targetNet = transfer_weights(targetNet, sourceNet, mode, numLayers, verbose)
%TRANSFER_WEIGHTS Transfer weights from source to target network
%
%   Attempts to copy weights from source network layers to matching
%   target network layers based on layer names and dimensions.

    targetLearnables = targetNet.Learnables;
    sourceLearnables = sourceNet.Learnables;

    numTargetLayers = height(targetLearnables);
    numSourceLayers = height(sourceLearnables);

    log_print(verbose, '  Source layers: %d\n', numSourceLayers);
    log_print(verbose, '  Target layers: %d\n', numTargetLayers);

    transferred = 0;
    skipped = 0;

    % Build mapping of source layer names to indices
    sourceMap = containers.Map();
    for i = 1:numSourceLayers
        layerName = sourceLearnables.Layer{i};
        paramName = sourceLearnables.Parameter{i};
        key = sprintf('%s/%s', layerName, paramName);
        sourceMap(key) = i;
    end

    % Iterate through target layers
    for t = 1:numTargetLayers
        targetLayerName = targetLearnables.Layer{t};
        targetParamName = targetLearnables.Parameter{t};
        targetKey = sprintf('%s/%s', targetLayerName, targetParamName);

        targetSize = size(targetLearnables.Value{t});

        % Try to find matching source layer
        matched = false;

        % Strategy 1: Exact name match
        if sourceMap.isKey(targetKey)
            sourceIdx = sourceMap(targetKey);
            sourceSize = size(sourceLearnables.Value{sourceIdx});

            if isequal(targetSize, sourceSize)
                targetNet.Learnables.Value{t} = sourceLearnables.Value{sourceIdx};
                matched = true;
                transferred = transferred + 1;
                log_print(verbose, '    [MATCH] %s: %s\n', targetKey, mat2str(targetSize));
            end
        end

        % Strategy 2: Match by layer number (fc1 -> fc1, etc.)
        if ~matched
            % Extract layer number from name (e.g., 'fc1' -> 1)
            tokens = regexp(targetLayerName, '(\d+)', 'tokens');
            if ~isempty(tokens)
                layerNum = str2double(tokens{1}{1});

                % Check mode
                if strcmp(mode, 'partial') && layerNum > numLayers
                    skipped = skipped + 1;
                    continue;
                end

                % Look for similar source layer
                for s = 1:numSourceLayers
                    sourceLayerName = sourceLearnables.Layer{s};
                    sourceParamName = sourceLearnables.Parameter{s};

                    % Check if same parameter type and similar layer number
                    if strcmp(targetParamName, sourceParamName)
                        sourceTokens = regexp(sourceLayerName, '(\d+)', 'tokens');
                        if ~isempty(sourceTokens)
                            sourceLayerNum = str2double(sourceTokens{1}{1});
                            if sourceLayerNum == layerNum
                                sourceSize = size(sourceLearnables.Value{s});

                                if isequal(targetSize, sourceSize)
                                    targetNet.Learnables.Value{t} = sourceLearnables.Value{s};
                                    matched = true;
                                    transferred = transferred + 1;
                                    log_print(verbose, '    [NUM]   %s <- %s/%s: %s\n', ...
                                        targetKey, sourceLayerName, sourceParamName, mat2str(targetSize));
                                    break;
                                end
                            end
                        end
                    end
                end
            end
        end

        if ~matched
            skipped = skipped + 1;
        end
    end

    log_print(verbose, '  Transferred: %d layers\n', transferred);
    log_print(verbose, '  Skipped:     %d layers\n', skipped);
end

function log_print(verbose, varargin)
    if verbose
        fprintf(varargin{:});
    end
end
