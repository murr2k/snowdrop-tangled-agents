function agent = reinitialize_critic(agent, options)
%REINITIALIZE_CRITIC Reset critic weights while keeping actor
%
%   agent = reinitialize_critic(agent)
%   agent = reinitialize_critic(agent, Name=Value)
%
%   Re-initializes the critic network with random weights when
%   the pre-trained critic has failed to learn useful values.
%   Keeps the actor (policy) network intact.
%
%   Name-Value Arguments:
%       InitMethod - 'xavier' (default), 'he', or 'narrow'
%       Verbose    - Print details (default: true)
%
%   Example:
%       agent = reinitialize_critic(agent);
%       % Now fine-tune with fresh critic

    arguments
        agent
        options.InitMethod string = "xavier"
        options.Verbose logical = true
    end

    log_print(options.Verbose, '\n=== Re-initializing Critic Network ===\n\n');

    %% Get current critic
    critic = getCritic(agent);
    criticNet = getModel(critic);

    learnables = criticNet.Learnables;
    numLayers = height(learnables);

    log_print(options.Verbose, 'Critic has %d learnable parameter sets\n', numLayers);

    %% Re-initialize weights
    for i = 1:numLayers
        layerName = learnables.Layer{i};
        paramName = learnables.Parameter{i};
        paramValue = learnables.Value{i};
        paramSize = size(paramValue);

        if strcmp(paramName, 'Weights')
            % Re-initialize weights
            fanIn = paramSize(2);
            fanOut = paramSize(1);

            switch options.InitMethod
                case 'xavier'
                    % Xavier/Glorot initialization
                    stddev = sqrt(2 / (fanIn + fanOut));
                case 'he'
                    % He initialization (good for ReLU)
                    stddev = sqrt(2 / fanIn);
                case 'narrow'
                    % Narrow initialization (start conservative)
                    stddev = sqrt(1 / (fanIn + fanOut));
                otherwise
                    stddev = sqrt(2 / (fanIn + fanOut));
            end

            newWeights = randn(paramSize) * stddev;
            criticNet.Learnables.Value{i} = dlarray(single(newWeights));

            log_print(options.Verbose, '  [RESET] %s/%s: %s (std=%.4f)\n', ...
                layerName, paramName, mat2str(paramSize), stddev);

        elseif strcmp(paramName, 'Bias')
            % Initialize biases to small values or zero
            newBias = zeros(paramSize, 'single') + 0.01;
            criticNet.Learnables.Value{i} = dlarray(newBias);

            log_print(options.Verbose, '  [RESET] %s/%s: %s (zeros+0.01)\n', ...
                layerName, paramName, mat2str(paramSize));
        else
            log_print(options.Verbose, '  [SKIP]  %s/%s: %s\n', ...
                layerName, paramName, mat2str(paramSize));
        end
    end

    %% Update agent with new critic
    critic = setModel(critic, criticNet);
    agent = setCritic(agent, critic);

    %% Verify
    log_print(options.Verbose, '\nVerifying new critic...\n');

    % Test on random states
    testStates = randn(50, 100);
    dlStates = dlarray(testStates, 'CB');
    newCriticNet = getModel(getCritic(agent));
    preds = forward(newCriticNet, dlStates);
    preds = extractdata(preds);

    log_print(options.Verbose, '  Prediction range: [%.4f, %.4f]\n', min(preds), max(preds));
    log_print(options.Verbose, '  Prediction std:   %.4f\n', std(preds));

    if std(preds) > 0.1
        log_print(options.Verbose, '  [OK] Critic now produces varied predictions\n');
    else
        log_print(options.Verbose, '  [WARNING] Critic still has low variance\n');
    end

    log_print(options.Verbose, '\nCritic re-initialized. Actor weights preserved.\n\n');
end

function log_print(verbose, varargin)
    if verbose
        fprintf(varargin{:});
    end
end
