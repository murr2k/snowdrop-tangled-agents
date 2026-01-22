function [actor, critic] = createPPONetworks(obsInfo, actInfo)
%CREATEPPONETWORKS Create actor and critic networks for PPO agent
%
%   [actor, critic] = createPPONetworks(obsInfo, actInfo)
%
%   Creates the neural network architectures for Proximal Policy
%   Optimization (PPO) agent. Both networks share a similar structure
%   but have different output layers.
%
%   Inputs:
%       obsInfo - Observation specification from environment
%       actInfo - Action specification from environment
%
%   Outputs:
%       actor  - Policy network (dlnetwork) outputting action probabilities
%       critic - Value network (dlnetwork) outputting state value
%
%   Architecture:
%       Input:  50-element observation vector
%       Shared: FC(128) → ReLU → FC(64) → ReLU
%       Actor:  FC(30) → Softmax (action probabilities)
%       Critic: FC(32) → ReLU → FC(1) (state value)
%
%   Example:
%       env = TangledEnvironment();
%       obsInfo = getObservationInfo(env);
%       actInfo = getActionInfo(env);
%       [actor, critic] = createPPONetworks(obsInfo, actInfo);

    arguments
        obsInfo
        actInfo
    end

    % Get dimensions
    obsSize = obsInfo.Dimension(1);  % 50
    numActions = length(actInfo.Elements);  % 30

    %% Actor Network (Policy)
    % Outputs probability distribution over actions

    actorLayers = [
        featureInputLayer(obsSize, 'Name', 'obs_actor', ...
            'Normalization', 'none')

        fullyConnectedLayer(128, 'Name', 'fc1_actor', ...
            'WeightsInitializer', 'he')
        reluLayer('Name', 'relu1_actor')
        dropoutLayer(0.2, 'Name', 'drop1_actor')

        fullyConnectedLayer(64, 'Name', 'fc2_actor', ...
            'WeightsInitializer', 'he')
        reluLayer('Name', 'relu2_actor')
        dropoutLayer(0.1, 'Name', 'drop2_actor')

        fullyConnectedLayer(numActions, 'Name', 'fc_out_actor', ...
            'WeightsInitializer', 'glorot')
        softmaxLayer('Name', 'actionProb')
    ];

    actor = dlnetwork(layerGraph(actorLayers));

    %% Critic Network (Value Function)
    % Outputs scalar state value estimate

    criticLayers = [
        featureInputLayer(obsSize, 'Name', 'obs_critic', ...
            'Normalization', 'none')

        fullyConnectedLayer(128, 'Name', 'fc1_critic', ...
            'WeightsInitializer', 'he')
        reluLayer('Name', 'relu1_critic')
        dropoutLayer(0.2, 'Name', 'drop1_critic')

        fullyConnectedLayer(64, 'Name', 'fc2_critic', ...
            'WeightsInitializer', 'he')
        reluLayer('Name', 'relu2_critic')

        fullyConnectedLayer(32, 'Name', 'fc3_critic', ...
            'WeightsInitializer', 'he')
        reluLayer('Name', 'relu3_critic')

        fullyConnectedLayer(1, 'Name', 'value', ...
            'WeightsInitializer', 'glorot')
    ];

    critic = dlnetwork(layerGraph(criticLayers));

    %% Summary
    fprintf('Actor network created:\n');
    fprintf('  Input: %d features\n', obsSize);
    fprintf('  Output: %d action probabilities\n', numActions);
    fprintf('  Parameters: %d\n', countParameters(actor));

    fprintf('Critic network created:\n');
    fprintf('  Input: %d features\n', obsSize);
    fprintf('  Output: 1 (state value)\n');
    fprintf('  Parameters: %d\n', countParameters(critic));
end

function n = countParameters(net)
%COUNTPARAMETERS Count total trainable parameters in network
    n = 0;
    for i = 1:length(net.Learnables.Value)
        n = n + numel(net.Learnables.Value{i});
    end
end
