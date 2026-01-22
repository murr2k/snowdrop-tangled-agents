function agent = createPPOAgent(env, options)
%CREATEPPOAGENT Create configured PPO agent for Tangled game
%
%   agent = createPPOAgent(env)
%   agent = createPPOAgent(env, Name=Value)
%
%   Creates a Proximal Policy Optimization (PPO) agent configured for
%   the Tangled game environment.
%
%   Inputs:
%       env - TangledEnvironment instance
%
%   Name-Value Arguments:
%       ExperienceHorizon    - Steps before update (default: 128)
%       ClipFactor           - PPO clip parameter (default: 0.2)
%       EntropyWeight        - Exploration bonus weight (default: 0.01)
%       MiniBatchSize        - Training batch size (default: 32)
%       NumEpochs            - Epochs per update (default: 4)
%       DiscountFactor       - Gamma for returns (default: 0.99)
%       GAEFactor            - Lambda for GAE (default: 0.95)
%       ActorLearnRate       - Actor learning rate (default: 3e-4)
%       CriticLearnRate      - Critic learning rate (default: 1e-3)
%
%   Outputs:
%       agent - Configured rlPPOAgent ready for training
%
%   Example:
%       env = TangledEnvironment();
%       agent = createPPOAgent(env, 'ExperienceHorizon', 256);
%       trainingStats = train(agent, env, trainOpts);

    arguments
        env
        options.ExperienceHorizon (1,1) double = 128
        options.ClipFactor (1,1) double = 0.2
        options.EntropyWeight (1,1) double = 0.01
        options.MiniBatchSize (1,1) double = 32
        options.NumEpochs (1,1) double = 4
        options.DiscountFactor (1,1) double = 0.99
        options.GAEFactor (1,1) double = 0.95
        options.ActorLearnRate (1,1) double = 3e-4
        options.CriticLearnRate (1,1) double = 1e-3
    end

    %% Get environment specs
    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);

    %% Create networks
    [actorNet, criticNet] = createPPONetworks(obsInfo, actInfo);

    %% Create actor representation
    % For discrete actions, use rlDiscreteCategoricalActor
    actor = rlDiscreteCategoricalActor(actorNet, obsInfo, actInfo, ...
        'ObservationInputNames', 'obs_actor');

    %% Create critic representation
    critic = rlValueFunction(criticNet, obsInfo, ...
        'ObservationInputNames', 'obs_critic');

    %% Configure PPO agent options
    agentOpts = rlPPOAgentOptions(...
        'ExperienceHorizon', options.ExperienceHorizon, ...
        'ClipFactor', options.ClipFactor, ...
        'EntropyLossWeight', options.EntropyWeight, ...
        'MiniBatchSize', options.MiniBatchSize, ...
        'NumEpoch', options.NumEpochs, ...
        'AdvantageEstimateMethod', 'gae', ...
        'GAEFactor', options.GAEFactor, ...
        'DiscountFactor', options.DiscountFactor, ...
        'SampleTime', 1);

    %% Configure optimizers
    agentOpts.ActorOptimizerOptions = rlOptimizerOptions(...
        'Algorithm', 'adam', ...
        'LearnRate', options.ActorLearnRate, ...
        'GradientThreshold', 1.0, ...
        'L2RegularizationFactor', 1e-4);

    agentOpts.CriticOptimizerOptions = rlOptimizerOptions(...
        'Algorithm', 'adam', ...
        'LearnRate', options.CriticLearnRate, ...
        'GradientThreshold', 1.0, ...
        'L2RegularizationFactor', 1e-4);

    %% Create agent
    agent = rlPPOAgent(actor, critic, agentOpts);

    %% Print summary
    fprintf('\nPPO Agent created:\n');
    fprintf('  Experience Horizon: %d\n', options.ExperienceHorizon);
    fprintf('  Clip Factor: %.2f\n', options.ClipFactor);
    fprintf('  Entropy Weight: %.3f\n', options.EntropyWeight);
    fprintf('  Mini-Batch Size: %d\n', options.MiniBatchSize);
    fprintf('  Epochs per Update: %d\n', options.NumEpochs);
    fprintf('  Discount Factor: %.2f\n', options.DiscountFactor);
    fprintf('  GAE Factor: %.2f\n', options.GAEFactor);
    fprintf('  Actor LR: %.1e\n', options.ActorLearnRate);
    fprintf('  Critic LR: %.1e\n', options.CriticLearnRate);
end
