function [policyNet, info] = train_policy_network(data, options)
%TRAIN_POLICY_NETWORK Train policy network for action imitation
%
%   [policyNet, info] = train_policy_network(data)
%   [policyNet, info] = train_policy_network(data, Name=Value)
%
%   Trains a classification neural network to imitate expert moves
%   using behavioral cloning (supervised learning).
%
%   Inputs:
%       data - Training data struct from extract_training_data()
%              Must have fields: .states [50 x N], .actions [N x 1]
%
%   Name-Value Arguments:
%       MaxEpochs        - Maximum training epochs (default: 100)
%       MiniBatchSize    - Mini-batch size (default: 64)
%       ValidationSplit  - Fraction for validation (default: 0.15)
%       LearningRate     - Initial learning rate (default: 0.001)
%       HiddenUnits      - Hidden layer sizes (default: [128, 64])
%       Dropout          - Dropout rates (default: [0.3, 0.2])
%       Patience         - Early stopping patience (default: 10)
%       OnlyOurMoves     - Only train on 'us' moves (default: true)
%       WeightByOutcome  - Weight samples by outcome (default: true)
%       OutputPath       - Path to save trained network (default: '')
%       Verbose          - Show training progress (default: true)
%       ShowPlots        - Show training plots (default: false)
%
%   Outputs:
%       policyNet - Trained dlnetwork object
%       info      - Training information struct
%
%   Example:
%       data = extract_training_data();
%       [policyNet, info] = train_policy_network(data, MaxEpochs=50);
%       fprintf('Top-1 accuracy: %.1f%%\n', info.top1Accuracy * 100);

    arguments
        data struct
        options.MaxEpochs (1,1) double = 100
        options.MiniBatchSize (1,1) double = 64
        options.ValidationSplit (1,1) double = 0.15
        options.LearningRate (1,1) double = 0.001
        options.HiddenUnits (1,:) double = [128, 64]
        options.Dropout (1,:) double = [0.3, 0.2]
        options.Patience (1,1) double = 10
        options.OnlyOurMoves logical = true
        options.WeightByOutcome logical = true
        options.OutputPath string = ""
        options.Verbose logical = true
        options.ShowPlots logical = false
    end

    log_print(options.Verbose, '\n=== Training Policy Network (Behavioral Cloning) ===\n\n');

    %% Validate input data
    if ~isfield(data, 'states') || ~isfield(data, 'actions')
        error('Data must have fields: states, actions');
    end

    X = data.states;  % [50 x N]
    actions = data.actions(:);  % [N x 1]

    %% Filter to only our moves if requested
    if options.OnlyOurMoves && isfield(data, 'players')
        ourMask = strcmp(data.players, 'us');
        X = X(:, ourMask);
        actions = actions(ourMask);

        if isfield(data, 'outcomes')
            outcomes = data.outcomes(ourMask);
        else
            outcomes = ones(sum(ourMask), 1);
        end

        log_print(options.Verbose, 'Filtered to our moves only: %d samples\n', sum(ourMask));
    else
        if isfield(data, 'outcomes')
            outcomes = data.outcomes(:);
        else
            outcomes = ones(length(actions), 1);
        end
    end

    numSamples = size(X, 2);
    numFeatures = size(X, 1);
    numActions = 30;  % 15 edges x 2 colors

    log_print(options.Verbose, 'Input data:\n');
    log_print(options.Verbose, '  Samples:  %d\n', numSamples);
    log_print(options.Verbose, '  Features: %d\n', numFeatures);
    log_print(options.Verbose, '  Actions:  %d classes\n', numActions);

    %% Convert actions to one-hot encoding
    y = zeros(numActions, numSamples);
    for i = 1:numSamples
        actionIdx = actions(i);
        if actionIdx >= 1 && actionIdx <= numActions
            y(actionIdx, i) = 1;
        end
    end

    %% Compute sample weights if requested
    if options.WeightByOutcome
        % Weight winning moves higher, losing moves lower
        % outcome in [-1, +1], convert to weight in [0.5, 2.0]
        weights = 1 + 0.75 * outcomes(:);
        weights = max(weights, 0.25);  % Minimum weight
        log_print(options.Verbose, '  Sample weights: min=%.2f, max=%.2f, mean=%.2f\n', ...
            min(weights), max(weights), mean(weights));
    else
        weights = ones(numSamples, 1);
    end

    %% Split into train/validation sets
    numVal = floor(numSamples * options.ValidationSplit);
    numTrain = numSamples - numVal;

    idx = randperm(numSamples);
    trainIdx = idx(1:numTrain);
    valIdx = idx(numTrain+1:end);

    X_train = X(:, trainIdx);
    y_train = y(:, trainIdx);
    weights_train = weights(trainIdx);

    X_val = X(:, valIdx);
    y_val = y(:, valIdx);
    actions_val = actions(valIdx);

    log_print(options.Verbose, '\nData split:\n');
    log_print(options.Verbose, '  Training:   %d samples\n', numTrain);
    log_print(options.Verbose, '  Validation: %d samples\n', numVal);

    %% Build network architecture
    layers = [
        featureInputLayer(numFeatures, 'Name', 'input', 'Normalization', 'zscore')
    ];

    for i = 1:length(options.HiddenUnits)
        units = options.HiddenUnits(i);
        layers = [layers
            fullyConnectedLayer(units, 'Name', sprintf('fc%d', i))
            batchNormalizationLayer('Name', sprintf('bn%d', i))
            reluLayer('Name', sprintf('relu%d', i))
        ];

        if i <= length(options.Dropout) && options.Dropout(i) > 0
            layers = [layers
                dropoutLayer(options.Dropout(i), 'Name', sprintf('drop%d', i))
            ];
        end
    end

    layers = [layers
        fullyConnectedLayer(numActions, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')  % Output: P(action)
    ];

    log_print(options.Verbose, '\nNetwork architecture:\n');
    log_print(options.Verbose, '  Input:  %d features\n', numFeatures);
    for i = 1:length(options.HiddenUnits)
        log_print(options.Verbose, '  Hidden: %d units', options.HiddenUnits(i));
        if i <= length(options.Dropout) && options.Dropout(i) > 0
            log_print(options.Verbose, ' (dropout %.1f%%)', options.Dropout(i)*100);
        end
        log_print(options.Verbose, '\n');
    end
    log_print(options.Verbose, '  Output: %d (softmax)\n', numActions);

    %% Training options
    if options.ShowPlots
        plotOption = 'training-progress';
    else
        plotOption = 'none';
    end

    trainOpts = trainingOptions('adam', ...
        'MaxEpochs', options.MaxEpochs, ...
        'MiniBatchSize', options.MiniBatchSize, ...
        'InitialLearnRate', options.LearningRate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', 30, ...
        'LearnRateDropFactor', 0.5, ...
        'ValidationData', {X_val', y_val'}, ...
        'ValidationFrequency', max(1, floor(numTrain / options.MiniBatchSize)), ...
        'ValidationPatience', options.Patience, ...
        'Shuffle', 'every-epoch', ...
        'Plots', plotOption, ...
        'Verbose', options.Verbose, ...
        'OutputNetwork', 'best-validation-loss');

    %% Train the network
    log_print(options.Verbose, '\nTraining...\n');
    tic;

    % Use trainnet with crossentropy loss
    [policyNet, trainInfo] = trainnet(X_train', y_train', layers, 'crossentropy', trainOpts);

    trainTime = toc;

    %% Evaluate on validation set
    y_pred_val = predict(policyNet, X_val');
    y_pred_val = extractdata(y_pred_val);

    % Prediction output is [numVal x 30] (samples as rows)
    % Transpose to [30 x numVal] to match y_val format
    y_pred_val = y_pred_val';

    % Top-1 accuracy
    [~, predActions] = max(y_pred_val, [], 1);  % Max across actions
    top1Acc = mean(predActions(:) == actions_val(:));

    % Top-5 accuracy
    top5Acc = 0;
    for i = 1:length(actions_val)
        [~, sortedIdx] = sort(y_pred_val(:, i), 'descend');
        if any(sortedIdx(1:min(5,end)) == actions_val(i))
            top5Acc = top5Acc + 1;
        end
    end
    top5Acc = top5Acc / length(actions_val);

    % Cross-entropy loss
    valLoss = -mean(sum(y_val .* log(y_pred_val + 1e-10), 1));

    %% Build output info struct
    info = struct();
    info.trainTime = trainTime;
    % R2023b+ uses TrainingHistory table instead of TrainingLoss
    if isprop(trainInfo, 'TrainingHistory')
        info.numEpochs = max(trainInfo.TrainingHistory.Epoch);
    else
        info.numEpochs = trainInfo.NumEpochs;
    end
    info.finalValLoss = valLoss;
    info.top1Accuracy = top1Acc;
    info.top5Accuracy = top5Acc;
    info.trainingHistory = trainInfo;
    info.numTrainSamples = numTrain;
    info.numValSamples = numVal;
    info.architecture = options.HiddenUnits;
    info.dropout = options.Dropout;

    %% Summary
    log_print(options.Verbose, '\n=== Training Complete ===\n');
    log_print(options.Verbose, '  Training time:    %.1f seconds\n', trainTime);
    log_print(options.Verbose, '  Epochs:           %d\n', info.numEpochs);
    log_print(options.Verbose, '  Validation loss:  %.4f\n', valLoss);
    log_print(options.Verbose, '  Top-1 accuracy:   %.1f%%\n', top1Acc * 100);
    log_print(options.Verbose, '  Top-5 accuracy:   %.1f%%\n', top5Acc * 100);

    %% Save if requested
    if options.OutputPath ~= ""
        save(options.OutputPath, 'policyNet', 'info');
        log_print(options.Verbose, '\nNetwork saved to: %s\n', options.OutputPath);
    end

    log_print(options.Verbose, '\n');
end

function log_print(verbose, varargin)
    if verbose
        fprintf(varargin{:});
    end
end
