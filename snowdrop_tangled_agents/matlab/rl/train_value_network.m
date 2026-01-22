function [valueNet, info] = train_value_network(data, options)
%TRAIN_VALUE_NETWORK Train value network to predict game outcomes
%
%   [valueNet, info] = train_value_network(data)
%   [valueNet, info] = train_value_network(data, Name=Value)
%
%   Trains a regression neural network to predict game outcomes from
%   board positions using supervised learning (Deep Learning Toolbox).
%
%   Inputs:
%       data - Training data struct from extract_training_data()
%              Must have fields: .states [50 x N], .outcomes [N x 1]
%
%   Name-Value Arguments:
%       MaxEpochs        - Maximum training epochs (default: 100)
%       MiniBatchSize    - Mini-batch size (default: 64)
%       ValidationSplit  - Fraction for validation (default: 0.15)
%       LearningRate     - Initial learning rate (default: 0.001)
%       HiddenUnits      - Hidden layer sizes (default: [128, 64, 32])
%       Dropout          - Dropout rates (default: [0.3, 0.2, 0])
%       Patience         - Early stopping patience (default: 10)
%       OutputPath       - Path to save trained network (default: '')
%       Verbose          - Show training progress (default: true)
%       ShowPlots        - Show training plots (default: false)
%
%   Outputs:
%       valueNet - Trained dlnetwork object
%       info     - Training information struct
%
%   Example:
%       data = extract_training_data();
%       [valueNet, info] = train_value_network(data, MaxEpochs=50);
%       fprintf('Final validation MSE: %.4f\n', info.finalValLoss);

    arguments
        data struct
        options.MaxEpochs (1,1) double = 100
        options.MiniBatchSize (1,1) double = 64
        options.ValidationSplit (1,1) double = 0.15
        options.LearningRate (1,1) double = 0.001
        options.HiddenUnits (1,:) double = [128, 64, 32]
        options.Dropout (1,:) double = [0.3, 0.2, 0]
        options.Patience (1,1) double = 10
        options.OutputPath string = ""
        options.Verbose logical = true
        options.ShowPlots logical = false
    end

    log_print(options.Verbose, '\n=== Training Value Network ===\n\n');

    %% Validate input data
    if ~isfield(data, 'states') || ~isfield(data, 'outcomes')
        error('Data must have fields: states, outcomes');
    end

    X = data.states;  % [50 x N]
    y = data.outcomes(:)';  % [1 x N]

    numSamples = size(X, 2);
    numFeatures = size(X, 1);

    log_print(options.Verbose, 'Input data:\n');
    log_print(options.Verbose, '  Samples:  %d\n', numSamples);
    log_print(options.Verbose, '  Features: %d\n', numFeatures);
    log_print(options.Verbose, '  Outcome range: [%.3f, %.3f]\n', min(y), max(y));

    %% Split into train/validation sets
    numVal = floor(numSamples * options.ValidationSplit);
    numTrain = numSamples - numVal;

    idx = randperm(numSamples);
    trainIdx = idx(1:numTrain);
    valIdx = idx(numTrain+1:end);

    X_train = X(:, trainIdx);
    y_train = y(trainIdx);
    X_val = X(:, valIdx);
    y_val = y(valIdx);

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
        fullyConnectedLayer(1, 'Name', 'fc_out')
        tanhLayer('Name', 'tanh_out')  % Output in [-1, +1]
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
    log_print(options.Verbose, '  Output: 1 (tanh -> [-1, +1])\n');

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

    % Use trainnet (recommended over trainNetwork)
    [valueNet, trainInfo] = trainnet(X_train', y_train', layers, 'mse', trainOpts);

    trainTime = toc;

    %% Evaluate on validation set
    y_pred_val = predict(valueNet, X_val');
    y_pred_val = extractdata(y_pred_val);
    valMSE = mean((y_pred_val' - y_val).^2);
    valCorr = corr(y_pred_val, y_val');

    % Evaluate on training set
    y_pred_train = predict(valueNet, X_train');
    y_pred_train = extractdata(y_pred_train);
    trainMSE = mean((y_pred_train' - y_train).^2);

    %% Build output info struct
    info = struct();
    info.trainTime = trainTime;
    % R2023b+ uses TrainingHistory table instead of TrainingLoss
    if isprop(trainInfo, 'TrainingHistory')
        info.numEpochs = max(trainInfo.TrainingHistory.Epoch);
    else
        info.numEpochs = trainInfo.NumEpochs;
    end
    info.finalTrainLoss = trainMSE;
    info.finalValLoss = valMSE;
    info.valCorrelation = valCorr;
    info.trainingHistory = trainInfo;
    info.numTrainSamples = numTrain;
    info.numValSamples = numVal;
    info.architecture = options.HiddenUnits;
    info.dropout = options.Dropout;

    %% Summary
    log_print(options.Verbose, '\n=== Training Complete ===\n');
    log_print(options.Verbose, '  Training time:    %.1f seconds\n', trainTime);
    log_print(options.Verbose, '  Epochs:           %d\n', info.numEpochs);
    log_print(options.Verbose, '  Train MSE:        %.4f\n', trainMSE);
    log_print(options.Verbose, '  Validation MSE:   %.4f\n', valMSE);
    log_print(options.Verbose, '  Val correlation:  %.4f\n', valCorr);

    %% Save if requested
    if options.OutputPath ~= ""
        save(options.OutputPath, 'valueNet', 'info');
        log_print(options.Verbose, '\nNetwork saved to: %s\n', options.OutputPath);
    end

    log_print(options.Verbose, '\n');
end

function log_print(verbose, varargin)
    if verbose
        fprintf(varargin{:});
    end
end
