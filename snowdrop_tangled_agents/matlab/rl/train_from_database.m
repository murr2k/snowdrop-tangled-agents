function results = train_from_database(options)
%TRAIN_FROM_DATABASE Complete pipeline: database -> pre-training -> fine-tuning
%
%   results = train_from_database()
%   results = train_from_database(Name=Value)
%
%   Orchestrates the complete supervised pre-training pipeline:
%   1. Extract training data from SQLite database
%   2. Train value network (outcome prediction)
%   3. Train policy network (action imitation)
%   4. Initialize PPO agent with pre-trained weights
%   5. Fine-tune with self-play (optional)
%
%   Name-Value Arguments:
%       DBPath              - Path to SQLite database (default: ~/.tangled/game_stats.db)
%       OutputDir           - Directory for outputs (default: temp dir)
%       ValueEpochs         - Epochs for value network (default: 100)
%       PolicyEpochs        - Epochs for policy network (default: 100)
%       FineTuneEpisodes    - Self-play episodes for fine-tuning (default: 0, skip)
%       MinGames            - Minimum games required (default: 20)
%       Verbose             - Print progress (default: true)
%       ShowPlots           - Show training plots (default: false)
%
%   Outputs:
%       results - Struct with fields:
%           .agent        - Final PPO agent
%           .valueNet     - Trained value network
%           .policyNet    - Trained policy network
%           .valueInfo    - Value network training info
%           .policyInfo   - Policy network training info
%           .data         - Extracted training data
%           .outputDir    - Path to output directory
%           .success      - Boolean success flag
%
%   Example:
%       % Train from database with fine-tuning
%       results = train_from_database('FineTuneEpisodes', 100);
%
%       % Check results
%       if results.success
%           fprintf('Training complete! Final win rate: %.1f%%\n', ...
%               results.finetuneStats.winRates(end) * 100);
%       end

    arguments
        options.DBPath string = ""
        options.OutputDir string = ""
        options.ValueEpochs (1,1) double = 100
        options.PolicyEpochs (1,1) double = 100
        options.FineTuneEpisodes (1,1) double = 0
        options.MinGames (1,1) double = 20
        options.Verbose logical = true
        options.ShowPlots logical = false
    end

    %% Setup
    if options.DBPath == ""
        options.DBPath = fullfile(getenv('USERPROFILE'), '.tangled', 'game_stats.db');
    end

    if options.OutputDir == ""
        timestamp = datestr(now, 'yyyymmdd_HHMMSS');
        options.OutputDir = fullfile(tempdir, ['tangled_pretrain_' timestamp]);
    end

    if ~exist(options.OutputDir, 'dir')
        mkdir(options.OutputDir);
    end

    results = struct();
    results.outputDir = options.OutputDir;
    results.success = false;
    results.error = '';

    log_print(options.Verbose, '\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  TANGLED RL - SUPERVISED PRE-TRAINING PIPELINE\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Database:   %s\n', options.DBPath);
    log_print(options.Verbose, '  Output:     %s\n', options.OutputDir);
    log_print(options.Verbose, '  Value epochs:  %d\n', options.ValueEpochs);
    log_print(options.Verbose, '  Policy epochs: %d\n', options.PolicyEpochs);
    log_print(options.Verbose, '  Fine-tune episodes: %d\n', options.FineTuneEpisodes);
    log_print(options.Verbose, '================================================================\n\n');

    %% Phase 1: Extract Training Data
    log_print(options.Verbose, '=== Phase 1: Extract Training Data ===\n');
    try
        dataPath = fullfile(options.OutputDir, 'training_data.mat');
        data = extract_training_data(options.DBPath, ...
            'OutputPath', dataPath, ...
            'Verbose', options.Verbose);

        results.data = data;

        % Check minimum requirements
        if data.metadata.numGames < options.MinGames
            error('Insufficient data: %d games (minimum: %d)', ...
                data.metadata.numGames, options.MinGames);
        end

        if data.metadata.numOurMoves < 50
            warning('Low sample count: %d our moves. Training may be unstable.', ...
                data.metadata.numOurMoves);
        end

        log_print(options.Verbose, 'Phase 1 PASSED\n\n');
    catch ME
        results.error = sprintf('Phase 1 failed: %s', ME.message);
        log_print(options.Verbose, 'Phase 1 FAILED: %s\n\n', ME.message);
        return;
    end

    %% Phase 2: Train Value Network
    log_print(options.Verbose, '=== Phase 2: Train Value Network ===\n');
    try
        valueNetPath = fullfile(options.OutputDir, 'value_network.mat');
        [valueNet, valueInfo] = train_value_network(data, ...
            'MaxEpochs', options.ValueEpochs, ...
            'OutputPath', valueNetPath, ...
            'Verbose', options.Verbose, ...
            'ShowPlots', options.ShowPlots);

        results.valueNet = valueNet;
        results.valueInfo = valueInfo;

        % Check quality
        if valueInfo.finalValLoss > 0.5
            warning('Value network validation MSE is high (%.3f). Model may underfit.', ...
                valueInfo.finalValLoss);
        end

        log_print(options.Verbose, 'Phase 2 PASSED\n\n');
    catch ME
        results.error = sprintf('Phase 2 failed: %s', ME.message);
        log_print(options.Verbose, 'Phase 2 FAILED: %s\n\n', ME.message);
        return;
    end

    %% Phase 3: Train Policy Network
    log_print(options.Verbose, '=== Phase 3: Train Policy Network ===\n');
    try
        policyNetPath = fullfile(options.OutputDir, 'policy_network.mat');
        [policyNet, policyInfo] = train_policy_network(data, ...
            'MaxEpochs', options.PolicyEpochs, ...
            'OutputPath', policyNetPath, ...
            'Verbose', options.Verbose, ...
            'ShowPlots', options.ShowPlots);

        results.policyNet = policyNet;
        results.policyInfo = policyInfo;

        % Check quality
        if policyInfo.top5Accuracy < 0.3
            warning('Policy network top-5 accuracy is low (%.1f%%). Model may underfit.', ...
                policyInfo.top5Accuracy * 100);
        end

        log_print(options.Verbose, 'Phase 3 PASSED\n\n');
    catch ME
        results.error = sprintf('Phase 3 failed: %s', ME.message);
        log_print(options.Verbose, 'Phase 3 FAILED: %s\n\n', ME.message);
        return;
    end

    %% Phase 4: Initialize PPO Agent
    log_print(options.Verbose, '=== Phase 4: Initialize PPO Agent ===\n');
    try
        env = TangledEnvironment();
        agent = initialize_ppo_from_pretrained(env, valueNet, policyNet, ...
            'Verbose', options.Verbose);

        results.agent = agent;

        % Save initial agent
        initialAgentPath = fullfile(options.OutputDir, 'agent_pretrained.mat');
        save(initialAgentPath, 'agent');
        log_print(options.Verbose, 'Pre-trained agent saved: %s\n', initialAgentPath);

        log_print(options.Verbose, 'Phase 4 PASSED\n\n');
    catch ME
        results.error = sprintf('Phase 4 failed: %s', ME.message);
        log_print(options.Verbose, 'Phase 4 FAILED: %s\n\n', ME.message);
        return;
    end

    %% Phase 5: Fine-Tune with Self-Play (optional)
    if options.FineTuneEpisodes > 0
        log_print(options.Verbose, '=== Phase 5: Fine-Tune with Self-Play ===\n');
        try
            dbPath = fullfile(options.OutputDir, 'finetune_buffer.db');
            [trainedAgent, finetuneStats] = trainParallel(agent, ...
                'MaxEpisodes', options.FineTuneEpisodes, ...
                'NumWorkers', 1, ...
                'DBPath', dbPath, ...
                'SavePath', options.OutputDir, ...
                'Verbose', options.Verbose);

            results.agent = trainedAgent;
            results.finetuneStats = finetuneStats;

            % Save fine-tuned agent
            fineTunedAgentPath = fullfile(options.OutputDir, 'agent_finetuned.mat');
            save(fineTunedAgentPath, 'trainedAgent', 'finetuneStats');
            log_print(options.Verbose, 'Fine-tuned agent saved: %s\n', fineTunedAgentPath);

            log_print(options.Verbose, 'Phase 5 PASSED\n\n');
        catch ME
            results.error = sprintf('Phase 5 failed: %s', ME.message);
            log_print(options.Verbose, 'Phase 5 FAILED: %s\n\n', ME.message);
            % Don't return - partial success is still useful
        end
    else
        log_print(options.Verbose, '=== Phase 5: Fine-Tune (SKIPPED) ===\n');
        log_print(options.Verbose, '  Set FineTuneEpisodes > 0 to enable self-play fine-tuning.\n\n');
    end

    %% Summary
    results.success = true;

    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  TRAINING COMPLETE\n');
    log_print(options.Verbose, '================================================================\n');
    log_print(options.Verbose, '  Games used:       %d\n', data.metadata.numGames);
    log_print(options.Verbose, '  Training samples: %d\n', data.metadata.numSamples);
    log_print(options.Verbose, '  Value MSE:        %.4f\n', valueInfo.finalValLoss);
    log_print(options.Verbose, '  Policy top-5 acc: %.1f%%\n', policyInfo.top5Accuracy * 100);
    if isfield(results, 'finetuneStats')
        log_print(options.Verbose, '  Final win rate:   %.1f%%\n', results.finetuneStats.winRates(end) * 100);
    end
    log_print(options.Verbose, '================================================================\n\n');

    log_print(options.Verbose, 'Artifacts saved to: %s\n', options.OutputDir);
    log_print(options.Verbose, '  - training_data.mat\n');
    log_print(options.Verbose, '  - value_network.mat\n');
    log_print(options.Verbose, '  - policy_network.mat\n');
    log_print(options.Verbose, '  - agent_pretrained.mat\n');
    if options.FineTuneEpisodes > 0
        log_print(options.Verbose, '  - agent_finetuned.mat\n');
    end
    log_print(options.Verbose, '\n');

    %% Cleanup
    if exist('env', 'var')
        delete(env);
    end

    %% Save summary
    summaryPath = fullfile(options.OutputDir, 'training_summary.json');
    summary = struct();
    summary.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    summary.success = results.success;
    summary.numGames = data.metadata.numGames;
    summary.numSamples = data.metadata.numSamples;
    summary.valueMSE = valueInfo.finalValLoss;
    summary.valueCorr = valueInfo.valCorrelation;
    summary.policyTop1 = policyInfo.top1Accuracy;
    summary.policyTop5 = policyInfo.top5Accuracy;
    if isfield(results, 'finetuneStats')
        summary.finalWinRate = results.finetuneStats.winRates(end);
    end

    jsonStr = jsonencode(summary, 'PrettyPrint', true);
    fid = fopen(summaryPath, 'w');
    fprintf(fid, '%s', jsonStr);
    fclose(fid);
end

function log_print(verbose, varargin)
    if verbose
        fprintf(varargin{:});
    end
end
