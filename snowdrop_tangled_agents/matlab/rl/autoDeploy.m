function deployed = autoDeploy(agent, metrics, registry, options)
%AUTODEPLOY Automatically deploy model if performance improves
%
%   deployed = autoDeploy(agent, metrics, registry)
%   deployed = autoDeploy(agent, metrics, registry, Name=Value)
%
%   Evaluates the new agent against the current deployed version and
%   automatically deploys if performance improves.
%
%   Inputs:
%       agent    - Trained PPO agent
%       metrics  - Training metrics struct (.episodes, .avgReward, .winRate)
%       registry - ModelRegistry instance
%
%   Name-Value Arguments:
%       MinImprovement - Minimum win rate improvement to deploy (default: 0.02)
%       MinWinRate     - Minimum absolute win rate to deploy (default: 0.45)
%       MinEpisodes    - Minimum training episodes required (default: 100)
%       Notes          - Notes for this deployment (default: 'Auto-deployed')
%       Verbose        - Print status messages (default: true)
%
%   Outputs:
%       deployed - True if model was deployed, false otherwise
%
%   Example:
%       registry = ModelRegistry('models.db', 'models');
%       [trainedAgent, stats] = trainParallel(agent, 'MaxEpisodes', 1000);
%       metrics = struct('episodes', 1000, 'winRate', 0.65, 'avgReward', 0.3);
%       deployed = autoDeploy(trainedAgent, metrics, registry);

    arguments
        agent
        metrics struct
        registry ModelRegistry
        options.MinImprovement (1,1) double = 0.02
        options.MinWinRate (1,1) double = 0.45
        options.MinEpisodes (1,1) double = 100
        options.Notes char = 'Auto-deployed'
        options.Verbose logical = true
    end

    deployed = false;

    %% Extract metrics
    newWinRate = getFieldOr(metrics, 'winRate', 0);
    newAvgReward = getFieldOr(metrics, 'avgReward', 0);
    newEpisodes = getFieldOr(metrics, 'episodes', 0);

    if options.Verbose
        fprintf('\n=== Auto-Deploy Evaluation ===\n');
        fprintf('New model:\n');
        fprintf('  Win Rate:   %.1f%%\n', newWinRate * 100);
        fprintf('  Avg Reward: %.3f\n', newAvgReward);
        fprintf('  Episodes:   %d\n', newEpisodes);
    end

    %% Check minimum requirements
    if newEpisodes < options.MinEpisodes
        if options.Verbose
            fprintf('SKIP: Not enough episodes (%d < %d)\n', ...
                newEpisodes, options.MinEpisodes);
        end
        return;
    end

    if newWinRate < options.MinWinRate
        if options.Verbose
            fprintf('SKIP: Win rate too low (%.1f%% < %.1f%%)\n', ...
                newWinRate * 100, options.MinWinRate * 100);
        end
        return;
    end

    %% Compare with currently deployed model
    currentInfo = registry.getDeployedInfo();

    if isempty(currentInfo)
        % No model deployed yet - deploy this one
        if options.Verbose
            fprintf('No model currently deployed. Deploying new model.\n');
        end
        shouldDeploy = true;
        currentWinRate = 0;
    else
        currentWinRate = currentInfo.winRate;

        if options.Verbose
            fprintf('Current deployed: %s\n', currentInfo.version);
            fprintf('  Win Rate: %.1f%%\n', currentWinRate * 100);
        end

        % Check if improvement is sufficient
        improvement = newWinRate - currentWinRate;

        if improvement >= options.MinImprovement
            shouldDeploy = true;
            if options.Verbose
                fprintf('Improvement: +%.1f%% (>= %.1f%% threshold)\n', ...
                    improvement * 100, options.MinImprovement * 100);
            end
        else
            shouldDeploy = false;
            if options.Verbose
                fprintf('SKIP: Insufficient improvement (+%.1f%% < %.1f%%)\n', ...
                    improvement * 100, options.MinImprovement * 100);
            end
        end
    end

    %% Deploy if criteria met
    if shouldDeploy
        % Register model
        notes = sprintf('%s (%.1f%% -> %.1f%%)', ...
            options.Notes, currentWinRate * 100, newWinRate * 100);
        version = registry.registerModel(agent, metrics, notes);

        % Deploy
        registry.deployModel(version);

        deployed = true;

        if options.Verbose
            fprintf('SUCCESS: Deployed %s\n', version);
            fprintf('==============================\n\n');
        end
    else
        if options.Verbose
            fprintf('==============================\n\n');
        end
    end
end

function val = getFieldOr(s, field, default)
%GETFIELDOR Get struct field with default value
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end
