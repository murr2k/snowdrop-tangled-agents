function quick_benchmark_mcts()
%QUICK_BENCHMARK_MCTS Fast MCTS performance test
%
%   Runs a quick benchmark to estimate performance at different iteration
%   counts. Much faster than full benchmark_mcts_performance.m.
%
%   Tests: 1K, 10K, 100K, 1M iterations with optimal worker count

    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║          QUICK MCTS BENCHMARK                                  ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    %% System Info
    fprintf('System Configuration:\n');
    maxWorkers = feature('numcores');
    fprintf('  CPU Cores: %d\n', maxWorkers);

    memInfo = memory;
    fprintf('  RAM Available: %.1f GB\n', memInfo.MemAvailableAllArrays / 1024^3);

    try
        gpuCount = gpuDeviceCount;
        if gpuCount > 0
            g = gpuDevice(1);
            fprintf('  GPU: %s (%.1f GB)\n', g.Name, g.TotalMemory / 1024^3);
        else
            fprintf('  GPU: None\n');
        end
    catch
        fprintf('  GPU: Not available\n');
    end

    fprintf('\n');

    %% Quick Test Configuration
    fprintf('Test Configuration:\n');

    % Use all available cores for parpool
    optimalWorkers = maxWorkers;
    fprintf('  Workers: %d (max available)\n', optimalWorkers);

    % Iteration counts to test
    iterCounts = [1000, 10000, 100000, 1000000];
    fprintf('  Iterations: [%s]\n', ...
        strjoin(arrayfun(@(x) sprintf('%s', formatNum(x)), iterCounts, 'Uniform', false), ', '));

    % Test position (mid-game)
    testState = 'GPGPGP---------';
    fprintf('  Position: Mid-game (6 moves played)\n\n');

    %% Initialize Parallel Pool
    fprintf('Initializing parallel pool (%d workers)...\n', optimalWorkers);
    delete(gcp('nocreate'));
    parpool(optimalWorkers);
    fprintf('  Ready\n\n');

    %% Run Benchmarks
    fprintf('Running benchmarks...\n');
    fprintf('─────────────────────────────────────────────────────────────────\n\n');

    results = table();

    for i = 1:length(iterCounts)
        numIters = iterCounts(i);

        fprintf('[%d/%d] Testing %s iterations...\n', i, length(iterCounts), formatNum(numIters));

        % Create MCTS
        mcts = TangledMCTS('Iterations', numIters, ...
                          'NumWorkers', optimalWorkers, ...
                          'UseParallel', true);

        % Warm-up run
        if i == 1
            fprintf('    Warm-up... ');
            [~, ~] = mcts.search(testState);
            fprintf('done\n');
        end

        % Timed runs (3 trials, take median)
        times = zeros(3, 1);
        for trial = 1:3
            tic;
            [edge, color, info] = mcts.search(testState);
            times(trial) = toc;
        end

        medianTime = median(times);
        rate = info.iterationsPerSecond;

        % Record
        row = table(numIters, optimalWorkers, medianTime, rate, ...
                   'VariableNames', {'Iterations', 'Workers', 'TimeSeconds', 'ItersPerSec'});
        results = [results; row];

        fprintf('    Time: %.2f seconds (%.0f iters/sec)\n', medianTime, rate);
        fprintf('    Move: E%d %s\n\n', edge-1, color);
    end

    %% Display Results Table
    fprintf('═══════════════════════════════════════════════════════════════\n');
    fprintf('║                        RESULTS                                ║\n');
    fprintf('═══════════════════════════════════════════════════════════════\n\n');

    fprintf('┌──────────────┬──────────┬────────────┬─────────────┬──────────────┐\n');
    fprintf('│  Iterations  │ Workers  │ Time/Move  │  iters/sec  │  Game Time   │\n');
    fprintf('├──────────────┼──────────┼────────────┼─────────────┼──────────────┤\n');

    for i = 1:height(results)
        row = results(i, :);
        gameTime = row.TimeSeconds * 15;  % 15 moves per game

        fprintf('│ %9s    │    %2d    │ %7.2fs   │  %9.0f  │  %7.1f min  │\n', ...
            formatNum(row.Iterations), row.Workers, row.TimeSeconds, ...
            row.ItersPerSec, gameTime / 60);
    end

    fprintf('└──────────────┴──────────┴────────────┴─────────────┴──────────────┘\n\n');

    fprintf('Note: Game Time assumes 15 moves per game (typical)\n\n');

    %% Recommendations
    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║                    RECOMMENDATIONS                             ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    % Find config for ~30-60 seconds per move
    mask = results.TimeSeconds >= 30 & results.TimeSeconds <= 60;
    if any(mask)
        recommended = results(find(mask, 1, 'last'), :);
    else
        % Otherwise take highest that's under 2 minutes
        mask = results.TimeSeconds <= 120;
        if any(mask)
            recommended = results(find(mask, 1, 'last'), :);
        else
            recommended = results(end, :);
        end
    end

    fprintf('FOR UNLIMITED TIME BUDGET:\n');
    fprintf('───────────────────────────────────────────────────────────────\n\n');

    fprintf('  ✦ RECOMMENDED CONFIGURATION:\n');
    fprintf('      Iterations: %s\n', formatNum(recommended.Iterations));
    fprintf('      Workers: %d\n', recommended.Workers);
    fprintf('      Time per move: ~%.1f seconds\n', recommended.TimeSeconds);
    fprintf('      Full game: ~%.1f minutes\n\n', recommended.TimeSeconds * 15 / 60);

    fprintf('  To use in MATLAB:\n');
    fprintf('    mcts = TangledMCTS(''Iterations'', %d, ''NumWorkers'', %d);\n\n', ...
        recommended.Iterations, recommended.Workers);

    fprintf('  To use in play_tangled.py:\n');
    fprintf('    python play_tangled.py --strategy hybrid_solver \\\n');
    fprintf('                           --mcts-iterations %d \\\n', recommended.Iterations);
    fprintf('                           --games 10\n\n');

    fprintf('SCALING ANALYSIS:\n');
    fprintf('───────────────────────────────────────────────────────────────\n\n');

    % Calculate scaling efficiency
    baseTime = results.TimeSeconds(1);
    baseIters = results.Iterations(1);

    fprintf('  Iteration scaling (compared to %s iters):\n\n', formatNum(baseIters));

    for i = 2:height(results)
        iterFactor = results.Iterations(i) / baseIters;
        timeFactor = results.TimeSeconds(i) / baseTime;
        efficiency = iterFactor / timeFactor;

        fprintf('    %8s iters: %.1fx iterations in %.1fx time (%.0f%% efficiency)\n', ...
            formatNum(results.Iterations(i)), iterFactor, timeFactor, efficiency * 100);
    end

    fprintf('\n');

    %% Competitive Recommendations
    fprintf('COMPETITIVE STRATEGY:\n');
    fprintf('───────────────────────────────────────────────────────────────\n\n');

    fprintf('  Against MCTS Melissa (strong opponent):\n');
    fprintf('    - Current performance: Poor\n');
    fprintf('    - Likely Melissa iterations: 50K-100K\n');
    fprintf('    - Your advantage: Unlimited time\n\n');

    fprintf('  Recommended approach:\n');
    fprintf('    1. Start with %s iterations (your max tested)\n', formatNum(results.Iterations(end)));
    fprintf('    2. Monitor win rate over 20+ games\n');
    fprintf('    3. If still losing, increase to 2-5M iterations\n');
    fprintf('    4. Time per move will be longer, but should improve play\n\n');

    fprintf('  Higher iterations = Better quality:\n');
    fprintf('    - More accurate position evaluation\n');
    fprintf('    - Better horizon handling\n');
    fprintf('    - Fewer tactical mistakes\n');
    fprintf('    - Stronger endgame play\n\n');

    %% Save Results
    saveFile = fullfile(fileparts(mfilename('fullpath')), 'data', 'quick_mcts_benchmark.mat');
    save(saveFile, 'results');
    fprintf('Results saved to: %s\n\n', saveFile);

    fprintf('═══════════════════════════════════════════════════════════════\n');
    fprintf('Benchmark complete! Run full benchmark for detailed analysis.\n');
end

function str = formatNum(num)
    if num >= 1e6
        str = sprintf('%.1fM', num / 1e6);
    elseif num >= 1e3
        str = sprintf('%dK', round(num / 1e3));
    else
        str = sprintf('%d', num);
    end
end
