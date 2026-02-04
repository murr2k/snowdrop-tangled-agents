function benchmark_mcts_unlimited()
%BENCHMARK_MCTS_UNLIMITED Benchmark with unlimited time budget
%
%   Tests MCTS with very high time limits so iterations control search depth.
%   Designed for competitive play where move time is unlimited.

    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║     MCTS BENCHMARK - UNLIMITED TIME BUDGET                     ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    %% System Info
    fprintf('System Configuration:\n');
    maxWorkers = feature('numcores');
    fprintf('  CPU Cores: %d\n', maxWorkers);

    memInfo = memory;
    fprintf('  RAM Available: %.1f GB\n', memInfo.MemAvailableAllArrays / 1024^3);

    try
        g = gpuDevice(1);
        fprintf('  GPU: %s (%.1f GB)\n', g.Name, g.TotalMemory / 1024^3);
    catch
        fprintf('  GPU: Not available\n');
    end

    fprintf('\n');

    %% Configuration
    fprintf('Test Configuration:\n');

    % Use all available cores for parpool (user confirmed 8 available)
    optimalWorkers = maxWorkers;
    fprintf('  Workers: %d (max available)\n', optimalWorkers);

    % Iteration counts
    iterCounts = [1000, 5000, 10000, 50000, 100000, 500000, 1000000];
    fprintf('  Iterations: [%s]\n', ...
        strjoin(arrayfun(@(x) formatNum(x), iterCounts, 'Uniform', false), ', '));

    % UNLIMITED time limit
    timeLimit = 3600;  % 1 hour max (effectively unlimited)
    fprintf('  Time Limit: %d seconds (effectively unlimited)\n', timeLimit);

    testState = 'GPGPGP---------';
    fprintf('  Position: Mid-game\n\n');

    %% Initialize Pool
    fprintf('Initializing parallel pool (%d workers)...\n', optimalWorkers);
    delete(gcp('nocreate'));
    parpool(optimalWorkers);
    fprintf('  Ready\n\n');

    %% Benchmarks
    fprintf('Running benchmarks (this will take time)...\n');
    fprintf('─────────────────────────────────────────────────────────────────\n\n');

    results = table();

    for i = 1:length(iterCounts)
        numIters = iterCounts(i);

        fprintf('[%d/%d] Testing %s iterations...\n', i, length(iterCounts), formatNum(numIters));

        % Create MCTS with unlimited time
        mcts = TangledMCTS('Iterations', numIters, ...
                          'NumWorkers', optimalWorkers, ...
                          'TimeLimit', timeLimit, ...
                          'UseParallel', true);

        % Warm-up
        if i == 1
            fprintf('    Warm-up... ');
            [~, ~] = mcts.search(testState);
            fprintf('done\n');
        end

        % Timed run
        tic;
        [edge, color, info] = mcts.search(testState);
        elapsed = toc;

        % Verify we completed all iterations
        completionPct = info.iterations / numIters * 100;

        % Record
        row = table(numIters, optimalWorkers, elapsed, info.iterations, ...
                   completionPct, info.iterationsPerSecond, ...
                   'VariableNames', {'TargetIters', 'Workers', 'TimeSeconds', ...
                                     'ActualIters', 'CompletionPct', 'ItersPerSec'});
        results = [results; row];

        fprintf('    Time: %.1f seconds\n', elapsed);
        fprintf('    Actual iterations: %d (%.0f%% of target)\n', info.iterations, completionPct);
        fprintf('    Rate: %.0f iters/sec\n', info.iterationsPerSecond);
        fprintf('    Move: E%d %s\n\n', edge-1, color);

        % Warn if time limit hit
        if completionPct < 95
            fprintf('    ⚠️  WARNING: Hit time limit before completing iterations!\n\n');
        end
    end

    %% Results Table
    fprintf('═══════════════════════════════════════════════════════════════\n');
    fprintf('║                        RESULTS                                ║\n');
    fprintf('═══════════════════════════════════════════════════════════════\n\n');

    fprintf('┌──────────────┬──────────┬────────────┬─────────────┬──────────────┬────────────┐\n');
    fprintf('│  Iterations  │ Workers  │ Time/Move  │  iters/sec  │  Game Time   │ Completed  │\n');
    fprintf('├──────────────┼──────────┼────────────┼─────────────┼──────────────┼────────────┤\n');

    for i = 1:height(results)
        row = results(i, :);
        gameTime = row.TimeSeconds * 15 / 60;  % 15 moves, convert to minutes

        fprintf('│ %9s    │    %2d    │ %7.1fs   │  %9.0f  │  %7.1f min  │   %5.1f%%   │\n', ...
            formatNum(row.TargetIters), row.Workers, row.TimeSeconds, ...
            row.ItersPerSec, gameTime, row.CompletionPct);
    end

    fprintf('└──────────────┴──────────┴────────────┴─────────────┴──────────────┴────────────┘\n\n');

    %% Analysis
    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║                         ANALYSIS                               ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    % Find rows that completed >95% iterations
    completed = results(results.CompletionPct >= 95, :);

    if height(completed) > 0
        fprintf('COMPLETED CONFIGURATIONS:\n');
        fprintf('───────────────────────────────────────────────────────────────\n\n');

        for i = 1:height(completed)
            row = completed(i, :);
            fprintf('  %s iterations:\n', formatNum(row.TargetIters));
            fprintf('    Time/move: %.1f seconds\n', row.TimeSeconds);
            fprintf('    Game time: %.1f minutes (%.1f hours)\n', ...
                row.TimeSeconds * 15 / 60, row.TimeSeconds * 15 / 3600);
            fprintf('    Quality: %s\n', getQuality(row.TargetIters));
            fprintf('\n');
        end
    end

    % Recommend configurations for different time budgets
    fprintf('RECOMMENDATIONS BY TIME BUDGET:\n');
    fprintf('───────────────────────────────────────────────────────────────\n\n');

    % <5 min per game
    mask = results.TimeSeconds * 15 / 60 <= 5;
    if any(mask)
        best = results(find(mask, 1, 'last'), :);
        fprintf('  ✦ Quick Games (<5 min/game):\n');
        fprintf('      %s iterations, %.1fs/move\n', formatNum(best.TargetIters), best.TimeSeconds);
        fprintf('      Quality: %s\n\n', getQuality(best.TargetIters));
    end

    % 5-30 min per game
    mask = results.TimeSeconds * 15 / 60 > 5 & results.TimeSeconds * 15 / 60 <= 30;
    if any(mask)
        best = results(find(mask, 1, 'last'), :);
        fprintf('  ✦ Moderate Games (5-30 min/game):\n');
        fprintf('      %s iterations, %.1fs/move\n', formatNum(best.TargetIters), best.TimeSeconds);
        fprintf('      Quality: %s\n\n', getQuality(best.TargetIters));
    end

    % 30-120 min per game
    mask = results.TimeSeconds * 15 / 60 > 30 & results.TimeSeconds * 15 / 60 <= 120;
    if any(mask)
        best = results(find(mask, 1, 'last'), :);
        fprintf('  ✦ Deep Games (30 min - 2 hr/game):\n');
        fprintf('      %s iterations, %.1fs/move\n', formatNum(best.TargetIters), best.TimeSeconds);
        fprintf('      Quality: %s\n\n', getQuality(best.TargetIters));
    end

    % >120 min per game
    mask = results.TimeSeconds * 15 / 60 > 120 & results.CompletionPct >= 95;
    if any(mask)
        best = results(find(mask, 1, 'last'), :);
        fprintf('  ✦ Maximum Quality (>2 hr/game):\n');
        fprintf('      %s iterations, %.1fs/move\n', formatNum(best.TargetIters), best.TimeSeconds);
        fprintf('      Quality: %s\n\n', getQuality(best.TargetIters));
    end

    %% Competitive Recommendations
    fprintf('FOR COMPETITIVE PLAY VS MCTS MELISSA:\n');
    fprintf('───────────────────────────────────────────────────────────────\n\n');

    % Find highest completed config
    completed = results(results.CompletionPct >= 95, :);
    if height(completed) > 0
        best = completed(end, :);

        fprintf('  Recommended: %s iterations\n', formatNum(best.TargetIters));
        fprintf('  Time/move: %.1f seconds (%.1f minutes per game)\n', ...
            best.TimeSeconds, best.TimeSeconds * 15 / 60);
        fprintf('  Expected quality: %s\n\n', getQuality(best.TargetIters));

        fprintf('  Configuration:\n');
        fprintf('    mcts = TangledMCTS(''Iterations'', %d, ...\n', best.TargetIters);
        fprintf('                       ''NumWorkers'', %d, ...\n', best.Workers);
        fprintf('                       ''TimeLimit'', inf);  %% Unlimited\n\n');

        fprintf('  For play_tangled.py:\n');
        fprintf('    python play_tangled.py --strategy hybrid_solver \\\n');
        fprintf('                           --mcts-iterations %d \\\n', best.TargetIters);
        fprintf('                           --games 20\n\n');
    end

    % Estimate vs Melissa
    fprintf('  Estimated win rate vs Melissa:\n');
    for i = 1:height(results)
        if results.CompletionPct(i) >= 95
            winRate = estimateWinRate(results.TargetIters(i));
            fprintf('    %9s iters: %s\n', formatNum(results.TargetIters(i)), winRate);
        end
    end

    fprintf('\n');

    %% Save
    saveFile = fullfile(fileparts(mfilename('fullpath')), 'data', 'mcts_unlimited_benchmark.mat');
    save(saveFile, 'results');
    fprintf('Results saved to: %s\n\n', saveFile);

    fprintf('═══════════════════════════════════════════════════════════════\n');
    fprintf('Benchmark complete!\n');
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

function quality = getQuality(iters)
    if iters < 5000
        quality = 'Weak';
    elseif iters < 20000
        quality = 'Good';
    elseif iters < 100000
        quality = 'Strong';
    elseif iters < 200000
        quality = 'Very Strong';
    elseif iters < 500000
        quality = 'Excellent';
    elseif iters < 1000000
        quality = 'Superb';
    else
        quality = 'Near-Perfect';
    end
end

function winRate = estimateWinRate(iters)
    if iters < 10000
        winRate = '<15% (Poor)';
    elseif iters < 50000
        winRate = '20-30% (Fair)';
    elseif iters < 100000
        winRate = '30-40% (Competitive)';
    elseif iters < 200000
        winRate = '40-50% (Strong)';
    elseif iters < 500000
        winRate = '50-60% (Very Strong)';
    else
        winRate = '55-65% (Elite)';
    end
end
