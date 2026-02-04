function results = benchmark_mcts_performance()
%BENCHMARK_MCTS_PERFORMANCE Comprehensive MCTS performance benchmarking
%
%   Measures iterations/sec and time/move across different configurations:
%   - Various iteration counts (1K - 1M)
%   - Worker counts (1, 4, 8, 12, 16)
%   - Early-game vs late-game positions
%   - With and without parallel pool
%
%   Creates performance tables for choosing optimal configuration.
%
%   Returns:
%       results - struct with all benchmark data and recommendations

    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║          MCTS PERFORMANCE BENCHMARK                            ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    %% System Information
    fprintf('[1/6] System Information\n');
    fprintf('─────────────────────────────────────────────────────────────────\n');

    % CPU info
    if ispc
        [~, cpuInfo] = system('wmic cpu get name');
        cpuLines = strsplit(cpuInfo, '\n');
        if length(cpuLines) > 1
            fprintf('CPU: %s\n', strtrim(cpuLines{2}));
        end
        [~, coreInfo] = system('wmic cpu get NumberOfCores');
        coreLines = strsplit(coreInfo, '\n');
        if length(coreLines) > 1
            fprintf('Cores: %s\n', strtrim(coreLines{2}));
        end
    else
        fprintf('CPU: (run on Windows for detection)\n');
    end

    % Memory info
    memInfo = memory;
    fprintf('RAM Available: %.1f GB\n', memInfo.MemAvailableAllArrays / 1024^3);

    % GPU info
    try
        gpuCount = gpuDeviceCount;
        if gpuCount > 0
            g = gpuDevice(1);
            fprintf('GPU: %s (%.1f GB, CC %s)\n', g.Name, ...
                g.TotalMemory / 1024^3, g.ComputeCapability);
        else
            fprintf('GPU: None detected\n');
        end
    catch
        fprintf('GPU: Not available\n');
    end

    % Parallel pool
    try
        pool = gcp('nocreate');
        if isempty(pool)
            fprintf('Parallel Pool: Not started\n');
        else
            fprintf('Parallel Pool: %d workers\n', pool.NumWorkers);
        end
    catch
        fprintf('Parallel Pool: Not available\n');
    end

    fprintf('\n');

    %% Test Positions
    fprintf('[2/6] Preparing test positions...\n');

    testPositions = {
        struct('state', '---------------', 'name', 'Empty board', 'type', 'opening'),
        struct('state', 'GPG------------', 'name', 'After 3 moves', 'type', 'early'),
        struct('state', 'GPGPGP---------', 'name', 'Mid-game', 'type', 'mid'),
        struct('state', 'GPGPGPGPG------', 'name', 'Late-game', 'type', 'late'),
        struct('state', 'GPGPGPGPGPGP---', 'name', 'Endgame', 'type', 'endgame'),
    };

    fprintf('    Testing %d positions\n\n', length(testPositions));

    %% Benchmark Configuration
    fprintf('[3/6] Benchmark configuration...\n');

    % Iteration counts to test
    iterationSets = [
        1000, 2000, 5000, 10000, 20000, 50000, ...
        100000, 200000, 500000, 1000000
    ];

    % Worker counts to test (based on available cores)
    maxWorkers = feature('numcores');
    workerCounts = [1, min(4, maxWorkers), min(8, maxWorkers), ...
                    min(12, maxWorkers), min(16, maxWorkers)];
    workerCounts = unique(workerCounts);

    fprintf('    Iteration counts: [%s]\n', ...
        strjoin(arrayfun(@(x) sprintf('%gK', x/1000), iterationSets, 'UniformOutput', false), ', '));
    fprintf('    Worker counts: [%s]\n', ...
        strjoin(arrayfun(@(x) sprintf('%d', x), workerCounts, 'UniformOutput', false), ', '));
    fprintf('    Test positions: %d\n', length(testPositions));
    fprintf('    Total tests: %d\n\n', ...
        length(iterationSets) * length(workerCounts) * length(testPositions));

    %% Initialize Results Storage
    results = struct();
    results.systemInfo = struct();
    results.systemInfo.maxWorkers = maxWorkers;
    results.systemInfo.ramGB = memInfo.MemAvailableAllArrays / 1024^3;
    results.systemInfo.hasGPU = gpuDeviceCount > 0;

    results.data = [];  % Will be table

    %% Run Benchmarks
    fprintf('[4/6] Running benchmarks...\n');
    fprintf('─────────────────────────────────────────────────────────────────\n');

    totalTests = length(iterationSets) * length(workerCounts) * length(testPositions);
    testNum = 0;
    lastPrintTime = tic;

    for iterIdx = 1:length(iterationSets)
        numIters = iterationSets(iterIdx);

        for workerIdx = 1:length(workerCounts)
            numWorkers = workerCounts(workerIdx);

            % Initialize or restart parallel pool if needed
            if numWorkers > 1
                pool = gcp('nocreate');
                if isempty(pool) || pool.NumWorkers ~= numWorkers
                    delete(gcp('nocreate'));
                    parpool(numWorkers);
                end
            else
                delete(gcp('nocreate'));
            end

            % Create MCTS instance
            mcts = TangledMCTS('Iterations', numIters, ...
                              'NumWorkers', numWorkers, ...
                              'UseParallel', numWorkers > 1);

            for posIdx = 1:length(testPositions)
                testNum = testNum + 1;
                pos = testPositions{posIdx};

                % Run search (warm-up first run)
                if testNum == 1
                    [~, ~] = mcts.search(pos.state);
                end

                % Timed run
                tic;
                [edge, color, info] = mcts.search(pos.state);
                elapsed = toc;

                % Record results
                row = struct();
                row.iterations = numIters;
                row.workers = numWorkers;
                row.position = pos.name;
                row.posType = pos.type;
                row.timeSeconds = elapsed;
                row.itersPerSec = info.iterationsPerSecond;
                row.actualIters = info.iterations;
                row.edge = edge;
                row.color = color;

                if isempty(results.data)
                    results.data = struct2table(row);
                else
                    results.data = [results.data; struct2table(row)];
                end

                % Progress (throttled)
                if toc(lastPrintTime) > 3
                    pct = testNum / totalTests * 100;
                    fprintf('    Progress: %d/%d (%.1f%%) - %s @ %gK iters, %d workers\n', ...
                        testNum, totalTests, pct, pos.name, numIters/1000, numWorkers);
                    lastPrintTime = tic;
                end
            end
        end
    end

    fprintf('    Completed %d tests\n\n', testNum);

    %% Analyze Results
    fprintf('[5/6] Analyzing results...\n');
    fprintf('─────────────────────────────────────────────────────────────────\n');

    % Find optimal worker count for each iteration level
    results.optimalWorkers = struct();

    for iterIdx = 1:length(iterationSets)
        numIters = iterationSets(iterIdx);

        % Get data for this iteration count
        mask = results.data.iterations == numIters;
        subset = results.data(mask, :);

        % Average time across positions for each worker count
        for wIdx = 1:length(workerCounts)
            w = workerCounts(wIdx);
            wMask = subset.workers == w;
            avgTime(wIdx) = mean(subset.timeSeconds(wMask));
            avgRate(wIdx) = mean(subset.itersPerSec(wMask));
        end

        [~, bestIdx] = min(avgTime);
        optWorkers = workerCounts(bestIdx);

        results.optimalWorkers.(sprintf('iters_%d', numIters)) = optWorkers;

        fprintf('    %7gK iters: Best = %2d workers (%.2fs avg, %7.0f iters/sec)\n', ...
            numIters/1000, optWorkers, avgTime(bestIdx), avgRate(bestIdx));
    end

    fprintf('\n');

    %% Generate Recommendations
    fprintf('[6/6] Generating recommendations...\n');
    fprintf('─────────────────────────────────────────────────────────────────\n');

    % Create summary table for unlimited time budget
    fprintf('\n╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║     ITERATIONS vs TIME (Unlimited Time Budget)                ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    fprintf('Configuration assumes: Best worker count for each iteration level\n');
    fprintf('Position tested: Mid-game (representative)\n\n');

    fprintf('┌──────────────┬──────────┬────────────┬─────────────┬──────────────┐\n');
    fprintf('│  Iterations  │ Workers  │ Time/Move  │  iters/sec  │  Quality     │\n');
    fprintf('├──────────────┼──────────┼────────────┼─────────────┼──────────────┤\n');

    qualityLevels = {'Baseline', 'Good', 'Strong', 'Very Strong', 'Excellent', ...
                     'Superb', 'Elite', 'Near-Perfect', 'Exhaustive', 'Overkill'};

    for iterIdx = 1:length(iterationSets)
        numIters = iterationSets(iterIdx);
        optWorkers = results.optimalWorkers.(sprintf('iters_%d', numIters));

        % Get mid-game data for this config
        mask = results.data.iterations == numIters & ...
               results.data.workers == optWorkers & ...
               strcmp(results.data.posType, 'mid');

        if any(mask)
            avgTime = mean(results.data.timeSeconds(mask));
            avgRate = mean(results.data.itersPerSec(mask));

            % Quality rating
            qualityIdx = min(iterIdx, length(qualityLevels));
            quality = qualityLevels{qualityIdx};

            fprintf('│ %9s    │    %2d    │ %7.2fs   │  %9.0f  │  %-11s │\n', ...
                formatNumber(numIters), optWorkers, avgTime, avgRate, quality);
        end
    end

    fprintf('└──────────────┴──────────┴────────────┴─────────────┴──────────────┘\n\n');

    %% Specific Recommendations
    fprintf('╔════════════════════════════════════════════════════════════════╗\n');
    fprintf('║                    RECOMMENDATIONS                             ║\n');
    fprintf('╚════════════════════════════════════════════════════════════════╝\n\n');

    % Find configurations for different time budgets
    configs = struct();

    % 1-2 second budget
    mask = results.data.timeSeconds <= 2 & strcmp(results.data.posType, 'mid');
    if any(mask)
        [maxIters, idx] = max(results.data.iterations(mask));
        configs.fast = results.data(find(mask, 1, 'first') + idx - 1, :);
    end

    % 5-10 second budget
    mask = results.data.timeSeconds >= 5 & results.data.timeSeconds <= 10 & ...
           strcmp(results.data.posType, 'mid');
    if any(mask)
        [maxIters, idx] = max(results.data.iterations(mask));
        configs.moderate = results.data(find(mask, 1, 'first') + idx - 1, :);
    end

    % 30-60 second budget
    mask = results.data.timeSeconds >= 30 & results.data.timeSeconds <= 60 & ...
           strcmp(results.data.posType, 'mid');
    if any(mask)
        [maxIters, idx] = max(results.data.iterations(mask));
        configs.deep = results.data(find(mask, 1, 'first') + idx - 1, :);
    end

    % Unlimited (>60 seconds)
    mask = results.data.timeSeconds > 60 & strcmp(results.data.posType, 'mid');
    if any(mask)
        [maxIters, idx] = max(results.data.iterations(mask));
        configs.unlimited = results.data(find(mask, 1, 'first') + idx - 1, :);
    end

    fprintf('FOR COMPETITIVE PLAY AGAINST MCTS MELISSA:\n');
    fprintf('───────────────────────────────────────────────────────────────\n\n');

    if isfield(configs, 'unlimited')
        fprintf('✦ RECOMMENDED (Unlimited Time):\n');
        fprintf('    Iterations: %s\n', formatNumber(configs.unlimited.iterations));
        fprintf('    Workers: %d\n', configs.unlimited.workers);
        fprintf('    Time per move: ~%.1f seconds\n', configs.unlimited.timeSeconds);
        fprintf('    Expected quality: Near-optimal\n\n');
    end

    if isfield(configs, 'deep')
        fprintf('✦ AGGRESSIVE (30-60s per move):\n');
        fprintf('    Iterations: %s\n', formatNumber(configs.deep.iterations));
        fprintf('    Workers: %d\n', configs.deep.workers);
        fprintf('    Time per move: ~%.1f seconds\n', configs.deep.timeSeconds);
        fprintf('    Expected quality: Very strong\n\n');
    end

    if isfield(configs, 'moderate')
        fprintf('✦ BALANCED (5-10s per move):\n');
        fprintf('    Iterations: %s\n', formatNumber(configs.moderate.iterations));
        fprintf('    Workers: %d\n', configs.moderate.workers);
        fprintf('    Time per move: ~%.1f seconds\n', configs.moderate.timeSeconds);
        fprintf('    Expected quality: Strong\n\n');
    end

    fprintf('MATLAB CONFIGURATION:\n');
    fprintf('───────────────────────────────────────────────────────────────\n\n');

    % Get best overall config (highest iterations that completed)
    [maxIters, idx] = max(results.data.iterations);
    bestRow = results.data(idx, :);

    fprintf('  To use the recommended configuration:\n\n');
    fprintf('  %% In MATLAB:\n');
    fprintf('  mcts = TangledMCTS(''Iterations'', %d, ''NumWorkers'', %d);\n\n', ...
        bestRow.iterations, bestRow.workers);

    fprintf('  %% In play_tangled.py (--mcts-iterations flag):\n');
    fprintf('  python play_tangled.py --strategy hybrid_solver \\\n');
    fprintf('                         --mcts-iterations %d \\\n', bestRow.iterations);
    fprintf('                         --games 10\n\n');

    fprintf('  %% In matlab_strategy.py (HybridSolverStrategy):\n');
    fprintf('  strategy = HybridSolverStrategy(mcts_iterations=%d)\n\n', bestRow.iterations);

    %% GPU Recommendations
    fprintf('GPU ACCELERATION:\n');
    fprintf('───────────────────────────────────────────────────────────────\n\n');

    if results.systemInfo.hasGPU
        fprintf('  ✓ GPU detected: Potential for acceleration\n');
        fprintf('  Current implementation: CPU-only\n');
        fprintf('  Opportunity: Port rollout simulation to GPU\n');
        fprintf('    - Expected speedup: 5-10x for large iteration counts\n');
        fprintf('    - Priority: High for >100K iterations\n\n');
    else
        fprintf('  ✗ No GPU detected\n');
        fprintf('  Recommendation: CPU parallelization only\n\n');
    end

    %% Save Results
    fprintf('═══════════════════════════════════════════════════════════════\n\n');

    results.configs = configs;
    results.recommendations = bestRow;

    % Save to file
    saveFile = fullfile(fileparts(mfilename('fullpath')), 'data', 'mcts_benchmark_results.mat');
    save(saveFile, 'results');
    fprintf('Results saved to: %s\n\n', saveFile);

    fprintf('Benchmark complete!\n');
end

function str = formatNumber(num)
    %FORMATNUMBER Format large numbers with K/M suffix
    if num >= 1e6
        str = sprintf('%.1fM', num / 1e6);
    elseif num >= 1e3
        str = sprintf('%.0fK', num / 1e3);
    else
        str = sprintf('%d', num);
    end
end
