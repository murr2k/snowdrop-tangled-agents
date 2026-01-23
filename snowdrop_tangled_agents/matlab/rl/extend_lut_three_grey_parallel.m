function extend_lut_three_grey_parallel()
%EXTEND_LUT_THREE_GREY_PARALLEL Parallel 3-grey LUT generation
%
%   Uses Parallel Computing Toolbox with parfor over triples.
%   Each of 455 triples is computed independently in parallel.
%
%   Expected time: 2-5 minutes with 8+ workers.

    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║   PARALLEL 3-GREY LUT EXTENSION                           ║\n');
    fprintf('║   Using Parallel Computing Toolbox                        ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

    scriptDir = fileparts(mfilename('fullpath'));

    %% Initialize parallel pool
    fprintf('[0/4] Initializing parallel pool...\n');
    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool('local');
    end
    numWorkers = pool.NumWorkers;
    fprintf('    Using %d parallel workers\n', numWorkers);

    %% Load existing LUT
    fprintf('\n[1/4] Loading existing LUT...\n');

    lutPath = fullfile(scriptDir, 'data', 'expanded_lut.mat');
    data = load(lutPath);
    terminalLUT = double(data.terminalLUT(:));

    fprintf('    Loaded %d terminal scores\n', length(terminalLUT));

    %% Setup
    fprintf('\n[2/4] Setting up computation...\n');

    greyTriples = nchoosek(1:15, 3);  % 455 triples
    numTriples = size(greyTriples, 1);
    numThreeGrey = 32768 * numTriples;  % 14,909,440

    fprintf('    Triples: %d\n', numTriples);
    fprintf('    Total states: %d\n', numThreeGrey);

    % Pre-compute powers of 2
    pow2 = 2.^(0:14);

    % Pre-compute all base state binary representations (32768 x 15)
    fprintf('    Pre-computing base states...\n');
    allBaseStates = zeros(32768, 15);
    for idx = 1:32768
        for j = 1:15
            allBaseStates(idx, j) = bitand(idx - 1, pow2(j)) > 0;
        end
    end

    %% Parallel computation over triples
    fprintf('\n[3/4] Computing 3-grey scores (parallel over %d triples)...\n', numTriples);

    tic;

    % Each triple produces 32768 scores
    % Store as cell array for parfor compatibility
    tripleScores = cell(numTriples, 1);

    parfor tripleIdx = 1:numTriples
        p1 = greyTriples(tripleIdx, 1);
        p2 = greyTriples(tripleIdx, 2);
        p3 = greyTriples(tripleIdx, 3);
        positions = [p1, p2, p3];

        % Compute scores for all 32768 base states
        scores = zeros(32768, 1, 'single');

        for baseIdx = 1:32768
            baseState = allBaseStates(baseIdx, :);

            % Depth-3 minimax
            bestScore = -Inf;

            % Our first move: 3 positions × 2 colors
            for ourPos1Idx = 1:3
                ourPos1 = positions(ourPos1Idx);
                for ourColor1 = [0, 1]
                    state1 = baseState;
                    state1(ourPos1) = ourColor1;

                    % Remaining 2 positions
                    remIdx = setdiff(1:3, ourPos1Idx);
                    rem = positions(remIdx);

                    % Opponent minimizes: 2 positions × 2 colors
                    worstForUs = Inf;

                    for oppPosIdx = 1:2
                        oppPos = rem(oppPosIdx);
                        for oppColor = [0, 1]
                            state2 = state1;
                            state2(oppPos) = oppColor;

                            % Final position
                            finalPos = rem(3 - oppPosIdx);

                            % Our second move maximizes
                            state3a = state2;
                            state3a(finalPos) = 0;
                            idx3a = 1 + state3a * pow2';

                            state3b = state2;
                            state3b(finalPos) = 1;
                            idx3b = 1 + state3b * pow2';

                            bestFinal = max(terminalLUT(idx3a), terminalLUT(idx3b));
                            worstForUs = min(worstForUs, bestFinal);
                        end
                    end

                    bestScore = max(bestScore, worstForUs);
                end
            end

            scores(baseIdx) = bestScore;
        end

        tripleScores{tripleIdx} = scores;
    end

    elapsed = toc;
    fprintf('    Completed in %.1f seconds\n', elapsed);
    fprintf('    Rate: %.0f states/sec\n', numThreeGrey / elapsed);

    %% Assemble and save
    fprintf('\n[4/4] Assembling and saving...\n');

    % Convert cell array to linear array
    % Index scheme: (baseIdx - 1) * numTriples + tripleIdx
    threeGreyScores = zeros(numThreeGrey, 1, 'single');
    for tripleIdx = 1:numTriples
        scores = tripleScores{tripleIdx};
        for baseIdx = 1:32768
            linearIdx = (baseIdx - 1) * numTriples + tripleIdx;
            threeGreyScores(linearIdx) = scores(baseIdx);
        end
    end

    fprintf('    Score range: [%.3f, %.3f]\n', min(threeGreyScores), max(threeGreyScores));

    % Load existing data
    oneGreyScores = data.oneGreyScores;
    twoGreyScores = data.twoGreyScores;
    greyPairs = data.greyPairs;
    metadata = data.metadata;

    % Update metadata
    metadata.version = '2.0';
    metadata.generated = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    metadata.threeGreyCount = length(threeGreyScores);
    metadata.totalCount = length(terminalLUT) + length(oneGreyScores) + ...
                          length(twoGreyScores) + length(threeGreyScores);
    metadata.parallelWorkers = numWorkers;

    save(lutPath, 'terminalLUT', 'oneGreyScores', 'twoGreyScores', ...
         'threeGreyScores', 'greyPairs', 'greyTriples', 'metadata', '-v7.3');

    fileInfo = dir(lutPath);
    fprintf('    File size: %.2f MB\n', fileInfo.bytes / 1024 / 1024);

    %% Summary
    fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║             3-GREY EXTENSION COMPLETE                      ║\n');
    fprintf('╠════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Terminal states:     %10d                           ║\n', length(terminalLUT));
    fprintf('║  One-grey states:     %10d                           ║\n', length(oneGreyScores));
    fprintf('║  Two-grey states:     %10d                           ║\n', length(twoGreyScores));
    fprintf('║  Three-grey states:   %10d (NEW)                     ║\n', length(threeGreyScores));
    fprintf('║  ─────────────────────────────────────────                 ║\n');
    fprintf('║  TOTAL ENTRIES:       %10d                           ║\n', metadata.totalCount);
    fprintf('╠════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Workers: %d    Time: %.1f sec                              ║\n', numWorkers, elapsed);
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
end
