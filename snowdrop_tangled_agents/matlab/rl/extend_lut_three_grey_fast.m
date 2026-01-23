function extend_lut_three_grey_fast()
%EXTEND_LUT_THREE_GREY_FAST Optimized 3-grey LUT generation
%
%   Uses vectorized operations and numeric indexing for speed.
%   Avoids string operations inside loops.
%
%   Target: ~15 million states in under 5 minutes.

    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║   FAST 3-GREY LUT EXTENSION (Vectorized)                  ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

    scriptDir = fileparts(mfilename('fullpath'));

    %% Load existing LUT
    fprintf('[1/3] Loading existing LUT...\n');

    lutPath = fullfile(scriptDir, 'data', 'expanded_lut.mat');
    data = load(lutPath);
    terminalLUT = double(data.terminalLUT(:));

    fprintf('    Loaded %d terminal scores\n', length(terminalLUT));

    %% Pre-compute bit masks for fast index computation
    fprintf('\n[2/3] Generating 3-grey states...\n');

    greyTriples = nchoosek(1:15, 3);  % 455 triples
    numTriples = size(greyTriples, 1);
    numThreeGrey = 32768 * numTriples;

    fprintf('    Target: %d states\n', numThreeGrey);

    tic;

    % Pre-compute powers of 2 for indexing
    pow2 = 2.^(0:14);

    % Allocate output
    threeGreyScores = zeros(numThreeGrey, 1, 'single');

    % Process in chunks for memory efficiency
    chunkSize = 4096;
    numChunks = ceil(32768 / chunkSize);

    for chunk = 1:numChunks
        startBase = (chunk - 1) * chunkSize + 1;
        endBase = min(chunk * chunkSize, 32768);
        chunkBases = startBase:endBase;
        numInChunk = length(chunkBases);

        % Convert base indices to binary representation (numInChunk x 15)
        baseStates = zeros(numInChunk, 15);
        for j = 1:15
            baseStates(:, j) = bitand(chunkBases' - 1, pow2(j)) > 0;
        end

        % Process each triple
        for tripleIdx = 1:numTriples
            p1 = greyTriples(tripleIdx, 1);
            p2 = greyTriples(tripleIdx, 2);
            p3 = greyTriples(tripleIdx, 3);

            % For each base state in chunk, compute depth-3 minimax
            scores = zeros(numInChunk, 1);

            for i = 1:numInChunk
                baseState = baseStates(i, :);

                % Depth-3 minimax using numeric states
                bestScore = -Inf;

                % Our move 1: position p1, p2, or p3 with color 0 or 1
                for ourPos1 = [p1, p2, p3]
                    for ourColor1 = [0, 1]
                        state1 = baseState;
                        state1(ourPos1) = ourColor1;

                        % Remaining positions
                        rem = setdiff([p1, p2, p3], ourPos1);

                        % Opponent minimizes
                        worstForUs = Inf;

                        for oppPos = rem
                            for oppColor = [0, 1]
                                state2 = state1;
                                state2(oppPos) = oppColor;

                                % Final position for our second move
                                finalPos = setdiff(rem, oppPos);

                                % We maximize final move
                                state3a = state2;
                                state3a(finalPos) = 0;
                                idx3a = 1 + sum(state3a .* pow2);

                                state3b = state2;
                                state3b(finalPos) = 1;
                                idx3b = 1 + sum(state3b .* pow2);

                                bestFinal = max(terminalLUT(idx3a), terminalLUT(idx3b));
                                worstForUs = min(worstForUs, bestFinal);
                            end
                        end

                        bestScore = max(bestScore, worstForUs);
                    end
                end

                scores(i) = bestScore;
            end

            % Store results
            linearStart = (startBase - 1) * numTriples + tripleIdx;
            linearEnd = (endBase - 1) * numTriples + tripleIdx;
            threeGreyScores(linearStart:numTriples:linearEnd) = scores;
        end

        % Progress
        elapsed = toc;
        pct = endBase / 32768 * 100;
        rate = endBase / elapsed;
        eta = (32768 - endBase) / rate;
        fprintf('    Progress: %d/%d (%.1f%%) - %.1f states/sec - ETA: %.0fs\n', ...
            endBase, 32768, pct, rate * numTriples, eta);
    end

    elapsed = toc;
    fprintf('    Completed %d states in %.1f seconds\n', numThreeGrey, elapsed);
    fprintf('    Rate: %.0f states/sec\n', numThreeGrey / elapsed);

    %% Save
    fprintf('\n[3/3] Saving extended LUT...\n');

    % Load all existing data
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

    save(lutPath, 'terminalLUT', 'oneGreyScores', 'twoGreyScores', ...
         'threeGreyScores', 'greyPairs', 'greyTriples', 'metadata', '-v7.3');

    fileInfo = dir(lutPath);
    fprintf('    File size: %.2f MB\n', fileInfo.bytes / 1024 / 1024);

    %% Summary
    fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║  TOTAL ENTRIES: %10d                                 ║\n', metadata.totalCount);
    fprintf('║  Three-grey:    %10d (NEW)                          ║\n', length(threeGreyScores));
    fprintf('║  Generation time: %.1f seconds                             ║\n', elapsed);
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
end
