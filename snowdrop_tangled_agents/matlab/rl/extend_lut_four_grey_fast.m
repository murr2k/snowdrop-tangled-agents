function extend_lut_four_grey_fast()
%EXTEND_LUT_FOUR_GREY_FAST Generate 4-grey LUT layer
%
%   Extends existing expanded_lut.mat with 4-grey minimax values.
%   Uses vectorized operations and optimized indexing.
%
%   Target: ~45 million states
%   Estimated time: 15-45 minutes (depending on hardware)
%   Memory usage: ~180MB additional
%
%   NOTE: 4-grey provides marginal benefit over 3-grey. Only generate
%   if you've confirmed 3-grey isn't sufficient for your use case.

    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║   FAST 4-GREY LUT EXTENSION (Vectorized)                  ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

    scriptDir = fileparts(mfilename('fullpath'));

    %% Load existing LUT
    fprintf('[1/4] Loading existing LUT...\n');

    lutPath = fullfile(scriptDir, 'data', 'expanded_lut.mat');
    if ~isfile(lutPath)
        error('expanded_lut.mat not found. Run generate_expanded_lut.m first.');
    end

    data = load(lutPath);
    terminalLUT = double(data.terminalLUT(:));

    % We need 3-grey data for depth-4 minimax
    if ~isfield(data, 'threeGreyScores')
        error('3-grey data not found. Run extend_lut_three_grey_fast.m first.');
    end

    threeGreyScores = double(data.threeGreyScores(:));
    greyTriples = data.greyTriples;

    fprintf('    Loaded %d terminal scores\n', length(terminalLUT));
    fprintf('    Loaded %d 3-grey scores\n', length(threeGreyScores));

    %% Build 3-grey index lookup
    fprintf('\n[2/4] Building 3-grey index map...\n');

    % Pre-compute mapping from (pos1,pos2,pos3) to triple index
    tripleIdxMap = containers.Map('KeyType', 'char', 'ValueType', 'uint32');
    for i = 1:size(greyTriples, 1)
        key = sprintf('%d_%d_%d', greyTriples(i,1), greyTriples(i,2), greyTriples(i,3));
        tripleIdxMap(key) = i;
    end

    fprintf('    Built index for %d triples\n', length(tripleIdxMap));

    %% Generate 4-grey states
    fprintf('\n[3/4] Generating 4-grey states...\n');

    greyQuads = nchoosek(1:15, 4);  % C(15,4) = 1365 quads
    numQuads = size(greyQuads, 1);
    numFourGrey = 32768 * numQuads;  % 44,748,800 states

    fprintf('    Target: %d states (%.1f million)\n', numFourGrey, numFourGrey/1e6);
    fprintf('    Quads: %d\n', numQuads);
    fprintf('    Expected time: 15-45 minutes\n\n');

    % Check available memory
    memInfo = memory;
    requiredMemory = numFourGrey * 4;  % 4 bytes per single
    availableMemory = memInfo.MemAvailableAllArrays;
    fprintf('    Memory required: %.1f MB\n', requiredMemory / 1024^2);
    fprintf('    Memory available: %.1f MB\n', availableMemory / 1024^2);

    if requiredMemory > 0.8 * availableMemory
        warning('Memory may be insufficient. Consider running on a machine with more RAM.');
    end

    tic;

    % Pre-compute powers of 2 for indexing
    pow2 = 2.^(0:14);

    % Allocate output
    fourGreyScores = zeros(numFourGrey, 1, 'single');

    % Process in chunks for memory efficiency
    chunkSize = 2048;  % Smaller chunks for 4-grey (more complex computation)
    numChunks = ceil(32768 / chunkSize);

    fprintf('    Processing in %d chunks of %d base states...\n\n', numChunks, chunkSize);

    lastPrintTime = tic;

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

        % Process each quad
        for quadIdx = 1:numQuads
            p1 = greyQuads(quadIdx, 1);
            p2 = greyQuads(quadIdx, 2);
            p3 = greyQuads(quadIdx, 3);
            p4 = greyQuads(quadIdx, 4);

            % For each base state in chunk, compute depth-4 minimax
            scores = zeros(numInChunk, 1);

            for i = 1:numInChunk
                baseState = baseStates(i, :);

                % Depth-4 minimax using 3-grey LUT as leaf evaluation
                bestScore = -Inf;

                % Our move 1: choose position and color
                for ourPos1 = [p1, p2, p3, p4]
                    for ourColor1 = [0, 1]
                        state1 = baseState;
                        state1(ourPos1) = ourColor1;

                        % Remaining positions after our move
                        rem1 = setdiff([p1, p2, p3, p4], ourPos1);

                        % Opponent minimizes
                        worstForUs = Inf;

                        for oppPos1 = rem1
                            for oppColor1 = [0, 1]
                                state2 = state1;
                                state2(oppPos1) = oppColor1;

                                % Remaining positions after opp move
                                rem2 = setdiff(rem1, oppPos1);

                                % Our move 2: maximize over remaining 2 positions
                                % This creates a 3-grey state, evaluate using 3-grey LUT
                                bestOurMove2 = -Inf;

                                for ourPos2 = rem2
                                    for ourColor2 = [0, 1]
                                        state3 = state2;
                                        state3(ourPos2) = ourColor2;

                                        % Final position (now 1-grey from terminal)
                                        finalPos = setdiff(rem2, ourPos2);

                                        % Opponent chooses final position
                                        % Evaluate both options and opponent takes min
                                        state4a = state3;
                                        state4a(finalPos) = 0;
                                        idx4a = 1 + sum(state4a .* pow2);

                                        state4b = state3;
                                        state4b(finalPos) = 1;
                                        idx4b = 1 + sum(state4b .* pow2);

                                        finalScore = min(terminalLUT(idx4a), terminalLUT(idx4b));
                                        bestOurMove2 = max(bestOurMove2, finalScore);
                                    end
                                end

                                worstForUs = min(worstForUs, bestOurMove2);
                            end
                        end

                        bestScore = max(bestScore, worstForUs);
                    end
                end

                scores(i) = bestScore;
            end

            % Store results
            linearStart = (startBase - 1) * numQuads + quadIdx;
            linearEnd = (endBase - 1) * numQuads + quadIdx;
            fourGreyScores(linearStart:numQuads:linearEnd) = scores;
        end

        % Progress update (throttled to once per 5 seconds)
        if toc(lastPrintTime) > 5
            elapsed = toc;
            pct = endBase / 32768 * 100;
            statesProcessed = endBase * numQuads;
            rate = statesProcessed / elapsed;
            eta = (numFourGrey - statesProcessed) / rate;

            fprintf('    Progress: %d/%d (%.1f%%) - %.0f states/sec - ETA: %.0f min\n', ...
                endBase, 32768, pct, rate, eta / 60);

            lastPrintTime = tic;
        end
    end

    elapsed = toc;
    fprintf('\n    Completed %d states in %.1f minutes\n', numFourGrey, elapsed / 60);
    fprintf('    Rate: %.0f states/sec\n', numFourGrey / elapsed);

    %% Verify data range
    fprintf('\n[4/4] Verifying and saving...\n');

    fprintf('    4-grey statistics:\n');
    fprintf('      Min score:  %+.4f\n', min(fourGreyScores));
    fprintf('      Max score:  %+.4f\n', max(fourGreyScores));
    fprintf('      Mean score: %+.4f\n', mean(fourGreyScores));
    fprintf('      Std dev:    %.4f\n', std(fourGreyScores));

    %% Save
    fprintf('    Saving extended LUT...\n');

    % Load all existing data
    oneGreyScores = data.oneGreyScores;
    twoGreyScores = data.twoGreyScores;
    greyPairs = data.greyPairs;
    metadata = data.metadata;

    % Update metadata
    metadata.version = '3.0';
    metadata.generated = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    metadata.fourGreyCount = length(fourGreyScores);
    metadata.totalCount = length(terminalLUT) + length(oneGreyScores) + ...
                          length(twoGreyScores) + length(threeGreyScores) + ...
                          length(fourGreyScores);
    metadata.generationTime4Grey = elapsed;

    % Save with compression to reduce file size
    save(lutPath, 'terminalLUT', 'oneGreyScores', 'twoGreyScores', ...
         'threeGreyScores', 'greyTriples', 'fourGreyScores', 'greyQuads', ...
         'greyPairs', 'metadata', '-v7.3');

    fprintf('    Saved to: %s\n', lutPath);
    fprintf('    Total entries: %d (%.1f million)\n', metadata.totalCount, metadata.totalCount / 1e6);

    % Estimate file size
    fileInfo = dir(lutPath);
    fprintf('    File size: %.1f MB\n', fileInfo.bytes / 1024^2);

    fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║   4-GREY LUT GENERATION COMPLETE                           ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
end
