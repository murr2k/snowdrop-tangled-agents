function extend_lut_three_grey()
%EXTEND_LUT_THREE_GREY Add 3-grey-edge states to expanded LUT
%
%   Extends the LUT from ~4M to ~19M entries by adding:
%   - All 14,909,440 states with 3 grey edges (depth-3 minimax)
%
%   This provides exact evaluation for the last 4 moves of the game.
%
%   Usage:
%       extend_lut_three_grey()

    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║      EXTENDING LUT WITH 3-GREY-EDGE STATES                ║\n');
    fprintf('║     Depth-3 Minimax (~15 million states)                  ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

    scriptDir = fileparts(mfilename('fullpath'));

    %% Load existing expanded LUT
    fprintf('[1/3] Loading existing expanded LUT...\n');

    lutPath = fullfile(scriptDir, 'data', 'expanded_lut.mat');
    if ~isfile(lutPath)
        error('Expanded LUT not found. Run generate_expanded_lut first.');
    end

    data = load(lutPath);
    terminalLUT = double(data.terminalLUT(:));
    oneGreyScores = data.oneGreyScores;
    twoGreyScores = data.twoGreyScores;
    greyPairs = data.greyPairs;
    metadata = data.metadata;

    fprintf('    Loaded existing LUT with %d entries\n', metadata.totalCount);

    %% Phase 2: Generate 3-grey-edge states (depth-3 minimax)
    fprintf('\n[2/3] Generating 3-grey-edge states (depth-3 minimax)...\n');

    greyTriples = nchoosek(1:15, 3);  % 455 triples
    numTriples = size(greyTriples, 1);
    numThreeGrey = 32768 * numTriples;  % 14,909,440

    fprintf('    Target: 32,768 x 455 = %d states\n', numThreeGrey);
    fprintf('    This may take a few minutes...\n');

    tic;

    threeGreyScores = zeros(numThreeGrey, 1, 'single');

    % Progress tracking
    progressInterval = 2000;
    lastReport = 0;

    for baseIdx = 1:32768
        baseState = idx2state(baseIdx);

        for tripleIdx = 1:numTriples
            pos1 = greyTriples(tripleIdx, 1);
            pos2 = greyTriples(tripleIdx, 2);
            pos3 = greyTriples(tripleIdx, 3);
            positions = [pos1, pos2, pos3];

            % Depth-3 minimax: we move (max) -> opp (min) -> we move (max) -> terminal
            bestScore = -Inf;

            % Our first move: 3 positions x 2 colors = 6 options
            for ourPos1 = positions
                for ourColor1 = ['G', 'P']
                    afterOur1 = baseState;
                    afterOur1(ourPos1) = ourColor1;

                    % Remaining positions after our first move
                    remaining2 = setdiff(positions, ourPos1);

                    % Opponent minimizes: 2 positions x 2 colors = 4 options
                    worstForUs = Inf;

                    for oppPos = remaining2
                        for oppColor = ['G', 'P']
                            afterOpp = afterOur1;
                            afterOpp(oppPos) = oppColor;

                            % Our second move: 1 position x 2 colors = 2 options
                            finalPos = setdiff(remaining2, oppPos);

                            % We maximize over our final move
                            bestFinal = -Inf;
                            for ourColor2 = ['G', 'P']
                                finalState = afterOpp;
                                finalState(finalPos) = ourColor2;
                                termScore = terminalLUT(state2idx(finalState));
                                bestFinal = max(bestFinal, termScore);
                            end

                            worstForUs = min(worstForUs, bestFinal);
                        end
                    end

                    bestScore = max(bestScore, worstForUs);
                end
            end

            linearIdx = (baseIdx - 1) * numTriples + tripleIdx;
            threeGreyScores(linearIdx) = bestScore;
        end

        % Progress report
        if baseIdx - lastReport >= progressInterval
            elapsed = toc;
            rate = baseIdx / elapsed;
            remaining = (32768 - baseIdx) / rate;
            pct = baseIdx / 32768 * 100;
            fprintf('    Progress: %d/%d (%.1f%%) - ETA: %.0fs\n', ...
                baseIdx, 32768, pct, remaining);
            lastReport = baseIdx;
        end
    end

    elapsed = toc;
    fprintf('    Completed %d states in %.1f seconds (%.0f states/sec)\n', ...
        numThreeGrey, elapsed, numThreeGrey/elapsed);
    fprintf('    Score range: [%.3f, %.3f]\n', min(threeGreyScores), max(threeGreyScores));

    %% Save extended LUT
    fprintf('\n[3/3] Saving extended LUT...\n');

    % Update metadata
    metadata.version = '2.0';
    metadata.generated = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    metadata.threeGreyCount = length(threeGreyScores);
    metadata.totalCount = length(terminalLUT) + length(oneGreyScores) + ...
                          length(twoGreyScores) + length(threeGreyScores);
    metadata.greyTriples = greyTriples;

    save(lutPath, 'terminalLUT', 'oneGreyScores', 'twoGreyScores', ...
         'threeGreyScores', 'greyPairs', 'greyTriples', 'metadata', '-v7.3');

    fileInfo = dir(lutPath);
    fprintf('    Saved to: %s\n', lutPath);
    fprintf('    File size: %.2f MB\n', fileInfo.bytes / 1024 / 1024);

    %% Summary
    fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║             EXTENDED LUT GENERATION COMPLETE               ║\n');
    fprintf('╠════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Terminal states (0 grey):    %10d                   ║\n', length(terminalLUT));
    fprintf('║  One-grey states:             %10d                   ║\n', length(oneGreyScores));
    fprintf('║  Two-grey states:             %10d                   ║\n', length(twoGreyScores));
    fprintf('║  Three-grey states:           %10d                   ║\n', length(threeGreyScores));
    fprintf('║  ─────────────────────────────────────────                 ║\n');
    fprintf('║  TOTAL ENTRIES:               %10d                   ║\n', metadata.totalCount);
    fprintf('╠════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Score Ranges:                                             ║\n');
    fprintf('║    Terminal:   [%+7.3f, %+7.3f]                         ║\n', min(terminalLUT), max(terminalLUT));
    fprintf('║    One-grey:   [%+7.3f, %+7.3f]                         ║\n', min(oneGreyScores), max(oneGreyScores));
    fprintf('║    Two-grey:   [%+7.3f, %+7.3f]                         ║\n', min(twoGreyScores), max(twoGreyScores));
    fprintf('║    Three-grey: [%+7.3f, %+7.3f]                         ║\n', min(threeGreyScores), max(threeGreyScores));
    fprintf('╠════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Coverage: Exact evaluation for last 4 moves of game      ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
end

function state = idx2state(idx)
%IDX2STATE Convert 1-based index to 15-char state string
    state = repmat('P', 1, 15);
    idx0 = idx - 1;
    for j = 1:15
        if bitand(idx0, 2^(j-1)) > 0
            state(j) = 'G';
        end
    end
end

function idx = state2idx(state)
%STATE2IDX Convert state string to 1-based index
    idx = 1;
    for j = 1:15
        if state(j) == 'G'
            idx = idx + 2^(j-1);
        end
    end
end
