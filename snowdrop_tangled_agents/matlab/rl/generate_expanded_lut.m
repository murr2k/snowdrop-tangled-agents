function generate_expanded_lut()
%GENERATE_EXPANDED_LUT Create expanded LUT with non-terminal states
%
%   Generates LUT entries for:
%   - All 32,768 terminal states (0 grey edges) - from existing LUT
%   - All 491,520 states with 1 grey edge (minimax depth-1)
%   - All 3,440,640 states with 2 grey edges (minimax depth-2)
%
%   Total: ~4 million exact minimax values
%
%   Output: data/expanded_lut.mat
%
%   Based on D-Wave's approach of precomputing subproblem solutions.
%   Reference: D-Wave qbsolv decomposition strategy
%
%   Usage:
%       generate_expanded_lut()
%
%   The generation takes approximately 10-30 minutes depending on hardware.

    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║         EXPANDED LUT GENERATION FOR TANGLED GAME          ║\n');
    fprintf('║     Inspired by D-Wave Hybrid Decomposition Strategy      ║\n');
    fprintf('╚════════════════════════════════════════════════════════════╝\n\n');

    scriptDir = fileparts(mfilename('fullpath'));

    %% Load existing terminal LUT
    fprintf('[1/4] Loading terminal state LUT...\n');

    terminalLutPath = fullfile(scriptDir, 'data', 'terminal_scores.mat');
    if ~isfile(terminalLutPath)
        error('Terminal LUT not found at %s.\nRun generate_terminal_lut.py first.', terminalLutPath);
    end

    data = load(terminalLutPath);
    terminalLUT = double(data.terminal_scores(:));

    if length(terminalLUT) ~= 32768
        error('Terminal LUT has %d entries, expected 32768.', length(terminalLUT));
    end

    fprintf('    Loaded %d terminal state scores\n', length(terminalLUT));
    fprintf('    Score range: [%.3f, %.3f]\n', min(terminalLUT), max(terminalLUT));

    %% Phase 2: States with 1 grey edge (depth-1 minimax)
    fprintf('\n[2/4] Generating 1-grey-edge states (depth-1 minimax)...\n');
    fprintf('    Target: 32,768 x 15 = 491,520 states\n');

    tic;

    % For each of 32768 base patterns and 15 possible grey positions
    % Value = min(green_completion, purple_completion) assuming opponent plays

    numOneGrey = 32768 * 15;
    oneGreyScores = zeros(numOneGrey, 1, 'single');

    % We'll use a compact indexing scheme:
    % index = (basePatternIdx - 1) * 15 + greyPos
    % where basePatternIdx is the index with the grey position treated as 'P'

    progressInterval = 5000;

    for baseIdx = 1:32768
        baseState = idx2state(baseIdx);

        for greyPos = 1:15
            % The base state has some color at greyPos
            % We compute what happens if that position were grey instead

            % Green completion: set greyPos to G
            greenState = baseState;
            greenState(greyPos) = 'G';
            greenScore = terminalLUT(state2idx(greenState));

            % Purple completion: set greyPos to P
            purpleState = baseState;
            purpleState(greyPos) = 'P';
            purpleScore = terminalLUT(state2idx(purpleState));

            % Opponent minimizes (it's their turn after this grey edge)
            minimaxScore = min(greenScore, purpleScore);

            % Store using linear index
            linearIdx = (baseIdx - 1) * 15 + greyPos;
            oneGreyScores(linearIdx) = minimaxScore;
        end

        if mod(baseIdx, progressInterval) == 0
            elapsed = toc;
            rate = baseIdx / elapsed;
            remaining = (32768 - baseIdx) / rate;
            fprintf('    Progress: %d/%d (%.1f%%) - ETA: %.0fs\n', ...
                baseIdx, 32768, baseIdx/32768*100, remaining);
        end
    end

    elapsed = toc;
    fprintf('    Completed 491,520 states in %.1f seconds (%.0f states/sec)\n', ...
        elapsed, numOneGrey/elapsed);
    fprintf('    Score range: [%.3f, %.3f]\n', min(oneGreyScores), max(oneGreyScores));

    %% Phase 3: States with 2 grey edges (depth-2 minimax)
    fprintf('\n[3/4] Generating 2-grey-edge states (depth-2 minimax)...\n');

    greyPairs = nchoosek(1:15, 2);  % 105 pairs
    numPairs = size(greyPairs, 1);
    numTwoGrey = 32768 * numPairs;  % 3,440,640

    fprintf('    Target: 32,768 x 105 = %d states\n', numTwoGrey);

    tic;

    twoGreyScores = zeros(numTwoGrey, 1, 'single');

    % Index scheme: (baseIdx - 1) * 105 + pairIdx
    % where pairIdx is 1-105 corresponding to nchoosek(1:15, 2)

    progressInterval = 2000;

    for baseIdx = 1:32768
        baseState = idx2state(baseIdx);

        for pairIdx = 1:numPairs
            pos1 = greyPairs(pairIdx, 1);
            pos2 = greyPairs(pairIdx, 2);

            % Depth-2 minimax: we move (max), opponent responds (min)
            % We try all 4 first moves (2 positions x 2 colors)
            % For each, opponent picks worst for us from 2 options

            bestScore = -Inf;  % We maximize

            % Our move options: pos1-G, pos1-P, pos2-G, pos2-P
            for ourPos = [pos1, pos2]
                for ourColor = ['G', 'P']
                    % Apply our move
                    afterOur = baseState;
                    afterOur(ourPos) = ourColor;

                    % Remaining grey position
                    if ourPos == pos1
                        oppPos = pos2;
                    else
                        oppPos = pos1;
                    end

                    % Opponent minimizes: tries both colors
                    oppGreen = afterOur;
                    oppGreen(oppPos) = 'G';
                    scoreG = terminalLUT(state2idx(oppGreen));

                    oppPurple = afterOur;
                    oppPurple(oppPos) = 'P';
                    scoreP = terminalLUT(state2idx(oppPurple));

                    worstForUs = min(scoreG, scoreP);
                    bestScore = max(bestScore, worstForUs);
                end
            end

            linearIdx = (baseIdx - 1) * numPairs + pairIdx;
            twoGreyScores(linearIdx) = bestScore;
        end

        if mod(baseIdx, progressInterval) == 0
            elapsed = toc;
            rate = baseIdx / elapsed;
            remaining = (32768 - baseIdx) / rate;
            fprintf('    Progress: %d/%d (%.1f%%) - ETA: %.0fs\n', ...
                baseIdx, 32768, baseIdx/32768*100, remaining);
        end
    end

    elapsed = toc;
    fprintf('    Completed %d states in %.1f seconds (%.0f states/sec)\n', ...
        numTwoGrey, elapsed, numTwoGrey/elapsed);
    fprintf('    Score range: [%.3f, %.3f]\n', min(twoGreyScores), max(twoGreyScores));

    %% Save expanded LUT
    fprintf('\n[4/4] Saving expanded LUT...\n');

    outputPath = fullfile(scriptDir, 'data', 'expanded_lut.mat');

    % Store metadata
    metadata = struct();
    metadata.version = '1.0';
    metadata.generated = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    metadata.terminalCount = length(terminalLUT);
    metadata.oneGreyCount = length(oneGreyScores);
    metadata.twoGreyCount = length(twoGreyScores);
    metadata.totalCount = length(terminalLUT) + length(oneGreyScores) + length(twoGreyScores);
    metadata.greyPairs = greyPairs;  % Store for index lookup

    save(outputPath, 'terminalLUT', 'oneGreyScores', 'twoGreyScores', ...
         'greyPairs', 'metadata', '-v7.3');

    fileInfo = dir(outputPath);
    fprintf('    Saved to: %s\n', outputPath);
    fprintf('    File size: %.2f MB\n', fileInfo.bytes / 1024 / 1024);

    %% Summary
    fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║                    LUT GENERATION COMPLETE                 ║\n');
    fprintf('╠════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Terminal states (0 grey):    %10d                   ║\n', length(terminalLUT));
    fprintf('║  One-grey states:             %10d                   ║\n', length(oneGreyScores));
    fprintf('║  Two-grey states:             %10d                   ║\n', length(twoGreyScores));
    fprintf('║  ─────────────────────────────────────────                 ║\n');
    fprintf('║  TOTAL ENTRIES:               %10d                   ║\n', metadata.totalCount);
    fprintf('╠════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Score Ranges:                                             ║\n');
    fprintf('║    Terminal: [%+7.3f, %+7.3f]                           ║\n', min(terminalLUT), max(terminalLUT));
    fprintf('║    One-grey: [%+7.3f, %+7.3f]                           ║\n', min(oneGreyScores), max(oneGreyScores));
    fprintf('║    Two-grey: [%+7.3f, %+7.3f]                           ║\n', min(twoGreyScores), max(twoGreyScores));
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
end

function state = idx2state(idx)
%IDX2STATE Convert 1-based index to 15-char state string
%   Index encoding: bit j = 1 means edge j is 'G' (green)
    state = repmat('P', 1, 15);
    idx0 = idx - 1;  % Convert to 0-based
    for j = 1:15
        if bitand(idx0, 2^(j-1)) > 0
            state(j) = 'G';
        end
    end
end

function idx = state2idx(state)
%STATE2IDX Convert state string to 1-based index
%   Only works for terminal states (no grey edges)
    idx = 1;  % MATLAB 1-indexed
    for j = 1:15
        if state(j) == 'G'
            idx = idx + 2^(j-1);
        end
    end
end
