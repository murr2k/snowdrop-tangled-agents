function generate_expanded_lut_parallel(terminalLutFile, outputFile)
%GENERATE_EXPANDED_LUT_PARALLEL Parallel version of expanded LUT generation
%
%   Uses MATLAB Parallel Computing Toolbox to accelerate generation.
%   Typically 5-10x faster than serial version with 8+ workers.
%
%   Generates LUT entries for:
%   - All 32,768 terminal states (0 grey edges) - from existing LUT
%   - All 491,520 states with 1 grey edge (minimax depth-1)
%   - All 3,440,640 states with 2 grey edges (minimax depth-2)
%
%   Total: ~4 million exact minimax values
%
%   Args:
%       terminalLutFile: Terminal LUT filename in data/ dir (default: terminal_scores.mat)
%       outputFile:      Output filename in data/ dir (default: expanded_lut.mat)
%
%   Based on D-Wave's approach of precomputing subproblem solutions.
%
%   Usage:
%       generate_expanded_lut_parallel()
%       generate_expanded_lut_parallel('terminal_scores_sa.mat', 'expanded_lut_sa.mat')

    if nargin < 1 || isempty(terminalLutFile)
        terminalLutFile = 'terminal_scores.mat';
    end
    if nargin < 2 || isempty(outputFile)
        outputFile = 'expanded_lut.mat';
    end

    fprintf('╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║      PARALLEL EXPANDED LUT GENERATION FOR TANGLED GAME    ║\n');
    fprintf('║     Inspired by D-Wave Hybrid Decomposition Strategy      ║\n');
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

    %% Load existing terminal LUT
    fprintf('\n[1/4] Loading terminal state LUT...\n');

    terminalLutPath = fullfile(scriptDir, 'data', terminalLutFile);
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

    %% Phase 2: States with 1 grey edge (depth-1 minimax) - PARALLEL
    fprintf('\n[2/4] Generating 1-grey-edge states (PARALLEL)...\n');
    fprintf('    Target: 32,768 x 15 = 491,520 states\n');

    tic;

    % Use 2D array for parfor compatibility (each row is independent)
    oneGreyScoresMatrix = zeros(32768, 15, 'single');

    % Parallel over base indices
    parfor baseIdx = 1:32768
        baseIdx0 = baseIdx - 1;  % 0-based for bit operations
        rowScores = zeros(1, 15, 'single');

        for greyPos = 1:15
            bitMask = 2^(greyPos-1);

            % Green completion: set bit at greyPos-1
            greenIdx0 = bitor(baseIdx0, bitMask);
            greenScore = terminalLUT(greenIdx0 + 1);

            % Purple completion: clear bit at greyPos-1
            % Create complement mask: all bits except greyPos-1
            clearMask = bitxor(32767, bitMask);  % 32767 = 2^15 - 1 (all 15 bits set)
            purpleIdx0 = bitand(baseIdx0, clearMask);
            purpleScore = terminalLUT(purpleIdx0 + 1);

            % Opponent minimizes
            rowScores(greyPos) = min(greenScore, purpleScore);
        end

        % Store entire row (parfor recognizes this pattern)
        oneGreyScoresMatrix(baseIdx, :) = rowScores;
    end

    % Flatten to column vector
    oneGreyScores = reshape(oneGreyScoresMatrix', [], 1);

    elapsed = toc;
    fprintf('    Completed 491,520 states in %.1f seconds (%.0f states/sec)\n', ...
        elapsed, length(oneGreyScores)/elapsed);
    fprintf('    Score range: [%.3f, %.3f]\n', min(oneGreyScores), max(oneGreyScores));

    %% Phase 3: States with 2 grey edges (depth-2 minimax) - PARALLEL
    fprintf('\n[3/4] Generating 2-grey-edge states (PARALLEL)...\n');

    greyPairs = nchoosek(1:15, 2);  % 105 pairs
    numPairs = size(greyPairs, 1);
    numTwoGrey = 32768 * numPairs;  % 3,440,640

    fprintf('    Target: 32,768 x 105 = %d states\n', numTwoGrey);

    tic;

    % Use 2D array for parfor compatibility (each row is independent)
    twoGreyScoresMatrix = zeros(32768, numPairs, 'single');

    % Parallel over base indices
    parfor baseIdx = 1:32768
        baseIdx0 = baseIdx - 1;  % 0-based for bit operations
        rowScores = zeros(1, numPairs, 'single');

        for pairIdx = 1:numPairs
            pos1 = greyPairs(pairIdx, 1);
            pos2 = greyPairs(pairIdx, 2);

            % Depth-2 minimax: we move (max), opponent responds (min)
            bestScore = -Inf;

            % Try all 4 first moves (2 positions x 2 colors)
            for ourPos = [pos1, pos2]
                for ourColorBit = [0, 1]  % 0=Purple (clear), 1=Green (set)
                    ourBitMask = 2^(ourPos-1);

                    % Apply our move
                    if ourColorBit == 1
                        afterOurIdx0 = bitor(baseIdx0, ourBitMask);
                    else
                        clearMask = bitxor(32767, ourBitMask);
                        afterOurIdx0 = bitand(baseIdx0, clearMask);
                    end

                    % Remaining position for opponent
                    if ourPos == pos1
                        oppPos = pos2;
                    else
                        oppPos = pos1;
                    end

                    oppBitMask = 2^(oppPos-1);

                    % Opponent tries Green
                    oppGreenIdx0 = bitor(afterOurIdx0, oppBitMask);
                    scoreG = terminalLUT(oppGreenIdx0 + 1);

                    % Opponent tries Purple
                    oppClearMask = bitxor(32767, oppBitMask);
                    oppPurpleIdx0 = bitand(afterOurIdx0, oppClearMask);
                    scoreP = terminalLUT(oppPurpleIdx0 + 1);

                    % Opponent minimizes
                    worstForUs = min(scoreG, scoreP);
                    bestScore = max(bestScore, worstForUs);
                end
            end

            rowScores(pairIdx) = bestScore;
        end

        % Store entire row (parfor recognizes this pattern)
        twoGreyScoresMatrix(baseIdx, :) = rowScores;
    end

    % Flatten to column vector
    twoGreyScores = reshape(twoGreyScoresMatrix', [], 1);

    elapsed = toc;
    fprintf('    Completed %d states in %.1f seconds (%.0f states/sec)\n', ...
        length(twoGreyScores), elapsed, length(twoGreyScores)/elapsed);
    fprintf('    Score range: [%.3f, %.3f]\n', min(twoGreyScores), max(twoGreyScores));

    %% Save expanded LUT
    fprintf('\n[4/4] Saving expanded LUT...\n');

    outputPath = fullfile(scriptDir, 'data', outputFile);

    % Store metadata
    metadata = struct();
    metadata.version = '1.0';
    metadata.generated = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    metadata.terminalCount = length(terminalLUT);
    metadata.oneGreyCount = length(oneGreyScores);
    metadata.twoGreyCount = length(twoGreyScores);
    metadata.totalCount = length(terminalLUT) + length(oneGreyScores) + length(twoGreyScores);
    metadata.greyPairs = greyPairs;
    metadata.parallelWorkers = numWorkers;

    save(outputPath, 'terminalLUT', 'oneGreyScores', 'twoGreyScores', ...
         'greyPairs', 'metadata', '-v7.3');

    fileInfo = dir(outputPath);
    fprintf('    Saved to: %s\n', outputPath);
    fprintf('    File size: %.2f MB\n', fileInfo.bytes / 1024 / 1024);

    %% Summary
    fprintf('\n╔════════════════════════════════════════════════════════════╗\n');
    fprintf('║             PARALLEL LUT GENERATION COMPLETE               ║\n');
    fprintf('╠════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Terminal states (0 grey):    %10d                   ║\n', length(terminalLUT));
    fprintf('║  One-grey states:             %10d                   ║\n', length(oneGreyScores));
    fprintf('║  Two-grey states:             %10d                   ║\n', length(twoGreyScores));
    fprintf('║  ─────────────────────────────────────────                 ║\n');
    fprintf('║  TOTAL ENTRIES:               %10d                   ║\n', metadata.totalCount);
    fprintf('╠════════════════════════════════════════════════════════════╣\n');
    fprintf('║  Workers used: %d                                          ║\n', numWorkers);
    fprintf('╚════════════════════════════════════════════════════════════╝\n');
end

function state = idx2state_local(idx)
%IDX2STATE_LOCAL Convert 1-based index to 15-char state string (parfor compatible)
    state = repmat('P', 1, 15);
    idx0 = idx - 1;
    for j = 1:15
        if bitand(idx0, 2^(j-1)) > 0
            state(j) = 'G';
        end
    end
end

function idx = state2idx_local(state)
%STATE2IDX_LOCAL Convert state string to 1-based index (parfor compatible)
    idx = 1;
    for j = 1:15
        if state(j) == 'G'
            idx = idx + 2^(j-1);
        end
    end
end
