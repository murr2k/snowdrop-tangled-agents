%TEST_LUT_EVALUATION Unit tests for terminal state LUT evaluation
%
%   Run with: runtests('test_lut_evaluation')
%
%   Tests:
%   - LUT loading
%   - Index conversion (state2idx, idx2state)
%   - Score lookup
%   - MCRollout integration

function tests = test_lut_evaluation
    tests = functiontests(localfunctions);
end

%% Test LUT Loading

function testTangledMCTSLUTLoad(testCase)
    % Test that TangledMCTS loads the LUT successfully
    mcts = TangledMCTS();

    % Check if LUT file exists
    scriptDir = fileparts(mfilename('fullpath'));
    lutPath = fullfile(scriptDir, 'data', 'terminal_scores.mat');

    if isfile(lutPath)
        verifyTrue(testCase, mcts.LUTLoaded, 'LUT should be loaded');
        verifyEqual(testCase, length(mcts.TerminalScoreLUT), 32768, 'LUT should have 32768 entries');
    else
        verifyFalse(testCase, mcts.LUTLoaded, 'LUT should not be loaded if file missing');
        warning('LUT file not found - run generate_terminal_lut.py first');
    end
end

function testMCRolloutLUTLoad(testCase)
    % Test that MCRollout can load the LUT
    lut = MCRollout.getTerminalLUT();

    scriptDir = fileparts(mfilename('fullpath'));
    lutPath = fullfile(scriptDir, 'data', 'terminal_scores.mat');

    if isfile(lutPath)
        verifyFalse(testCase, isempty(lut), 'LUT should be loaded');
        verifyEqual(testCase, length(lut), 32768, 'LUT should have 32768 entries');
    else
        verifyTrue(testCase, isempty(lut), 'LUT should be empty if file missing');
    end
end

%% Test Index Conversion

function testState2Idx_AllPurple(testCase)
    % All purple state should map to index 1
    mcts = TangledMCTS();
    state = 'PPPPPPPPPPPPPPP';
    idx = mcts.state2idx(state);
    verifyEqual(testCase, idx, 1);
end

function testState2Idx_AllGreen(testCase)
    % All green state should map to index 32768
    mcts = TangledMCTS();
    state = 'GGGGGGGGGGGGGGG';
    idx = mcts.state2idx(state);
    verifyEqual(testCase, idx, 32768);
end

function testState2Idx_FirstEdgeGreen(testCase)
    % Only first edge green -> index 2
    mcts = TangledMCTS();
    state = 'GPPPPPPPPPPPPPP';
    idx = mcts.state2idx(state);
    verifyEqual(testCase, idx, 2);
end

function testIdx2State_AllPurple(testCase)
    % Index 1 should map to all purple
    mcts = TangledMCTS();
    state = mcts.idx2state(1);
    verifyEqual(testCase, state, 'PPPPPPPPPPPPPPP');
end

function testIdx2State_AllGreen(testCase)
    % Index 32768 should map to all green
    mcts = TangledMCTS();
    state = mcts.idx2state(32768);
    verifyEqual(testCase, state, 'GGGGGGGGGGGGGGG');
end

function testIndexRoundTrip(testCase)
    % Test round-trip conversion for multiple indices
    mcts = TangledMCTS();
    testIndices = [1, 2, 100, 1000, 16384, 32768];

    for idx = testIndices
        state = mcts.idx2state(idx);
        roundtrip = mcts.state2idx(state);
        verifyEqual(testCase, roundtrip, idx, ...
            sprintf('Round-trip failed for index %d', idx));
    end
end

function testMCRolloutState2Idx(testCase)
    % Test MCRollout's static state2idx
    state = 'PPPPPPPPPPPPPPP';
    idx = MCRollout.state2idx(state);
    verifyEqual(testCase, idx, 1);

    state = 'GGGGGGGGGGGGGGG';
    idx = MCRollout.state2idx(state);
    verifyEqual(testCase, idx, 32768);
end

%% Test Score Lookup

function testEvaluateTerminalLUT(testCase)
    % Test that LUT lookup returns valid scores
    scriptDir = fileparts(mfilename('fullpath'));
    lutPath = fullfile(scriptDir, 'data', 'terminal_scores.mat');

    if ~isfile(lutPath)
        warning('Skipping LUT evaluation test - file not found');
        return;
    end

    mcts = TangledMCTS();
    verifyTrue(testCase, mcts.LUTLoaded, 'LUT should be loaded');

    % Test a few states
    states = {
        'PPPPPPPPPPPPPPP',  % All purple
        'GGGGGGGGGGGGGGG',  % All green
        'GPGPGPGPGPGPGPG',  % Alternating
    };

    for i = 1:length(states)
        score = mcts.evaluateTerminal(states{i});
        verifyTrue(testCase, isfinite(score), ...
            sprintf('Score should be finite for state %s', states{i}));
        verifyTrue(testCase, score >= -10 && score <= 10, ...
            sprintf('Score should be in reasonable range for state %s', states{i}));
    end
end

function testMCRolloutEvaluateTerminal(testCase)
    % Test MCRollout terminal evaluation
    score = MCRollout.evaluateTerminal('PPPPPPPPPPPPPPP');
    verifyTrue(testCase, isfinite(score), 'Score should be finite');
end

%% Test Player Perspective

function testPlayerPerspectiveP1(testCase)
    % Test score from P1 perspective
    scriptDir = fileparts(mfilename('fullpath'));
    lutPath = fullfile(scriptDir, 'data', 'terminal_scores.mat');

    if ~isfile(lutPath)
        warning('Skipping perspective test - LUT file not found');
        return;
    end

    mcts1 = TangledMCTS('Player', 1);
    state = 'GGGGGPPPPPPPPPP';  % Some mixed state
    scoreP1 = mcts1.evaluateTerminal(state);

    mcts2 = TangledMCTS('Player', 2);
    scoreP2 = mcts2.evaluateTerminal(state);

    % P2 score should be negative of P1 score
    verifyEqual(testCase, scoreP2, -scoreP1, 'AbsTol', 0.001, ...
        'P2 score should be negative of P1 score');
end

%% Test Heuristic Fallback

function testHeuristicFallback(testCase)
    % Test that heuristic works when LUT not loaded
    mcts = TangledMCTS();
    state = 'GGGGGPPPPPPPPPP';

    % Get heuristic score directly
    heuristicScore = mcts.evaluateTerminalHeuristic(state);

    verifyTrue(testCase, isfinite(heuristicScore), ...
        'Heuristic score should be finite');
end

%% Test Performance

function testLUTPerformance(testCase)
    % Test that LUT lookup is fast
    scriptDir = fileparts(mfilename('fullpath'));
    lutPath = fullfile(scriptDir, 'data', 'terminal_scores.mat');

    if ~isfile(lutPath)
        warning('Skipping performance test - LUT file not found');
        return;
    end

    mcts = TangledMCTS();
    if ~mcts.LUTLoaded
        warning('Skipping performance test - LUT not loaded');
        return;
    end

    numEvals = 10000;
    state = 'GPGPGPGPGPGPGPG';

    tic;
    for i = 1:numEvals
        mcts.evaluateTerminal(state);
    end
    elapsed = toc;

    avgTimeMs = elapsed / numEvals * 1000;
    fprintf('LUT lookup: %.4f ms per evaluation\n', avgTimeMs);

    % Should be sub-millisecond
    verifyTrue(testCase, avgTimeMs < 1.0, ...
        sprintf('LUT lookup should be < 1ms, got %.4f ms', avgTimeMs));
end

%% Test LUT Statistics

function testLUTStatistics(testCase)
    % Test LUT score distribution
    scriptDir = fileparts(mfilename('fullpath'));
    lutPath = fullfile(scriptDir, 'data', 'terminal_scores.mat');

    if ~isfile(lutPath)
        warning('Skipping statistics test - LUT file not found');
        return;
    end

    mcts = TangledMCTS();
    if ~mcts.LUTLoaded
        warning('Skipping statistics test - LUT not loaded');
        return;
    end

    lut = mcts.TerminalScoreLUT;

    fprintf('\nLUT Statistics:\n');
    fprintf('  Min: %.3f\n', min(lut));
    fprintf('  Max: %.3f\n', max(lut));
    fprintf('  Mean: %.3f\n', mean(lut));
    fprintf('  Std: %.3f\n', std(lut));

    % Count favorable states
    p1Wins = sum(lut > 0.5);
    p2Wins = sum(lut < -0.5);
    draws = length(lut) - p1Wins - p2Wins;

    fprintf('  P1 favorable (>0.5): %d (%.1f%%)\n', p1Wins, p1Wins/length(lut)*100);
    fprintf('  P2 favorable (<-0.5): %d (%.1f%%)\n', p2Wins, p2Wins/length(lut)*100);
    fprintf('  Balanced: %d (%.1f%%)\n', draws, draws/length(lut)*100);

    % Verify reasonable distribution (SA scores can range to ~+/-16)
    verifyTrue(testCase, min(lut) > -20, 'Min score should be > -20');
    verifyTrue(testCase, max(lut) < 20, 'Max score should be < 20');
end
