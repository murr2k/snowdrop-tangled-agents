function tests = test_expected_value_solver
%TEST_EXPECTED_VALUE_SOLVER Unit tests for AlphaQPolicy and the
%   expected-value adversary mode in HybridTangledSolver (Phase 4).
%
%   Verifies:
%     - AlphaQPolicy MATLAB featuriser matches the Python featuriser on
%       reference samples (Python values from
%       scripts/_phase4_reference_dump.py).
%     - predict() returns a 30-elt distribution with illegal-action mask
%       applied and renormalisation to 1.
%     - HybridTangledSolver in 'expected' mode returns a legal move and
%       reports strategy 'oracle_expected' at our turns.
%
%   Run with: runtests('test_expected_value_solver')

    tests = functiontests(localfunctions);
end

%% AlphaQPolicy featuriser parity tests

function testFeaturiserInitialState(testCase)
    pol = AlphaQPolicy();
    f = pol.featurise('---------------');
    verifyEqual(testCase, numel(f), 92, 'feature dim');
    verifyEqual(testCase, sum(f(1:45)), 15.0, 'edge ohe sums to 15');
    verifyTrue(testCase, all(f(46:75) == 0), 'vertex degrees zero');
    verifyTrue(testCase, all(f(76:87) == 0.5), 'all cycles incomplete -> 0.5');
    verifyEqual(testCase, f(88), 1.0, 'AbsTol', 1e-9, 'grey fraction 1');
    verifyEqual(testCase, f(89), 0.0, 'AbsTol', 1e-9);
    verifyEqual(testCase, f(90), 0.0, 'AbsTol', 1e-9);
    verifyEqual(testCase, f(91), 1.0);
    verifyEqual(testCase, f(92), 1.0, 'grey count odd (15)');
end

function testFeaturiserSparsePartial(testCase)
    pol = AlphaQPolicy();
    state = 'PPGPGGG--------';
    f = pol.featurise(state);
    % Reference values from scripts/_phase4_reference_dump.py:
    % vertex_deg (45..74) per Python = [1,2,3,2,1,3,1,1,2,0,2,2,2,0,2,0,0,0,1,0,1,1,0,1,0,0,0,0,0,0]
    expectedDeg = [1,2,3,2,1,3,1,1,2,0,2,2,2,0,2,0,0,0,1,0,1,1,0,1,0,0,0,0,0,0];
    verifyEqual(testCase, f(46:75), expectedDeg, 'vertex degrees');
    % cycle parities (Python 75..86): all 0.5 except cycle 12 (index 86) = 1.0
    expectedCyc = [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 1.0];
    verifyEqual(testCase, f(76:87), expectedCyc, 'AbsTol', 1e-9);
    verifyEqual(testCase, f(88), 8/15, 'AbsTol', 1e-9, 'grey fraction');
    verifyEqual(testCase, f(89), 4/15, 'AbsTol', 1e-9, 'green fraction');
    verifyEqual(testCase, f(90), 3/15, 'AbsTol', 1e-9, 'purple fraction');
    verifyEqual(testCase, f(91), 1.0);
    verifyEqual(testCase, f(92), 0.0, 'grey count 8 even');
end

function testFeaturiserDeepGame(testCase)
    pol = AlphaQPolicy();
    state = 'GPGPGGGP-PP-PGP';
    f = pol.featurise(state);
    expectedDeg = [2,1,3,2,1,3,2,1,3,0,2,2,2,1,3,0,2,2,1,2,3,2,1,3,1,2,3,0,1,1];
    verifyEqual(testCase, f(46:75), expectedDeg, 'vertex degrees');
    expectedCyc = [0.5 1.0 0.5 1.0 0.0 0.5 0.5 0.5 0.0 1.0 0.5 0.0];
    verifyEqual(testCase, f(76:87), expectedCyc, 'AbsTol', 1e-9);
    verifyEqual(testCase, f(88), 2/15, 'AbsTol', 1e-9, 'grey fraction');
    verifyEqual(testCase, f(89), 6/15, 'AbsTol', 1e-9, 'green fraction');
    verifyEqual(testCase, f(90), 7/15, 'AbsTol', 1e-9, 'purple fraction');
end

%% Action <-> (edge, color) helpers

function testActionRoundTrip(testCase)
    % Action 1 = (edge 1, G); action 2 = (edge 1, P); action 30 = (edge 15, P)
    for e = 1:15
        for c = ['G', 'P']
            a = AlphaQPolicy.edgeColorToAction(e, c);
            [e2, c2] = AlphaQPolicy.actionToEdgeColor(a);
            verifyEqual(testCase, e2, e);
            verifyEqual(testCase, c2, c);
        end
    end
end

%% predict() shape and masking

function testPredictDistributionShape(testCase)
    pol = AlphaQPolicy();
    if ~pol.Loaded
        warning('AlphaQPolicy not loaded - skipping predict test');
        return;
    end
    state = 'PPGPGGG--------';
    p = pol.predict(state);
    verifyEqual(testCase, size(p), [30, 1], 'predict returns 30x1');
    verifyEqual(testCase, sum(p), 1.0, 'AbsTol', 1e-9, 'sums to 1');

    % Legal mask: edges 8..15 (1-indexed) are grey; actions 15..30 should
    % have non-negative probability, actions 1..14 must be zero.
    illegalActions = 1:14;
    verifyTrue(testCase, all(p(illegalActions) == 0), ...
        'illegal actions must be masked to zero');
    legalActions = 15:30;
    verifyTrue(testCase, sum(p(legalActions)) > 0.99, ...
        'legal mass should be ~1');
end

function testPredictTerminalNoLegalMoves(testCase)
    pol = AlphaQPolicy();
    state = repmat('G', 1, 15);
    p = pol.predict(state);
    verifyEqual(testCase, size(p), [30, 1]);
    verifyEqual(testCase, sum(p), 0, 'AbsTol', 1e-12, ...
        'no legal moves -> distribution is all zero');
end

%% HybridTangledSolver expected-value mode integration

function testSolverExpectedModeReturnsLegalMove(testCase)
    % Construct solver with adversary='expected' and the MLP policy file.
    solver = HybridTangledSolver( ...
        'TimeLimit', 5.0, ...
        'Player', int32(1), ...
        'ExpandedLUTFile', 'expanded_lut_calib.mat', ...
        'AdversaryMode', 'expected', ...
        'OpponentPolicyFile', 'alphaq_policy_mlp.mat');

    if ~solver.LUTLoaded
        warning('Expanded LUT not loaded - skipping solver test');
        return;
    end
    if ~solver.OpponentPolicyLoaded
        warning('Opponent policy not loaded - skipping solver test');
        return;
    end

    % Use one of the exploit candidate states from Phase 1 / Phase 2.
    % grey=8 means it's P1's turn (15 minus 8 = 7 colored, 8th will be P1).
    % Actually grey=8 with P1=odd-turn convention: numGrey=8 -> P1 needs to
    % decide; mod(8,2)==0 and mod(1,2)==1 -> not our turn for P1. Use grey=5
    % candidate: 'P-G-P---GGPGPPP' (grey=5, odd -> P1's turn).
    state = 'P-G-P---GGPGPPP';
    [edge, color, info] = solver.solve(state);
    verifyTrue(testCase, edge >= 0 && edge <= 14, 'edge in [0,14]');
    verifyTrue(testCase, any(color == 'GP'), 'color is G or P');
    verifyEqual(testCase, info.strategy, 'oracle_expected', ...
        'should use expected-value oracle path');
    % Sanity: returned move targets a grey edge
    verifyEqual(testCase, state(edge + 1), '-', 'edge must have been grey');
end

function testSolverDefaultModeIsMinimax(testCase)
    solver = HybridTangledSolver( ...
        'TimeLimit', 5.0, ...
        'Player', int32(1), ...
        'ExpandedLUTFile', 'expanded_lut_calib.mat');
    verifyEqual(testCase, solver.AdversaryMode, 'minimax', ...
        'default adversary mode is minimax');
    verifyFalse(testCase, solver.OpponentPolicyLoaded, ...
        'opponent policy should not be loaded by default');
end
