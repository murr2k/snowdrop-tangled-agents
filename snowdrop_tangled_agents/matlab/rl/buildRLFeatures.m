function features = buildRLFeatures(state, turnFlag, currentScore, scoreHistory)
%BUILDRLFEATURES Build 50-element observation vector for RL agent
%
%   features = buildRLFeatures(state, turnFlag)
%   features = buildRLFeatures(state, turnFlag, currentScore)
%   features = buildRLFeatures(state, turnFlag, currentScore, scoreHistory)
%
%   Inputs:
%       state        - 15-character board state string (G/P/-)
%       turnFlag     - +1 if our turn, -1 if opponent's turn
%       currentScore - Current game score (default: 0)
%       scoreHistory - Array of recent scores for momentum (default: [])
%
%   Outputs:
%       features - 50x1 feature vector:
%                  [1-15]   Board state encoding (-1=P, 0=grey, +1=G)
%                  [16]     Turn indicator
%                  [17-31]  Edge category encoding
%                  [32]     Grey edge count (normalized)
%                  [33-35]  Score momentum (last 3 deltas)
%                  [36-50]  Game phase one-hot encoding
%
%   Example:
%       state = '---------------';
%       features = buildRLFeatures(state, 1, 0);

    arguments
        state (1,15) char
        turnFlag (1,1) double = 1
        currentScore (1,1) double = 0
        scoreHistory (:,1) double = []
    end

    features = zeros(50, 1);

    %% Features 1-15: Board state encoding
    for i = 1:15
        if state(i) == 'G'
            features(i) = 1;
        elseif state(i) == 'P'
            features(i) = -1;
        else
            features(i) = 0;
        end
    end

    %% Feature 16: Turn indicator
    features(16) = turnFlag;

    %% Features 17-31: Edge category encoding
    % Categories based on Petersen graph structure:
    %   Inner edges (pentagram): E0-E4 connect inner vertices
    %   Spoke edges: E5-E9 connect inner to outer
    %   Outer edges (pentagon): E10-E14 connect outer vertices
    %
    % Encoding: importance weight based on strategic value
    % MY_VERTEX edges (good for us): +0.5
    % OPP_VERTEX edges (good for opponent): -0.5
    % HUB edges (critical): +1.0
    % NEUTRAL: 0

    % Edge categories (empirically derived from game analysis)
    % Good green edges: E9, E10, E11 (outer left triangle)
    % Good purple edges: E5, E12, E13 (right side)
    % Hub edges: E0, E3, E6 (central connections)

    edgeCategories = [
        0.8;   % E0: Hub edge
        0.3;   % E1: Inner
        0.2;   % E2: Inner
        0.8;   % E3: Hub edge
        0.2;   % E4: Inner
        0.6;   % E5: Good purple
        0.8;   % E6: Hub edge
        0.3;   % E7: Spoke
        0.3;   % E8: Spoke
        0.7;   % E9: Good green
        0.7;   % E10: Good green
        0.7;   % E11: Good green
        0.6;   % E12: Good purple
        0.6;   % E13: Good purple
        0.4;   % E14: Outer
    ];

    features(17:31) = edgeCategories;

    %% Feature 32: Grey edge count (normalized 0-1)
    greyCount = sum(state == '-');
    features(32) = greyCount / 15;

    %% Features 33-35: Score momentum (last 3 deltas)
    if ~isempty(scoreHistory) && length(scoreHistory) >= 2
        deltas = diff(scoreHistory);
        deltas = deltas(max(1, end-2):end);  % Last 3 deltas
        deltas = tanh(deltas);  % Normalize to [-1, 1]

        % Pad with zeros if needed
        n = length(deltas);
        features(33:32+n) = deltas;
    end

    %% Features 36-50: Game phase one-hot encoding
    % Divide game into 5 phases based on moves made (grey edges remaining)
    % Each phase has 3 substates: early, mid, late

    if greyCount >= 13
        phase = 1;  % Opening (0-2 moves)
    elseif greyCount >= 10
        phase = 2;  % Early game (3-5 moves)
    elseif greyCount >= 6
        phase = 3;  % Mid game (6-9 moves)
    elseif greyCount >= 3
        phase = 4;  % Late game (10-12 moves)
    else
        phase = 5;  % Endgame (13-15 moves)
    end

    % One-hot encode: 3 features per phase
    phaseIdx = 36 + (phase - 1) * 3;

    % Substate within phase
    if greyCount == 15 || greyCount == 12 || greyCount == 9 || greyCount == 6 || greyCount == 3
        substate = 1;  % Start of phase
    elseif greyCount == 14 || greyCount == 11 || greyCount == 8 || greyCount == 5 || greyCount == 2
        substate = 2;  % Middle of phase
    else
        substate = 3;  % End of phase
    end

    features(phaseIdx + substate - 1) = 1;
end
