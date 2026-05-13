classdef ExpandedLUT < handle
%EXPANDEDLUT Lookup table for Tangled game state evaluation
%
%   Provides O(1) lookup for:
%   - Terminal states (0 grey edges): 32,768 entries
%   - One-grey states: 491,520 entries
%   - Two-grey states: 3,440,640 entries
%   - Three-grey states: 14,909,440 entries
%
%   For states with 4+ grey edges, falls back to heuristic or
%   Monte Carlo estimation.
%
%   Based on D-Wave's precomputation strategy for hybrid solvers.
%
%   Example:
%       lut = ExpandedLUT();
%       score = lut.evaluate('GGGGGGGGGGGGGG-');  % One grey
%       score = lut.evaluate('GGGGGGGGGGGGGGG');  % Terminal

    properties (SetAccess = private)
        TerminalLUT         % 32768x1 double
        OneGreyScores       % 491520x1 single
        TwoGreyScores       % 3440640x1 single
        ThreeGreyScores     % 14909440x1 single
        FourGreyScores      % 44748800x1 single
        GreyPairs           % 105x2 - pairs of grey positions
        GreyTriples         % 455x3 - triples of grey positions
        GreyQuads           % 1365x4 - quads of grey positions
        GreyPairIndex       % Map from (pos1,pos2) to pair index
        GreyTripleIndex     % Map from (pos1,pos2,pos3) to triple index
        GreyQuadIndex       % Map from (pos1,pos2,pos3,pos4) to quad index

        Loaded logical = false
        HasExpandedData logical = false
        HasThreeGreyData logical = false
        HasFourGreyData logical = false

        % LUT filename to load (override for SA vs Schrödinger)
        LUTFile char = 'expanded_lut.mat'

        % Metadata
        Version char = ''
        Generated char = ''
        TotalEntries int32 = 0
    end

    properties (Constant)
        NUM_EDGES = 15
        NUM_TERMINAL = 32768
        NUM_ONE_GREY = 491520
        NUM_TWO_GREY = 3440640
        NUM_THREE_GREY = 14909440
        NUM_FOUR_GREY = 44748800
    end

    methods
        function this = ExpandedLUT(options)
            %EXPANDEDLUT Constructor - loads LUT data
            arguments
                options.LUTFile char = 'expanded_lut.mat'
            end
            this.LUTFile = options.LUTFile;
            this.loadLUT();
        end

        function loadLUT(this)
            %LOADLUT Load LUT data from file

            scriptDir = fileparts(mfilename('fullpath'));

            % Try expanded LUT first
            expandedPath = fullfile(scriptDir, 'data', this.LUTFile);
            terminalPath = fullfile(scriptDir, 'data', 'terminal_scores.mat');

            if isfile(expandedPath)
                this.loadExpandedLUT(expandedPath);
            elseif isfile(terminalPath)
                this.loadTerminalOnly(terminalPath);
            else
                warning('ExpandedLUT:NoData', ...
                    'No LUT data found. Run generate_expanded_lut.m first.');
                this.Loaded = false;
            end
        end

        function loadExpandedLUT(this, path)
            %LOADEXPANDEDLUT Load full expanded LUT

            try
                data = load(path);

                this.TerminalLUT = double(data.terminalLUT(:));
                this.OneGreyScores = single(data.oneGreyScores(:));
                this.TwoGreyScores = single(data.twoGreyScores(:));
                this.GreyPairs = data.greyPairs;

                % Build grey pair index for fast lookup
                this.GreyPairIndex = containers.Map('KeyType', 'char', 'ValueType', 'uint32');
                for i = 1:size(this.GreyPairs, 1)
                    key = sprintf('%d_%d', this.GreyPairs(i,1), this.GreyPairs(i,2));
                    this.GreyPairIndex(key) = i;
                end

                % Load three-grey data if available
                if isfield(data, 'threeGreyScores')
                    this.ThreeGreyScores = single(data.threeGreyScores(:));
                    this.GreyTriples = data.greyTriples;

                    % Build grey triple index for fast lookup
                    this.GreyTripleIndex = containers.Map('KeyType', 'char', 'ValueType', 'uint32');
                    for i = 1:size(this.GreyTriples, 1)
                        key = sprintf('%d_%d_%d', this.GreyTriples(i,1), this.GreyTriples(i,2), this.GreyTriples(i,3));
                        this.GreyTripleIndex(key) = i;
                    end

                    this.HasThreeGreyData = true;
                end

                % Load four-grey data if available
                if isfield(data, 'fourGreyScores')
                    this.FourGreyScores = single(data.fourGreyScores(:));
                    this.GreyQuads = data.greyQuads;

                    % Build grey quad index for fast lookup
                    this.GreyQuadIndex = containers.Map('KeyType', 'char', 'ValueType', 'uint32');
                    for i = 1:size(this.GreyQuads, 1)
                        key = sprintf('%d_%d_%d_%d', this.GreyQuads(i,1), this.GreyQuads(i,2), this.GreyQuads(i,3), this.GreyQuads(i,4));
                        this.GreyQuadIndex(key) = i;
                    end

                    this.HasFourGreyData = true;
                end

                if isfield(data, 'metadata')
                    this.Version = data.metadata.version;
                    this.Generated = data.metadata.generated;
                    this.TotalEntries = data.metadata.totalCount;
                end

                this.Loaded = true;
                this.HasExpandedData = true;

            catch ME
                warning('ExpandedLUT:LoadError', 'Failed to load expanded LUT: %s', ME.message);
                this.Loaded = false;
            end
        end

        function loadTerminalOnly(this, path)
            %LOADTERMINALONLY Load terminal-only LUT as fallback

            try
                data = load(path);
                this.TerminalLUT = double(data.terminal_scores(:));
                this.Loaded = true;
                this.HasExpandedData = false;
                this.TotalEntries = length(this.TerminalLUT);

            catch ME
                warning('ExpandedLUT:LoadError', 'Failed to load terminal LUT: %s', ME.message);
                this.Loaded = false;
            end
        end

        function score = evaluate(this, state)
            %EVALUATE Evaluate a game state
            %
            %   score = evaluate(lut, state)
            %
            %   Returns minimax-optimal score for states with 0-3 grey edges.
            %   For states with 4+ grey edges, returns heuristic estimate.
            %
            %   Score is from Player 1's perspective.

            if ~this.Loaded
                score = 0;
                return;
            end

            greyPositions = find(state == '-');
            numGrey = length(greyPositions);

            switch numGrey
                case 0
                    % Terminal state - direct lookup
                    idx = this.state2idx(state);
                    score = this.TerminalLUT(idx);

                case 1
                    % One grey edge
                    if this.HasExpandedData
                        score = this.lookupOneGrey(state, greyPositions(1));
                    else
                        score = this.evaluateOneGreyDirect(state, greyPositions(1));
                    end

                case 2
                    % Two grey edges
                    if this.HasExpandedData
                        score = this.lookupTwoGrey(state, greyPositions);
                    else
                        score = this.evaluateTwoGreyDirect(state, greyPositions);
                    end

                case 3
                    % Three grey edges
                    if this.HasThreeGreyData
                        score = this.lookupThreeGrey(state, greyPositions);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                case 4
                    % Four grey edges
                    if this.HasFourGreyData
                        score = this.lookupFourGrey(state, greyPositions);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                otherwise
                    % 5+ grey edges - use heuristic
                    score = this.evaluateHeuristic(state, greyPositions);
            end
        end

        function score = lookupOneGrey(this, state, greyPos)
            %LOOKUPONEGREYLOOKUP O(1) lookup for one-grey state

            % Get base pattern (with grey position as 'P')
            baseState = state;
            baseState(greyPos) = 'P';
            baseIdx = this.state2idx(baseState);

            % Linear index into oneGreyScores
            linearIdx = (baseIdx - 1) * 15 + greyPos;
            score = double(this.OneGreyScores(linearIdx));
        end

        function score = lookupTwoGrey(this, state, greyPositions)
            %LOOKUPTWOGREY O(1) lookup for two-grey state

            pos1 = min(greyPositions);
            pos2 = max(greyPositions);

            % Get base pattern (with both grey positions as 'P')
            baseState = state;
            baseState(pos1) = 'P';
            baseState(pos2) = 'P';
            baseIdx = this.state2idx(baseState);

            % Find pair index
            key = sprintf('%d_%d', pos1, pos2);
            pairIdx = this.GreyPairIndex(key);

            % Linear index into twoGreyScores
            linearIdx = (baseIdx - 1) * 105 + pairIdx;
            score = double(this.TwoGreyScores(linearIdx));
        end

        function score = lookupThreeGrey(this, state, greyPositions)
            %LOOKUPTHREEGREY O(1) lookup for three-grey state

            sortedPos = sort(greyPositions);
            pos1 = sortedPos(1);
            pos2 = sortedPos(2);
            pos3 = sortedPos(3);

            % Get base pattern (with all grey positions as 'P')
            baseState = state;
            baseState(pos1) = 'P';
            baseState(pos2) = 'P';
            baseState(pos3) = 'P';
            baseIdx = this.state2idx(baseState);

            % Find triple index
            key = sprintf('%d_%d_%d', pos1, pos2, pos3);
            tripleIdx = this.GreyTripleIndex(key);

            % Linear index into threeGreyScores
            linearIdx = (baseIdx - 1) * 455 + tripleIdx;
            score = double(this.ThreeGreyScores(linearIdx));
        end

        function score = evaluateOneGreyDirect(this, state, greyPos)
            %EVALUATEONEGREYDIRECT Compute one-grey value directly

            greenState = state;
            greenState(greyPos) = 'G';
            greenScore = this.TerminalLUT(this.state2idx(greenState));

            purpleState = state;
            purpleState(greyPos) = 'P';
            purpleScore = this.TerminalLUT(this.state2idx(purpleState));

            % Opponent minimizes
            score = min(greenScore, purpleScore);
        end

        function score = evaluateTwoGreyDirect(this, state, greyPositions)
            %EVALUATETWOGREYDIRECT Compute two-grey value directly

            pos1 = greyPositions(1);
            pos2 = greyPositions(2);

            bestScore = -Inf;

            for ourPos = [pos1, pos2]
                for ourColor = ['G', 'P']
                    afterOur = state;
                    afterOur(ourPos) = ourColor;

                    oppPos = pos1 + pos2 - ourPos;  % The other position

                    greenState = afterOur;
                    greenState(oppPos) = 'G';
                    scoreG = this.TerminalLUT(this.state2idx(greenState));

                    purpleState = afterOur;
                    purpleState(oppPos) = 'P';
                    scoreP = this.TerminalLUT(this.state2idx(purpleState));

                    worstForUs = min(scoreG, scoreP);
                    bestScore = max(bestScore, worstForUs);
                end
            end

            score = bestScore;
        end

        function score = lookupFourGrey(this, state, greyPositions)
            %LOOKUPFOURGREY O(1) lookup for four-grey state

            sortedPos = sort(greyPositions);
            pos1 = sortedPos(1);
            pos2 = sortedPos(2);
            pos3 = sortedPos(3);
            pos4 = sortedPos(4);

            % Get base pattern (with all grey positions as 'P')
            baseState = state;
            baseState(pos1) = 'P';
            baseState(pos2) = 'P';
            baseState(pos3) = 'P';
            baseState(pos4) = 'P';
            baseIdx = this.state2idx(baseState);

            % Find quad index
            key = sprintf('%d_%d_%d_%d', pos1, pos2, pos3, pos4);
            quadIdx = this.GreyQuadIndex(key);

            % Linear index into fourGreyScores
            linearIdx = (baseIdx - 1) * 1365 + quadIdx;
            score = double(this.FourGreyScores(linearIdx));
        end

        function score = evaluateHeuristic(this, state, greyPositions)
            %EVALUATEHEURISTIC Heuristic for states with 3+ grey edges
            %
            %   Uses average of random completions weighted by priors.

            numSamples = 10;
            scores = zeros(numSamples, 1);

            for s = 1:numSamples
                sampleState = state;
                for pos = greyPositions'
                    % Weighted random by simple heuristic
                    if rand() < 0.55
                        sampleState(pos) = 'G';
                    else
                        sampleState(pos) = 'P';
                    end
                end
                scores(s) = this.TerminalLUT(this.state2idx(sampleState));
            end

            score = mean(scores);
        end

        function idx = state2idx(~, state)
            %STATE2IDX Convert terminal state to index
            idx = 1;
            for j = 1:15
                if state(j) == 'G'
                    idx = idx + 2^(j-1);
                end
            end
        end

        function state = idx2state(~, idx)
            %IDX2STATE Convert index to state string
            state = repmat('P', 1, 15);
            idx0 = idx - 1;
            for j = 1:15
                if bitand(idx0, 2^(j-1)) > 0
                    state(j) = 'G';
                end
            end
        end

        function info = getInfo(this)
            %GETINFO Return LUT statistics

            info = struct();
            info.loaded = this.Loaded;
            info.hasExpandedData = this.HasExpandedData;
            info.hasThreeGreyData = this.HasThreeGreyData;
            info.hasFourGreyData = this.HasFourGreyData;
            info.totalEntries = this.TotalEntries;
            info.version = this.Version;
            info.generated = this.Generated;

            if this.Loaded
                info.terminalCount = length(this.TerminalLUT);
                info.terminalRange = [min(this.TerminalLUT), max(this.TerminalLUT)];

                if this.HasExpandedData
                    info.oneGreyCount = length(this.OneGreyScores);
                    info.twoGreyCount = length(this.TwoGreyScores);
                    info.oneGreyRange = [min(this.OneGreyScores), max(this.OneGreyScores)];
                    info.twoGreyRange = [min(this.TwoGreyScores), max(this.TwoGreyScores)];
                end

                if this.HasThreeGreyData
                    info.threeGreyCount = length(this.ThreeGreyScores);
                    info.threeGreyRange = [min(this.ThreeGreyScores), max(this.ThreeGreyScores)];
                end

                if this.HasFourGreyData
                    info.fourGreyCount = length(this.FourGreyScores);
                    info.fourGreyRange = [min(this.FourGreyScores), max(this.FourGreyScores)];
                end
            end
        end
    end
end
