classdef ExpandedLUT < handle
%EXPANDEDLUT Lookup table for Tangled game state evaluation
%
%   Provides O(1) lookup for terminal states and states with 1-13 grey edges
%   via retrograde minimax oracle (generate_sa_oracle.py).
%
%   Turn conventions (correct): k-odd -> P1 maximizes, k-even -> P2 minimizes.
%
%   Example:
%       lut = ExpandedLUT('LUTFile', 'expanded_lut_sa.mat');
%       score = lut.evaluate('GGGGGGGGGGGGGG-');  % One grey
%       score = lut.evaluate('GGGGGGGGGGGGGGG');  % Terminal

    properties (SetAccess = private)
        TerminalLUT         % 32768x1 double
        OneGreyScores       % 491520x1 single
        TwoGreyScores       % 3440640x1 single
        ThreeGreyScores     % 14909440x1 single
        FourGreyScores      % 44748800x1 single
        FiveGreyScores      % 98402304x1 single
        SixGreyScores       % 164003840x1 single
        SevenGreyScores     % 210862080x1 single
        EightGreyScores     % 210862080x1 single
        NineGreyScores      % 164003840x1 single
        TenGreyScores        % 98402304x1 single
        ElevenGreyScores     % 44748800x1 single
        TwelveGreyScores     % 14909440x1 single
        ThirteenGreyScores   % 3440640x1 single
        GreyPairs           % 105x2
        GreyTriples         % 455x3
        GreyQuads           % 1365x4
        GreyFives           % 3003x5
        GreySixes           % 5005x6
        GreySevens          % 6435x7
        GreyEights          % 6435x8
        GreyNines           % 5005x9
        GreyTens             % 3003x10
        GreyElevens          % 1365x11
        GreyTwelves          % 455x12
        GreyThirteens        % 105x13
        GreyPairIndex       % containers.Map: key -> 1-based combo index
        GreyTripleIndex
        GreyQuadIndex
        GreyFiveIndex
        GreySixIndex
        GreySevenIndex
        GreyEightIndex
        GreyNineIndex
        GreyTenIndex
        GreyElevenIndex
        GreyTwelveIndex
        GreyThirteenIndex

        Loaded logical = false
        HasExpandedData logical = false
        HasThreeGreyData logical = false
        HasFourGreyData logical = false
        HasFiveGreyData logical = false
        HasSixGreyData logical = false
        HasSevenGreyData logical = false
        HasEightGreyData logical = false
        HasNineGreyData logical = false
        HasTenGreyData logical = false
        HasElevenGreyData logical = false
        HasTwelveGreyData logical = false
        HasThirteenGreyData logical = false

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
        NUM_FIVE_GREY = 98402304
        NUM_SIX_GREY = 164003840
        NUM_SEVEN_GREY = 210862080
        NUM_EIGHT_GREY = 210862080
        NUM_NINE_GREY = 164003840
        NUM_TEN_GREY = 98402304
        NUM_ELEVEN_GREY = 44748800
        NUM_TWELVE_GREY = 14909440
        NUM_THIRTEEN_GREY = 3440640
        % C(15,k) combo counts for linear index formula
        N_COMBOS = [1, 15, 105, 455, 1365, 3003, 5005, 6435, 6435, 5005, 3003, 1365, 455, 105]
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

        function data = readMatOrH5(~, path)
            %READMATORH5 Read .mat file via load(), falling back to h5read()
            %   h5py-written HDF5 files are valid HDF5 but not in MATLAB's
            %   internal format, so load() fails. h5read() handles both.
            try
                data = load(path);
            catch
                info = h5info(path);
                data = struct();
                for i = 1:length(info.Datasets)
                    name = info.Datasets(i).Name;
                    try
                        val = h5read(path, ['/' name]);
                        data.(name) = val;
                    catch
                    end
                end
            end
        end

        function loadExpandedLUT(this, path)
            %LOADEXPANDEDLUT Load full expanded LUT

            try
                data = this.readMatOrH5(path);

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

                % Load levels 5-9 (oracle extension)
                levelDefs = {
                    'fiveGreyScores',   'greyFives',    'GreyFiveIndex',   'HasFiveGreyData',   'FiveGreyScores',   'GreyFives';
                    'sixGreyScores',    'greySixes',    'GreySixIndex',    'HasSixGreyData',    'SixGreyScores',    'GreySixes';
                    'sevenGreyScores',  'greySevens',   'GreySevenIndex',  'HasSevenGreyData',  'SevenGreyScores',  'GreySevens';
                    'eightGreyScores',  'greyEights',   'GreyEightIndex',  'HasEightGreyData',  'EightGreyScores',  'GreyEights';
                    'nineGreyScores',   'greyNines',    'GreyNineIndex',   'HasNineGreyData',   'NineGreyScores',   'GreyNines';
                    'tenGreyScores',      'greyTens',      'GreyTenIndex',      'HasTenGreyData',      'TenGreyScores',      'GreyTens';
                    'elevenGreyScores',  'greyElevens',   'GreyElevenIndex',   'HasElevenGreyData',   'ElevenGreyScores',   'GreyElevens';
                    'twelveGreyScores',  'greyTwelves',   'GreyTwelveIndex',   'HasTwelveGreyData',   'TwelveGreyScores',   'GreyTwelves';
                    'thirteenGreyScores','greyThirteens', 'GreyThirteenIndex', 'HasThirteenGreyData', 'ThirteenGreyScores', 'GreyThirteens';
                };
                for row = 1:size(levelDefs, 1)
                    scoresField  = levelDefs{row, 1};
                    combosField  = levelDefs{row, 2};
                    indexProp    = levelDefs{row, 3};
                    flagProp     = levelDefs{row, 4};
                    scoresProp   = levelDefs{row, 5};
                    combosProp   = levelDefs{row, 6};
                    if isfield(data, scoresField) && isfield(data, combosField)
                        this.(scoresProp) = single(data.(scoresField)(:));
                        comboArr = data.(combosField);
                        this.(combosProp) = comboArr;
                        idx = containers.Map('KeyType', 'char', 'ValueType', 'uint32');
                        for i = 1:size(comboArr, 1)
                            idx(this.comboKey(comboArr(i,:))) = i;
                        end
                        this.(indexProp) = idx;
                        this.(flagProp) = true;
                    end
                end

                if isfield(data, 'metadata')
                    try
                        % MATLAB-generated file: metadata is a struct
                        this.Version = data.metadata.version;
                        this.Generated = data.metadata.generated;
                        this.TotalEntries = data.metadata.totalCount;
                    catch
                        % Python-generated oracle file: metadata is a plain string
                        this.Version = 'oracle_sa';
                        this.Generated = char(data.metadata);
                    end
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
                data = this.readMatOrH5(path);
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
                    if this.HasThreeGreyData
                        score = this.lookupKGrey(state, greyPositions, ...
                            this.ThreeGreyScores, this.GreyTripleIndex, 455);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                case 4
                    if this.HasFourGreyData
                        score = this.lookupKGrey(state, greyPositions, ...
                            this.FourGreyScores, this.GreyQuadIndex, 1365);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                case 5
                    if this.HasFiveGreyData
                        score = this.lookupKGrey(state, greyPositions, ...
                            this.FiveGreyScores, this.GreyFiveIndex, 3003);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                case 6
                    if this.HasSixGreyData
                        score = this.lookupKGrey(state, greyPositions, ...
                            this.SixGreyScores, this.GreySixIndex, 5005);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                case 7
                    if this.HasSevenGreyData
                        score = this.lookupKGrey(state, greyPositions, ...
                            this.SevenGreyScores, this.GreySevenIndex, 6435);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                case 8
                    if this.HasEightGreyData
                        score = this.lookupKGrey(state, greyPositions, ...
                            this.EightGreyScores, this.GreyEightIndex, 6435);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                case 9
                    if this.HasNineGreyData
                        score = this.lookupKGrey(state, greyPositions, ...
                            this.NineGreyScores, this.GreyNineIndex, 5005);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                case 10
                    if this.HasTenGreyData
                        score = this.lookupKGrey(state, greyPositions, ...
                            this.TenGreyScores, this.GreyTenIndex, 3003);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                case 11
                    if this.HasElevenGreyData
                        score = this.lookupKGrey(state, greyPositions, ...
                            this.ElevenGreyScores, this.GreyElevenIndex, 1365);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                case 12
                    if this.HasTwelveGreyData
                        score = this.lookupKGrey(state, greyPositions, ...
                            this.TwelveGreyScores, this.GreyTwelveIndex, 455);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                case 13
                    if this.HasThirteenGreyData
                        score = this.lookupKGrey(state, greyPositions, ...
                            this.ThirteenGreyScores, this.GreyThirteenIndex, 105);
                    else
                        score = this.evaluateHeuristic(state, greyPositions);
                    end

                otherwise
                    % 14+ grey edges — heuristic only (opening moves, not oracle-covered)
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

        function score = lookupKGrey(this, state, greyPositions, scores, indexMap, nCombos)
            %LOOKUPKGREY Generic O(1) lookup for k-grey state (k=3..9)

            sortedPos = sort(greyPositions);

            baseState = state;
            for i = 1:length(sortedPos)
                baseState(sortedPos(i)) = 'P';
            end
            baseIdx = this.state2idx(baseState);

            comboIdx = indexMap(this.comboKey(sortedPos));
            linearIdx = (baseIdx - 1) * nCombos + comboIdx;
            score = double(scores(linearIdx));
        end

        function key = comboKey(~, sortedPos)
            %COMBOKEY Build underscore-joined string key from sorted position vector
            parts = arrayfun(@(x) sprintf('%d', x), sortedPos, 'UniformOutput', false);
            key = strjoin(parts, '_');
        end

        function score = evaluateOneGreyDirect(this, state, greyPos)
            %EVALUATEONEGREYDIRECT Fallback: compute one-grey value without LUT
            % k=1 is odd -> P1 maximizes

            greenState = state; greenState(greyPos) = 'G';
            purpleState = state; purpleState(greyPos) = 'P';
            score = max(this.TerminalLUT(this.state2idx(greenState)), ...
                        this.TerminalLUT(this.state2idx(purpleState)));
        end

        function score = evaluateTwoGreyDirect(this, state, greyPositions)
            %EVALUATETWOGREYDIRECT Fallback: compute two-grey value without LUT
            % k=2 is even -> P2 minimizes first, then P1 maximizes at k=1

            pos1 = greyPositions(1);
            pos2 = greyPositions(2);
            worstCase = Inf;  % P2 minimizes

            for p2Pos = [pos1, pos2]
                for p2Color = ['G', 'P']
                    afterP2 = state;
                    afterP2(p2Pos) = p2Color;
                    p1Pos = pos1 + pos2 - p2Pos;

                    g = afterP2; g(p1Pos) = 'G';
                    p = afterP2; p(p1Pos) = 'P';
                    p1Best = max(this.TerminalLUT(this.state2idx(g)), ...
                                 this.TerminalLUT(this.state2idx(p)));
                    worstCase = min(worstCase, p1Best);
                end
            end

            score = worstCase;
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
            info.oracleLevels = this.oracleLevels();
            info.totalEntries = this.TotalEntries;
            info.version = this.Version;
            info.generated = this.Generated;

            if this.Loaded
                info.terminalCount = length(this.TerminalLUT);
                info.terminalRange = [min(this.TerminalLUT), max(this.TerminalLUT)];

                % Report each level without eagerly copying large arrays into a cell
                if this.HasExpandedData
                    info.oneGreyCount  = length(this.OneGreyScores);
                    info.twoGreyCount  = length(this.TwoGreyScores);
                end
                if this.HasThreeGreyData, info.threeGreyCount = length(this.ThreeGreyScores); end
                if this.HasFourGreyData,  info.fourGreyCount  = length(this.FourGreyScores);  end
                if this.HasFiveGreyData,  info.fiveGreyCount  = length(this.FiveGreyScores);  end
                if this.HasSixGreyData,   info.sixGreyCount   = length(this.SixGreyScores);   end
                if this.HasSevenGreyData, info.sevenGreyCount = length(this.SevenGreyScores); end
                if this.HasEightGreyData, info.eightGreyCount = length(this.EightGreyScores); end
                if this.HasNineGreyData,   info.nineGreyCount   = length(this.NineGreyScores);   end
                if this.HasTenGreyData,      info.tenGreyCount      = length(this.TenGreyScores);      end
                if this.HasElevenGreyData,   info.elevenGreyCount   = length(this.ElevenGreyScores);   end
                if this.HasTwelveGreyData,   info.twelveGreyCount   = length(this.TwelveGreyScores);   end
                if this.HasThirteenGreyData, info.thirteenGreyCount = length(this.ThirteenGreyScores); end
            end
        end

        function ok = hasLevel(this, k)
            %HASLEVEL True if oracle covers states with exactly k grey edges
            switch k
                case 0,  ok = this.Loaded;
                case 1,  ok = this.HasExpandedData;
                case 2,  ok = this.HasExpandedData;
                case 3,  ok = this.HasThreeGreyData;
                case 4,  ok = this.HasFourGreyData;
                case 5,  ok = this.HasFiveGreyData;
                case 6,  ok = this.HasSixGreyData;
                case 7,  ok = this.HasSevenGreyData;
                case 8,  ok = this.HasEightGreyData;
                case 9,  ok = this.HasNineGreyData;
                case 10, ok = this.HasTenGreyData;
                case 11, ok = this.HasElevenGreyData;
                case 12, ok = this.HasTwelveGreyData;
                case 13, ok = this.HasThirteenGreyData;
                otherwise, ok = false;
            end
        end

        function levels = oracleLevels(this)
            %ORACLELEVELS Return vector of levels covered by oracle (0-indexed grey count)
            flags = [this.HasExpandedData, this.HasExpandedData, ...
                     this.HasThreeGreyData, this.HasFourGreyData, ...
                     this.HasFiveGreyData, this.HasSixGreyData, ...
                     this.HasSevenGreyData, this.HasEightGreyData, ...
                     this.HasNineGreyData, this.HasTenGreyData, ...
                     this.HasElevenGreyData, this.HasTwelveGreyData, ...
                     this.HasThirteenGreyData];
            levels = [0, find(flags)];  % 0 always covered (terminal LUT)
        end
    end
end
