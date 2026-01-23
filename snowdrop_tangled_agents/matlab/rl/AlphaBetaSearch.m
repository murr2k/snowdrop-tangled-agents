classdef AlphaBetaSearch < handle
%ALPHABETASEARCH Minimax with alpha-beta pruning for Tangled
%
%   Inspired by D-Wave's hybrid branch-and-bound approach.
%   Uses exact search at shallow depths with LUT evaluation.
%
%   Key features:
%   - Alpha-beta pruning for efficient tree search
%   - Transposition table to avoid re-evaluating positions
%   - Move ordering using game-specific priors
%   - Expanded LUT for leaf evaluation
%
%   Example:
%       ab = AlphaBetaSearch('MaxDepth', 4);
%       [edge, color, score, info] = ab.search(state, true);

    properties
        MaxDepth int32 = 4             % Maximum search depth
        UseTransposition logical = true % Use transposition table

        % Transposition table (state -> {score, bestMove, depth, flag})
        TransTable containers.Map
        TransHits int32 = 0
        TransMisses int32 = 0

        % Move ordering priors (same as MCTSNode categories)
        MY_EDGES = [10, 11, 12]        % Player 1's vertex edges (1-indexed)
        OPP_EDGES = [6, 13, 14]        % Player 2's vertex edges
        HUB_EDGES = [1, 2, 3, 4, 5]    % Shared hub edges

        % LUT for evaluation
        LUT ExpandedLUT
        LUTLoaded logical = false

        % Statistics
        NodesSearched int32 = 0
        PruneCount int32 = 0
    end

    properties (Constant)
        % Transposition table entry flags
        TT_EXACT = 0
        TT_LOWER = 1   % Alpha bound (failed high)
        TT_UPPER = 2   % Beta bound (failed low)
    end

    methods
        function this = AlphaBetaSearch(options)
            %ALPHABETASEARCH Constructor
            %
            %   ab = AlphaBetaSearch()
            %   ab = AlphaBetaSearch('MaxDepth', 6, 'UseTransposition', true)

            arguments
                options.MaxDepth int32 = 4
                options.UseTransposition logical = true
            end

            this.MaxDepth = options.MaxDepth;
            this.UseTransposition = options.UseTransposition;

            if this.UseTransposition
                this.TransTable = containers.Map('KeyType', 'char', 'ValueType', 'any');
            end

            this.loadLUT();
        end

        function loadLUT(this)
            %LOADLUT Load expanded LUT for evaluation

            try
                this.LUT = ExpandedLUT();
                this.LUTLoaded = this.LUT.Loaded;
            catch ME
                warning('AlphaBetaSearch:LUTError', 'Failed to load LUT: %s', ME.message);
                this.LUTLoaded = false;
            end
        end

        function [bestEdge, bestColor, bestScore, info] = search(this, state, isOurTurn)
            %SEARCH Find best move using alpha-beta search
            %
            %   [edge, color, score, info] = search(ab, state, isOurTurn)
            %
            %   Returns:
            %   - edge: 0-indexed edge number
            %   - color: 'G' or 'P'
            %   - score: minimax value
            %   - info: search statistics

            this.NodesSearched = 0;
            this.PruneCount = 0;
            this.TransHits = 0;
            this.TransMisses = 0;

            tic;

            [bestScore, bestMove] = this.alphabeta(state, this.MaxDepth, ...
                -Inf, Inf, isOurTurn);

            elapsed = toc;

            if ~isempty(bestMove)
                bestEdge = bestMove.edge - 1;  % Convert to 0-indexed
                bestColor = bestMove.color;
            else
                bestEdge = -1;
                bestColor = '-';
            end

            info = struct();
            info.score = bestScore;
            info.nodesSearched = this.NodesSearched;
            info.pruneCount = this.PruneCount;
            info.transHits = this.TransHits;
            info.transMisses = this.TransMisses;
            info.time = elapsed;
            info.nodesPerSecond = this.NodesSearched / max(elapsed, 0.001);
            info.depth = this.MaxDepth;
        end

        function [score, bestMove] = alphabeta(this, state, depth, alpha, beta, maximizing)
            %ALPHABETA Recursive alpha-beta search with transposition table
            %
            %   Implements negamax-style alpha-beta with:
            %   - Transposition table lookup/store
            %   - Move ordering for better pruning
            %   - LUT evaluation at leaves

            this.NodesSearched = this.NodesSearched + 1;
            alphaOrig = alpha;

            % Check transposition table
            if this.UseTransposition
                ttKey = state;
                if isKey(this.TransTable, ttKey)
                    this.TransHits = this.TransHits + 1;
                    entry = this.TransTable(ttKey);

                    if entry.depth >= depth
                        switch entry.flag
                            case this.TT_EXACT
                                score = entry.score;
                                bestMove = entry.bestMove;
                                return;
                            case this.TT_LOWER
                                alpha = max(alpha, entry.score);
                            case this.TT_UPPER
                                beta = min(beta, entry.score);
                        end

                        if alpha >= beta
                            score = entry.score;
                            bestMove = entry.bestMove;
                            return;
                        end
                    end
                else
                    this.TransMisses = this.TransMisses + 1;
                end
            end

            % Terminal check or depth limit
            greyEdges = find(state == '-');
            if isempty(greyEdges) || depth == 0
                score = this.evaluate(state);
                bestMove = struct('edge', 0, 'color', '-');

                if this.UseTransposition
                    entry = struct('score', score, 'bestMove', bestMove, ...
                                   'depth', depth, 'flag', this.TT_EXACT);
                    this.TransTable(ttKey) = entry;
                end
                return;
            end

            % Generate moves ordered by prior (best first for better pruning)
            moves = this.getOrderedMoves(state, greyEdges, maximizing);
            bestMove = moves{1};  % Default to first move

            if maximizing
                score = -Inf;
                for i = 1:length(moves)
                    move = moves{i};
                    childState = state;
                    childState(move.edge) = move.color;

                    [childScore, ~] = this.alphabeta(childState, depth-1, ...
                        alpha, beta, false);

                    if childScore > score
                        score = childScore;
                        bestMove = move;
                    end

                    alpha = max(alpha, score);
                    if beta <= alpha
                        this.PruneCount = this.PruneCount + 1;
                        break;  % Beta cutoff
                    end
                end
            else
                score = Inf;
                for i = 1:length(moves)
                    move = moves{i};
                    childState = state;
                    childState(move.edge) = move.color;

                    [childScore, ~] = this.alphabeta(childState, depth-1, ...
                        alpha, beta, true);

                    if childScore < score
                        score = childScore;
                        bestMove = move;
                    end

                    beta = min(beta, score);
                    if beta <= alpha
                        this.PruneCount = this.PruneCount + 1;
                        break;  % Alpha cutoff
                    end
                end
            end

            % Store in transposition table
            if this.UseTransposition
                if score <= alphaOrig
                    flag = this.TT_UPPER;
                elseif score >= beta
                    flag = this.TT_LOWER;
                else
                    flag = this.TT_EXACT;
                end

                entry = struct('score', score, 'bestMove', bestMove, ...
                               'depth', depth, 'flag', flag);
                this.TransTable(ttKey) = entry;
            end
        end

        function moves = getOrderedMoves(this, state, greyEdges, isOurTurn)
            %GETORDEREDMOVES Return moves sorted by prior (best first)
            %
            %   Good move ordering is critical for alpha-beta efficiency.
            %   With perfect ordering, alpha-beta searches sqrt(N) nodes.

            numMoves = length(greyEdges) * 2;
            moveList = cell(numMoves, 1);
            priors = zeros(numMoves, 1);

            idx = 1;
            for i = 1:length(greyEdges)
                edge = greyEdges(i);
                for color = ['G', 'P']
                    moveList{idx} = struct('edge', edge, 'color', color);
                    priors(idx) = this.computePrior(edge, color, isOurTurn);
                    idx = idx + 1;
                end
            end

            % Sort by prior descending (best moves first)
            [~, sortIdx] = sort(priors, 'descend');
            moves = moveList(sortIdx);
        end

        function prior = computePrior(this, edge, color, isOurTurn)
            %COMPUTEPRIOR Heuristic prior for move ordering
            %
            %   Uses game-specific knowledge to estimate move quality.
            %   Based on edge categories and known good moves.

            isGreen = (color == 'G');

            if isOurTurn
                % We maximize - prefer our favorable colorings
                if ismember(edge, this.MY_EDGES)
                    % Our vertex edges - strongly prefer green
                    prior = 0.95 * isGreen + 0.05 * ~isGreen;
                elseif ismember(edge, this.OPP_EDGES)
                    % Opponent's vertex edges - strongly prefer purple
                    prior = 0.05 * isGreen + 0.95 * ~isGreen;
                elseif ismember(edge, this.HUB_EDGES)
                    % Hub edges - slight green preference
                    prior = 0.6 * isGreen + 0.4 * ~isGreen;
                else
                    % Other edges - neutral with slight green bias
                    prior = 0.55 * isGreen + 0.45 * ~isGreen;
                end
            else
                % Opponent minimizes - expect opposite preferences
                if ismember(edge, this.OPP_EDGES)
                    % Their vertex edges - expect green
                    prior = 0.95 * isGreen + 0.05 * ~isGreen;
                elseif ismember(edge, this.MY_EDGES)
                    % Our vertex edges - expect purple
                    prior = 0.05 * isGreen + 0.95 * ~isGreen;
                elseif ismember(edge, this.HUB_EDGES)
                    % Hub edges - slight green
                    prior = 0.55 * isGreen + 0.45 * ~isGreen;
                else
                    prior = 0.5;  % Neutral
                end
            end
        end

        function score = evaluate(this, state)
            %EVALUATE Evaluate state using expanded LUT
            %
            %   Returns score from Player 1's perspective.
            %   Uses LUT for 0-2 grey edges, heuristic otherwise.

            if this.LUTLoaded
                score = this.LUT.evaluate(state);
            else
                score = this.evaluateHeuristic(state);
            end
        end

        function score = evaluateHeuristic(this, state)
            %EVALUATEHEURISTIC Fallback heuristic when LUT not available

            numGrey = sum(state == '-');

            if numGrey == 0
                % Terminal state - use simple count heuristic
                score = 0;
                for e = this.MY_EDGES
                    if state(e) == 'G'
                        score = score + 4;
                    else
                        score = score - 4;
                    end
                end
                for e = this.OPP_EDGES
                    if state(e) == 'P'
                        score = score + 4;
                    else
                        score = score - 4;
                    end
                end
                for e = this.HUB_EDGES
                    if state(e) == 'G'
                        score = score + 1;
                    end
                end
            else
                % Non-terminal - sample a few completions
                numSamples = 5;
                scores = zeros(numSamples, 1);
                greyEdges = find(state == '-');

                for s = 1:numSamples
                    sampleState = state;
                    for e = greyEdges'
                        prior = this.computePrior(e, 'G', true);
                        if rand() < prior
                            sampleState(e) = 'G';
                        else
                            sampleState(e) = 'P';
                        end
                    end
                    scores(s) = this.evaluateHeuristic(sampleState);
                end

                score = mean(scores);
            end
        end

        function clearTransTable(this)
            %CLEARTRANSTABLE Clear the transposition table

            if this.UseTransposition
                this.TransTable = containers.Map('KeyType', 'char', 'ValueType', 'any');
            end
        end

        function stats = getStats(this)
            %GETSTATS Return search statistics

            stats = struct();
            stats.nodesSearched = this.NodesSearched;
            stats.pruneCount = this.PruneCount;
            stats.transHits = this.TransHits;
            stats.transMisses = this.TransMisses;
            stats.transTableSize = 0;

            if this.UseTransposition
                stats.transTableSize = length(this.TransTable);
            end

            if this.NodesSearched > 0
                stats.pruneRate = this.PruneCount / this.NodesSearched;
            else
                stats.pruneRate = 0;
            end

            if this.TransHits + this.TransMisses > 0
                stats.transHitRate = this.TransHits / (this.TransHits + this.TransMisses);
            else
                stats.transHitRate = 0;
            end
        end
    end
end
