classdef HybridTangledSolver < handle
%HYBRIDTANGLEDSOLVER D-Wave inspired hybrid minimax-MCTS solver
%
%   Combines:
%   - Alpha-beta minimax at shallow depths (exact)
%   - MCTS with progressive widening at deep levels
%   - Tabu search for rollout refinement
%   - Expanded LUT for evaluation
%
%   Inspired by D-Wave's hybrid solver architecture:
%   - qbsolv decomposition strategy
%   - MST2 tabu search
%   - Branch-and-bound with subproblem solving
%
%   The solver automatically selects the best strategy based on:
%   - Number of grey edges remaining
%   - Available time budget
%   - Estimated tree size
%
%   Example:
%       solver = HybridTangledSolver('TimeLimit', 10.0);
%       [edge, color, info] = solver.solve(state);
%
%   Reference:
%   - D-Wave qbsolv: https://github.com/dwavesystems/qbsolv
%   - D-Wave tabu: https://github.com/dwavesystems/dwave-tabu
%   - Palubeckis (2004) "Multistart Tabu Search Strategies"

    properties
        % Search parameters
        TimeLimit double = 10.0         % Total time budget (seconds)
        MinimaxDepth int32 = 4          % Depth for exact alpha-beta search
        MCTSIterations int32 = 5000     % MCTS iterations after minimax

        % Component solvers
        AlphaBeta AlphaBetaSearch
        TabuSearcher TabuSearch
        MCTS TangledMCTS

        % Strategy thresholds
        MinimaxNodeThreshold int32 = 200000  % Max nodes for pure minimax
        TabuThreshold int32 = 9              % Max grey edges for hybrid path
        MCTSThreshold int32 = 4              % Min grey edges for MCTS

        % Time allocation (fractions of TimeLimit)
        MinimaxTimeFraction double = 0.35
        MCTSTimeFraction double = 0.55
        TabuTimeFraction double = 0.10

        % Player perspective
        PlayerPerspective int32 = 1

        % Statistics
        LastSearchTime double = 0
        LastMethod char = ''
        LastMinimaxNodes int32 = 0
        LastMCTSIterations int32 = 0
        LastTabuRestarts int32 = 0
        LastScore double = 0

        % LUT for direct evaluation
        LUT ExpandedLUT
        LUTLoaded logical = false

        % Learned edge bias from REINFORCE (1x15, default zeros)
        EdgeBias double = zeros(1, 15)

        % Opponent name for conditional calibration (empty = generic)
        OpponentName char = ''
    end

    methods
        function this = HybridTangledSolver(options)
            %HYBRIDTANGLEDSOLVER Constructor
            %
            %   solver = HybridTangledSolver()
            %   solver = HybridTangledSolver('TimeLimit', 15.0, 'Player', 2)

            arguments
                options.TimeLimit double = 10.0
                options.MinimaxDepth int32 = 4
                options.MCTSIterations int32 = 5000
                options.Player int32 = 1
                options.Opponent char = ''
            end

            this.TimeLimit = options.TimeLimit;
            this.MinimaxDepth = options.MinimaxDepth;
            this.MCTSIterations = options.MCTSIterations;
            this.PlayerPerspective = options.Player;
            this.OpponentName = options.Opponent;

            % Initialize component solvers
            this.initializeSolvers();

            % Load LUT
            this.loadLUT();
        end

        function initializeSolvers(this)
            %INITIALIZESOLVERS Create component solver instances

            % Alpha-beta search
            this.AlphaBeta = AlphaBetaSearch('MaxDepth', this.MinimaxDepth, ...
                                             'UseTransposition', true);

            % Tabu search
            this.TabuSearcher = TabuSearch('MaxIterations', 500, ...
                                           'NumRestarts', 3, ...
                                           'TabuTenure', 7);

            % MCTS - initialized with fraction of time limit
            mctsTime = this.TimeLimit * this.MCTSTimeFraction;
            this.MCTS = TangledMCTS('Iterations', this.MCTSIterations, ...
                                    'TimeLimit', mctsTime, ...
                                    'Player', this.PlayerPerspective, ...
                                    'Opponent', this.OpponentName);
        end

        function loadLUT(this)
            %LOADLUT Load expanded LUT for evaluation

            try
                this.LUT = ExpandedLUT();
                this.LUTLoaded = this.LUT.Loaded;
            catch
                this.LUTLoaded = false;
            end
        end

        function [edge, color, info] = solve(this, state)
            %SOLVE Find best move using hybrid approach
            %
            %   [edge, color, info] = solve(solver, state)
            %
            %   Automatically selects strategy based on game state:
            %   - Pure minimax for small trees (late game)
            %   - Hybrid minimax+MCTS for medium trees
            %   - MCTS+Tabu for large trees (early game)
            %
            %   Returns:
            %   - edge: 0-indexed edge number
            %   - color: 'G' or 'P'
            %   - info: struct with search details

            startTime = tic;

            greyEdges = find(state == '-');
            numGrey = length(greyEdges);

            info = struct();
            info.numGrey = numGrey;
            info.strategy = 'unknown';

            % Handle terminal state
            if numGrey == 0
                edge = -1;
                color = '-';
                info.strategy = 'terminal';
                info.score = this.evaluate(state);
                info.time = toc(startTime);
                return;
            end

            % Opening book: Secure our vertex edges (E9, E10, E11) Green.
            % Empirically correct against Melissa (fitted calibration exists).
            % Skip for named opponents without a fitted calibration curve —
            % those opponents counter the opening and the 30 s budget is better
            % spent on MCTS/hybrid search at 12–13 grey edges.
            useOpeningBook = isempty(this.OpponentName) || this.MCTS.CalibrationLoaded;
            if useOpeningBook
                openingEdges = [10, 11, 12];  % E9, E10, E11
                for e = openingEdges
                    if state(e) == '-'
                        edge = e - 1;  % Convert to 0-indexed
                        color = 'G';
                        info.strategy = 'opening';
                        info.score = 0.9;
                        info.time = toc(startTime);
                        return;
                    end
                end
            end

            % Strategy selection based on game phase
            estimatedNodes = this.estimateMinimaxNodes(numGrey);

            if numGrey <= 3 || estimatedNodes < this.MinimaxNodeThreshold
                % Late game: Use pure minimax (guaranteed optimal)
                [edge, color, info] = this.solveMinimax(state, startTime);

            elseif numGrey <= this.TabuThreshold
                % Mid game (grey 9): Hybrid minimax candidate gen + MCTS eval
                [edge, color, info] = this.solveHybrid(state, startTime);

            else
                % Early game: MCTS + Tabu refinement
                [edge, color, info] = this.solveMCTS(state, startTime);
            end

            % Record statistics
            this.LastSearchTime = info.time;
            this.LastMethod = info.strategy;
            this.LastScore = info.score;
        end

        function [edge, color, info] = solveMinimax(this, state, startTime)
            %SOLVEMINIMAX Pure minimax for late game

            greyEdges = find(state == '-');
            numGrey = length(greyEdges);

            % Search to completion if possible
            depth = min(numGrey, 10);  % Limit for safety
            this.AlphaBeta.MaxDepth = depth;
            this.AlphaBeta.clearTransTable();

            [edge, color, score, abInfo] = this.AlphaBeta.search(state, true);

            info = struct();
            info.strategy = 'minimax';
            info.score = score;
            info.depth = depth;
            info.nodesSearched = abInfo.nodesSearched;
            info.pruneCount = abInfo.pruneCount;
            info.time = toc(startTime);

            this.LastMinimaxNodes = abInfo.nodesSearched;
        end

        function [edge, color, info] = solveHybrid(this, state, startTime)
            %SOLVEHYBRID Hybrid minimax + MCTS for mid game

            % Time allocation
            minimaxTime = this.TimeLimit * this.MinimaxTimeFraction;
            mctsTime = this.TimeLimit * this.MCTSTimeFraction;
            tabuTime = this.TimeLimit * this.TabuTimeFraction;

            % Phase 1: Get candidate moves from alpha-beta
            this.AlphaBeta.MaxDepth = this.MinimaxDepth;
            this.AlphaBeta.clearTransTable();

            topMoves = this.getTopMovesFromMinimax(state, 5, minimaxTime);

            if isempty(topMoves)
                % Fallback to first available move
                greyEdges = find(state == '-');
                edge = greyEdges(1) - 1;
                color = 'G';
                info = struct('strategy', 'fallback', 'score', 0, 'time', toc(startTime));
                return;
            end

            % Phase 2: Evaluate top moves with MCTS
            this.MCTS.TimeLimit = mctsTime / max(length(topMoves), 1);

            bestScore = -Inf;
            bestMove = topMoves{1};
            moveScores = zeros(length(topMoves), 1);

            for i = 1:length(topMoves)
                if toc(startTime) > this.TimeLimit - 0.5
                    break;
                end

                move = topMoves{i};
                afterState = state;
                afterState(move.edge) = move.color;

                % MCTS evaluation from opponent's perspective
                [~, ~, mctsInfo] = this.MCTS.search(afterState);

                % Score is negated opponent's best (minimax)
                if isfield(mctsInfo, 'children') && ~isempty(mctsInfo.children)
                    moveScores(i) = -mctsInfo.children{1}.value;
                else
                    % Fallback: use LUT evaluation
                    moveScores(i) = -this.evaluate(afterState);
                end

                if moveScores(i) > bestScore
                    bestScore = moveScores(i);
                    bestMove = move;
                end
            end

            this.LastMCTSIterations = this.MCTS.LastIterations;

            % Phase 3: Optional tabu refinement
            tabuImproved = false;
            if toc(startTime) < this.TimeLimit - 0.3 && tabuTime > 0.1
                [tabuMoves, tabuScore] = this.TabuSearcher.search(state);
                if tabuScore > bestScore && ~isempty(tabuMoves)
                    bestScore = tabuScore;
                    bestMove = tabuMoves{1};
                    tabuImproved = true;
                    this.LastTabuRestarts = this.TabuSearcher.NumRestarts;
                end
            end

            edge = bestMove.edge - 1;  % 0-indexed
            color = bestMove.color;

            info = struct();
            info.strategy = 'hybrid';
            info.score = bestScore;
            info.numCandidates = length(topMoves);
            info.moveScores = moveScores;
            info.tabuImproved = tabuImproved;
            info.time = toc(startTime);
        end

        function [edge, color, info] = solveMCTS(this, state, startTime)
            %SOLVEMCTS MCTS + Tabu for early game

            % Give most time to MCTS
            mctsTime = this.TimeLimit * 0.8;
            tabuTime = this.TimeLimit * 0.2;

            this.MCTS.TimeLimit = mctsTime;

            % Run MCTS
            [mctsEdge, mctsColor, mctsInfo] = this.MCTS.search(state);

            bestEdge = mctsEdge;
            bestColor = mctsColor;
            % Get best score from top child (most visited)
            if isfield(mctsInfo, 'children') && ~isempty(mctsInfo.children)
                bestScore = mctsInfo.children{1}.value;
            else
                % Fallback: use LUT or heuristic evaluation
                bestScore = this.evaluate(state);
            end

            this.LastMCTSIterations = mctsInfo.iterations;

            % Tabu refinement
            tabuImproved = false;
            if toc(startTime) < this.TimeLimit - 0.2
                [tabuMoves, tabuScore] = this.TabuSearcher.search(state);

                if tabuScore > bestScore && ~isempty(tabuMoves)
                    bestEdge = tabuMoves{1}.edge - 1;
                    bestColor = tabuMoves{1}.color;
                    bestScore = tabuScore;
                    tabuImproved = true;
                    this.LastTabuRestarts = this.TabuSearcher.NumRestarts;
                end
            end

            edge = bestEdge;
            color = bestColor;

            info = struct();
            info.strategy = 'mcts';
            info.score = bestScore;
            info.mctsIterations = mctsInfo.iterations;
            info.mctsRootVisits = mctsInfo.rootVisits;
            info.tabuImproved = tabuImproved;
            info.time = toc(startTime);
        end

        function topMoves = getTopMovesFromMinimax(this, state, numMoves, timeLimit)
            %GETTOPMOVESFROMMINIMAX Get candidate moves ranked by alpha-beta

            greyEdges = find(state == '-');
            moves = {};
            moveScores = [];

            startTime = tic;

            for edge = greyEdges'
                for color = ['G', 'P']
                    if toc(startTime) > timeLimit
                        break;
                    end

                    afterState = state;
                    afterState(edge) = color;

                    % Quick minimax evaluation
                    [score, ~] = this.AlphaBeta.alphabeta(afterState, ...
                        this.MinimaxDepth - 1, -Inf, Inf, false);

                    moves{end+1} = struct('edge', edge, 'color', color);
                    moveScores(end+1) = score;
                end
                if toc(startTime) > timeLimit
                    break;
                end
            end

            if isempty(moves)
                topMoves = {};
                return;
            end

            % Sort and return top N
            [~, sortIdx] = sort(moveScores, 'descend');
            numMoves = min(numMoves, length(moves));
            topMoves = moves(sortIdx(1:numMoves));
        end

        function nodes = estimateMinimaxNodes(~, numGrey)
            %ESTIMATEMINIMAXNODES Estimate minimax tree size
            %
            %   With good move ordering, alpha-beta examines ~sqrt(full tree)
            %   Each position has 2*numGrey moves initially

            if numGrey == 0
                nodes = 1;
            elseif numGrey <= 3
                % Small enough to enumerate
                nodes = prod(2 * (numGrey:-1:1));
            else
                % Estimate with alpha-beta pruning factor
                % Full tree: prod(2*numGrey, 2*(numGrey-1), ...)
                % With alpha-beta: approximately b^(d/2) where b is branching
                avgBranching = 2 * numGrey;  % Rough estimate
                nodes = ceil(avgBranching^(numGrey/2));
            end
        end

        function score = evaluate(this, state)
            %EVALUATE Evaluate state using LUT

            if this.LUTLoaded
                score = this.LUT.evaluate(state);
            else
                % Fallback to MCTS evaluation
                greyPos = find(state == '-');
                if isempty(greyPos)
                    score = 0;  % Unknown terminal
                else
                    % Quick MCTS probe
                    this.MCTS.TimeLimit = 0.5;
                    [~, ~, info] = this.MCTS.search(state);
                    score = info.bestValue;
                end
            end
        end

        function setPlayer(this, player)
            %SETPLAYER Set player perspective (1 or 2)

            this.PlayerPerspective = player;

            % Reinitialize MCTS with new perspective
            mctsTime = this.TimeLimit * this.MCTSTimeFraction;
            this.MCTS = TangledMCTS('Iterations', this.MCTSIterations, ...
                                    'TimeLimit', mctsTime, ...
                                    'Player', player, ...
                                    'Opponent', this.OpponentName);

            % Re-apply any previously set edge bias to new MCTS instance
            if any(this.EdgeBias ~= 0)
                this.MCTS.setEdgeBias(this.EdgeBias);
            end
        end

        function setEdgeBias(this, bias)
            %SETEDGEBIAS Set learned edge bias and propagate to MCTS
            this.EdgeBias = bias;
            this.MCTS.setEdgeBias(bias);
        end

        function stats = getStats(this)
            %GETSTATS Return solver statistics

            stats = struct();
            stats.lastSearchTime = this.LastSearchTime;
            stats.lastMethod = this.LastMethod;
            stats.lastScore = this.LastScore;
            stats.lastMinimaxNodes = this.LastMinimaxNodes;
            stats.lastMCTSIterations = this.LastMCTSIterations;
            stats.lastTabuRestarts = this.LastTabuRestarts;
            stats.lutLoaded = this.LUTLoaded;

            if this.LUTLoaded
                lutInfo = this.LUT.getInfo();
                stats.lutEntries = lutInfo.totalEntries;
            else
                stats.lutEntries = 0;
            end
        end

        function printStats(this)
            %PRINTSTATS Print solver statistics

            stats = this.getStats();

            fprintf('\n=== HybridTangledSolver Statistics ===\n');
            fprintf('Last search: %.3f seconds\n', stats.lastSearchTime);
            fprintf('Strategy: %s\n', stats.lastMethod);
            fprintf('Score: %.3f\n', stats.lastScore);
            fprintf('LUT loaded: %s (%d entries)\n', ...
                string(stats.lutLoaded), stats.lutEntries);

            if stats.lastMinimaxNodes > 0
                fprintf('Minimax nodes: %d\n', stats.lastMinimaxNodes);
            end
            if stats.lastMCTSIterations > 0
                fprintf('MCTS iterations: %d\n', stats.lastMCTSIterations);
            end
            if stats.lastTabuRestarts > 0
                fprintf('Tabu restarts: %d\n', stats.lastTabuRestarts);
            end
        end
    end
end
