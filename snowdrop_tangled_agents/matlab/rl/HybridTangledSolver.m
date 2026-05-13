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

        % LUT filename for MCTS terminal evaluation (override for SA vs Schrödinger)
        MCTSLUTFile char = 'terminal_scores.mat'

        % Expanded LUT filename (override for SA vs Schrödinger)
        ExpandedLUTFile char = 'expanded_lut.mat'

        % Opponent model filename for MCTS rollout policy
        OpponentModelFile char = 'opponent_model.mat'

        % Early-game fast selection threshold (0 = disabled).
        % When numGrey >= this value, skip MCTS and use fast heuristic:
        %   grey=9  -> shallow minimax (depth=MinimaxDepth, <1s)
        %   grey>=10 -> greedy prior (EdgeBias + heuristic, sub-ms)
        EarlyGameThreshold int32 = 0
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
                options.MCTSLUTFile char = 'terminal_scores.mat'
                options.ExpandedLUTFile char = 'expanded_lut.mat'
                options.OpponentModelFile char = 'opponent_model.mat'
                options.EarlyGameThreshold int32 = 0
            end

            this.TimeLimit = options.TimeLimit;
            this.MinimaxDepth = options.MinimaxDepth;
            this.MCTSIterations = options.MCTSIterations;
            this.PlayerPerspective = options.Player;
            this.OpponentName = options.Opponent;
            this.MCTSLUTFile = options.MCTSLUTFile;
            this.ExpandedLUTFile = options.ExpandedLUTFile;
            this.OpponentModelFile = options.OpponentModelFile;
            this.EarlyGameThreshold = options.EarlyGameThreshold;

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
                                    'Opponent', this.OpponentName, ...
                                    'Exploration', 1.8, ...
                                    'LUTFile', this.MCTSLUTFile, ...
                                    'OpponentModelFile', this.OpponentModelFile);
        end

        function loadLUT(this)
            %LOADLUT Load expanded LUT for evaluation

            try
                this.LUT = ExpandedLUT('LUTFile', this.ExpandedLUTFile);
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
            % Only use when an opponent-specific calibration curve exists
            % (e.g., Melissa). Skip for opponents using generic calibration
            % so MCTS explores freely and parallel workers are utilized.
            useOpeningBook = isempty(this.OpponentName) || this.MCTS.CalibrationIsOpponentSpecific;
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

            % Early-game fast selection: skip MCTS for grey >= threshold.
            % Moves 2-4 (grey=13,11,9) have trees too large for useful MCTS.
            if this.EarlyGameThreshold > 0 && numGrey >= this.EarlyGameThreshold
                [edge, color, info] = this.solveEarlyGame(state, startTime);
                this.LastSearchTime = info.time;
                this.LastMethod = info.strategy;
                this.LastScore = info.score;
                return;
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
            info.mctsSimulations = mctsInfo.simulations;
            info.tabuImproved = tabuImproved;
            info.time = toc(startTime);
        end

        function [edge, color, info] = solveEarlyGame(this, state, startTime)
            %SOLVEEARLYEGAME Fast move selection for early game (grey >= EarlyGameThreshold)
            %
            %   grey == 9: shallow alpha-beta at MinimaxDepth (depth 4), no MCTS.
            %   grey >= 10: greedy prior — argmax over EdgeBias + heuristic priors.
            %
            %   Both avoid spawning the parallel pool.

            greyEdges = find(state == '-');
            numGrey = length(greyEdges);

            if numGrey <= 9
                % Shallow minimax: exact search 4-ply deep. Fast (~0.1s).
                this.AlphaBeta.MaxDepth = this.MinimaxDepth;
                this.AlphaBeta.clearTransTable();
                [edge, color, score, abInfo] = this.AlphaBeta.search(state, true);
                info = struct('strategy', 'early_minimax', 'score', score, ...
                    'time', toc(startTime), 'nodesSearched', abInfo.nodesSearched, ...
                    'tabuImproved', false);
            else
                % Greedy prior: sub-millisecond heuristic selection.
                [edge, color, score] = this.greedyPrior(state, greyEdges);
                info = struct('strategy', 'early_prior', 'score', score, ...
                    'time', toc(startTime), 'tabuImproved', false);
            end
        end

        function [edge, color, score] = greedyPrior(this, state, greyEdges)
            %GREEDYPRIOR Argmax move selection using heuristic priors + EdgeBias
            %
            %   Scores every (edge, color) pair with computeRolloutPriorStatic
            %   (which incorporates the learned EdgeBias) and returns the best.
            %   Edge is returned 0-indexed to match the convention of search().

            bestScore = -Inf;
            edge = greyEdges(1) - 1;  % 0-indexed fallback
            color = 'G';
            score = 0;

            myE  = this.MCTS.MyEdges;
            oppE = this.MCTS.OppEdges;
            hubE = this.MCTS.HubEdges;
            bias = this.EdgeBias;

            for e = greyEdges'
                for c = 1:2
                    isGreen = (c == 1);
                    col = char('G' + (c - 1) * ('P' - 'G'));
                    w = TangledMCTS.computeRolloutPriorStatic( ...
                        e, isGreen, true, myE, oppE, hubE, bias);
                    if w > bestScore
                        bestScore = w;
                        edge = e - 1;  % 0-indexed
                        color = col;
                        score = w;
                    end
                end
            end
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
                                    'Opponent', this.OpponentName, ...
                                    'Exploration', 1.8, ...
                                    'LUTFile', this.MCTSLUTFile, ...
                                    'OpponentModelFile', this.OpponentModelFile);

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
