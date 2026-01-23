classdef TabuSearch < handle
%TABUSEARCH MST2-inspired tabu search for Tangled game
%
%   Implements multistart tabu search based on D-Wave's approach.
%   Reference: Palubeckis (2004) "Multistart Tabu Search Strategies
%              for the Unconstrained Binary Quadratic Optimization Problem"
%
%   D-Wave uses this as their primary classical solver in qbsolv/dwave-tabu.
%
%   Example:
%       ts = TabuSearch('TabuTenure', 7, 'MaxIterations', 1000);
%       [bestMoves, bestScore] = ts.search(state);
%
%   The search explores the space of move sequences to find optimal
%   terminal state colorings from any given game state.

    properties
        TabuTenure int32 = 7           % How long moves stay tabu
        MaxIterations int32 = 1000     % Max iterations per restart
        NumRestarts int32 = 5          % Number of random restarts
        AspirationEnabled logical = true  % Allow tabu if improves best

        % LUT for evaluation
        LUT ExpandedLUT
        LUTLoaded logical = false
    end

    properties (Constant)
        % Edge categories for intelligent initialization
        MY_EDGES = [10, 11, 12]     % Player 1 favorable edges (1-indexed)
        OPP_EDGES = [6, 13, 14]     % Player 2 favorable edges
        HUB_EDGES = [1, 2, 3, 4, 5] % Shared hub edges
    end

    methods
        function this = TabuSearch(options)
            %TABUSEARCH Constructor
            %
            %   ts = TabuSearch()
            %   ts = TabuSearch('TabuTenure', 10, 'MaxIterations', 500)

            arguments
                options.TabuTenure int32 = 7
                options.MaxIterations int32 = 1000
                options.NumRestarts int32 = 5
                options.AspirationEnabled logical = true
            end

            this.TabuTenure = options.TabuTenure;
            this.MaxIterations = options.MaxIterations;
            this.NumRestarts = options.NumRestarts;
            this.AspirationEnabled = options.AspirationEnabled;

            this.loadLUT();
        end

        function loadLUT(this)
            %LOADLUT Load expanded LUT for evaluation

            try
                this.LUT = ExpandedLUT();
                this.LUTLoaded = this.LUT.Loaded;
            catch ME
                warning('TabuSearch:LUTError', 'Failed to load LUT: %s', ME.message);
                this.LUTLoaded = false;
            end
        end

        function [bestMoves, bestScore] = search(this, initialState)
            %SEARCH Run multistart tabu search from given state
            %
            %   [bestMoves, bestScore] = search(ts, state)
            %
            %   Returns cell array of moves {edge, color} to reach best
            %   terminal state found. Score is from Player 1's perspective.

            globalBestScore = -Inf;
            globalBestMoves = {};

            for restart = 1:this.NumRestarts
                [moves, score] = this.singleSearch(initialState, restart);

                if score > globalBestScore
                    globalBestScore = score;
                    globalBestMoves = moves;
                end
            end

            bestMoves = globalBestMoves;
            bestScore = globalBestScore;
        end

        function [bestMoves, bestScore] = singleSearch(this, initialState, seed)
            %SINGLESEARCH Single tabu search run
            %
            %   Uses D-Wave's MST2-inspired approach:
            %   1. Start from random or heuristic initialization
            %   2. Iteratively flip edges (change G<->P)
            %   3. Track tabu list to prevent cycling
            %   4. Use aspiration to override tabu for improvements

            rng(seed);  % Reproducible restarts

            greyEdges = find(initialState == '-');
            numGrey = length(greyEdges);

            if numGrey == 0
                % Terminal state - nothing to search
                bestMoves = {};
                bestScore = this.evaluate(initialState);
                return;
            end

            % Initialize with biased random completion
            currentState = initialState;
            currentMoves = cell(numGrey, 1);

            for i = 1:numGrey
                edge = greyEdges(i);
                % Use heuristic bias for initialization
                bias = this.getInitializationBias(edge);
                if rand() < bias
                    color = 'G';
                else
                    color = 'P';
                end
                currentState(edge) = color;
                currentMoves{i} = struct('edge', edge, 'color', color);
            end

            currentScore = this.evaluate(currentState);
            bestScore = currentScore;
            bestMoves = currentMoves;
            bestState = currentState;

            % Tabu list: stores iteration when move becomes non-tabu
            % tabuUntil(edge, colorIdx) where colorIdx: 1=G, 2=P
            tabuUntil = zeros(15, 2);

            % Track iterations without improvement for termination
            iterationsWithoutImprovement = 0;
            maxStagnation = ceil(this.MaxIterations / 4);

            for iter = 1:this.MaxIterations
                % Find best non-tabu move (flip one edge's color)
                bestNeighborScore = -Inf;
                bestFlipEdge = 0;
                bestFlipColor = '-';

                for i = 1:numGrey
                    edge = greyEdges(i);
                    currentColor = currentState(edge);

                    % New color is opposite of current
                    if currentColor == 'G'
                        newColor = 'P';
                        colorIdx = 2;
                    else
                        newColor = 'G';
                        colorIdx = 1;
                    end

                    % Check if tabu
                    isTabu = (tabuUntil(edge, colorIdx) > iter);

                    % Compute neighbor score
                    neighborState = currentState;
                    neighborState(edge) = newColor;
                    neighborScore = this.evaluate(neighborState);

                    % Aspiration: allow if improves global best
                    if isTabu && this.AspirationEnabled
                        if neighborScore > bestScore
                            isTabu = false;
                        end
                    end

                    if ~isTabu && neighborScore > bestNeighborScore
                        bestNeighborScore = neighborScore;
                        bestFlipEdge = edge;
                        bestFlipColor = newColor;
                    end
                end

                if bestFlipEdge == 0
                    % All moves tabu - pick least tabu move
                    minTabu = Inf;
                    for i = 1:numGrey
                        edge = greyEdges(i);
                        currentColor = currentState(edge);
                        if currentColor == 'G'
                            colorIdx = 2;
                            newColor = 'P';
                        else
                            colorIdx = 1;
                            newColor = 'G';
                        end

                        if tabuUntil(edge, colorIdx) < minTabu
                            minTabu = tabuUntil(edge, colorIdx);
                            bestFlipEdge = edge;
                            bestFlipColor = newColor;
                        end
                    end
                end

                % Apply move
                oldColor = currentState(bestFlipEdge);
                currentState(bestFlipEdge) = bestFlipColor;
                currentScore = this.evaluate(currentState);

                % Update tabu list (old color becomes tabu)
                if oldColor == 'G'
                    oldColorIdx = 1;
                else
                    oldColorIdx = 2;
                end
                tabuUntil(bestFlipEdge, oldColorIdx) = iter + this.TabuTenure;

                % Update move sequence
                for i = 1:numGrey
                    if currentMoves{i}.edge == bestFlipEdge
                        currentMoves{i}.color = bestFlipColor;
                        break;
                    end
                end

                % Track best
                if currentScore > bestScore
                    bestScore = currentScore;
                    bestMoves = currentMoves;
                    bestState = currentState;
                    iterationsWithoutImprovement = 0;
                else
                    iterationsWithoutImprovement = iterationsWithoutImprovement + 1;
                end

                % Early termination on stagnation
                if iterationsWithoutImprovement > maxStagnation
                    break;
                end
            end
        end

        function bias = getInitializationBias(this, edge)
            %GETINITIALIZATIONBIAS Get probability of choosing green for edge
            %
            %   Uses game-specific knowledge for intelligent initialization

            if ismember(edge, this.MY_EDGES)
                bias = 0.9;  % Strongly prefer green for our edges
            elseif ismember(edge, this.OPP_EDGES)
                bias = 0.1;  % Strongly prefer purple for opponent edges
            elseif ismember(edge, this.HUB_EDGES)
                bias = 0.5;  % Neutral for hub edges
            else
                bias = 0.55; % Slight green bias for other edges
            end
        end

        function score = evaluate(this, state)
            %EVALUATE Evaluate state using LUT
            %
            %   Returns score from Player 1's perspective.

            if this.LUTLoaded
                score = this.LUT.evaluate(state);
            else
                % Fallback heuristic if LUT not loaded
                score = this.evaluateHeuristic(state);
            end
        end

        function score = evaluateHeuristic(~, state)
            %EVALUATEHEURISTIC Simple heuristic evaluation
            %
            %   Counts green vs purple in important edge categories.

            MY_EDGES = [10, 11, 12];
            OPP_EDGES = [6, 13, 14];
            HUB_EDGES = [1, 2, 3, 4, 5];

            score = 0;

            % My edges - prefer green
            for e = MY_EDGES
                if state(e) == 'G'
                    score = score + 3;
                elseif state(e) == 'P'
                    score = score - 3;
                end
            end

            % Opponent edges - prefer purple
            for e = OPP_EDGES
                if state(e) == 'P'
                    score = score + 3;
                elseif state(e) == 'G'
                    score = score - 3;
                end
            end

            % Hub edges - slight green bias
            for e = HUB_EDGES
                if state(e) == 'G'
                    score = score + 0.5;
                end
            end
        end

        function [edge, color] = getBestFirstMove(this, state)
            %GETBESTFIRSTMOVE Get the best first move from search result
            %
            %   [edge, color] = getBestFirstMove(ts, state)
            %
            %   Runs search and returns only the first move to make.

            [moves, ~] = this.search(state);

            if isempty(moves)
                % No moves available - shouldn't happen for non-terminal
                edge = -1;
                color = '-';
            else
                % Return first move (0-indexed for game interface)
                edge = moves{1}.edge - 1;
                color = moves{1}.color;
            end
        end
    end
end
