classdef SimulatedOpponent < handle
%SIMULATEDOPPONENT Simulated opponent for RL training
%
%   This class simulates opponent behavior for self-play training.
%   Multiple styles are available to create a curriculum of opponents.
%
%   Styles:
%       'random'    - Uniform random move selection
%       'heuristic' - Simple heuristic-based play
%       'mcts'      - MCTS-style play (simulates MCTS Melissa)
%       'defensive' - Prioritizes blocking our good edges
%       'aggressive'- Prioritizes taking strategic edges
%       'petersen'  - Petersen strategy (our web bot logic)
%
%   Example:
%       opp = SimulatedOpponent('Style', 'mcts');
%       move = opp.selectMove(state);

    properties
        Style char = 'mcts'

        % MCTS parameters
        Iterations int32 = 500
        ExplorationConstant double = 1.41
        MCTSTimeLimit double = 1.0

        % Real MCTS engine (lazy initialized)
        MCTSEngine TangledMCTS

        % Heuristic weights (for non-random styles)
        EdgeWeights (15,1) double
    end

    methods
        function this = SimulatedOpponent(options)
            %SIMULATEDOPPONENT Construct opponent
            %
            %   opp = SimulatedOpponent()
            %   opp = SimulatedOpponent('Style', 'mcts')
            %   opp = SimulatedOpponent('Style', 'random')
            %   opp = SimulatedOpponent('Style', 'mcts', 'Iterations', 1000)
            %
            %   Styles:
            %       'random'     - Uniform random
            %       'heuristic'  - Weighted heuristic
            %       'mcts'       - Real MCTS (matches Melissa)
            %       'fast_mcts'  - Quick heuristic approximation
            %       'petersen'   - Petersen strategy
            %       'defensive'  - Block opponent edges
            %       'aggressive' - Take strategic edges

            arguments
                options.Style char = 'mcts'
                options.Iterations int32 = 500
                options.TimeLimit double = 1.0
            end

            this.Style = options.Style;
            this.Iterations = options.Iterations;
            this.MCTSTimeLimit = options.TimeLimit;

            % Initialize edge weights based on Petersen graph analysis
            % Higher weight = more likely to play
            this.EdgeWeights = [
                0.7;   % E0: Hub
                0.5;   % E1
                0.4;   % E2
                0.7;   % E3: Hub
                0.4;   % E4
                0.8;   % E5: Strategic
                0.7;   % E6: Hub
                0.5;   % E7
                0.5;   % E8
                0.6;   % E9
                0.6;   % E10
                0.6;   % E11
                0.8;   % E12: Strategic
                0.8;   % E13: Strategic
                0.5;   % E14
            ];
        end

        function move = selectMove(this, state)
            %SELECTMOVE Select a move given current state
            %
            %   move = selectMove(opp, state)
            %
            %   Inputs:
            %       state - 15-character board state string
            %
            %   Outputs:
            %       move - struct with fields:
            %              .edge  - 0-indexed edge number
            %              .color - 'G' or 'P'

            switch this.Style
                case 'random'
                    move = this.randomMove(state);
                case 'heuristic'
                    move = this.heuristicMove(state);
                case 'mcts'
                    move = this.mctsMove(state);
                case 'fast_mcts'
                    move = this.fastMctsMove(state);
                case 'defensive'
                    move = this.defensiveMove(state);
                case 'aggressive'
                    move = this.aggressiveMove(state);
                case 'petersen'
                    move = this.petersenMove(state);
                otherwise
                    move = this.randomMove(state);
            end
        end
    end

    methods (Access = private)
        function move = randomMove(~, state)
            %RANDOMMOVE Uniform random move selection

            % Find grey edges
            greyEdges = find(state == '-') - 1;  % 0-indexed

            if isempty(greyEdges)
                move = struct('edge', -1, 'color', '-');
                return;
            end

            % Random edge and color
            move.edge = greyEdges(randi(length(greyEdges)));
            move.color = char('G' + (rand() > 0.5) * ('P' - 'G'));
        end

        function move = heuristicMove(this, state)
            %HEURISTICMOVE Weighted random based on edge importance

            greyEdges = find(state == '-');  % 1-indexed

            if isempty(greyEdges)
                move = struct('edge', -1, 'color', '-');
                return;
            end

            % Weight by edge importance
            weights = this.EdgeWeights(greyEdges);
            weights = weights / sum(weights);

            % Sample edge
            cumWeights = cumsum(weights);
            r = rand();
            edgeIdx = find(cumWeights >= r, 1);
            edge = greyEdges(edgeIdx) - 1;  % 0-indexed

            % Choose color based on edge type
            color = this.chooseColor(edge, state);

            move = struct('edge', edge, 'color', color);
        end

        function move = mctsMove(this, state)
            %MCTSMOVE Real MCTS move using TangledMCTS engine
            %
            %   Uses full Monte Carlo Tree Search with UCB1 selection
            %   and heuristic-guided rollouts to match MCTS Melissa.

            greyEdges = find(state == '-');

            if isempty(greyEdges)
                move = struct('edge', -1, 'color', '-');
                return;
            end

            % Lazy initialize MCTS engine
            if isempty(this.MCTSEngine)
                this.MCTSEngine = TangledMCTS(...
                    'Iterations', this.Iterations, ...
                    'TimeLimit', this.MCTSTimeLimit, ...
                    'Exploration', this.ExplorationConstant, ...
                    'PriorWeight', 2.0, ...
                    'UseParallel', false);  % Disable parallel for opponent
            end

            % Run MCTS search
            [edge, color] = this.MCTSEngine.search(state);

            move = struct('edge', edge, 'color', color);
        end

        function move = fastMctsMove(this, state)
            %FASTMCTSMOVE Fast heuristic approximation of MCTS
            %
            %   Uses weighted selection with exploration noise.
            %   Much faster than real MCTS but less accurate.

            greyEdges = find(state == '-');  % 1-indexed

            if isempty(greyEdges)
                move = struct('edge', -1, 'color', '-');
                return;
            end

            % Compute scores for each possible move
            scores = zeros(length(greyEdges), 2);  % [edge_idx, color_idx]

            for i = 1:length(greyEdges)
                edge = greyEdges(i);

                % Base score from edge weight
                baseScore = this.EdgeWeights(edge);

                % Evaluate both colors
                for c = 1:2
                    color = char('G' + (c-1) * ('P' - 'G'));

                    % Simulate move
                    testState = state;
                    testState(edge) = color;

                    % Quick evaluation
                    scores(i, c) = baseScore + this.quickEval(testState, color);

                    % Add exploration noise
                    scores(i, c) = scores(i, c) + 0.1 * randn();
                end
            end

            % Select best move (with some randomness for variety)
            [~, idx] = max(scores(:));
            [edgeIdx, colorIdx] = ind2sub(size(scores), idx);

            move.edge = greyEdges(edgeIdx) - 1;  % 0-indexed
            move.color = char('G' + (colorIdx-1) * ('P' - 'G'));
        end

        function move = defensiveMove(this, state)
            %DEFENSIVEMOVE Prioritize blocking opponent's good edges

            % Opponent's good edges (our good edges from their perspective)
            opponentGoodEdges = [10, 11, 12];  % E9, E10, E11 (1-indexed)

            greyEdges = find(state == '-');

            if isempty(greyEdges)
                move = struct('edge', -1, 'color', '-');
                return;
            end

            % Check if any opponent good edges are available
            availableGood = intersect(greyEdges, opponentGoodEdges);

            if ~isempty(availableGood)
                % Block one of their good edges
                edge = availableGood(randi(length(availableGood))) - 1;
                move = struct('edge', edge, 'color', 'P');  % Purple to counter
            else
                % Fall back to heuristic
                move = this.heuristicMove(state);
            end
        end

        function move = aggressiveMove(this, state)
            %AGGRESSIVEMOVE Prioritize strategic edges

            % Strategic edges for opponent
            strategicEdges = [6, 13, 14];  % E5, E12, E13 (1-indexed)

            greyEdges = find(state == '-');

            if isempty(greyEdges)
                move = struct('edge', -1, 'color', '-');
                return;
            end

            % Check if any strategic edges are available
            availableStrategic = intersect(greyEdges, strategicEdges);

            if ~isempty(availableStrategic)
                edge = availableStrategic(randi(length(availableStrategic))) - 1;
                move = struct('edge', edge, 'color', 'P');
            else
                move = this.heuristicMove(state);
            end
        end

        function move = petersenMove(~, state)
            %PETERSENMOVE Petersen strategy from web bot
            %
            %   Implements the same logic as petersen_strategy.py:
            %   - Opening: Secure MY_EDGES (E9, E10, E11) with Green
            %   - Then prioritize by edge category weights
            %   - Color: MY_EDGES -> Green, OPP_EDGES -> Purple
            %
            %   Note: From opponent's perspective, edge assignments are:
            %   - Opponent's MY_EDGES: [6, 13, 14] (1-indexed) = E5, E12, E13
            %   - Opponent's OPP_EDGES (us): [10, 11, 12] = E9, E10, E11
            %   - HUB_EDGES: [3, 11, 13] = E2, E10, E12

            % Edge classifications (1-indexed for MATLAB)
            % From opponent's perspective (they are Player 2)
            OPP_MY_EDGES = [6, 13, 14];    % E5, E12, E13 - opponent secures these
            OPP_OPP_EDGES = [10, 11, 12];  % E9, E10, E11 - they attack these
            HUB_EDGES = [3, 11, 13];       % E2, E10, E12 - hub edges

            % Edge values from petersen_strategy.py
            edgeValues = [
                0.0;   % E0
                0.0;   % E1
                0.6;   % E2 - hub
                0.0;   % E3
                0.0;   % E4
                0.9;   % E5 - opponent spoke
                0.0;   % E6
                0.0;   % E7
                0.0;   % E8
                1.1;   % E9 - MY edge
                1.0;   % E10 - MY + hub
                1.1;   % E11 - MY edge
                0.9;   % E12 - opponent hub
                0.7;   % E13 - opponent edge
                0.0;   % E14
            ];

            % Weights
            W_MY = 10.0;
            W_OPP = 8.0;
            W_HUB = 5.0;
            W_NEUTRAL = 1.0;
            HUB_PRIORITY = 0.8;

            greyEdges = find(state == '-');  % 1-indexed

            if isempty(greyEdges)
                move = struct('edge', -1, 'color', '-');
                return;
            end

            % Opening sequence: Secure opponent's MY edges first
            moveNum = 15 - length(greyEdges);
            openingSequence = [6, 13, 14];  % E5, E12, E13 (1-indexed)

            if moveNum < length(openingSequence)
                forcedEdge = openingSequence(moveNum + 1);
                if state(forcedEdge) == '-'
                    move = struct('edge', forcedEdge - 1, 'color', 'G');
                    return;
                end
            end

            % Score all available edges
            scores = zeros(length(greyEdges), 1);

            for i = 1:length(greyEdges)
                idx = greyEdges(i);
                score = edgeValues(idx);

                % Category bonuses
                if ismember(idx, OPP_MY_EDGES)
                    score = score + W_MY;
                elseif ismember(idx, OPP_OPP_EDGES)
                    score = score + W_OPP;
                elseif ismember(idx, HUB_EDGES)
                    score = score + W_HUB;
                else
                    score = score + W_NEUTRAL;
                end

                % Hub preference
                if ismember(idx, HUB_EDGES)
                    score = score + HUB_PRIORITY;
                end

                scores(i) = score;
            end

            % Select highest scoring edge
            [~, bestIdx] = max(scores);
            bestEdge = greyEdges(bestIdx);

            % Choose color
            if ismember(bestEdge, OPP_MY_EDGES)
                color = 'G';  % Green on MY edges
            elseif ismember(bestEdge, OPP_OPP_EDGES)
                color = 'P';  % Purple on opponent's edges
            else
                color = 'G';  % Default to Green
            end

            move = struct('edge', bestEdge - 1, 'color', color);  % 0-indexed
        end

        function color = chooseColor(~, edge, ~)
            %CHOOSECOLOR Choose color based on edge position
            %
            %   Opponent prefers:
            %   - Purple on outer edges (counter our green strategy)
            %   - Green on inner edges (control center)

            % Outer edges: E9-E14 (0-indexed: 9-14)
            if edge >= 9
                color = 'P';  % Purple on outer
            elseif edge <= 4
                color = 'G';  % Green on inner
            else
                % Spoke edges - random
                color = char('G' + (rand() > 0.5) * ('P' - 'G'));
            end
        end

        function score = quickEval(~, state, lastColor)
            %QUICKEVAL Quick position evaluation

            greenCount = sum(state == 'G');
            purpleCount = sum(state == 'P');

            % Opponent wants balance or slight purple advantage
            score = (purpleCount - greenCount) * 0.1;

            % Bonus for last move color
            if lastColor == 'P'
                score = score + 0.05;
            end
        end
    end
end
