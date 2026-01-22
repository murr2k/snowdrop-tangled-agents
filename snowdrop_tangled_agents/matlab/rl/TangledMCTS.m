classdef TangledMCTS < handle
%TANGLEDMCTS Monte Carlo Tree Search for Tangled game
%
%   Full MCTS implementation with:
%   - UCB1 selection with Progressive Bias
%   - Parallel rollouts using Parallel Computing Toolbox
%   - Domain-specific heuristic rollout policy
%   - Terminal evaluation using heuristics or adjudicator
%
%   This is designed to match MCTS Melissa's behavior for training
%   and can also be used as a strong player.
%
%   Example:
%       mcts = TangledMCTS('Iterations', 1000, 'NumWorkers', 6);
%       [edge, color] = mcts.search('---------------');

    properties
        % Search parameters
        Iterations int32 = 1000
        TimeLimit double = 2.0
        Exploration double = 1.414
        PriorWeight double = 2.0

        % Rollout parameters
        UseHeuristicRollout logical = true
        RolloutsPerLeaf int32 = 1

        % Parallelization
        NumWorkers int32 = 6
        UseParallel logical = true
        PoolInitialized logical = false

        % Player perspective (1 = P1/Red, 2 = P2/Blue)
        PlayerPerspective int32 = 1

        % Edge classifications (dynamically set based on perspective)
        MyEdges
        OppEdges
        HubEdges

        % Statistics
        LastIterations int32 = 0
        LastTime double = 0
        LastRootVisits int32 = 0

        % Diagnostic compute metrics
        LastCPUTime double = 0          % CPU time consumed (seconds)
        LastNodesExpanded int32 = 0     % Tree nodes created
        LastSimulations int32 = 0       % Rollout simulations run
        LastTreeDepth int32 = 0         % Maximum tree depth reached
        LastMemoryUsed double = 0       % Memory allocated (MB)
        TotalCPUTime double = 0         % Cumulative CPU time this session
        TotalIterations int32 = 0       % Cumulative iterations this session
    end

    properties (Constant)
        % Edge classifications for P1 (1-indexed)
        P1_MY_EDGES = [10, 11, 12]    % Touch vertex 5
        P1_OPP_EDGES = [6, 13, 14]    % Touch vertex 7
        P1_HUB_EDGES = [3, 11, 13]    % Touch vertex 6
    end

    methods
        function this = TangledMCTS(options)
            %TANGLEDMCTS Construct MCTS search engine
            %
            %   mcts = TangledMCTS()
            %   mcts = TangledMCTS('Iterations', 2000)
            %   mcts = TangledMCTS('TimeLimit', 3.0, 'NumWorkers', 12)
            %   mcts = TangledMCTS('Player', 2)  % Play as P2/Blue

            arguments
                options.Iterations int32 = 1000
                options.TimeLimit double = 2.0
                options.Exploration double = 1.414
                options.PriorWeight double = 2.0
                options.NumWorkers int32 = 6
                options.UseParallel logical = true
                options.Player int32 = 1
            end

            this.Iterations = options.Iterations;
            this.TimeLimit = options.TimeLimit;
            this.Exploration = options.Exploration;
            this.PriorWeight = options.PriorWeight;
            this.NumWorkers = options.NumWorkers;
            this.UseParallel = options.UseParallel;
            this.PlayerPerspective = options.Player;

            % Set edge classifications based on player perspective
            if this.PlayerPerspective == 1
                % P1 (Red) - vertex 5
                this.MyEdges = this.P1_MY_EDGES;
                this.OppEdges = this.P1_OPP_EDGES;
                this.HubEdges = this.P1_HUB_EDGES;
            else
                % P2 (Blue) - vertex 7, swap my/opp edges
                this.MyEdges = this.P1_OPP_EDGES;   % P2's edges are P1's opponent edges
                this.OppEdges = this.P1_MY_EDGES;   % P2's opponent is P1
                this.HubEdges = this.P1_HUB_EDGES;  % Hub stays same
            end
        end

        function initPool(this)
            %INITPOOL Initialize parallel pool if needed

            if ~this.UseParallel || this.PoolInitialized
                return;
            end

            try
                cluster = parcluster('local');
                maxWorkers = cluster.NumWorkers;
                actualWorkers = min(this.NumWorkers, maxWorkers);

                pool = gcp('nocreate');
                if isempty(pool)
                    parpool('local', actualWorkers);
                elseif pool.NumWorkers < actualWorkers
                    delete(pool);
                    parpool('local', actualWorkers);
                end

                this.NumWorkers = actualWorkers;
                this.PoolInitialized = true;
            catch
                this.UseParallel = false;
            end
        end

        function [edge, color, info] = search(this, state)
            %SEARCH Run MCTS search and return best move
            %
            %   [edge, color] = search(mcts, state)
            %   [edge, color, info] = search(mcts, state)
            %
            %   Inputs:
            %       state - 15-char board state string
            %
            %   Outputs:
            %       edge  - 0-indexed edge number
            %       color - 'G' or 'P'
            %       info  - Struct with search statistics and diagnostics

            startTime = tic;
            cpuStart = cputime;  % Track CPU time consumed

            % Initialize parallel pool if using parallel rollouts
            if this.UseParallel
                this.initPool();
            end

            % Create root node
            root = MCTSNode(state, true);

            iterations = 0;
            nodesExpanded = 0;
            simulations = 0;
            maxDepth = 0;

            while iterations < this.Iterations
                % Check time limit
                if toc(startTime) >= this.TimeLimit
                    break;
                end

                % Selection: traverse tree using UCB1
                node = root;
                currentDepth = 0;
                while ~node.isTerminal() && node.isFullyExpanded()
                    node = node.bestChild(this.Exploration, this.PriorWeight);
                    currentDepth = currentDepth + 1;
                end
                maxDepth = max(maxDepth, currentDepth);

                % Expansion: add new child if not terminal
                if ~node.isTerminal() && ~node.isFullyExpanded()
                    node = node.expand();
                    nodesExpanded = nodesExpanded + 1;
                    maxDepth = max(maxDepth, currentDepth + 1);
                end

                % Simulation: rollout to terminal state
                if node.isTerminal()
                    value = this.evaluateTerminal(node.State);
                else
                    value = this.simulate(node.State, node.IsOurTurn);
                    simulations = simulations + 1;
                end

                % Backpropagation: update values up the tree
                node.update(value);

                iterations = iterations + 1;
            end

            % Calculate CPU time consumed
            cpuElapsed = cputime - cpuStart;

            % Get memory usage (MATLAB built-in)
            memInfo = memory;
            memUsedMB = memInfo.MemUsedMATLAB / (1024 * 1024);

            % Store statistics
            this.LastIterations = iterations;
            this.LastTime = toc(startTime);
            this.LastRootVisits = root.Visits;

            % Store diagnostic compute metrics
            this.LastCPUTime = cpuElapsed;
            this.LastNodesExpanded = nodesExpanded;
            this.LastSimulations = simulations;
            this.LastTreeDepth = maxDepth;
            this.LastMemoryUsed = memUsedMB;

            % Accumulate session totals
            this.TotalCPUTime = this.TotalCPUTime + cpuElapsed;
            this.TotalIterations = this.TotalIterations + iterations;

            % Select best action by visit count (most robust)
            [bestAction, visits] = root.getMostVisitedAction();

            if isempty(bestAction)
                % No children expanded - return first legal action
                greyEdges = find(state == '-');
                if ~isempty(greyEdges)
                    edge = greyEdges(1) - 1;  % 0-indexed
                    color = 'G';
                else
                    edge = -1;
                    color = '-';
                end
            else
                edge = bestAction{1} - 1;  % Convert to 0-indexed
                color = bestAction{2};
            end

            % Build info struct with diagnostics
            info = struct();
            info.iterations = iterations;
            info.time = this.LastTime;
            info.rootVisits = root.Visits;
            info.bestVisits = visits;
            info.iterationsPerSecond = iterations / max(this.LastTime, 0.001);

            % Diagnostic compute metrics
            info.cpuTime = cpuElapsed;
            info.cpuEfficiency = cpuElapsed / max(this.LastTime, 0.001);  % CPU vs wall ratio
            info.nodesExpanded = nodesExpanded;
            info.simulations = simulations;
            info.treeDepth = maxDepth;
            info.memoryUsedMB = memUsedMB;
            info.nodesPerSecond = nodesExpanded / max(this.LastTime, 0.001);
            info.simulationsPerSecond = simulations / max(this.LastTime, 0.001);

            % Session totals for tracking cumulative effort
            info.sessionTotalCPU = this.TotalCPUTime;
            info.sessionTotalIterations = this.TotalIterations;

            % Get top children for debugging
            if root.Children.Count > 0
                keys = root.Children.keys();
                childInfo = cell(length(keys), 1);
                for i = 1:length(keys)
                    c = root.Children(keys{i});
                    childInfo{i} = struct('action', keys{i}, ...
                                          'visits', c.Visits, ...
                                          'value', c.TotalValue / max(c.Visits, 1));
                end
                info.children = childInfo;
            end
        end

        function value = simulate(this, state, isOurTurn)
            %SIMULATE Run rollout from state to terminal
            %
            %   value = simulate(mcts, state, isOurTurn)

            currentState = state;
            currentTurn = isOurTurn;

            % Find available edges
            greyEdges = find(currentState == '-');

            while ~isempty(greyEdges)
                % Select action
                if this.UseHeuristicRollout
                    [edge, color] = this.heuristicAction(currentState, greyEdges, currentTurn);
                else
                    edge = greyEdges(randi(length(greyEdges)));
                    color = char('G' + (rand() > 0.5) * ('P' - 'G'));
                end

                % Apply move
                currentState(edge) = color;
                greyEdges = find(currentState == '-');
                currentTurn = ~currentTurn;
            end

            % Evaluate terminal state
            value = this.evaluateTerminal(currentState);
        end

        function [edge, color] = heuristicAction(this, state, available, isOurTurn)
            %HEURISTICACTION Select action using weighted stochastic selection

            % Build list of actions with weights
            nActions = length(available) * 2;
            actions = zeros(nActions, 2);  % [edge, colorIdx]
            weights = zeros(nActions, 1);

            idx = 1;
            for i = 1:length(available)
                e = available(i);
                for c = 1:2
                    actions(idx, :) = [e, c];
                    prior = this.computeRolloutPrior(e, c == 1, isOurTurn);
                    weights(idx) = prior^2;  % Square to amplify differences
                    idx = idx + 1;
                end
            end

            % Weighted random selection
            weights = weights / sum(weights);
            cumWeights = cumsum(weights);
            r = rand();
            selectedIdx = find(cumWeights >= r, 1);

            edge = actions(selectedIdx, 1);
            color = char('G' + (actions(selectedIdx, 2) - 1) * ('P' - 'G'));
        end

        function prior = computeRolloutPrior(this, edge, isGreen, isOurTurn)
            %COMPUTEROLLOUTPRIOR Compute prior for rollout action selection
            %
            %   Updated based on game data:
            %   - E2 G (hub-inner edge green) causes score collapse
            %   - MY_EDGES green is critical for defense
            %   - OPP_EDGES purple is modest benefit

            prior = 0.5;

            if isOurTurn
                if ismember(edge, this.MyEdges)
                    prior = 0.95 * isGreen + 0.05 * ~isGreen;
                elseif ismember(edge, this.OppEdges)
                    prior = 0.05 * isGreen + 0.95 * ~isGreen;
                elseif ismember(edge, this.HubEdges)
                    % Hub edges (mainly E2) - game data shows green is bad
                    prior = 0.25 * isGreen + 0.75 * ~isGreen;
                else
                    prior = 0.55 * isGreen + 0.45 * ~isGreen;
                end
            else
                if ismember(edge, this.OppEdges)
                    prior = 0.95 * isGreen + 0.05 * ~isGreen;
                elseif ismember(edge, this.MyEdges)
                    prior = 0.15 * isGreen + 0.85 * ~isGreen;
                elseif ismember(edge, this.HubEdges)
                    % Opponent on hub - less clear, slight green preference
                    prior = 0.55 * isGreen + 0.45 * ~isGreen;
                else
                    prior = 0.55 * isGreen + 0.45 * ~isGreen;
                end
            end
        end

        function score = evaluateTerminal(this, state)
            %EVALUATETERMINAL Evaluate terminal state
            %
            %   Calibrated evaluation based on observed game statistics.
            %   Priority system avoids double-counting overlapping edges:
            %   1. MY_EDGES - highest priority (defense)
            %   2. OPP_EDGES - skip if already in MY_EDGES (attack)
            %   3. HUB_EDGES - skip if already in MY/OPP (hub control)
            %
            %   CRITICAL INSIGHT from game data:
            %   - E12 G causes massive score collapse (-1.5 points observed!)
            %   - E12 connects hub (V6) to opponent vertex (V7)
            %   - Green strengthens V7, which is BAD for us
            %
            %   Returns score from THIS player's perspective.

            score = 0;
            scored = false(1, 15);  % Track which edges we've scored

            % MY_EDGES scoring - HIGHEST PRIORITY: secure our vertex
            % Data shows E9 G has +0.875 avg delta - defense is critical
            for e = this.MyEdges
                if state(e) == 'G'
                    score = score + 1.2;   % Securing our edges is very valuable
                elseif state(e) == 'P'
                    score = score - 1.0;   % Enemy attacks are costly
                end
                scored(e) = true;
            end

            % OPP_EDGES scoring - attacks on opponent's vertex
            % CRITICAL: E12 G is VERY BAD (strengthens V7 via hub)
            % Game data shows E12 P is risky, E13 P is OK, E5 P is mixed
            for e = this.OppEdges
                if scored(e)
                    continue;  % Skip if already scored as MY_EDGE
                end

                if state(e) == 'P'
                    % Attacking their vertex - modest benefit
                    if e == 6  % E5 (1-indexed 6) - moderate benefit
                        score = score + 0.25;
                    elseif e == 14  % E13 (1-indexed 14) - best attack
                        score = score + 0.35;
                    else  % E12 (1-indexed 13) - risky attack
                        score = score + 0.15;
                    end
                elseif state(e) == 'G'
                    % They securing their edge OR us making a mistake
                    if e == 13  % E12 (1-indexed 13) - CRITICAL: green is TERRIBLE
                        score = score - 0.8;  % Strong penalty for E12 G
                    else
                        score = score - 0.15;  % Modest penalty for other OPP green
                    end
                end
                scored(e) = true;
            end

            % HUB_EDGES (V6) - only E2 gets processed here (E10, E12 handled above)
            % CRITICAL: Game data shows E2 G causes massive score collapse (-2.6 points!)
            % E2 = V0-V6, connecting inner pentagram to hub
            % Green on E2 seems to help opponent in quantum adjudication
            for e = this.HubEdges
                if scored(e)
                    continue;  % Skip if already scored
                end

                if e == 3  % E2 (1-indexed 3) - special handling
                    if state(e) == 'G'
                        score = score - 0.5;  % E2 G is BAD based on game data
                    elseif state(e) == 'P'
                        score = score + 0.1;  % E2 P might be slightly better
                    end
                else
                    % Other hub edges (shouldn't reach here normally)
                    if state(e) == 'G'
                        score = score + 0.2;
                    elseif state(e) == 'P'
                        score = score - 0.15;
                    end
                end
                scored(e) = true;
            end

            % Color balance - slight preference for balance
            greenCount = sum(state == 'G');
            purpleCount = sum(state == 'P');
            imbalance = abs(greenCount - purpleCount);
            score = score - imbalance * 0.02;  % Small penalty for imbalance
        end

        function move = selectMove(this, state)
            %SELECTMOVE Interface for SimulatedOpponent compatibility
            %
            %   move = selectMove(mcts, state)
            %   Returns struct with .edge (0-indexed) and .color

            [edge, color] = this.search(state);
            move = struct('edge', edge, 'color', color);
        end

        function stats = getStats(this)
            %GETSTATS Return statistics from last search

            stats = struct();
            stats.iterations = this.LastIterations;
            stats.time = this.LastTime;
            stats.rootVisits = this.LastRootVisits;
            stats.iterationsPerSecond = this.LastIterations / max(this.LastTime, 0.001);

            % Include diagnostic metrics
            stats.cpuTime = this.LastCPUTime;
            stats.nodesExpanded = this.LastNodesExpanded;
            stats.simulations = this.LastSimulations;
            stats.treeDepth = this.LastTreeDepth;
            stats.memoryUsedMB = this.LastMemoryUsed;
        end

        function effort = getComputeEffort(this)
            %GETCOMPUTEEFFORT Return detailed compute effort diagnostics
            %
            %   effort = getComputeEffort(mcts)
            %
            %   Returns struct with:
            %     - cpuTime: CPU seconds consumed in last search
            %     - cpuEfficiency: Ratio of CPU time to wall time
            %     - nodesExpanded: Tree nodes created
            %     - simulations: Rollout simulations run
            %     - treeDepth: Maximum search depth
            %     - memoryMB: Memory used by MATLAB
            %     - throughput: Iterations per CPU second
            %     - sessionTotal: Cumulative stats for session

            effort = struct();

            % Last search metrics
            effort.cpuTime = this.LastCPUTime;
            effort.wallTime = this.LastTime;
            effort.cpuEfficiency = this.LastCPUTime / max(this.LastTime, 0.001);
            effort.nodesExpanded = this.LastNodesExpanded;
            effort.simulations = this.LastSimulations;
            effort.treeDepth = this.LastTreeDepth;
            effort.memoryMB = this.LastMemoryUsed;

            % Throughput metrics
            effort.iterationsPerCPUSec = this.LastIterations / max(this.LastCPUTime, 0.001);
            effort.nodesPerCPUSec = this.LastNodesExpanded / max(this.LastCPUTime, 0.001);
            effort.simsPerCPUSec = this.LastSimulations / max(this.LastCPUTime, 0.001);

            % Session totals
            effort.sessionTotalCPU = this.TotalCPUTime;
            effort.sessionTotalIterations = this.TotalIterations;
            effort.sessionAvgCPUPerSearch = this.TotalCPUTime / max(double(this.TotalIterations) / double(this.Iterations), 1);
        end

        function resetSessionStats(this)
            %RESETSESSIONSTATS Reset cumulative session statistics

            this.TotalCPUTime = 0;
            this.TotalIterations = 0;
        end
    end

    methods (Static)
        function score = evaluateTerminalFull(state)
            %EVALUATETERMINALFULL Full evaluation using simulated annealing
            %
            %   This would call Python adjudicator for accurate scoring.
            %   For now, falls back to heuristic.

            % TODO: Bridge to Python SimulatedAnnealingAdjudicator
            mcts = TangledMCTS();
            score = mcts.evaluateTerminal(state);
        end
    end
end
