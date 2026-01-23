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

        % Terminal state LUT for accurate evaluation
        TerminalScoreLUT               % 32768x1 double array
        LUTLoaded logical = false      % Whether LUT was successfully loaded
        LUTPath char = ''              % Path to the LUT file

        % Opponent model for learned rollout policy
        OpponentModel struct           % Loaded from opponent_model.mat
        OpponentModelLoaded logical = false
        UseOpponentModel logical = true  % Whether to use opponent model in rollouts
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

            % Load terminal state LUT for accurate evaluation
            this.loadLUT();

            % Load opponent model for learned rollout policy
            this.loadOpponentModel();
        end

        function loadLUT(this)
            %LOADLUT Load terminal state lookup table from .mat file
            %
            %   Attempts to load pre-computed terminal scores from:
            %   <script_dir>/data/terminal_scores.mat
            %
            %   Falls back to heuristic evaluation if LUT not available.

            % Find the LUT file relative to this script
            scriptPath = mfilename('fullpath');
            scriptDir = fileparts(scriptPath);
            lutPath = fullfile(scriptDir, 'data', 'terminal_scores.mat');

            if ~isfile(lutPath)
                % Try alternative paths
                altPaths = {
                    fullfile(pwd, 'data', 'terminal_scores.mat'),
                    fullfile(pwd, 'snowdrop_tangled_agents', 'matlab', 'rl', 'data', 'terminal_scores.mat')
                };
                for i = 1:length(altPaths)
                    if isfile(altPaths{i})
                        lutPath = altPaths{i};
                        break;
                    end
                end
            end

            if ~isfile(lutPath)
                warning('TangledMCTS:LUTNotFound', ...
                    'Terminal score LUT not found at %s. Using heuristic evaluation.', lutPath);
                this.LUTLoaded = false;
                return;
            end

            try
                data = load(lutPath);
                if isfield(data, 'terminal_scores')
                    this.TerminalScoreLUT = double(data.terminal_scores(:));
                    expectedSize = 2^15;  % 32768
                    if length(this.TerminalScoreLUT) == expectedSize
                        this.LUTLoaded = true;
                        this.LUTPath = lutPath;
                    else
                        warning('TangledMCTS:LUTWrongSize', ...
                            'LUT has %d entries, expected %d. Using heuristic.', ...
                            length(this.TerminalScoreLUT), expectedSize);
                        this.LUTLoaded = false;
                    end
                else
                    warning('TangledMCTS:LUTMissingField', ...
                        'LUT file missing terminal_scores field. Using heuristic.');
                    this.LUTLoaded = false;
                end
            catch ME
                warning('TangledMCTS:LUTLoadError', ...
                    'Failed to load LUT: %s. Using heuristic.', ME.message);
                this.LUTLoaded = false;
            end
        end

        function loadOpponentModel(this)
            %LOADOPPONENTMODEL Load opponent model for learned rollout policy
            %
            %   Loads opponent response probabilities from:
            %   <script_dir>/data/opponent_model.mat
            %
            %   The model contains:
            %   - response_probs: 30x30 matrix of P(opp_move | our_move)
            %   - phase_probs: 4x30 matrix of P(opp_move | phase)
            %   - total_moves: Number of training samples

            % Find the model file relative to this script
            scriptPath = mfilename('fullpath');
            scriptDir = fileparts(scriptPath);
            modelPath = fullfile(scriptDir, 'data', 'opponent_model.mat');

            if ~isfile(modelPath)
                % Try alternative paths
                altPaths = {
                    fullfile(pwd, 'data', 'opponent_model.mat'),
                    fullfile(pwd, 'snowdrop_tangled_agents', 'matlab', 'rl', 'data', 'opponent_model.mat')
                };
                for i = 1:length(altPaths)
                    if isfile(altPaths{i})
                        modelPath = altPaths{i};
                        break;
                    end
                end
            end

            if ~isfile(modelPath)
                % Opponent model not found - use heuristic only
                this.OpponentModelLoaded = false;
                return;
            end

            try
                data = load(modelPath);
                if isfield(data, 'response_probs') && isfield(data, 'phase_probs')
                    this.OpponentModel = struct();
                    this.OpponentModel.response_probs = double(data.response_probs);
                    this.OpponentModel.phase_probs = double(data.phase_probs);
                    this.OpponentModel.total_moves = double(data.total_moves);
                    this.OpponentModel.response_totals = double(data.response_totals);

                    % Validate shapes
                    if size(this.OpponentModel.response_probs, 1) == 30 && ...
                       size(this.OpponentModel.response_probs, 2) == 30 && ...
                       size(this.OpponentModel.phase_probs, 1) == 4 && ...
                       size(this.OpponentModel.phase_probs, 2) == 30
                        this.OpponentModelLoaded = true;
                        fprintf('Loaded opponent model with %d training moves\n', ...
                            this.OpponentModel.total_moves);
                    else
                        warning('TangledMCTS:OpponentModelWrongShape', ...
                            'Opponent model has unexpected shape. Using heuristic.');
                        this.OpponentModelLoaded = false;
                    end
                else
                    warning('TangledMCTS:OpponentModelMissingField', ...
                        'Opponent model missing required fields. Using heuristic.');
                    this.OpponentModelLoaded = false;
                end
            catch ME
                warning('TangledMCTS:OpponentModelLoadError', ...
                    'Failed to load opponent model: %s. Using heuristic.', ME.message);
                this.OpponentModelLoaded = false;
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
            %
            %   When opponent model is loaded and it's opponent's turn,
            %   uses learned response probabilities instead of heuristics.

            % Build list of actions with weights
            nActions = length(available) * 2;
            actions = zeros(nActions, 2);  % [edge, colorIdx]
            weights = zeros(nActions, 1);

            % Use opponent model for opponent's turn if available
            useOppModel = ~isOurTurn && this.UseOpponentModel && this.OpponentModelLoaded;

            if useOppModel
                % Get phase from grey count
                greyCount = sum(state == '-');
                phaseIdx = this.greyToPhaseIdx(greyCount);

                % Get phase probabilities from model
                phaseProbs = this.OpponentModel.phase_probs(phaseIdx, :);
            end

            idx = 1;
            for i = 1:length(available)
                e = available(i);
                for c = 1:2
                    actions(idx, :) = [e, c];

                    if useOppModel
                        % Use opponent model probability
                        % Move index: edge * 2 + colorIdx (0 for G, 1 for P)
                        % Note: edges are 1-indexed in MATLAB, model uses 0-indexed
                        moveIdx = (e - 1) * 2 + (c - 1) + 1;  % +1 for MATLAB 1-indexing
                        prior = phaseProbs(moveIdx);
                    else
                        % Use heuristic
                        prior = this.computeRolloutPrior(e, c == 1, isOurTurn);
                    end

                    weights(idx) = prior + 0.001;  % Small floor to prevent zero weights
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

        function phaseIdx = greyToPhaseIdx(~, greyCount)
            %GREYTOPHASEIDX Convert grey count to phase index (1-4)
            %
            %   Phases: early (12-15), mid (8-11), late (4-7), endgame (0-3)

            if greyCount >= 12
                phaseIdx = 1;  % early
            elseif greyCount >= 8
                phaseIdx = 2;  % mid
            elseif greyCount >= 4
                phaseIdx = 3;  % late
            else
                phaseIdx = 4;  % endgame
            end
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
            %EVALUATETERMINAL Evaluate terminal state using LUT or heuristic
            %
            %   If LUT is loaded, performs O(1) lookup for exact adjudicator score.
            %   Otherwise falls back to heuristic evaluation.
            %
            %   Returns score from THIS player's perspective.

            if this.LUTLoaded
                % O(1) lookup in pre-computed LUT
                idx = this.state2idx(state);
                score = this.TerminalScoreLUT(idx);

                % Adjust for player perspective (LUT stores P1 perspective)
                if this.PlayerPerspective == 2
                    score = -score;
                end
            else
                % Fallback to heuristic
                score = this.evaluateTerminalHeuristic(state);
            end
        end

        function idx = state2idx(~, state)
            %STATE2IDX Convert state string to 1-based LUT index
            %
            %   Index encoding: bit j = 1 means edge j is 'G'
            %   Returns 1-based index for MATLAB array access.

            idx = 1;  % MATLAB is 1-indexed
            for j = 1:15
                if state(j) == 'G'
                    idx = idx + 2^(j-1);
                end
            end
        end

        function state = idx2state(~, idx)
            %IDX2STATE Convert 1-based LUT index to state string
            %
            %   Inverse of state2idx for verification.

            state = repmat('P', 1, 15);
            idx0 = idx - 1;  % Convert to 0-based
            for j = 1:15
                if bitand(idx0, 2^(j-1)) > 0
                    state(j) = 'G';
                end
            end
        end

        function score = evaluateTerminalHeuristic(this, state)
            %EVALUATETERMINALHEURISTIC Heuristic terminal evaluation (fallback)
            %
            %   Calibrated evaluation based on observed game statistics.
            %   Used when LUT is not available.

            score = 0;
            scored = false(1, 15);

            % MY_EDGES scoring - HIGHEST PRIORITY: secure our vertex
            for e = this.MyEdges
                if state(e) == 'G'
                    score = score + 1.2;
                elseif state(e) == 'P'
                    score = score - 1.0;
                end
                scored(e) = true;
            end

            % OPP_EDGES scoring - attacks on opponent's vertex
            for e = this.OppEdges
                if scored(e)
                    continue;
                end

                if state(e) == 'P'
                    if e == 6
                        score = score + 0.25;
                    elseif e == 14
                        score = score + 0.35;
                    else
                        score = score + 0.15;
                    end
                elseif state(e) == 'G'
                    if e == 13
                        score = score - 0.8;
                    else
                        score = score - 0.15;
                    end
                end
                scored(e) = true;
            end

            % HUB_EDGES scoring
            for e = this.HubEdges
                if scored(e)
                    continue;
                end

                if e == 3
                    if state(e) == 'G'
                        score = score - 0.5;
                    elseif state(e) == 'P'
                        score = score + 0.1;
                    end
                else
                    if state(e) == 'G'
                        score = score + 0.2;
                    elseif state(e) == 'P'
                        score = score - 0.15;
                    end
                end
                scored(e) = true;
            end

            % Color balance penalty
            greenCount = sum(state == 'G');
            purpleCount = sum(state == 'P');
            imbalance = abs(greenCount - purpleCount);
            score = score - imbalance * 0.02;
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

        function info = getLUTInfo(this)
            %GETLUTINFO Return information about LUT status
            %
            %   info = getLUTInfo(mcts)
            %
            %   Returns struct with:
            %     - loaded: Whether LUT is loaded
            %     - path: Path to LUT file (if loaded)
            %     - numEntries: Number of entries in LUT
            %     - minScore: Minimum score in LUT
            %     - maxScore: Maximum score in LUT

            info = struct();
            info.loaded = this.LUTLoaded;

            if this.LUTLoaded
                info.path = this.LUTPath;
                info.numEntries = length(this.TerminalScoreLUT);
                info.minScore = min(this.TerminalScoreLUT);
                info.maxScore = max(this.TerminalScoreLUT);
                info.meanScore = mean(this.TerminalScoreLUT);
            else
                info.path = '';
                info.numEntries = 0;
                info.minScore = NaN;
                info.maxScore = NaN;
                info.meanScore = NaN;
            end
        end
    end

    methods (Static)
        function score = evaluateTerminalFull(state)
            %EVALUATETERMINALFULL Full evaluation using LUT
            %
            %   Uses pre-computed terminal scores from SimulatedAnnealingAdjudicator.
            %   Falls back to heuristic if LUT not available.

            mcts = TangledMCTS();
            score = mcts.evaluateTerminal(state);
        end
    end
end
