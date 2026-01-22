classdef MCRollout < handle
%MCROLLOUT Parallel Monte Carlo rollout engine for Tangled game
%
%   Performs parallel Monte Carlo simulations to estimate action values.
%   Uses MATLAB's Parallel Computing Toolbox for multi-threaded execution.
%
%   Example:
%       mc = MCRollout('NumWorkers', 22, 'RolloutsPerAction', 100);
%       values = mc.evaluateActions(state, candidateActions);
%
%   The rollout plays random moves to terminal state and uses heuristic
%   evaluation to estimate the final score.

    properties
        NumWorkers (1,1) int32 = 22          % Parallel workers (threads)
        RolloutsPerAction (1,1) int32 = 50   % Rollouts per candidate action
        MaxDepth (1,1) int32 = 15            % Max moves (Petersen has 15 edges)

        % Strategic edge knowledge for smarter rollouts
        MyEdges = [10, 11, 12]      % E9, E10, E11 (1-indexed)
        OppEdges = [6, 13, 14]      % E5, E12, E13 (1-indexed)
        HubEdges = [3, 11, 13]      % E2, E10, E12 (1-indexed)

        % Pool management
        PoolInitialized logical = false
    end

    methods
        function this = MCRollout(options)
            %MCROLLOUT Construct the rollout engine

            arguments
                options.NumWorkers (1,1) int32 = 22
                options.RolloutsPerAction (1,1) int32 = 50
            end

            this.NumWorkers = options.NumWorkers;
            this.RolloutsPerAction = options.RolloutsPerAction;
        end

        function initPool(this)
            %INITPOOL Initialize parallel pool if needed

            if this.PoolInitialized
                return;
            end

            % Auto-detect max workers
            cluster = parcluster('local');
            maxWorkers = cluster.NumWorkers;
            actualWorkers = min(this.NumWorkers, maxWorkers);

            pool = gcp('nocreate');
            if isempty(pool)
                fprintf('Starting parallel pool with %d workers...\n', actualWorkers);
                parpool('local', actualWorkers);
            elseif pool.NumWorkers < actualWorkers
                fprintf('Resizing pool to %d workers...\n', actualWorkers);
                delete(pool);
                parpool('local', actualWorkers);
            end

            this.NumWorkers = actualWorkers;
            this.PoolInitialized = true;
        end

        function values = evaluateActions(this, state, actions)
            %EVALUATEACTIONS Evaluate candidate actions via parallel rollouts
            %
            %   values = evaluateActions(mc, state, actions)
            %
            %   Inputs:
            %       state   - 15-char board state string
            %       actions - Vector of action indices (1-30)
            %
            %   Outputs:
            %       values  - Estimated value for each action [-1, 1]

            this.initPool();

            numActions = length(actions);
            values = zeros(numActions, 1);
            totalRollouts = this.RolloutsPerAction;

            % Parallel evaluation of all actions
            parfor i = 1:numActions
                action = actions(i);

                % Decode action
                if action <= 15
                    edge = action;
                    color = 'G';
                else
                    edge = action - 15;
                    color = 'P';
                end

                % Skip if invalid action
                if state(edge) ~= '-'
                    values(i) = -1;  % Invalid action penalty
                    continue;
                end

                % Apply our action
                afterState = state;
                afterState(edge) = color;

                % Run rollouts from this state
                totalScore = 0;
                for r = 1:totalRollouts
                    score = MCRollout.singleRollout(afterState);
                    totalScore = totalScore + score;
                end

                values(i) = totalScore / totalRollouts;
            end
        end

        function [values, stats] = evaluateActionsDetailed(this, state, actions)
            %EVALUATEACTIONSDETAILED Detailed evaluation with statistics
            %
            %   [values, stats] = evaluateActionsDetailed(mc, state, actions)
            %
            %   Returns mean values and detailed stats (wins/losses/draws)

            this.initPool();

            numActions = length(actions);
            values = zeros(numActions, 1);
            stats = struct('wins', zeros(numActions,1), ...
                          'losses', zeros(numActions,1), ...
                          'draws', zeros(numActions,1), ...
                          'scores', cell(numActions,1));

            totalRollouts = this.RolloutsPerAction;

            parfor i = 1:numActions
                action = actions(i);

                % Decode action
                if action <= 15
                    edge = action;
                    color = 'G';
                else
                    edge = action - 15;
                    color = 'P';
                end

                % Skip if invalid
                if state(edge) ~= '-'
                    values(i) = -1;
                    continue;
                end

                % Apply our action
                afterState = state;
                afterState(edge) = color;

                % Run rollouts
                scores = zeros(totalRollouts, 1);
                wins = 0; losses = 0; draws = 0;

                for r = 1:totalRollouts
                    score = MCRollout.singleRollout(afterState);
                    scores(r) = score;

                    if score > 0.5
                        wins = wins + 1;
                    elseif score < -0.5
                        losses = losses + 1;
                    else
                        draws = draws + 1;
                    end
                end

                values(i) = mean(scores);
                stats(i).wins = wins;
                stats(i).losses = losses;
                stats(i).draws = draws;
                stats(i).scores = scores;
            end
        end
    end

    methods (Static)
        function lut = getTerminalLUT()
            %GETTERMINALLUT Get the cached terminal score LUT
            %
            %   Uses persistent variable to load LUT once and share across calls.
            %   Thread-safe for parfor since persistent is per-worker.

            persistent cachedLUT;
            persistent lutLoaded;

            if isempty(lutLoaded)
                lutLoaded = false;
            end

            if ~lutLoaded
                % Find and load LUT file
                scriptPath = mfilename('fullpath');
                scriptDir = fileparts(scriptPath);
                lutPath = fullfile(scriptDir, 'data', 'terminal_scores.mat');

                if isfile(lutPath)
                    try
                        data = load(lutPath);
                        if isfield(data, 'terminal_scores') && length(data.terminal_scores) == 32768
                            cachedLUT = double(data.terminal_scores(:));
                            lutLoaded = true;
                        end
                    catch
                        lutLoaded = false;
                    end
                end
            end

            if lutLoaded
                lut = cachedLUT;
            else
                lut = [];
            end
        end

        function idx = state2idx(state)
            %STATE2IDX Convert state string to 1-based LUT index
            idx = 1;
            for j = 1:15
                if state(j) == 'G'
                    idx = idx + 2^(j-1);
                end
            end
        end

        function score = singleRollout(state)
            %SINGLEROLLOUT Execute one random rollout to terminal
            %
            %   score = singleRollout(state)
            %
            %   Plays random moves alternating between players until
            %   all edges are colored, then evaluates terminal state.

            currentState = state;
            isOurTurn = false;  % Opponent moves next (we just played)

            while true
                % Find grey edges
                greyEdges = find(currentState == '-');

                if isempty(greyEdges)
                    % Terminal state - evaluate
                    score = MCRollout.evaluateTerminal(currentState);
                    return;
                end

                % Random move
                edge = greyEdges(randi(length(greyEdges)));

                % Smart color selection based on edge type
                color = MCRollout.smartColorChoice(edge, isOurTurn);

                currentState(edge) = color;
                isOurTurn = ~isOurTurn;
            end
        end

        function color = smartColorChoice(edge, isOurTurn)
            %SMARTCOLORCHOICE Informed color selection for rollouts
            %
            %   Instead of pure random, use domain knowledge:
            %   - Our turn: Green on our edges, Purple on opponent edges
            %   - Opponent turn: Opposite strategy

            % Edge classifications (1-indexed)
            myEdges = [10, 11, 12];    % E9, E10, E11
            oppEdges = [6, 13, 14];    % E5, E12, E13

            if isOurTurn
                if ismember(edge, myEdges)
                    color = 'G';  % Secure our edges
                elseif ismember(edge, oppEdges)
                    color = 'P';  % Attack their edges
                else
                    % Random for neutral edges
                    color = char('G' + (rand() > 0.5) * ('P' - 'G'));
                end
            else
                % Opponent's perspective (reversed)
                if ismember(edge, oppEdges)
                    color = 'G';  % They secure their edges
                elseif ismember(edge, myEdges)
                    color = 'P';  % They attack our edges
                else
                    color = char('G' + (rand() > 0.5) * ('P' - 'G'));
                end
            end
        end

        function score = evaluateTerminal(state)
            %EVALUATETERMINAL Evaluate terminal state using LUT or heuristic
            %
            %   Returns score from Player 1's perspective.
            %   Uses pre-computed LUT if available, otherwise heuristic.

            lut = MCRollout.getTerminalLUT();

            if ~isempty(lut)
                % O(1) lookup in pre-computed LUT
                idx = MCRollout.state2idx(state);
                score = lut(idx);
            else
                % Fallback to heuristic
                score = MCRollout.evaluateTerminalHeuristic(state);
            end
        end

        function score = evaluateTerminalHeuristic(state)
            %EVALUATETERMINALHEURISTIC Heuristic terminal evaluation (fallback)
            %
            %   Returns score in [-1, 1] from our perspective
            %   Positive = good for us, Negative = good for opponent

            % Edge classifications (1-indexed)
            myEdges = [10, 11, 12];    % E9, E10, E11
            oppEdges = [6, 13, 14];    % E5, E12, E13
            hubEdges = [3, 11, 13];    % E2, E10, E12

            score = 0;

            % MY_EDGES scoring
            for e = myEdges
                if state(e) == 'G'
                    score = score + 0.4;   % We secured it
                elseif state(e) == 'P'
                    score = score - 0.35;  % They attacked it
                end
            end

            % OPP_EDGES scoring
            for e = oppEdges
                if state(e) == 'P'
                    score = score + 0.35;  % We attacked it
                elseif state(e) == 'G'
                    score = score - 0.3;   % They secured it
                end
            end

            % HUB_EDGES bonus
            for e = hubEdges
                if state(e) == 'G'
                    score = score + 0.15;
                elseif state(e) == 'P'
                    score = score - 0.1;
                end
            end

            % Clamp to [-1, 1]
            score = max(-1, min(1, score));
        end

        function score = evaluateTerminalFull(state)
            %EVALUATETERMINALFULL Full evaluation using LUT
            %
            %   Uses pre-computed LUT with SimulatedAnnealing adjudicator scores.
            %   Falls back to heuristic if LUT not available.

            score = MCRollout.evaluateTerminal(state);
        end

        function loaded = isLUTLoaded()
            %ISLUTLOADED Check if the terminal score LUT is loaded
            lut = MCRollout.getTerminalLUT();
            loaded = ~isempty(lut);
        end
    end
end
