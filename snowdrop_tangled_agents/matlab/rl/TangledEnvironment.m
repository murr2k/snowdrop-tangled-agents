classdef TangledEnvironment < rl.env.MATLABEnvironment
%TANGLEDENVIRONMENT RL environment for the Tangled quantum game
%
%   This environment wraps the Tangled game for use with MATLAB's
%   Reinforcement Learning Toolbox. It implements the standard RL
%   interface: reset(), step(action), and observation/action specs.
%
%   Observation: 50-element feature vector
%   Action: Discrete 1-30 (15 edges x 2 colors)
%
%   Reward Shaping (v2):
%   - Bonus for securing MY_EDGES (E9, E10, E11) with Green
%   - Bonus for attacking OPP_EDGES (E5, E12, E13) with Purple
%   - Urgency bonus for early strategic moves
%   - Penalty for invalid actions
%   - Terminal reward based on final score
%
%   Example:
%       env = TangledEnvironment();
%       obs = reset(env);
%       [nextObs, reward, isDone, info] = step(env, action);

    properties
        % Current game state
        State char = repmat('-', 1, 15)  % 15-char board: G/P/-

        % Game tracking
        Score double = 0                  % Current position score
        MoveCount int32 = 0               % Moves made this game
        MaxMoves int32 = 15               % Petersen graph has 15 edges
        OurMoveCount int32 = 0            % Our moves only

        % Components
        Opponent                          % Opponent policy (SimulatedOpponent)

        % Configuration
        UseShapingReward logical = true   % Use intermediate rewards
        InvalidActionPenalty double = -0.5  % Penalty for invalid action
        AutoCorrectInvalidActions logical = false  % Don't auto-correct - teach valid moves

        % Strategic edge sets (1-indexed for MATLAB)
        MyEdges = [10, 11, 12]    % E9, E10, E11 - edges connected to our vertex
        OppEdges = [6, 13, 14]    % E5, E12, E13 - edges connected to opponent vertex
        HubEdges = [3, 11, 13]    % E2, E10, E12 - high-connectivity edges
    end

    properties (Access = private)
        % Cache for adjudicator (expensive to create)
        AdjudicatorCache

        % Track previous state for opponent move detection
        PrevState char
    end

    methods
        function this = TangledEnvironment(options)
            %TANGLEDENVIRONMENT Construct the environment
            %
            %   env = TangledEnvironment()
            %   env = TangledEnvironment('Opponent', opponent)

            arguments
                options.Opponent = []
                options.AutoCorrect logical = false
            end

            % Define observation specification
            % 50-element feature vector in range [-1, 1]
            obsInfo = rlNumericSpec([50 1], ...
                'LowerLimit', -1, ...
                'UpperLimit', 1, ...
                'Name', 'TangledObservation', ...
                'Description', 'Board state, turn, categories, momentum, phase');

            % Define action specification
            % Discrete: 1-15 = Green on edge 0-14, 16-30 = Purple on edge 0-14
            actInfo = rlFiniteSetSpec(1:30);
            actInfo.Name = 'TangledAction';
            actInfo.Description = 'Edge and color selection';

            % Call superclass constructor
            this = this@rl.env.MATLABEnvironment(obsInfo, actInfo);

            % Set opponent
            if isempty(options.Opponent)
                this.Opponent = SimulatedOpponent('Style', 'mcts');
            else
                this.Opponent = options.Opponent;
            end

            this.AutoCorrectInvalidActions = options.AutoCorrect;
        end

        function [observation, reward, isDone, info] = step(this, action)
            %STEP Execute one step in the environment
            %
            %   [obs, reward, isDone, info] = step(env, action)
            %
            %   action: 1-30 integer
            %     1-15:  Play Green on edge (action-1)
            %     16-30: Play Purple on edge (action-16)

            info = struct();
            info.InvalidAction = false;
            info.ShapingReward = 0;
            reward = 0;

            % Store previous state
            this.PrevState = this.State;

            % Decode action to edge and color
            if action <= 15
                edge = action;  % 1-indexed edge (1-15)
                color = 'G';
            else
                edge = action - 15;
                color = 'P';
            end

            % Validate action (must be grey edge)
            if this.State(edge) ~= '-'
                if this.AutoCorrectInvalidActions
                    % Remap to a random valid action
                    greyEdges = find(this.State == '-');
                    if isempty(greyEdges)
                        % No valid moves - game should be over
                        reward = 0;
                        observation = this.getObservation();
                        isDone = true;
                        return;
                    end
                    edge = greyEdges(randi(length(greyEdges)));
                    info.RemappedAction = true;
                else
                    % Invalid action - penalize and return
                    reward = this.InvalidActionPenalty;
                    observation = this.getObservation();
                    isDone = false;
                    info.InvalidAction = true;
                    info.Message = sprintf('Edge %d already colored', edge-1);
                    return;
                end
            end

            % Apply our move
            this.State(edge) = color;
            this.MoveCount = this.MoveCount + 1;
            this.OurMoveCount = this.OurMoveCount + 1;

            % Calculate shaping reward for our move
            if this.UseShapingReward
                shapingReward = this.calculateShapingReward(edge, color);
                info.ShapingReward = shapingReward;
                reward = reward + shapingReward;
            end

            % Check if game over (all edges colored)
            greyCount = sum(this.State == '-');

            if greyCount == 0
                % Terminal state - evaluate final score
                finalScore = this.evaluateTerminal();
                terminalReward = this.calculateTerminalReward(finalScore);
                reward = reward + terminalReward;
                isDone = true;
                info.FinalScore = finalScore;
                info.TerminalReward = terminalReward;
                info.Result = this.getResult(finalScore);
            else
                % Opponent's turn
                oppMove = this.Opponent.selectMove(this.State);

                if oppMove.edge >= 0 && oppMove.edge < 15
                    oppEdge = oppMove.edge + 1;  % Convert 0-indexed to 1-indexed
                    if this.State(oppEdge) == '-'
                        this.State(oppEdge) = oppMove.color;
                        this.MoveCount = this.MoveCount + 1;

                        % Penalty if opponent took one of our strategic edges
                        oppPenalty = this.calculateOpponentPenalty(oppEdge, oppMove.color);
                        reward = reward + oppPenalty;
                        info.OpponentPenalty = oppPenalty;
                    end
                end

                % Check again for terminal
                greyCount = sum(this.State == '-');

                if greyCount == 0
                    % Terminal after opponent move
                    finalScore = this.evaluateTerminal();
                    terminalReward = this.calculateTerminalReward(finalScore);
                    reward = reward + terminalReward;
                    isDone = true;
                    info.FinalScore = finalScore;
                    info.TerminalReward = terminalReward;
                    info.Result = this.getResult(finalScore);
                else
                    isDone = false;
                end
            end

            observation = this.getObservation();
            info.State = this.State;
            info.MoveCount = this.MoveCount;
            info.OurMoveCount = this.OurMoveCount;
        end

        function observation = reset(this)
            %RESET Reset environment to initial state
            %
            %   obs = reset(env)

            this.State = repmat('-', 1, 15);
            this.PrevState = this.State;
            this.Score = 0;
            this.MoveCount = 0;
            this.OurMoveCount = 0;
            observation = this.getObservation();
        end

        function observation = getObservation(this)
            %GETOBSERVATION Build 50-element observation vector

            observation = buildRLFeatures(this.State, 1, this.Score);
        end

        function mask = getActionMask(this)
            %GETACTIONMASK Get valid action mask
            %
            %   mask = getActionMask(env)
            %   mask(i) = 1 if action i is valid, 0 otherwise

            mask = getActionMask(this.State);
        end
    end

    methods (Access = private)
        function reward = calculateShapingReward(this, edge, color)
            %CALCULATESHAPINGREWARD Compute immediate reward for a move
            %
            %   Rewards strategic moves:
            %   - Green on MY_EDGES: +0.15 (securing our territory)
            %   - Purple on OPP_EDGES: +0.10 (attacking opponent)
            %   - Green on HUB_EDGES: +0.05 (controlling connectivity)
            %   - Urgency bonus for early strategic moves

            reward = 0;

            % Base strategic rewards
            if color == 'G' && ismember(edge, this.MyEdges)
                % Securing our edges with Green - most important
                reward = reward + 0.15;

                % Urgency bonus: higher reward for doing this early
                if this.OurMoveCount <= 3
                    reward = reward + 0.10;  % Extra bonus in first 3 moves
                end

            elseif color == 'P' && ismember(edge, this.OppEdges)
                % Attacking opponent edges with Purple
                reward = reward + 0.10;

            elseif color == 'G' && ismember(edge, this.HubEdges)
                % Controlling hub edges with Green
                reward = reward + 0.05;
            end

            % Small penalty for playing Purple on our own edges (bad)
            if color == 'P' && ismember(edge, this.MyEdges)
                reward = reward - 0.10;
            end

            % Small penalty for playing Green on opponent edges (helps them)
            if color == 'G' && ismember(edge, this.OppEdges)
                reward = reward - 0.05;
            end
        end

        function penalty = calculateOpponentPenalty(this, oppEdge, oppColor)
            %CALCULATEOPPONENTPENALTY Penalty when opponent takes strategic edges
            %
            %   If opponent secures their edges or attacks ours, we get penalized

            penalty = 0;

            % Opponent took one of OUR edges before we could
            if ismember(oppEdge, this.MyEdges)
                if oppColor == 'P'
                    % They attacked our edge with Purple - very bad
                    penalty = -0.15;
                else
                    % They took our edge with Green - moderately bad
                    penalty = -0.08;
                end
            end

            % Opponent secured their own edges
            if ismember(oppEdge, this.OppEdges) && oppColor == 'G'
                penalty = penalty - 0.05;
            end
        end

        function reward = calculateTerminalReward(~, finalScore)
            %CALCULATETERMINALREWARD Convert final score to reward
            %
            %   Uses tanh scaling with bonus for decisive wins

            % Base reward from score
            reward = tanh(finalScore / 2);  % More sensitive than /3

            % Bonus for decisive outcomes
            if finalScore > 2
                reward = reward + 0.2;  % Bonus for big win
            elseif finalScore < -2
                reward = reward - 0.2;  % Extra penalty for big loss
            end
        end

        function score = evaluateTerminal(this)
            %EVALUATETERMINAL Evaluate terminal state using adjudicator

            % Use simulated annealing adjudicator
            try
                if isempty(this.AdjudicatorCache)
                    score = this.heuristicEval();
                else
                    score = this.AdjudicatorCache.evaluate(this.State);
                end
            catch
                score = this.heuristicEval();
            end
        end

        function score = heuristicEval(this)
            %HEURISTICEVAL Heuristic position evaluation
            %
            %   More sophisticated evaluation based on edge ownership

            score = 0;

            % MY_EDGES scoring (E9, E10, E11)
            for e = this.MyEdges
                if this.State(e) == 'G'
                    score = score + 0.5;   % We secured it
                elseif this.State(e) == 'P'
                    score = score - 0.4;   % Opponent attacked it
                end
            end

            % OPP_EDGES scoring (E5, E12, E13)
            for e = this.OppEdges
                if this.State(e) == 'P'
                    score = score + 0.4;   % We attacked it
                elseif this.State(e) == 'G'
                    score = score - 0.3;   % Opponent secured it
                end
            end

            % HUB_EDGES scoring (E2, E10, E12)
            for e = this.HubEdges
                if this.State(e) == 'G'
                    score = score + 0.2;
                end
            end

            % Color balance consideration
            greenCount = sum(this.State == 'G');
            purpleCount = sum(this.State == 'P');
            greyCount = sum(this.State == '-');

            % In terminal state, slight preference for balanced colors
            if greyCount == 0
                imbalance = abs(greenCount - purpleCount);
                score = score - imbalance * 0.03;
            end
        end

        function result = getResult(~, score)
            %GETRESULT Convert score to result string

            if score > 0.5
                result = 'win';
            elseif score < -0.5
                result = 'loss';
            else
                result = 'draw';
            end
        end
    end
end
