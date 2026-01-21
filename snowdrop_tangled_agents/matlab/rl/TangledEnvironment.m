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

        % Components
        Opponent                          % Opponent policy (SimulatedOpponent)

        % Configuration
        UseShapingReward logical = true   % Use intermediate rewards
        InvalidActionPenalty double = -1.0
    end

    properties (Access = private)
        % Cache for adjudicator (expensive to create)
        AdjudicatorCache
    end

    methods
        function this = TangledEnvironment(options)
            %TANGLEDENVIRONMENT Construct the environment
            %
            %   env = TangledEnvironment()
            %   env = TangledEnvironment('Opponent', opponent)

            arguments
                options.Opponent = []
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
                % Invalid action - edge already colored
                reward = this.InvalidActionPenalty;
                observation = this.getObservation();
                isDone = false;
                info.InvalidAction = true;
                info.Message = sprintf('Edge %d already colored', edge-1);
                return;
            end

            % Apply our move
            this.State(edge) = color;
            this.MoveCount = this.MoveCount + 1;

            % Check if game over (all edges colored)
            greyCount = sum(this.State == '-');

            if greyCount == 0
                % Terminal state - evaluate final score
                finalScore = this.evaluateTerminal();
                reward = tanh(finalScore / 3);  % Normalize to [-1, 1]
                isDone = true;
                info.FinalScore = finalScore;
                info.Result = this.getResult(finalScore);
            else
                % Opponent's turn
                oppMove = this.Opponent.selectMove(this.State);
                this.State(oppMove.edge + 1) = oppMove.color;  % Convert 0-indexed to 1-indexed
                this.MoveCount = this.MoveCount + 1;

                % Check again for terminal
                greyCount = sum(this.State == '-');

                if greyCount == 0
                    % Terminal after opponent move
                    finalScore = this.evaluateTerminal();
                    reward = tanh(finalScore / 3);
                    isDone = true;
                    info.FinalScore = finalScore;
                    info.Result = this.getResult(finalScore);
                else
                    % Game continues - compute shaping reward
                    if this.UseShapingReward
                        newScore = this.evaluatePosition();
                        reward = (newScore - this.Score) * 0.1;
                        this.Score = newScore;
                    else
                        reward = 0;
                    end
                    isDone = false;
                end
            end

            observation = this.getObservation();
            info.State = this.State;
            info.MoveCount = this.MoveCount;
        end

        function observation = reset(this)
            %RESET Reset environment to initial state
            %
            %   obs = reset(env)

            this.State = repmat('-', 1, 15);
            this.Score = 0;
            this.MoveCount = 0;
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
        function score = evaluateTerminal(this)
            %EVALUATETERMINAL Evaluate terminal state using adjudicator

            % Use simulated annealing adjudicator
            % This matches the official tangled-game.com scoring
            try
                if isempty(this.AdjudicatorCache)
                    % Would need Python bridge or pure MATLAB implementation
                    % For now, use heuristic approximation
                    score = this.heuristicEval();
                else
                    score = this.AdjudicatorCache.evaluate(this.State);
                end
            catch
                score = this.heuristicEval();
            end
        end

        function score = evaluatePosition(this)
            %EVALUATEPOSITION Quick position evaluation for shaping reward

            score = this.heuristicEval();
        end

        function score = heuristicEval(this)
            %HEURISTICEVAL Heuristic position evaluation
            %
            %   Simple evaluation based on edge ownership patterns

            % Count colors
            greenCount = sum(this.State == 'G');
            purpleCount = sum(this.State == 'P');

            % Good edges for each color (from empirical analysis)
            goodGreen = [10, 11, 12];  % E9, E10, E11 (1-indexed)
            goodPurple = [6, 13, 14];  % E5, E12, E13 (1-indexed)

            score = 0;

            % Bonus for good green edges
            for e = goodGreen
                if this.State(e) == 'G'
                    score = score + 0.3;
                elseif this.State(e) == 'P'
                    score = score - 0.2;
                end
            end

            % Bonus for good purple edges
            for e = goodPurple
                if this.State(e) == 'P'
                    score = score + 0.3;
                elseif this.State(e) == 'G'
                    score = score - 0.2;
                end
            end

            % Slight penalty for color imbalance
            imbalance = abs(greenCount - purpleCount);
            score = score - imbalance * 0.05;
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
