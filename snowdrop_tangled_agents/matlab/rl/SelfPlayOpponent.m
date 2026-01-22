classdef SelfPlayOpponent < handle
%SELFPLAYOPPONENT Opponent that uses the same agent for self-play
%
%   This opponent wrapper uses a copy of the training agent to select
%   moves, enabling self-play training where both sides improve.
%
%   Example:
%       agent = createPPOAgent(env);
%       opp = SelfPlayOpponent(agent);
%       env = TangledEnvironment('Opponent', opp);
%       train(agent, env, opts);  % Both sides use same policy

    properties
        Agent           % The PPO agent to use for move selection
        ActorNetwork    % Cached actor network for fast inference
        Epsilon double = 0.1  % Exploration rate for opponent
    end

    methods
        function this = SelfPlayOpponent(agent, options)
            arguments
                agent
                options.Epsilon double = 0.1
            end

            this.Agent = agent;
            this.Epsilon = options.Epsilon;

            % Cache actor network for faster inference
            actor = getActor(agent);
            this.ActorNetwork = getModel(actor);
        end

        function updateAgent(this, agent)
            %UPDATEAGENT Update the opponent's agent (for periodic sync)
            this.Agent = agent;
            actor = getActor(agent);
            this.ActorNetwork = getModel(actor);
        end

        function move = selectMove(this, state)
            %SELECTMOVE Select move using the agent's policy
            %
            %   move = selectMove(opp, state)
            %
            %   Uses the actor network to get action probabilities,
            %   then samples an action (with optional exploration).

            % Convert state to observation
            obs = this.stateToObservation(state);

            % Get action probabilities from actor
            dlObs = dlarray(obs, 'CB');
            probs = forward(this.ActorNetwork, dlObs);
            probs = extractdata(probs);

            % Build action mask (only grey edges are valid)
            mask = zeros(30, 1);
            for i = 1:15
                if state(i) == '-'
                    mask(i) = 1;      % Green on edge i-1
                    mask(i+15) = 1;   % Purple on edge i-1
                end
            end

            % Apply mask
            maskedProbs = probs(:) .* mask;

            % Epsilon-greedy exploration
            if rand() < this.Epsilon
                % Random valid action
                validActions = find(mask > 0);
                if isempty(validActions)
                    move = struct('edge', -1, 'color', '-');
                    return;
                end
                action = validActions(randi(length(validActions)));
            else
                % Sample from masked probabilities
                if sum(maskedProbs) > 0
                    maskedProbs = maskedProbs / sum(maskedProbs);
                    cumProbs = cumsum(maskedProbs);
                    action = find(cumProbs >= rand(), 1);
                else
                    validActions = find(mask > 0);
                    if isempty(validActions)
                        move = struct('edge', -1, 'color', '-');
                        return;
                    end
                    action = validActions(randi(length(validActions)));
                end
            end

            % Decode action to edge and color
            if action <= 15
                edge = action - 1;  % 0-indexed
                color = 'G';
            else
                edge = action - 16;  % 0-indexed
                color = 'P';
            end

            move = struct('edge', edge, 'color', color);
        end
    end

    methods (Access = private)
        function obs = stateToObservation(~, state)
            %STATETOOBSERVATION Convert state string to 50-dim observation
            %
            %   Same feature encoding as TangledEnvironment.getObservation()

            obs = zeros(50, 1);

            % [1:15] Board state
            for i = 1:15
                if state(i) == 'G'
                    obs(i) = 1;
                elseif state(i) == 'P'
                    obs(i) = -1;
                else
                    obs(i) = 0;
                end
            end

            % [16] Turn indicator (opponent's turn = -1 from their view)
            obs(16) = -1;

            % [17:31] Edge categories
            myEdges = [10, 11, 12];   % 1-indexed
            oppEdges = [6, 13, 14];
            hubEdges = [3, 11, 13];

            for i = 1:15
                if ismember(i, myEdges)
                    obs(16+i) = 0.5;
                elseif ismember(i, oppEdges)
                    obs(16+i) = -0.5;
                elseif ismember(i, hubEdges)
                    obs(16+i) = 0.25;
                else
                    obs(16+i) = 0;
                end
            end

            % [32] Grey count
            greyCount = sum(state == '-');
            obs(32) = greyCount / 15;

            % [33:35] Score momentum (placeholder)
            obs(33:35) = 0;

            % [36:50] Game phase
            if greyCount > 10
                obs(36:40) = 1;
            elseif greyCount >= 5
                obs(41:45) = 1;
            else
                obs(46:50) = 1;
            end
        end
    end
end
