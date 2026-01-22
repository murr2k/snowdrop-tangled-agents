classdef EnsembleSelfPlayOpponent < handle
%ENSEMBLESELFPLAYOPPONENT Self-play opponent using ensemble policy
%
%   This opponent uses the EnsemblePolicy (RL + MC) for move selection,
%   providing a stronger training signal than pure RL self-play.
%
%   The ensemble opponent is periodically updated to match the current
%   training agent, creating a curriculum of increasingly strong opponents.
%
%   Example:
%       ensemble = EnsemblePolicy(agent);
%       opp = EnsembleSelfPlayOpponent(ensemble);
%       env = TangledEnvironment('Opponent', opp);
%       train(agent, env, opts);

    properties
        Ensemble EnsemblePolicy            % The ensemble policy
        Epsilon double = 0.1               % Exploration rate
        UseFullEnsemble logical = true     % Use MC search or just RL
    end

    methods
        function this = EnsembleSelfPlayOpponent(ensemble, options)
            %ENSEMBLESELFPLAYOPPONENT Construct the opponent

            arguments
                ensemble EnsemblePolicy
                options.Epsilon double = 0.1
                options.UseFullEnsemble logical = true
            end

            this.Ensemble = ensemble;
            this.Epsilon = options.Epsilon;
            this.UseFullEnsemble = options.UseFullEnsemble;
        end

        function updateEnsemble(this, ensemble)
            %UPDATEENSEMBLE Update the ensemble (for periodic sync)

            this.Ensemble = ensemble;
        end

        function move = selectMove(this, state)
            %SELECTMOVE Select move using ensemble policy
            %
            %   move = selectMove(opp, state)
            %
            %   Returns struct with .edge (0-indexed) and .color

            % Epsilon-greedy exploration
            if rand() < this.Epsilon
                move = this.randomMove(state);
                return;
            end

            % Use ensemble for selection
            if this.UseFullEnsemble
                action = this.Ensemble.selectAction(state);
            else
                % Just use RL priors (faster)
                priors = this.Ensemble.getRLPriors(state);
                validMask = this.getValidMask(state);
                maskedPriors = priors .* validMask;

                if sum(maskedPriors) == 0
                    move = this.randomMove(state);
                    return;
                end

                maskedPriors = maskedPriors / sum(maskedPriors);

                % Sample from distribution
                cumProbs = cumsum(maskedPriors);
                action = find(cumProbs >= rand(), 1);
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
        function move = randomMove(~, state)
            %RANDOMMOVE Random valid move selection

            greyEdges = find(state == '-') - 1;  % 0-indexed

            if isempty(greyEdges)
                move = struct('edge', -1, 'color', '-');
                return;
            end

            move.edge = greyEdges(randi(length(greyEdges)));
            move.color = char('G' + (rand() > 0.5) * ('P' - 'G'));
        end

        function mask = getValidMask(~, state)
            %GETVALIDMASK Get mask of valid actions

            mask = zeros(30, 1);
            for i = 1:15
                if state(i) == '-'
                    mask(i) = 1;
                    mask(i+15) = 1;
                end
            end
        end
    end
end
