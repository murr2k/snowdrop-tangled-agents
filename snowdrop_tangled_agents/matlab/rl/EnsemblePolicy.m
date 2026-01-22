classdef EnsemblePolicy < handle
%ENSEMBLEPOLICY Ensemble policy combining RL neural network with MC search
%
%   This implements an AlphaZero-style ensemble:
%   1. RL policy provides prior probabilities P(a) for all actions
%   2. MC rollouts evaluate top-K candidate actions
%   3. Final selection combines RL priors with MC value estimates
%
%   The combination uses: score(a) = P(a)^alpha * softmax(V(a))^beta
%   where alpha and beta control the RL vs MC weighting.
%
%   Example:
%       ensemble = EnsemblePolicy(agent, 'TopK', 5, 'RolloutsPerAction', 100);
%       action = ensemble.selectAction(state);
%
%   For training, the ensemble can generate improved targets for the RL policy.

    properties
        Agent                              % The PPO agent
        MCEngine MCRollout                 % Monte Carlo rollout engine

        % Ensemble parameters
        TopK (1,1) int32 = 5               % Number of top candidates to evaluate
        RolloutsPerAction (1,1) int32 = 50 % MC rollouts per candidate
        NumWorkers (1,1) int32 = 22        % Parallel workers

        % Combination weights
        AlphaPrior (1,1) double = 0.5      % Weight for RL prior (0-1)
        BetaMC (1,1) double = 0.5          % Weight for MC value (0-1)
        Temperature (1,1) double = 1.0     % Softmax temperature

        % Caching
        ActorNetwork                       % Cached actor network
        LastState char = ''
        LastPriors double
    end

    methods
        function this = EnsemblePolicy(agent, options)
            %ENSEMBLEPOLICY Construct the ensemble policy

            arguments
                agent
                options.TopK (1,1) int32 = 5
                options.RolloutsPerAction (1,1) int32 = 50
                options.NumWorkers (1,1) int32 = 22
                options.AlphaPrior (1,1) double = 0.5
                options.BetaMC (1,1) double = 0.5
            end

            this.Agent = agent;
            this.TopK = options.TopK;
            this.RolloutsPerAction = options.RolloutsPerAction;
            this.NumWorkers = options.NumWorkers;
            this.AlphaPrior = options.AlphaPrior;
            this.BetaMC = options.BetaMC;

            % Initialize MC engine
            this.MCEngine = MCRollout('NumWorkers', this.NumWorkers, ...
                                      'RolloutsPerAction', this.RolloutsPerAction);

            % Cache actor network for fast inference
            this.cacheActorNetwork();
        end

        function updateAgent(this, agent)
            %UPDATEAGENT Update the underlying agent (e.g., during training)

            this.Agent = agent;
            this.cacheActorNetwork();
            this.LastState = '';  % Invalidate cache
        end

        function action = selectAction(this, state)
            %SELECTACTION Select action using ensemble (RL + MC)
            %
            %   action = selectAction(ensemble, state)
            %
            %   Returns 1-indexed action (1-30)

            % Get RL prior probabilities
            priors = this.getRLPriors(state);

            % Get valid actions (grey edges)
            validMask = this.getValidMask(state);
            maskedPriors = priors .* validMask;

            % Handle case where no valid actions
            if sum(maskedPriors) == 0
                validActions = find(validMask > 0);
                if isempty(validActions)
                    action = 1;  % Fallback
                    return;
                end
                action = validActions(randi(length(validActions)));
                return;
            end

            % Normalize masked priors
            maskedPriors = maskedPriors / sum(maskedPriors);

            % Select top-K candidates
            [~, sortIdx] = sort(maskedPriors, 'descend');
            validActions = find(validMask > 0);
            numCandidates = min(this.TopK, length(validActions));

            candidates = [];
            for i = 1:length(sortIdx)
                if validMask(sortIdx(i)) > 0
                    candidates = [candidates; sortIdx(i)];
                    if length(candidates) >= numCandidates
                        break;
                    end
                end
            end

            % If only one candidate, return it
            if length(candidates) == 1
                action = candidates(1);
                return;
            end

            % Run MC rollouts on candidates
            mcValues = this.MCEngine.evaluateActions(state, candidates);

            % Combine RL priors with MC values
            candidatePriors = maskedPriors(candidates);

            % Softmax on MC values
            mcProbs = this.softmax(mcValues / this.Temperature);

            % Weighted combination
            combinedScores = (candidatePriors .^ this.AlphaPrior) .* (mcProbs .^ this.BetaMC);
            combinedScores = combinedScores / sum(combinedScores);

            % Select best action
            [~, bestIdx] = max(combinedScores);
            action = candidates(bestIdx);
        end

        function [action, info] = selectActionDetailed(this, state)
            %SELECTACTIONDETAILED Select action with detailed diagnostics
            %
            %   [action, info] = selectActionDetailed(ensemble, state)

            info = struct();

            % Get RL priors
            priors = this.getRLPriors(state);
            validMask = this.getValidMask(state);
            maskedPriors = priors .* validMask;

            if sum(maskedPriors) == 0
                validActions = find(validMask > 0);
                action = validActions(randi(length(validActions)));
                info.method = 'random_fallback';
                return;
            end

            maskedPriors = maskedPriors / sum(maskedPriors);

            % Top-K candidates
            [~, sortIdx] = sort(maskedPriors, 'descend');
            candidates = [];
            for i = 1:length(sortIdx)
                if validMask(sortIdx(i)) > 0
                    candidates = [candidates; sortIdx(i)];
                    if length(candidates) >= this.TopK
                        break;
                    end
                end
            end

            info.candidates = candidates;
            info.candidatePriors = maskedPriors(candidates);

            % MC evaluation
            [mcValues, mcStats] = this.MCEngine.evaluateActionsDetailed(state, candidates);
            info.mcValues = mcValues;
            info.mcStats = mcStats;

            % Combine
            candidatePriors = maskedPriors(candidates);
            mcProbs = this.softmax(mcValues / this.Temperature);
            combinedScores = (candidatePriors .^ this.AlphaPrior) .* (mcProbs .^ this.BetaMC);
            combinedScores = combinedScores / sum(combinedScores);

            info.mcProbs = mcProbs;
            info.combinedScores = combinedScores;

            [~, bestIdx] = max(combinedScores);
            action = candidates(bestIdx);
            info.selectedIdx = bestIdx;
            info.method = 'ensemble';
        end

        function priors = getRLPriors(this, state)
            %GETRLPRIORS Get RL policy prior probabilities

            % Check cache
            if strcmp(this.LastState, state)
                priors = this.LastPriors;
                return;
            end

            % Build observation
            obs = buildRLFeatures(state, 1, 0);

            % Forward pass through actor
            dlObs = dlarray(obs, 'CB');
            dlProbs = forward(this.ActorNetwork, dlObs);
            priors = extractdata(dlProbs);
            priors = priors(:);  % Column vector

            % Cache
            this.LastState = state;
            this.LastPriors = priors;
        end

        function targets = generateImprovedTargets(this, state)
            %GENERATEIMPROVEDTARGETS Generate improved policy targets from ensemble
            %
            %   targets = generateImprovedTargets(ensemble, state)
            %
            %   Returns 30-element probability vector that can be used as
            %   training target (policy distillation from ensemble).

            % Get valid mask
            validMask = this.getValidMask(state);
            validActions = find(validMask > 0);

            if isempty(validActions)
                targets = zeros(30, 1);
                return;
            end

            % Evaluate ALL valid actions with MC (for best targets)
            mcValues = this.MCEngine.evaluateActions(state, validActions);

            % Create target distribution
            targets = zeros(30, 1);
            mcProbs = this.softmax(mcValues / this.Temperature);
            targets(validActions) = mcProbs;
        end
    end

    methods (Access = private)
        function cacheActorNetwork(this)
            %CACHEACTORNETWORK Cache the actor network for fast inference

            actor = getActor(this.Agent);
            this.ActorNetwork = getModel(actor);
        end

        function mask = getValidMask(~, state)
            %GETVALIDMASK Get mask of valid actions (grey edges only)

            mask = zeros(30, 1);
            for i = 1:15
                if state(i) == '-'
                    mask(i) = 1;      % Green on edge i
                    mask(i+15) = 1;   % Purple on edge i
                end
            end
        end

        function p = softmax(~, x)
            %SOFTMAX Compute softmax probabilities

            x = x - max(x);  % Numerical stability
            expX = exp(x);
            p = expX / sum(expX);
        end
    end
end
