classdef MCTSNode < handle
%MCTSNODE Node in Monte Carlo Tree Search tree
%
%   Represents a game state in the MCTS tree with UCB1 selection
%   and Progressive Bias for exploration guidance.
%
%   Properties:
%       State       - 15-char board state string
%       IsOurTurn   - True if it's our turn at this node
%       Parent      - Parent node (empty for root)
%       Action      - (edge, color) that led to this node
%       Prior       - Heuristic prior for this action [0,1]
%       Children    - Map of action -> child node
%       Visits      - Number of times this node was visited
%       TotalValue  - Sum of backpropagated values
%
%   Example:
%       root = MCTSNode('---------------', true);
%       child = root.expand();
%       root.update(0.5);

    properties
        State char
        IsOurTurn logical
        Parent MCTSNode
        Action cell = {}  % {edge, color}
        Prior double = 0.5
        Children containers.Map
        Visits int32 = 0
        TotalValue double = 0
        UntriedActions cell = {}
        ActionPriors containers.Map
    end

    properties (Constant)
        % Edge classifications (1-indexed for MATLAB)
        MY_EDGES = [10, 11, 12]     % E9, E10, E11 - touch vertex 5
        OPP_EDGES = [6, 13, 14]     % E5, E12, E13 - touch vertex 7
        HUB_EDGES = [3, 11, 13]     % E2, E10, E12 - touch vertex 6

        % Empirically good/bad purple edges
        GOOD_PURPLE = [1, 2, 4, 6, 13, 14]  % Inner + opponent edges
        BAD_PURPLE = [3, 5, 7, 8, 9, 15]    % These often backfire
    end

    methods
        function this = MCTSNode(state, isOurTurn, parent, action, prior)
            %MCTSNODE Construct a tree node
            %
            %   node = MCTSNode(state, isOurTurn)
            %   node = MCTSNode(state, isOurTurn, parent, action, prior)

            this.State = state;
            this.IsOurTurn = isOurTurn;
            this.Children = containers.Map('KeyType', 'char', 'ValueType', 'any');
            this.ActionPriors = containers.Map('KeyType', 'char', 'ValueType', 'double');

            if nargin >= 3 && ~isempty(parent)
                this.Parent = parent;
            end
            if nargin >= 4 && ~isempty(action)
                this.Action = action;
            end
            if nargin >= 5
                this.Prior = prior;
            end

            % Initialize untried actions with priorities
            this.UntriedActions = this.getPrioritizedActions();
        end

        function actions = getPrioritizedActions(this)
            %GETPRIORITIZEDACTIONS Get legal actions sorted by heuristic prior

            actionList = {};
            priorList = [];

            for i = 1:15
                if this.State(i) == '-'
                    for c = {'G', 'P'}
                        color = c{1};
                        prior = this.computeActionPrior(i, color);
                        actionList{end+1} = {i, color}; %#ok<AGROW>
                        priorList(end+1) = prior; %#ok<AGROW>

                        % Store prior for later use
                        key = sprintf('%d_%s', i, color);
                        this.ActionPriors(key) = prior;
                    end
                end
            end

            % Sort by prior descending
            [~, sortIdx] = sort(priorList, 'descend');
            actions = actionList(sortIdx);
        end

        function prior = computeActionPrior(this, edge, color)
            %COMPUTEACTIONPRIOR Compute heuristic prior for an action
            %
            %   prior = computeActionPrior(node, edge, color)
            %   Returns value in [0, 1]

            prior = 0.5;  % Base prior

            if this.IsOurTurn
                % OUR TURN
                if ismember(edge, this.MY_EDGES)
                    if color == 'G'
                        prior = 0.99;  % Always Green on our edges
                    else
                        prior = 0.01;
                    end
                elseif ismember(edge, this.OPP_EDGES)
                    if color == 'P'
                        prior = 0.95;  % Purple on opponent edges
                    else
                        prior = 0.05;
                    end
                elseif ismember(edge, this.GOOD_PURPLE) && color == 'P'
                    prior = 0.80;
                elseif ismember(edge, this.BAD_PURPLE) && color == 'P'
                    prior = 0.10;
                elseif ismember(edge, this.BAD_PURPLE) && color == 'G'
                    prior = 0.90;
                elseif ismember(edge, this.HUB_EDGES)
                    prior = 0.70 * (color == 'G') + 0.30 * (color == 'P');
                else
                    prior = 0.60 * (color == 'G') + 0.40 * (color == 'P');
                end
            else
                % OPPONENT'S TURN
                if ismember(edge, this.OPP_EDGES)
                    if color == 'G'
                        prior = 0.95;  % They secure their edges
                    else
                        prior = 0.05;
                    end
                elseif ismember(edge, this.MY_EDGES)
                    if color == 'P'
                        prior = 0.85;  % They attack our edges
                    else
                        prior = 0.15;
                    end
                elseif ismember(edge, this.HUB_EDGES)
                    prior = 0.65 * (color == 'G') + 0.35 * (color == 'P');
                else
                    prior = 0.55 * (color == 'G') + 0.45 * (color == 'P');
                end
            end
        end

        function tf = isTerminal(this)
            %ISTERMINAL Check if this is a terminal state
            tf = ~any(this.State == '-');
        end

        function tf = isFullyExpanded(this)
            %ISFULLYEXPANDED Check if all children have been expanded
            tf = isempty(this.UntriedActions);
        end

        function value = ucb1Value(this, exploration, priorWeight)
            %UCB1VALUE Calculate UCB1 value with Progressive Bias
            %
            %   value = ucb1Value(node, exploration, priorWeight)
            %
            %   Formula: Q/N + c*sqrt(ln(parent.N)/N) + w*(prior-0.5)/(N+1)

            if nargin < 2, exploration = 1.414; end
            if nargin < 3, priorWeight = 1.0; end

            if this.Visits == 0
                value = inf;
                return;
            end

            exploitation = this.TotalValue / double(this.Visits);
            explorationTerm = exploration * sqrt(log(double(this.Parent.Visits)) / double(this.Visits));

            % Progressive bias: prior influence decreases with visits
            priorBonus = priorWeight * (this.Prior - 0.5) / (double(this.Visits) + 1);

            % Negate when parent is opponent's turn (minimax)
            if this.IsOurTurn
                value = -exploitation + explorationTerm + priorBonus;
            else
                value = exploitation + explorationTerm + priorBonus;
            end
        end

        function child = bestChild(this, exploration, priorWeight)
            %BESTCHILD Select best child using UCB1

            if nargin < 2, exploration = 1.414; end
            if nargin < 3, priorWeight = 1.0; end

            bestValue = -inf;
            child = [];

            keys = this.Children.keys();
            for i = 1:length(keys)
                c = this.Children(keys{i});
                v = c.ucb1Value(exploration, priorWeight);
                if v > bestValue
                    bestValue = v;
                    child = c;
                end
            end
        end

        function child = expand(this)
            %EXPAND Expand by adding a new child node

            if isempty(this.UntriedActions)
                child = [];
                return;
            end

            % Pop first action (highest priority)
            action = this.UntriedActions{1};
            this.UntriedActions(1) = [];

            edge = action{1};
            color = action{2};

            % Create new state
            newState = this.State;
            newState(edge) = color;

            % Get stored prior
            key = sprintf('%d_%s', edge, color);
            if this.ActionPriors.isKey(key)
                prior = this.ActionPriors(key);
            else
                prior = 0.5;
            end

            % Create child node
            child = MCTSNode(newState, ~this.IsOurTurn, this, action, prior);
            this.Children(key) = child;
        end

        function update(this, value)
            %UPDATE Backpropagate value up the tree

            this.Visits = this.Visits + 1;
            this.TotalValue = this.TotalValue + value;

            if ~isempty(this.Parent)
                this.Parent.update(value);
            end
        end

        function [bestAction, visits] = getMostVisitedAction(this)
            %GETMOSTVISITEDACTION Get action with most visits (robust selection)

            bestVisits = -1;
            bestAction = {};
            visits = 0;

            keys = this.Children.keys();
            for i = 1:length(keys)
                c = this.Children(keys{i});
                if c.Visits > bestVisits
                    bestVisits = c.Visits;
                    bestAction = c.Action;
                    visits = c.Visits;
                end
            end
        end
    end
end
