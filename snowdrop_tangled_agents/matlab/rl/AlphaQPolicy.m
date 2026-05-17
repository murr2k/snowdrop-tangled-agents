classdef AlphaQPolicy < handle
%ALPHAQPOLICY Phase 2 AlphaQ predictive policy loader and evaluator.
%
%   Loads the .mat file produced by scripts/train_alphaq_policy.py and
%   exposes a predict(state) method that returns a 30-element distribution
%   over (edge, color) actions, with illegal actions masked to zero and
%   the remainder renormalised.
%
%   Action indexing (1-based MATLAB):
%       a = 2 * edge - 1  -> (edge=1..15, color='G')
%       a = 2 * edge      -> (edge=1..15, color='P')
%   Equivalent to Python label = (edge - 1) * 2 + (0 if 'G' else 1).
%
%   Featurisation MUST match scripts/train_alphaq_policy.py:featurise.
%   The 12 Petersen 5-cycles are hard-coded with the same edge ordering
%   the Python script produces at module load.
%
%   Example:
%       pol = AlphaQPolicy('PolicyFile', 'alphaq_policy_mlp.mat');
%       p = pol.predict('PPGPGGG--------');  % 30x1 column, sums to 1

    properties (SetAccess = private)
        ModelType char = ''       % 'mlp' or 'logreg'
        FeatureMean double        % 1x92
        FeatureScale double       % 1x92
        Loaded logical = false

        % MLP layers (all empty for logreg)
        W1 double = []
        b1 double = []
        W2 double = []
        b2 double = []
        W3 double = []
        b3 double = []

        % LogReg (empty for MLP)
        LR_W double = []          % (30, 92)
        LR_b double = []          % (30, 1)

        PolicyFile char = 'alphaq_policy_mlp.mat'
    end

    properties (Constant)
        N_EDGES = 15
        N_VERTICES = 10
        N_ACTIONS = 30
        N_FEATURES = 92

        % Petersen edge list (0-indexed vertex labels). Matches Python
        % EDGE_LIST in scripts/train_alphaq_policy.py.
        EDGE_LIST = [ ...
            0, 2;  0, 3;  0, 6;  ...     % E0, E1, E2
            1, 3;  1, 4;  1, 7;  ...     % E3, E4, E5
            2, 4;  2, 8;          ...    % E6, E7
            3, 9;                  ...   % E8
            4, 5;                  ...   % E9
            5, 6;  5, 9;           ...   % E10, E11
            6, 7;                  ...   % E12
            7, 8;                  ...   % E13
            8, 9                          ... % E14
        ]

        % 12 Petersen 5-cycles, each as a sorted list of 5 *0-indexed* edge
        % labels. Matches Python PETERSEN_5_CYCLES exactly (verified
        % 2026-05-17 by scripts/_phase4_reference_dump.py).
        CYCLES = [ ...
             6,  7,  9, 11, 14;  ...    % C1
             4,  5,  9, 10, 12;  ...    % C2
             1,  2,  8, 10, 11;  ...    % C3
             1,  2,  3,  5, 12;  ...    % C4
             0,  2,  6,  9, 10;  ...    % C5
             3,  4,  8,  9, 11;  ...    % C6
             0,  1,  7,  8, 14;  ...    % C7
             3,  5,  8, 13, 14;  ...    % C8
             0,  2,  7, 12, 13;  ...    % C9
             4,  5,  6,  7, 13;  ...    % C10
            10, 11, 12, 13, 14;  ...    % C11
             0,  1,  3,  4,  6       ...% C12
        ]
    end

    methods
        function this = AlphaQPolicy(options)
            arguments
                options.PolicyFile char = 'alphaq_policy_mlp.mat'
            end
            this.PolicyFile = options.PolicyFile;
            this.loadPolicy();
        end

        function loadPolicy(this)
            scriptDir = fileparts(mfilename('fullpath'));
            path = fullfile(scriptDir, 'data', this.PolicyFile);
            if ~isfile(path)
                warning('AlphaQPolicy:NotFound', ...
                    'Policy file not found: %s', path);
                this.Loaded = false;
                return;
            end
            try
                data = load(path);
            catch ME
                warning('AlphaQPolicy:LoadError', ...
                    'Failed to load policy file %s: %s', path, ME.message);
                this.Loaded = false;
                return;
            end

            mt = strtrim(char(data.model_type));
            this.ModelType = mt;
            this.FeatureMean = double(data.feature_mean(:))';
            this.FeatureScale = double(data.feature_scale(:))';

            % Guard against zero scale (would NaN the standardiser)
            tiny = this.FeatureScale < 1e-12;
            this.FeatureScale(tiny) = 1.0;

            switch mt
                case 'mlp'
                    this.W1 = double(data.W1);
                    this.b1 = double(data.b1(:))';
                    this.W2 = double(data.W2);
                    this.b2 = double(data.b2(:))';
                    this.W3 = double(data.W3);
                    this.b3 = double(data.b3(:))';
                case 'logreg'
                    this.LR_W = double(data.W);
                    this.LR_b = double(data.b(:));
                otherwise
                    warning('AlphaQPolicy:UnknownType', ...
                        'Unknown model_type "%s"', mt);
                    this.Loaded = false;
                    return;
            end
            this.Loaded = true;
        end

        function p = predict(this, state)
            %PREDICT Return a 30x1 probability over (edge, color) actions.
            %   Illegal actions (non-grey edges) are masked to zero and the
            %   remainder renormalised to sum to 1. If the model is not
            %   loaded, returns a uniform distribution over legal actions.

            mask = this.legalMask(state);
            if ~this.Loaded
                p = double(mask(:));
                s = sum(p);
                if s > 0
                    p = p / s;
                end
                return;
            end

            x_raw = this.featurise(state);                 % 1x92
            x = (x_raw - this.FeatureMean) ./ this.FeatureScale;

            switch this.ModelType
                case 'mlp'
                    h1 = max(0, x * this.W1 + this.b1);     % 1x64
                    h2 = max(0, h1 * this.W2 + this.b2);    % 1x64
                    z  = h2 * this.W3 + this.b3;            % 1x30
                case 'logreg'
                    % LR_W is (30, 92); LR_b is (30, 1)
                    z = (this.LR_W * x' + this.LR_b)';      % 1x30
                otherwise
                    z = zeros(1, this.N_ACTIONS);
            end

            % Mask illegal logits to -inf, then softmax
            z_masked = z;
            z_masked(~mask) = -inf;

            % Numerically stable softmax over legal entries only
            mz = max(z_masked);
            if isinf(mz) && mz < 0
                % No legal actions -- shouldn't happen for non-terminal
                p = zeros(this.N_ACTIONS, 1);
                return;
            end
            ez = exp(z_masked - mz);
            ez(~mask) = 0;
            s = sum(ez);
            if s > 0
                p = (ez / s)';
            else
                % All masked-out somehow; uniform over legal as fallback
                p = double(mask(:));
                p = p / max(sum(p), 1);
            end
        end

        function [edge, color] = sample(this, state)
            %SAMPLE Draw one action from predict(state). Returns 1-indexed
            %       edge and 'G'/'P' color. Useful for synthetic tests.
            p = this.predict(state);
            r = rand();
            c = cumsum(p);
            a = find(c >= r, 1, 'first');
            if isempty(a), a = find(p > 0, 1, 'first'); end
            [edge, color] = AlphaQPolicy.actionToEdgeColor(a);
        end

        function f = featurise(this, state)
            %FEATURISE Compute 92-dim feature vector matching Python.
            f = zeros(1, this.N_FEATURES);

            % 1. Per-edge one-hot (indices 1..45, mapping to Python 0..44)
            for i = 1:this.N_EDGES
                base = (i - 1) * 3;
                switch state(i)
                    case '-', f(base + 1) = 1.0;
                    case 'G', f(base + 2) = 1.0;
                    case 'P', f(base + 3) = 1.0;
                end
            end

            % 2. Per-vertex degree counts (indices 46..75)
            edges = this.EDGE_LIST;
            for v = 0:(this.N_VERTICES - 1)
                ng = 0; npp = 0; nc = 0;
                for ei = 1:this.N_EDGES
                    if v == edges(ei, 1) || v == edges(ei, 2)
                        c = state(ei);
                        if c == 'G'
                            ng = ng + 1;
                        elseif c == 'P'
                            npp = npp + 1;
                        end
                        if c ~= '-'
                            nc = nc + 1;
                        end
                    end
                end
                f(46 + v * 3) = ng;
                f(47 + v * 3) = npp;
                f(48 + v * 3) = nc;
            end

            % 3. Frustration indicators per 5-cycle (indices 76..87)
            for ci = 1:size(this.CYCLES, 1)
                n_purple = 0;
                any_grey = false;
                for k = 1:5
                    e = this.CYCLES(ci, k) + 1;  % to 1-based
                    c = state(e);
                    if c == 'P'
                        n_purple = n_purple + 1;
                    elseif c == '-'
                        any_grey = true;
                        break;
                    end
                end
                if any_grey
                    f(75 + ci) = 0.5;
                else
                    f(75 + ci) = double(mod(n_purple, 2) == 1);
                end
            end

            % 4. Aggregate counts (indices 88..90)
            n_grey = sum(state == '-');
            n_green = sum(state == 'G');
            n_purple = sum(state == 'P');
            f(88) = n_grey / 15.0;
            f(89) = n_green / 15.0;
            f(90) = n_purple / 15.0;

            % 5. Parity (indices 91..92)
            f(91) = 1.0;
            f(92) = double(mod(n_grey, 2));
        end

        function m = legalMask(this, state)
            %LEGALMASK 1x30 logical mask of legal (edge, color) actions.
            m = false(1, this.N_ACTIONS);
            for e = 1:this.N_EDGES
                if state(e) == '-'
                    m(2*e - 1) = true;
                    m(2*e)     = true;
                end
            end
        end
    end

    methods (Static)
        function [edge, color] = actionToEdgeColor(a)
            %ACTIONTOEDGECOLOR Convert 1-indexed MATLAB action -> (edge,color)
            edge = floor((a - 1) / 2) + 1;
            if mod(a - 1, 2) == 0
                color = 'G';
            else
                color = 'P';
            end
        end

        function a = edgeColorToAction(edge, color)
            %EDGECOLORTOACTION Convert (1-indexed edge, color) -> action idx
            if color == 'G' || color == 'g'
                a = 2 * edge - 1;
            else
                a = 2 * edge;
            end
        end
    end
end
