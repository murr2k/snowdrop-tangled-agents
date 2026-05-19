classdef PetersenGeometry
    %PETERSENGEOMETRY  Static structural-feature utilities for the
    %   Petersen graph in Tangled play.
    %
    %   Used by HybridTangledSolver's switchback mode.
    %   Encodes:
    %     - 12 5-cycles (edge index tuples)
    %     - Edge-orbit partition under Stab(p1=5, p2=7):
    %         6 swap-pair orbits, 3 fixed-edge orbits.
    %     - State encoding: 15-char G/P/'-' string, position j = edge j.
    %
    %   Feature methods (all static, take a 15-char state):
    %     frustrationLocked(state)      -> count of 5-cycles locked frustrated
    %     satisfiedLocked(state)        -> count of 5-cycles locked satisfied
    %     undecidedCycles(state)        -> count of 5-cycles still flexible
    %     orbitColorBalance(state)      -> balance score across the 9 edge-orbits
    %     switchbackScore(state, w)     -> linear combination of the above

    methods (Static)
        function cycles = cycles5()
            %CYCLES5 Return the 12 five-cycles as a 12x5 array of edge indices
            %   (0-indexed). Verified via networkx enumeration against
            %   GraphProperties().graph_database[5]['edge_list'].
            cycles = [
                 0,  1,  3,  4,  6;
                 0,  1,  7,  8, 14;
                 0,  2,  6,  9, 10;
                 0,  2,  7, 12, 13;
                 1,  2,  3,  5, 12;
                 1,  2,  8, 10, 11;
                 3,  4,  8,  9, 11;
                 3,  5,  8, 13, 14;
                 4,  5,  6,  7, 13;
                 4,  5,  9, 10, 12;
                 6,  7,  9, 11, 14;
                10, 11, 12, 13, 14;
            ];
        end

        function orbits = edgeOrbits()
            %EDGEORBITS Stab(p1=5, p2=7) acts on edges. Return a 1x15 vector
            %   mapping each edge to its orbit ID (1..9). The 9 orbits are:
            %     orbit 1: {0, 1}       (swap pair)
            %     orbit 2: {2}          (fixed)
            %     orbit 3: {3, 7}
            %     orbit 4: {4, 14}
            %     orbit 5: {5, 13}
            %     orbit 6: {6, 8}
            %     orbit 7: {9, 11}
            %     orbit 8: {10}         (fixed)
            %     orbit 9: {12}         (fixed)
            orbits = zeros(1, 15);
            orbits(0+1) = 1; orbits(1+1)  = 1;
            orbits(2+1) = 2;
            orbits(3+1) = 3; orbits(7+1)  = 3;
            orbits(4+1) = 4; orbits(14+1) = 4;
            orbits(5+1) = 5; orbits(13+1) = 5;
            orbits(6+1) = 6; orbits(8+1)  = 6;
            orbits(9+1) = 7; orbits(11+1) = 7;
            orbits(10+1) = 8;
            orbits(12+1) = 9;
        end

        function n = frustrationLocked(state)
            %FRUSTRATIONLOCKED Count of 5-cycles fully colored with an odd
            %   number of P (purple) edges, i.e., locked frustrated.
            cs = PetersenGeometry.cycles5();
            n = 0;
            for k = 1:size(cs, 1)
                idxs = cs(k, :) + 1;  % 1-indexed
                chars = state(idxs);
                if any(chars == '-')
                    continue;
                end
                p_count = sum(chars == 'P');
                if mod(p_count, 2) == 1
                    n = n + 1;
                end
            end
        end

        function n = satisfiedLocked(state)
            %SATISFIEDLOCKED Count of 5-cycles fully colored with an even
            %   number of P (purple) edges, i.e., locked satisfied.
            cs = PetersenGeometry.cycles5();
            n = 0;
            for k = 1:size(cs, 1)
                idxs = cs(k, :) + 1;
                chars = state(idxs);
                if any(chars == '-')
                    continue;
                end
                p_count = sum(chars == 'P');
                if mod(p_count, 2) == 0
                    n = n + 1;
                end
            end
        end

        function n = undecidedCycles(state)
            %UNDECIDEDCYCLES Count of 5-cycles with at least one grey edge.
            cs = PetersenGeometry.cycles5();
            n = 0;
            for k = 1:size(cs, 1)
                idxs = cs(k, :) + 1;
                if any(state(idxs) == '-')
                    n = n + 1;
                end
            end
        end

        function score = orbitColorBalance(state)
            %ORBITCOLORBALANCE Higher score = colored edges are more evenly
            %   distributed across the 9 Stab(5,7) edge-orbits, AND within
            %   each orbit, G/P are balanced. Encourages "switchback"
            %   distribution rather than exhausting one structural class.
            %
            %   Strategy: for each orbit, count (#colored, #G - #P). Penalize
            %   high deviation from the mean across orbits. Negative penalty,
            %   higher (less negative) = better.
            orbits = PetersenGeometry.edgeOrbits();
            colored_per = zeros(1, 9);
            net_per = zeros(1, 9);  % #G - #P per orbit
            for j = 1:15
                o = orbits(j);
                c = state(j);
                if c == 'G'
                    colored_per(o) = colored_per(o) + 1;
                    net_per(o) = net_per(o) + 1;
                elseif c == 'P'
                    colored_per(o) = colored_per(o) + 1;
                    net_per(o) = net_per(o) - 1;
                end
            end
            % Penalize uneven coverage across orbits AND large |G-P| within orbits.
            % Both deviations are "switchback violations" — committing to one
            % orbit or one color too quickly.
            sigma_cov = std(colored_per);
            sigma_net = std(abs(net_per));
            score = -(sigma_cov + sigma_net);
        end

        function score = switchbackScore(state, weights)
            %SWITCHBACKSCORE Linear combination of structural features.
            %
            %   weights is a 1x4 vector [w_sat, w_frust, w_undec, w_balance].
            %   Defaults to [+1, -2, +0.3, +1].
            %
            %   Higher = better move target.
            if nargin < 2 || isempty(weights)
                weights = [+1, -2, +0.3, +1];
            end
            score = weights(1) * PetersenGeometry.satisfiedLocked(state) ...
                  + weights(2) * PetersenGeometry.frustrationLocked(state) ...
                  + weights(3) * PetersenGeometry.undecidedCycles(state) ...
                  + weights(4) * PetersenGeometry.orbitColorBalance(state);
        end
    end
end
