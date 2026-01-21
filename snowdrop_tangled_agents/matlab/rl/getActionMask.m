function mask = getActionMask(state)
%GETACTIONMASK Return valid action mask for current board state
%
%   mask = getActionMask(state)
%
%   Inputs:
%       state - 15-character string representing board state
%               'G' = Green (Ferromagnetic)
%               'P' = Purple (Antiferromagnetic)
%               '-' = Grey (Uncolored)
%
%   Outputs:
%       mask - 30x1 vector where mask(i) = 1 if action i is valid
%              Actions 1-15:  Play Green on edges 0-14
%              Actions 16-30: Play Purple on edges 0-14
%
%   Example:
%       state = 'GP-------------';  % Edges 0,1 colored
%       mask = getActionMask(state);
%       validActions = find(mask);  % Returns [3:15, 18:30]
%
%   Note:
%       Only grey edges can be played on. The mask ensures the agent
%       cannot select invalid actions (already colored edges).

    arguments
        state (1,15) char
    end

    mask = zeros(30, 1);

    for i = 1:15
        if state(i) == '-'
            mask(i) = 1;       % Green on edge (i-1) is valid
            mask(i + 15) = 1;  % Purple on edge (i-1) is valid
        end
    end
end
