function data = extract_training_data(dbPath, options)
%EXTRACT_TRAINING_DATA Extract training data from game database
%
%   data = extract_training_data(dbPath)
%   data = extract_training_data(dbPath, Name=Value)
%
%   Extracts (state, action, outcome) tuples from the SQLite database
%   for supervised pre-training of value and policy networks.
%
%   Inputs:
%       dbPath - Path to SQLite database (default: ~/.tangled/game_stats.db)
%
%   Name-Value Arguments:
%       OnlyWins       - Only include winning games for policy training (default: false)
%       MinMoves       - Minimum moves in game to include (default: 5)
%       OutputPath     - Path to save extracted data (default: '', no save)
%       Verbose        - Print progress (default: true)
%
%   Outputs:
%       data - Struct with fields:
%           .states       - [50 x N] state feature vectors
%           .actions      - [N x 1] action indices (1-30)
%           .outcomes     - [N x 1] normalized game outcomes [-1, +1]
%           .gameIds      - {N x 1} cell array of game IDs
%           .moveNums     - [N x 1] move numbers
%           .players      - {N x 1} 'us' or 'opponent'
%           .metadata     - Struct with extraction info
%
%   Example:
%       data = extract_training_data();
%       fprintf('Extracted %d samples from %d games\n', size(data.states, 2), data.metadata.numGames);

    arguments
        dbPath string = ""
        options.OnlyWins logical = false
        options.MinMoves (1,1) double = 5
        options.OutputPath string = ""
        options.Verbose logical = true
    end

    %% Default database path
    if dbPath == ""
        dbPath = fullfile(getenv('USERPROFILE'), '.tangled', 'game_stats.db');
    end

    if ~exist(dbPath, 'file')
        error('Database not found: %s', dbPath);
    end

    log_print(options.Verbose, '\n=== Extracting Training Data ===\n');
    log_print(options.Verbose, 'Database: %s\n\n', dbPath);

    %% Connect to database
    conn = sqlite(dbPath);

    %% Query completed games
    if options.OnlyWins
        gameQuery = sprintf([...
            'SELECT id, result, final_score ' ...
            'FROM games ' ...
            'WHERE result = ''win'' AND total_moves >= %d ' ...
            'ORDER BY timestamp'], options.MinMoves);
    else
        gameQuery = sprintf([...
            'SELECT id, result, final_score ' ...
            'FROM games ' ...
            'WHERE result IS NOT NULL AND total_moves >= %d ' ...
            'ORDER BY timestamp'], options.MinMoves);
    end

    games = fetch(conn, gameQuery);

    if isempty(games) || height(games) == 0
        close(conn);
        error('No completed games found in database');
    end

    numGames = height(games);
    log_print(options.Verbose, 'Found %d completed games\n', numGames);

    %% Initialize output arrays
    allStates = [];
    allActions = [];
    allOutcomes = [];
    allGameIds = {};
    allMoveNums = [];
    allPlayers = {};

    %% Process each game
    for g = 1:numGames
        gameId = games.id{g};
        result = games.result{g};

        % Handle final_score which may be NULL
        if isfield(games, 'final_score') && ~isnan(games.final_score(g))
            finalScore = games.final_score(g);
        else
            finalScore = 0;
        end

        % Normalize outcome to [-1, +1]
        % Using result and final_score: win ~ +1, loss ~ -1, draw ~ 0
        switch result
            case 'win'
                outcome = 0.5 + 0.5 * tanh(finalScore / 3);  % Range [0.5, 1.0]
            case 'loss'
                outcome = -0.5 + 0.5 * tanh(finalScore / 3);  % Range [-1.0, -0.5]
            case 'draw'
                outcome = tanh(finalScore / 6);  % Small range around 0
            otherwise
                outcome = 0;
        end

        % Query moves for this game
        moveQuery = sprintf([...
            'SELECT move_number, player, edge, color, state_after, ' ...
            'score_after, score_delta ' ...
            'FROM moves ' ...
            'WHERE game_id = ''%s'' ' ...
            'ORDER BY move_number'], gameId);

        moves = fetch(conn, moveQuery);

        if isempty(moves) || height(moves) == 0
            continue;
        end

        % Process each move
        % We need state_before, which is previous move's state_after
        % For first move, state is '---------------'
        prevState = '---------------';

        for m = 1:height(moves)
            player = moves.player{m};
            edge = moves.edge(m);
            color = moves.color{m};

            % State before this move
            stateBefore = prevState;

            % Update prevState for next iteration
            if isfield(moves, 'state_after') && ~isempty(moves.state_after{m})
                prevState = moves.state_after{m};
            end

            % Skip if missing data
            if isempty(stateBefore) || length(stateBefore) ~= 15
                continue;
            end

            % Convert state to feature vector
            stateVec = state_to_features(stateBefore, strcmp(player, 'us'));

            % Convert action to index (1-30)
            % Action = edge * 2 + (color == 'P') + 1
            if strcmp(color, 'G')
                actionIdx = edge * 2 + 1;
            else  % 'P'
                actionIdx = edge * 2 + 2;
            end

            % Store sample
            allStates = [allStates, stateVec];
            allActions = [allActions; actionIdx];
            allOutcomes = [allOutcomes; outcome];
            allGameIds{end+1, 1} = gameId;
            allMoveNums = [allMoveNums; moves.move_number(m)];
            allPlayers{end+1, 1} = player;
        end

        if mod(g, 10) == 0
            log_print(options.Verbose, '  Processed %d/%d games (%d samples)\n', ...
                g, numGames, size(allStates, 2));
        end
    end

    %% Close database
    close(conn);

    %% Build output struct
    data = struct();
    data.states = allStates;
    data.actions = allActions;
    data.outcomes = allOutcomes;
    data.gameIds = allGameIds;
    data.moveNums = allMoveNums;
    data.players = allPlayers;

    % Metadata
    data.metadata = struct();
    data.metadata.numGames = numGames;
    data.metadata.numSamples = size(allStates, 2);
    data.metadata.numOurMoves = sum(strcmp(allPlayers, 'us'));
    data.metadata.numOppMoves = sum(strcmp(allPlayers, 'opponent'));
    data.metadata.dbPath = dbPath;
    data.metadata.extractedAt = datestr(now, 'yyyy-mm-dd HH:MM:SS');
    data.metadata.onlyWins = options.OnlyWins;
    data.metadata.minMoves = options.MinMoves;

    % Outcome distribution
    data.metadata.outcomeStats = struct();
    data.metadata.outcomeStats.mean = mean(allOutcomes);
    data.metadata.outcomeStats.std = std(allOutcomes);
    data.metadata.outcomeStats.min = min(allOutcomes);
    data.metadata.outcomeStats.max = max(allOutcomes);

    %% Summary
    log_print(options.Verbose, '\n=== Extraction Complete ===\n');
    log_print(options.Verbose, '  Total samples: %d\n', data.metadata.numSamples);
    log_print(options.Verbose, '  Our moves:     %d\n', data.metadata.numOurMoves);
    log_print(options.Verbose, '  Opponent moves: %d\n', data.metadata.numOppMoves);
    log_print(options.Verbose, '  Games:         %d\n', data.metadata.numGames);
    log_print(options.Verbose, '  Outcome mean:  %.3f\n', data.metadata.outcomeStats.mean);
    log_print(options.Verbose, '  Outcome std:   %.3f\n', data.metadata.outcomeStats.std);

    %% Save if requested
    if options.OutputPath ~= ""
        save(options.OutputPath, 'data');
        log_print(options.Verbose, '\nData saved to: %s\n', options.OutputPath);
    end

    log_print(options.Verbose, '\n');
end

%% Helper Functions

function features = state_to_features(stateStr, isOurTurn)
%STATE_TO_FEATURES Convert 15-char state string to 50-element feature vector
%
%   Feature vector layout (50 elements):
%       [1:15]  - Board state: -1 (Purple), 0 (Grey), +1 (Green)
%       [16]    - Turn indicator: +1 (our turn), -1 (opponent turn)
%       [17:31] - Edge category encoding (MY_EDGE, OPP_EDGE, HUB, NEUTRAL)
%       [32]    - Grey count (normalized 0-1)
%       [33:35] - Score momentum (placeholder, filled with 0)
%       [36:50] - Game phase one-hot (opening/mid/end)

    features = zeros(50, 1);

    % [1:15] Board state encoding
    for i = 1:15
        c = stateStr(i);
        if c == 'G'
            features(i) = 1;
        elseif c == 'P'
            features(i) = -1;
        else
            features(i) = 0;
        end
    end

    % [16] Turn indicator
    if isOurTurn
        features(16) = 1;
    else
        features(16) = -1;
    end

    % [17:31] Edge category encoding
    % Petersen graph edge categories (simplified)
    % MY_EDGES (outer pentagon): 0, 1, 2, 3, 4
    % OPP_EDGES (inner pentagram): 5, 6, 7, 8, 9
    % HUB_EDGES (spokes): 10, 11, 12, 13, 14
    myEdges = [1, 2, 3, 4, 5];       % 1-indexed
    oppEdges = [6, 7, 8, 9, 10];
    hubEdges = [11, 12, 13, 14, 15];

    for i = 1:15
        if ismember(i, myEdges)
            features(16 + i) = 0.5;   % MY_EDGE
        elseif ismember(i, oppEdges)
            features(16 + i) = -0.5;  % OPP_EDGE
        elseif ismember(i, hubEdges)
            features(16 + i) = 0.25;  % HUB_EDGE
        else
            features(16 + i) = 0;     % NEUTRAL
        end
    end

    % [32] Grey count (normalized)
    greyCount = sum(stateStr == '-');
    features(32) = greyCount / 15;

    % [33:35] Score momentum (placeholder)
    features(33:35) = 0;

    % [36:50] Game phase one-hot
    % Opening: > 10 grey edges
    % Mid: 5-10 grey edges
    % End: < 5 grey edges
    if greyCount > 10
        features(36:40) = 1;  % Opening
    elseif greyCount >= 5
        features(41:45) = 1;  % Mid
    else
        features(46:50) = 1;  % End
    end
end

function log_print(verbose, varargin)
    if verbose
        fprintf(varargin{:});
    end
end
