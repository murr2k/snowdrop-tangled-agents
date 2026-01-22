classdef SQLiteExperienceBuffer < handle
%SQLITEEXPERIENCEBUFFER Persistent experience buffer backed by SQLite
%
%   Stores RL experience tuples (state, action, reward, next_state, done)
%   in a SQLite database for persistent replay memory across sessions.
%
%   Example:
%       buffer = SQLiteExperienceBuffer('experience.db', 100000);
%       buffer.add(state, action, reward, nextState, done);
%       batch = buffer.sample(32);
%       buffer.close();

    properties (SetAccess = private)
        DBPath char          % Path to SQLite database
        MaxSize double       % Maximum buffer size
        Connection           % SQLite connection object
    end

    properties (Dependent)
        CurrentSize          % Current number of experiences
    end

    methods
        function this = SQLiteExperienceBuffer(dbPath, maxSize)
            %SQLITEEXPERIENCEBUFFER Construct experience buffer
            %
            %   buffer = SQLiteExperienceBuffer(dbPath, maxSize)
            %
            %   Inputs:
            %       dbPath  - Path to SQLite database file
            %       maxSize - Maximum number of experiences to store

            arguments
                dbPath char = 'experience.db'
                maxSize (1,1) double = 100000
            end

            this.DBPath = dbPath;
            this.MaxSize = maxSize;

            % Connect to database
            this.Connection = sqlite(dbPath, 'create');

            % Initialize table
            this.initializeTable();

            fprintf('Experience buffer initialized: %s (max %d)\n', ...
                dbPath, maxSize);
        end

        function add(this, state, action, reward, nextState, done)
            %ADD Add experience tuple to buffer
            %
            %   add(buffer, state, action, reward, nextState, done)
            %
            %   Inputs:
            %       state     - Current state (50x1 vector)
            %       action    - Action taken (1-30)
            %       reward    - Reward received
            %       nextState - Next state (50x1 vector)
            %       done      - Episode done flag (0 or 1)

            % Serialize states to byte arrays
            stateBlob = getByteStreamFromArray(state(:));
            nextStateBlob = getByteStreamFromArray(nextState(:));

            % Insert into database
            insert(this.Connection, 'experience', ...
                {'state', 'action', 'reward', 'next_state', 'done'}, ...
                {stateBlob, action, reward, nextStateBlob, double(done)});

            % Prune if over max size
            this.pruneOldest();
        end

        function addBatch(this, states, actions, rewards, nextStates, dones)
            %ADDBATCH Add multiple experience tuples
            %
            %   addBatch(buffer, states, actions, rewards, nextStates, dones)

            n = size(states, 2);
            for i = 1:n
                this.add(states(:,i), actions(i), rewards(i), ...
                    nextStates(:,i), dones(i));
            end
        end

        function batch = sample(this, batchSize)
            %SAMPLE Random sample from buffer
            %
            %   batch = sample(buffer, batchSize)
            %
            %   Returns struct with fields:
            %       states     - [obsSize x batchSize]
            %       actions    - [1 x batchSize]
            %       rewards    - [1 x batchSize]
            %       nextStates - [obsSize x batchSize]
            %       dones      - [1 x batchSize]

            % Query random samples
            query = sprintf([...
                'SELECT state, action, reward, next_state, done ' ...
                'FROM experience ORDER BY RANDOM() LIMIT %d'], batchSize);

            data = fetch(this.Connection, query);

            if isempty(data) || height(data) == 0
                batch = [];
                return;
            end

            n = height(data);

            % Deserialize states
            states = zeros(50, n);
            nextStates = zeros(50, n);

            for i = 1:n
                states(:, i) = getArrayFromByteStream(data.state{i});
                nextStates(:, i) = getArrayFromByteStream(data.next_state{i});
            end

            batch = struct();
            batch.states = states;
            batch.actions = data.action';
            batch.rewards = data.reward';
            batch.nextStates = nextStates;
            batch.dones = data.done';
        end

        function n = get.CurrentSize(this)
            %GET.CURRENTSIZE Get current buffer size

            result = fetch(this.Connection, ...
                'SELECT COUNT(*) as cnt FROM experience');
            n = result.cnt;
        end

        function clear(this)
            %CLEAR Clear all experiences from buffer

            exec(this.Connection, 'DELETE FROM experience');
            fprintf('Experience buffer cleared\n');
        end

        function close(this)
            %CLOSE Close database connection

            if ~isempty(this.Connection)
                close(this.Connection);
                this.Connection = [];
            end
        end

        function delete(this)
            %DELETE Destructor - ensure connection is closed

            this.close();
        end
    end

    methods (Access = private)
        function initializeTable(this)
            %INITIALIZETABLE Create experience table if not exists

            exec(this.Connection, [...
                'CREATE TABLE IF NOT EXISTS experience (' ...
                '  id INTEGER PRIMARY KEY AUTOINCREMENT,' ...
                '  state BLOB NOT NULL,' ...
                '  action INTEGER NOT NULL,' ...
                '  reward REAL NOT NULL,' ...
                '  next_state BLOB NOT NULL,' ...
                '  done INTEGER NOT NULL,' ...
                '  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP' ...
                ')']);

            % Create index for faster random sampling
            exec(this.Connection, [...
                'CREATE INDEX IF NOT EXISTS idx_experience_random ' ...
                'ON experience (id)']);
        end

        function pruneOldest(this)
            %PRUNEOLDEST Remove oldest experiences if over max size

            result = fetch(this.Connection, ...
                'SELECT COUNT(*) as cnt FROM experience');

            if result.cnt > this.MaxSize
                excess = result.cnt - this.MaxSize;

                exec(this.Connection, sprintf([...
                    'DELETE FROM experience WHERE id IN (' ...
                    '  SELECT id FROM experience ORDER BY timestamp LIMIT %d' ...
                    ')'], excess));
            end
        end
    end
end
