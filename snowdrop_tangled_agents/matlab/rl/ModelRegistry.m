classdef ModelRegistry < handle
%MODELREGISTRY Manage trained model versions and deployment
%
%   Tracks model versions, performance metrics, and handles deployment
%   of trained agents for production use.
%
%   Example:
%       registry = ModelRegistry('models.db', 'checkpoints');
%       version = registry.registerModel(agent, metrics, 'Initial training');
%       registry.deployModel(version);
%       agent = registry.loadDeployed();

    properties (SetAccess = private)
        DBPath char              % Path to SQLite metadata database
        ModelDir char            % Directory for model files
        DeployDir char           % Directory for deployed models
        Connection               % SQLite connection
    end

    methods
        function this = ModelRegistry(dbPath, modelDir)
            %MODELREGISTRY Construct model registry
            %
            %   registry = ModelRegistry(dbPath, modelDir)
            %
            %   Inputs:
            %       dbPath   - Path to SQLite database for metadata
            %       modelDir - Directory to store model files

            arguments
                dbPath char = fullfile(tempdir, 'tangled_models.db')
                modelDir char = fullfile(tempdir, 'tangled_models')
            end

            this.DBPath = dbPath;
            this.ModelDir = modelDir;
            this.DeployDir = fullfile(modelDir, 'deployed');

            % Create directories
            if ~exist(this.ModelDir, 'dir')
                mkdir(this.ModelDir);
            end
            if ~exist(this.DeployDir, 'dir')
                mkdir(this.DeployDir);
            end

            % Connect to database
            this.Connection = sqlite(dbPath, 'create');
            this.initializeTable();

            fprintf('Model registry initialized:\n');
            fprintf('  Database: %s\n', dbPath);
            fprintf('  Models:   %s\n', modelDir);
            fprintf('  Deploy:   %s\n', this.DeployDir);
        end

        function version = registerModel(this, agent, metrics, notes)
            %REGISTERMODEL Register a trained model version
            %
            %   version = registerModel(registry, agent, metrics, notes)
            %
            %   Inputs:
            %       agent   - Trained PPO agent
            %       metrics - Struct with training metrics:
            %                 .episodes, .avgReward, .winRate
            %       notes   - Description string
            %
            %   Outputs:
            %       version - Version string (e.g., 'v20260121_153045')

            arguments
                this
                agent
                metrics struct
                notes char = ''
            end

            % Generate version string
            version = sprintf('v%s', datestr(now, 'yyyymmdd_HHMMSS'));

            % Save model file
            filePath = fullfile(this.ModelDir, [version '.mat']);
            save(filePath, 'agent', 'metrics');

            % Extract metrics with defaults
            episodes = getFieldOr(metrics, 'episodes', 0);
            avgReward = getFieldOr(metrics, 'avgReward', 0);
            winRate = getFieldOr(metrics, 'winRate', 0);

            % Register in database
            insert(this.Connection, 'model_versions', ...
                {'version', 'file_path', 'training_episodes', 'avg_reward', 'win_rate', 'notes', 'deployed'}, ...
                {version, filePath, episodes, avgReward, winRate, notes, 0});

            fprintf('Registered model: %s\n', version);
            fprintf('  Episodes:   %d\n', episodes);
            fprintf('  Avg Reward: %.3f\n', avgReward);
            fprintf('  Win Rate:   %.1f%%\n', winRate * 100);
        end

        function deployModel(this, version)
            %DEPLOYMODEL Deploy a specific model version
            %
            %   deployModel(registry, version)
            %
            %   Marks the specified version as deployed and copies
            %   it to the deployment directory.

            % Get model info
            query = sprintf(...
                'SELECT file_path FROM model_versions WHERE version = ''%s''', ...
                version);
            data = fetch(this.Connection, query);

            if isempty(data) || height(data) == 0
                error('Model version not found: %s', version);
            end

            sourcePath = data.file_path{1};
            if ~exist(sourcePath, 'file')
                error('Model file not found: %s', sourcePath);
            end

            % Unmark previous deployed version
            exec(this.Connection, 'UPDATE model_versions SET deployed = 0 WHERE deployed = 1');

            % Mark new version as deployed
            exec(this.Connection, sprintf(...
                'UPDATE model_versions SET deployed = 1 WHERE version = ''%s''', ...
                version));

            % Copy to deployment location
            deployPath = fullfile(this.DeployDir, 'current_model.mat');
            copyfile(sourcePath, deployPath);

            % Also save version info
            versionFile = fullfile(this.DeployDir, 'current_version.txt');
            fid = fopen(versionFile, 'w');
            fprintf(fid, '%s\n', version);
            fprintf(fid, 'Deployed: %s\n', datestr(now));
            fclose(fid);

            fprintf('Deployed model: %s\n', version);
            fprintf('  Location: %s\n', deployPath);
        end

        function [agent, version] = loadDeployed(this)
            %LOADDEPLOYED Load the currently deployed model
            %
            %   [agent, version] = loadDeployed(registry)
            %
            %   Outputs:
            %       agent   - Deployed PPO agent
            %       version - Version string of deployed model

            deployPath = fullfile(this.DeployDir, 'current_model.mat');

            if ~exist(deployPath, 'file')
                error('No model currently deployed');
            end

            data = load(deployPath, 'agent');
            agent = data.agent;

            % Get version
            versionFile = fullfile(this.DeployDir, 'current_version.txt');
            if exist(versionFile, 'file')
                fid = fopen(versionFile, 'r');
                version = strtrim(fgetl(fid));
                fclose(fid);
            else
                version = 'unknown';
            end
        end

        function versions = listVersions(this, limit)
            %LISTVERSIONS List all registered model versions
            %
            %   versions = listVersions(registry)
            %   versions = listVersions(registry, limit)

            arguments
                this
                limit (1,1) double = 20
            end

            query = sprintf([...
                'SELECT version, training_episodes, avg_reward, win_rate, ' ...
                'deployed, created, notes ' ...
                'FROM model_versions ' ...
                'ORDER BY created DESC LIMIT %d'], limit);

            versions = fetch(this.Connection, query);
        end

        function info = getDeployedInfo(this)
            %GETDEPLOYEDINFO Get information about deployed model
            %
            %   info = getDeployedInfo(registry)

            query = [...
                'SELECT version, training_episodes, avg_reward, win_rate, ' ...
                'created, notes ' ...
                'FROM model_versions WHERE deployed = 1'];

            data = fetch(this.Connection, query);

            if isempty(data) || height(data) == 0
                info = [];
            else
                info = struct();
                info.version = data.version{1};
                info.episodes = data.training_episodes;
                info.avgReward = data.avg_reward;
                info.winRate = data.win_rate;
                info.created = data.created{1};
                info.notes = data.notes{1};
            end
        end

        function deleteVersion(this, version)
            %DELETEVERSION Delete a model version
            %
            %   deleteVersion(registry, version)
            %
            %   Cannot delete the currently deployed version.

            % Check if deployed
            query = sprintf(...
                'SELECT deployed, file_path FROM model_versions WHERE version = ''%s''', ...
                version);
            data = fetch(this.Connection, query);

            if isempty(data) || height(data) == 0
                error('Version not found: %s', version);
            end

            if data.deployed == 1
                error('Cannot delete deployed version. Deploy another version first.');
            end

            % Delete file
            filePath = data.file_path{1};
            if exist(filePath, 'file')
                delete(filePath);
            end

            % Delete from database
            exec(this.Connection, sprintf(...
                'DELETE FROM model_versions WHERE version = ''%s''', version));

            fprintf('Deleted version: %s\n', version);
        end

        function close(this)
            %CLOSE Close database connection

            if ~isempty(this.Connection)
                close(this.Connection);
                this.Connection = [];
            end
        end

        function delete(this)
            %DELETE Destructor

            this.close();
        end
    end

    methods (Access = private)
        function initializeTable(this)
            %INITIALIZETABLE Create model_versions table

            exec(this.Connection, [...
                'CREATE TABLE IF NOT EXISTS model_versions (' ...
                '  id INTEGER PRIMARY KEY AUTOINCREMENT,' ...
                '  version TEXT UNIQUE NOT NULL,' ...
                '  file_path TEXT NOT NULL,' ...
                '  training_episodes INTEGER,' ...
                '  avg_reward REAL,' ...
                '  win_rate REAL,' ...
                '  deployed INTEGER DEFAULT 0,' ...
                '  created DATETIME DEFAULT CURRENT_TIMESTAMP,' ...
                '  notes TEXT' ...
                ')']);

            % Create index for fast lookup
            exec(this.Connection, [...
                'CREATE INDEX IF NOT EXISTS idx_deployed ' ...
                'ON model_versions (deployed)']);
        end
    end
end

function val = getFieldOr(s, field, default)
%GETFIELDOR Get struct field with default value
    if isfield(s, field)
        val = s.(field);
    else
        val = default;
    end
end
