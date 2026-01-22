function vecEnv = createParallelEnv(numWorkers, options)
%CREATEPARALLELENV Create vectorized parallel environment for training
%
%   vecEnv = createParallelEnv(numWorkers)
%   vecEnv = createParallelEnv(numWorkers, Name=Value)
%
%   Creates multiple TangledEnvironment instances for parallel self-play
%   training. Supports both local parallel pools and vectorized simulation.
%
%   Inputs:
%       numWorkers - Number of parallel environments (default: 4)
%
%   Name-Value Arguments:
%       UseParpool  - Use parallel pool for true parallelism (default: true)
%       OpponentType - Opponent type: 'heuristic', 'mcts', 'self' (default: 'heuristic')
%       Verbose     - Print status messages (default: true)
%
%   Outputs:
%       vecEnv - Vectorized environment or cell array of environments
%
%   Example:
%       % Create 8 parallel environments
%       vecEnv = createParallelEnv(8);
%
%       % Create environments without parallel pool
%       vecEnv = createParallelEnv(4, 'UseParpool', false);

    arguments
        numWorkers (1,1) double = 4
        options.UseParpool logical = true
        options.OpponentType char = 'heuristic'
        options.Verbose logical = true
    end

    %% Validate
    if numWorkers < 1
        error('numWorkers must be at least 1');
    end

    if options.Verbose
        fprintf('Creating %d parallel environments...\n', numWorkers);
    end

    %% Create environment instances
    envs = cell(numWorkers, 1);
    for i = 1:numWorkers
        envs{i} = TangledEnvironment();

        % Configure opponent type
        switch options.OpponentType
            case 'heuristic'
                envs{i}.Opponent = SimulatedOpponent('Style', 'heuristic');
            case 'mcts'
                envs{i}.Opponent = SimulatedOpponent('Style', 'mcts');
            case 'self'
                % Self-play: opponent will use same agent (set later)
                envs{i}.Opponent = [];
            otherwise
                error('Unknown opponent type: %s', options.OpponentType);
        end
    end

    %% Setup parallel execution
    if options.UseParpool
        % Check for Parallel Computing Toolbox
        if ~license('test', 'Distrib_Computing_Toolbox')
            warning('Parallel Computing Toolbox not available. Using sequential mode.');
            options.UseParpool = false;
        else
            % Create or get existing parallel pool
            pool = gcp('nocreate');
            if isempty(pool)
                if options.Verbose
                    fprintf('Starting parallel pool with %d workers...\n', numWorkers);
                end
                pool = parpool('local', min(numWorkers, feature('numcores')));
            elseif pool.NumWorkers < numWorkers
                if options.Verbose
                    fprintf('Resizing parallel pool to %d workers...\n', numWorkers);
                end
                delete(pool);
                pool = parpool('local', min(numWorkers, feature('numcores')));
            end

            if options.Verbose
                fprintf('Parallel pool ready with %d workers\n', pool.NumWorkers);
            end
        end
    end

    %% Create vectorized environment
    if options.UseParpool && exist('rlSimulationEnvironment', 'file')
        % Use RL Toolbox's simulation environment for parallel rollouts
        try
            vecEnv = rlSimulationEnvironment(envs);
            if options.Verbose
                fprintf('Created rlSimulationEnvironment with %d envs\n', numWorkers);
            end
        catch ME
            warning('Could not create rlSimulationEnvironment: %s', ME.message);
            warning('Falling back to cell array of environments');
            vecEnv = envs;
        end
    else
        % Return cell array for manual parallel iteration
        vecEnv = envs;
        if options.Verbose
            fprintf('Created %d independent environments (manual parallelism)\n', numWorkers);
        end
    end

    %% Summary
    if options.Verbose
        fprintf('\nParallel Environment Configuration:\n');
        fprintf('  Workers:      %d\n', numWorkers);
        fprintf('  Parallel:     %s\n', string(options.UseParpool));
        fprintf('  Opponent:     %s\n', options.OpponentType);
        fprintf('  Obs Size:     50\n');
        fprintf('  Action Space: 30 (15 edges × 2 colors)\n');
    end
end
