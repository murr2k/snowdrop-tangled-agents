function workerInit(workerID, options)
%WORKERINIT Initialize parallel worker for training
%
%   workerInit(workerID)
%   workerInit(workerID, Name=Value)
%
%   Initializes a parallel worker with necessary resources for training.
%   Called at the start of each parfor worker's lifecycle.
%
%   Inputs:
%       workerID - Unique identifier for this worker (1 to NumWorkers)
%
%   Name-Value Arguments:
%       Seed      - Random seed base (actual seed = Seed + workerID)
%       UseGPU    - Enable GPU on this worker (default: false)
%       Verbose   - Print initialization info (default: true)
%       CachePath - Path for worker-local cache (default: tempdir)
%
%   Example:
%       % Initialize worker 3 with deterministic seed
%       workerInit(3, 'Seed', 42);
%
%   Notes:
%       - Each worker gets a unique random seed to ensure diversity
%       - GPU can be shared across workers or disabled per-worker
%       - Worker-local caches prevent file contention

    arguments
        workerID (1,1) double
        options.Seed (1,1) double = 0
        options.UseGPU logical = false
        options.Verbose logical = true
        options.CachePath char = tempdir
    end

    %% Set random seed for reproducibility with diversity
    if options.Seed > 0
        actualSeed = options.Seed + workerID;
    else
        % Use time-based seed with worker offset
        actualSeed = mod(round(posixtime(datetime('now')) * 1000) + workerID * 1000, 2^31);
    end

    rng(actualSeed, 'twister');

    if options.Verbose
        fprintf('Worker %d: Random seed set to %d\n', workerID, actualSeed);
    end

    %% Setup GPU if requested
    if options.UseGPU
        try
            % Get number of available GPUs
            numGPUs = gpuDeviceCount();

            if numGPUs > 0
                % Assign GPU to worker (round-robin)
                gpuIdx = mod(workerID - 1, numGPUs) + 1;
                gpuDevice(gpuIdx);

                if options.Verbose
                    gpu = gpuDevice();
                    fprintf('Worker %d: Using GPU %d (%s)\n', ...
                        workerID, gpuIdx, gpu.Name);
                end
            else
                if options.Verbose
                    fprintf('Worker %d: No GPU available, using CPU\n', workerID);
                end
            end
        catch ME
            warning('Worker %d: GPU setup failed: %s', workerID, ME.message);
        end
    end

    %% Setup worker-local cache directory
    workerCachePath = fullfile(options.CachePath, sprintf('worker_%d', workerID));
    if ~exist(workerCachePath, 'dir')
        mkdir(workerCachePath);
    end

    if options.Verbose
        fprintf('Worker %d: Cache path: %s\n', workerID, workerCachePath);
    end

    %% Verify environment classes are available
    try
        % Test that TangledEnvironment can be instantiated
        testEnv = TangledEnvironment();
        delete(testEnv);

        if options.Verbose
            fprintf('Worker %d: TangledEnvironment verified\n', workerID);
        end
    catch ME
        error('Worker %d: Failed to create TangledEnvironment: %s', ...
            workerID, ME.message);
    end

    %% Worker ready
    if options.Verbose
        fprintf('Worker %d: Initialization complete\n', workerID);
    end
end

function count = gpuDeviceCount()
%GPUDEVICECOUNT Get number of available GPUs

    count = 0;

    if ~license('test', 'Distrib_Computing_Toolbox')
        return;
    end

    try
        % Query all GPU devices
        count = parallel.gpu.GPUDevice.getDeviceCount();
    catch
        count = 0;
    end
end
