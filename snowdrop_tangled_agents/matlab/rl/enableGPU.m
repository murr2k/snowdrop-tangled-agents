function agent = enableGPU(agent)
%ENABLEGPU Move agent networks to GPU for faster training
%
%   agent = enableGPU(agent)
%
%   Transfers the actor and critic networks to GPU memory for accelerated
%   training. Falls back gracefully if GPU is not available.
%
%   Inputs:
%       agent - PPO agent from createPPOAgent
%
%   Outputs:
%       agent - Agent with networks on GPU (or unchanged if GPU unavailable)
%
%   Notes:
%       - Requires NVIDIA GPU with CUDA support
%       - Requires Parallel Computing Toolbox
%       - Automatically detects GPU availability
%
%   Example:
%       agent = createPPOAgent(env);
%       agent = enableGPU(agent);

    %% Check GPU availability
    if ~canUseGPU()
        warning('enableGPU:NoGPU', ...
            'GPU not available. Training will use CPU.\n%s', ...
            getGPUDiagnostics());
        return;
    end

    %% Get GPU info
    gpu = gpuDevice();
    fprintf('GPU detected: %s (%.1f GB memory)\n', ...
        gpu.Name, gpu.TotalMemory / 1e9);

    %% Move actor network to GPU
    try
        actor = getActor(agent);
        actorNet = getModel(actor);

        % Move network parameters to GPU
        actorNet = dlupdate(@gpuArray, actorNet);

        % Set back to actor
        actor = setModel(actor, actorNet);
        agent = setActor(agent, actor);

        fprintf('  Actor network moved to GPU\n');
    catch ME
        warning('enableGPU:ActorFailed', ...
            'Failed to move actor to GPU: %s', ME.message);
    end

    %% Move critic network to GPU
    try
        critic = getCritic(agent);
        criticNet = getModel(critic);

        % Move network parameters to GPU
        criticNet = dlupdate(@gpuArray, criticNet);

        % Set back to critic
        critic = setModel(critic, criticNet);
        agent = setCritic(agent, critic);

        fprintf('  Critic network moved to GPU\n');
    catch ME
        warning('enableGPU:CriticFailed', ...
            'Failed to move critic to GPU: %s', ME.message);
    end

    %% Verify
    fprintf('GPU acceleration enabled\n');
end

function available = canUseGPU()
%CANUSEGPU Check if GPU is available for computation

    available = false;

    % Check for Parallel Computing Toolbox
    if ~license('test', 'Distrib_Computing_Toolbox')
        return;
    end

    % Check for GPU device
    try
        gpu = gpuDevice();
        available = gpu.DeviceAvailable;
    catch
        available = false;
    end
end

function msg = getGPUDiagnostics()
%GETGPUDIAGNOSTICS Get diagnostic information about GPU availability

    msg = '';

    % Check toolbox
    if ~license('test', 'Distrib_Computing_Toolbox')
        msg = [msg 'Parallel Computing Toolbox not licensed.\n'];
        return;
    end

    % Check CUDA
    try
        gpu = gpuDevice();
        if ~gpu.DeviceAvailable
            msg = [msg sprintf('GPU %s is not available.\n', gpu.Name)];
        end
    catch ME
        msg = [msg sprintf('GPU detection failed: %s\n', ME.message)];
    end

    if isempty(msg)
        msg = 'Unknown GPU issue.';
    end
end
