function build_rl_package(outputDir, options)
%BUILD_RL_PACKAGE Build Python package from MATLAB RL agent
%
%   build_rl_package()
%   build_rl_package(outputDir)
%   build_rl_package(outputDir, Name=Value)
%
%   Compiles the Tangled RL agent inference functions into a Python-callable
%   package using MATLAB Compiler SDK. The resulting package can be used
%   without a MATLAB installation (only MATLAB Runtime required).
%
%   Inputs:
%       outputDir - Output directory for compiled package (default: 'compiled')
%
%   Name-Value Arguments:
%       PackageName   - Name of Python package (default: 'tangled_rl_agent')
%       IncludeModel  - Include deployed model in package (default: true)
%       ModelPath     - Path to model to include (default: auto-detect)
%       Verbose       - Print build progress (default: true)
%
%   Requirements:
%       - MATLAB Compiler SDK license
%       - Reinforcement Learning Toolbox
%       - Deep Learning Toolbox
%
%   Example:
%       % Build package with default settings
%       build_rl_package();
%
%       % Build to specific directory
%       build_rl_package('C:\tangled_compiled', 'PackageName', 'tangled_agent');
%
%   After building:
%       1. Install MATLAB Runtime (free from MathWorks)
%       2. pip install <outputDir>/tangled_rl_agent
%       3. In Python:
%          import tangled_rl_agent
%          pkg = tangled_rl_agent.initialize()
%          action, value, probs = pkg.tangled_agent_inference(state, mask)

    arguments
        outputDir char = fullfile(pwd, 'compiled')
        options.PackageName char = 'tangled_rl_agent'
        options.IncludeModel logical = true
        options.ModelPath char = ''
        options.Verbose logical = true
    end

    %% Check prerequisites
    if options.Verbose
        fprintf('\n=== Building Tangled RL Agent Package ===\n\n');
    end

    % Check for Compiler SDK
    if ~license('test', 'MATLAB_Builder_for_Python_Interface')
        error(['MATLAB Compiler SDK not licensed. ' ...
               'Required for building Python packages.']);
    end

    % Check for required toolboxes
    requiredToolboxes = {
        'Reinforcement Learning Toolbox'
        'Deep Learning Toolbox'
    };

    v = ver;
    installedToolboxes = {v.Name};

    for i = 1:length(requiredToolboxes)
        if ~any(strcmp(installedToolboxes, requiredToolboxes{i}))
            error('Required toolbox not installed: %s', requiredToolboxes{i});
        end
    end

    if options.Verbose
        fprintf('Prerequisites checked:\n');
        fprintf('  [OK] MATLAB Compiler SDK\n');
        fprintf('  [OK] Reinforcement Learning Toolbox\n');
        fprintf('  [OK] Deep Learning Toolbox\n\n');
    end

    %% Prepare output directory
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    packageDir = fullfile(outputDir, options.PackageName);

    %% Locate source files
    sourceDir = fileparts(mfilename('fullpath'));

    functionFiles = {
        fullfile(sourceDir, 'tangled_agent_inference.m')
    };

    % Verify files exist
    for i = 1:length(functionFiles)
        if ~exist(functionFiles{i}, 'file')
            error('Source file not found: %s', functionFiles{i});
        end
    end

    if options.Verbose
        fprintf('Source files:\n');
        for i = 1:length(functionFiles)
            [~, name, ext] = fileparts(functionFiles{i});
            fprintf('  %s%s\n', name, ext);
        end
        fprintf('\n');
    end

    %% Handle model inclusion
    additionalFiles = {};

    if options.IncludeModel
        if isempty(options.ModelPath)
            % Auto-detect deployed model
            deployedModel = fullfile(sourceDir, 'deployed', 'current_model.mat');
            if exist(deployedModel, 'file')
                options.ModelPath = deployedModel;
            end
        end

        if ~isempty(options.ModelPath) && exist(options.ModelPath, 'file')
            % Create deployed directory structure
            deployDir = fullfile(outputDir, 'deployed');
            if ~exist(deployDir, 'dir')
                mkdir(deployDir);
            end

            % Copy model
            copyfile(options.ModelPath, fullfile(deployDir, 'current_model.mat'));
            additionalFiles{end+1} = fullfile(deployDir, 'current_model.mat');

            if options.Verbose
                fprintf('Including deployed model: %s\n\n', options.ModelPath);
            end
        else
            if options.Verbose
                fprintf('No deployed model found - package will use fallback inference\n\n');
            end
        end
    end

    %% Build package
    if options.Verbose
        fprintf('Building Python package: %s\n', options.PackageName);
        fprintf('Output directory: %s\n\n', packageDir);
    end

    try
        % Build using compiler.build.pythonPackage
        buildResults = compiler.build.pythonPackage(...
            functionFiles, ...
            'PackageName', options.PackageName, ...
            'OutputDir', packageDir, ...
            'AdditionalFiles', additionalFiles, ...
            'Verbose', options.Verbose);

        if options.Verbose
            fprintf('\n=== Build Successful ===\n\n');
            fprintf('Package location: %s\n', packageDir);
            fprintf('\nTo install:\n');
            fprintf('  1. Install MATLAB Runtime R%s from:\n', version('-release'));
            fprintf('     https://www.mathworks.com/products/compiler/matlab-runtime.html\n');
            fprintf('  2. pip install %s\n', packageDir);
            fprintf('\nTo use in Python:\n');
            fprintf('  import %s\n', options.PackageName);
            fprintf('  pkg = %s.initialize()\n', options.PackageName);
            fprintf('  action, value, probs = pkg.tangled_agent_inference(state, mask)\n');
            fprintf('\n');
        end

    catch ME
        if options.Verbose
            fprintf('\n=== Build Failed ===\n');
            fprintf('Error: %s\n', ME.message);
        end
        rethrow(ME);
    end
end
