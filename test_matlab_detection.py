#!/usr/bin/env python3
"""Test MATLAB detection system."""

import sys
from pathlib import Path

# Direct import without going through package __init__
matlab_config_path = Path(__file__).parent / "snowdrop_tangled_agents" / "matlab"
sys.path.insert(0, str(matlab_config_path))

import matlab_config

print("=" * 60)
print("MATLAB Detection Test")
print("=" * 60)

# Test find_matlab_installation
print("\n1. Finding MATLAB installation...")
matlab_root = matlab_config.find_matlab_installation()

if matlab_root:
    print(f"   [OK] Found: {matlab_root}")
else:
    print("   [FAIL] Not found")
    sys.exit(1)

# Test get_matlab_paths
print("\n2. Getting MATLAB paths...")
root, bin_path, runtime_path = matlab_config.get_matlab_paths()

print(f"   Root:    {root}")
print(f"   Bin:     {bin_path}")
print(f"   Runtime: {runtime_path}")

# Test setup_matlab_path
print("\n3. Setting up MATLAB PATH...")
success = matlab_config.setup_matlab_path()
print(f"   {'[OK] Success' if success else '[FAIL] Failed'}")

# Test strategies directory
print("\n4. Finding strategies directory...")
strategies_dir = matlab_config.get_strategies_dir()
if strategies_dir:
    print(f"   [OK] Found: {strategies_dir}")
else:
    print("   [SKIP] Not found (optional)")

print("\n" + "=" * 60)
print("MATLAB detection successful!")
print("=" * 60)

# Part 2: Query MATLAB capabilities (requires MATLAB Engine)
print("\n" + "=" * 60)
print("MATLAB Capabilities (requires Engine API)")
print("=" * 60)

try:
    import matlab.engine
    print("\n6. Starting MATLAB Engine...")

    # Try to connect to existing session first
    sessions = matlab.engine.find_matlab()
    if sessions:
        print(f"   [OK] Found existing session: {sessions[0]}")
        eng = matlab.engine.connect_matlab(sessions[0])
    else:
        print("   Starting new MATLAB session (this may take 10-20 seconds)...")
        eng = matlab.engine.start_matlab("-nodesktop -nosplash")
        print("   [OK] New session started")

    # Query parallel computing configuration
    print("\n7. Parallel Computing Configuration...")
    parallel_config = matlab_config.get_parallel_config(eng)

    if parallel_config['available']:
        print(f"   [OK] Workers: {parallel_config['workers']}")
        print(f"   [OK] Cluster: {parallel_config['cluster_type']}")
    else:
        print(f"   [WARN] Not available: {parallel_config.get('error', 'unknown')}")

    # Query GPU configuration
    print("\n8. GPU Configuration...")
    gpu_config = matlab_config.get_gpu_config(eng)

    if gpu_config['available']:
        print(f"   [OK] GPU Count: {gpu_config['count']}")

        for device in gpu_config['devices']:
            if 'error' in device:
                print(f"   [WARN] GPU {device['index']}: {device['error']}")
            else:
                print(f"\n   GPU {device['index']}: {device['name']}")
                print(f"      Compute Capability: {device['compute_capability']}")
                print(f"      Total Memory: {device['total_memory_gb']:.2f} GB")
                print(f"      Available Memory: {device['available_memory_gb']:.2f} GB")

        if gpu_config.get('validated'):
            print(f"\n   [OK] GPU Validation: PASSED")
        else:
            print(f"\n   [WARN] GPU Validation: FAILED or SKIPPED")

        # Show validation output if available
        if gpu_config.get('validation_output'):
            print("\n   Validation Output:")
            for line in gpu_config['validation_output'].split('\n'):
                if line.strip():
                    print(f"      {line}")
    else:
        error_msg = gpu_config.get('error', 'No GPUs found')
        if 'No CUDA' in error_msg or gpu_config['count'] == 0:
            print(f"   [INFO] No GPUs available")
        else:
            print(f"   [WARN] GPU query failed: {error_msg}")

    # Don't quit if connected to existing session
    if not sessions:
        print("\n9. Closing MATLAB session...")
        eng.quit()
        print("   [OK] Session closed")
    else:
        print("\n9. Leaving existing session running...")

    print("\n" + "=" * 60)
    print("MATLAB capabilities query complete!")
    print("=" * 60)

except ImportError:
    print("\n[SKIP] MATLAB Engine API not installed")
    print("   Run: pip install matlabengine")
    print("\n" + "=" * 60)
    print("Basic detection successful, Engine API unavailable")
    print("=" * 60)

except Exception as e:
    print(f"\n[ERROR] Failed to query MATLAB capabilities: {e}")
    print("\n" + "=" * 60)
    print("Basic detection successful, capability query failed")
    print("=" * 60)
