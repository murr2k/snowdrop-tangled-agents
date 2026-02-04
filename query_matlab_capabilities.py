#!/usr/bin/env python3
"""
Query MATLAB Parallel Computing and GPU Capabilities.

This script connects to MATLAB Engine and queries:
- Parallel computing workers
- GPU device count and specifications
- GPU validation status

Prerequisites:
- MATLAB Engine API for Python installed (pip install matlabengine)
- MATLAB installation detected by matlab_config
"""

import sys
from pathlib import Path

# Direct import without going through package __init__
matlab_config_path = Path(__file__).parent / "snowdrop_tangled_agents" / "matlab"
sys.path.insert(0, str(matlab_config_path))

import matlab_config

print("=" * 70)
print("MATLAB Capabilities Query")
print("=" * 70)

# Check MATLAB installation first
print("\nStep 1: Checking MATLAB installation...")
matlab_root = matlab_config.find_matlab_installation()
if not matlab_root:
    print("   [ERROR] MATLAB not found!")
    print("   Run test_matlab_detection.py first")
    sys.exit(1)

print(f"   [OK] MATLAB found: {matlab_root}")

# Try to import MATLAB Engine
print("\nStep 2: Importing MATLAB Engine API...")
try:
    import matlab.engine
    print("   [OK] MATLAB Engine API available")
except ImportError:
    print("   [ERROR] MATLAB Engine API not installed")
    print("\n   To install:")
    print("   1. Navigate to MATLAB's extern/engines/python directory:")
    print(f"      cd \"{matlab_root}\\extern\\engines\\python\"")
    print("   2. Run: python setup.py install")
    print("   3. Or install from PyPI: pip install matlabengine")
    sys.exit(1)

# Connect to MATLAB
print("\nStep 3: Connecting to MATLAB Engine...")
print("   Looking for existing sessions...")

sessions = matlab.engine.find_matlab()
if sessions:
    print(f"   [OK] Found existing session: {sessions[0]}")
    print("   Connecting...")
    eng = matlab.engine.connect_matlab(sessions[0])
    print("   [OK] Connected to existing session")
    close_session = False
else:
    print("   No existing sessions found")
    print("   Starting new MATLAB session (10-20 seconds)...")
    eng = matlab.engine.start_matlab("-nodesktop -nosplash")
    print("   [OK] New session started")
    close_session = True

# Query parallel computing configuration
print("\n" + "=" * 70)
print("Parallel Computing Configuration")
print("=" * 70)

parallel_config = matlab_config.get_parallel_config(eng)

if parallel_config['available']:
    print(f"\nStatus: Available")
    print(f"Workers: {parallel_config['workers']}")
    print(f"Cluster Profile: {parallel_config['cluster_type']}")

    if parallel_config['workers'] > 1:
        print(f"\n[INFO] Parallel training can use up to {parallel_config['workers']} workers")
    else:
        print(f"\n[WARN] Only 1 worker configured - parallel training will be limited")
        print("       To increase workers, in MATLAB run:")
        print("       >> pc = parcluster;")
        print("       >> pc.NumWorkers = 6;  % or desired number")
        print("       >> saveProfile(pc);")
else:
    print(f"\nStatus: Not Available")
    print(f"Error: {parallel_config.get('error', 'Unknown error')}")
    print("\n[WARN] Parallel Computing Toolbox may not be installed")

# Query GPU configuration
print("\n" + "=" * 70)
print("GPU Configuration")
print("=" * 70)

gpu_config = matlab_config.get_gpu_config(eng)

if gpu_config['available']:
    print(f"\nGPU Count: {gpu_config['count']}")

    if gpu_config['count'] == 0:
        print("\n[INFO] No GPUs found")
    else:
        for i, device in enumerate(gpu_config['devices'], 1):
            print(f"\n{'─' * 70}")
            print(f"GPU {device['index']}: {device['name']}")
            print(f"{'─' * 70}")

            if 'error' in device:
                print(f"[ERROR] {device['error']}")
            else:
                print(f"Compute Capability:  {device['compute_capability']}")
                print(f"Total Memory:        {device['total_memory_gb']:.2f} GB")
                print(f"Available Memory:    {device['available_memory_gb']:.2f} GB")

                # Check compute capability for deep learning
                try:
                    major, minor = device['compute_capability'].split('.')
                    compute_cap_num = float(f"{major}.{minor}")

                    if compute_cap_num >= 3.5:
                        print(f"Deep Learning:       Supported (CC >= 3.5)")
                    else:
                        print(f"Deep Learning:       Not Supported (CC < 3.5)")
                except Exception:
                    pass

        # Show GPU validation results
        print(f"\n{'─' * 70}")
        print("GPU Validation")
        print(f"{'─' * 70}")

        if gpu_config.get('validated'):
            print("\n[OK] GPU Validation: PASSED")
            print("     All GPU tests completed successfully")
        else:
            print("\n[WARN] GPU Validation: FAILED or SKIPPED")

        # Show detailed validation output
        if gpu_config.get('validation_output'):
            print("\nDetailed Validation Output:")
            print("─" * 70)
            for line in gpu_config['validation_output'].split('\n'):
                if line.strip():
                    # Highlight PASSED/FAILED
                    if 'PASSED' in line:
                        print(f"   {line}")
                    elif 'FAILED' in line:
                        print(f"   [!] {line}")
                    else:
                        print(f"   {line}")
            print("─" * 70)

        # Summary recommendations
        print("\n" + "=" * 70)
        print("Training Recommendations")
        print("=" * 70)

        if gpu_config['count'] > 0 and gpu_config.get('validated'):
            print("\n[OK] GPU-accelerated training is available!")
            print("     Configure your training to use GPU:")
            print("     - Set training option: UseGPU=true")
            print("     - Ensure batch size fits in GPU memory")
            print(f"     - Available GPU memory: {gpu_config['devices'][0]['available_memory_gb']:.2f} GB")
        elif gpu_config['count'] > 0:
            print("\n[WARN] GPU detected but validation failed")
            print("       Training may fall back to CPU")
        else:
            print("\n[INFO] No GPU available - training will use CPU")
            print("       Consider using GPU for faster training")

else:
    error_msg = gpu_config.get('error', 'Unknown error')
    if 'No CUDA' in error_msg or gpu_config['count'] == 0:
        print("\nStatus: No GPUs Available")
        print("\n[INFO] System does not have CUDA-capable GPUs")
        print("       Training will use CPU only")
    else:
        print("\nStatus: Query Failed")
        print(f"Error: {error_msg}")

# Close session if we started it
print("\n" + "=" * 70)
if close_session:
    print("Closing MATLAB session...")
    try:
        eng.quit()
        print("[OK] Session closed")
    except Exception as e:
        print(f"[WARN] Error closing session: {e}")
else:
    print("Leaving existing MATLAB session running...")
    print("[OK] Session still active")

print("=" * 70)
print("\nCapability query complete!")
print("=" * 70)
