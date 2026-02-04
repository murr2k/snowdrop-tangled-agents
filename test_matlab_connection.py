#!/usr/bin/env python3
"""Test MATLAB Engine connection."""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from snowdrop_tangled_agents.matlab.matlab_config import (
    find_matlab_installation,
    setup_matlab_path,
    get_matlab_paths,
    get_strategies_dir
)

# Detect and setup MATLAB paths
matlab_root, matlab_bin, matlab_runtime = get_matlab_paths()

if not matlab_root:
    print("ERROR: MATLAB installation not found!")
    print("\nSearched locations:")
    print("  - Environment variable MATLAB_ROOT")
    print("  - C:\\Program Files\\MATLAB\\R20XXx")
    print("  - PATH for matlab executable")
    print("\nTo fix:")
    print("  1. Set MATLAB_ROOT environment variable, OR")
    print("  2. Ensure MATLAB is installed in a standard location")
    sys.exit(1)

print(f"Found MATLAB installation: {matlab_root}")
print(f"  Bin: {matlab_bin}")
print(f"  Runtime: {matlab_runtime}")

# Add to PATH
setup_matlab_path()
print(f"MATLAB paths added to PATH")

try:
    import matlab.engine
    print("MATLAB Engine module imported successfully")
except ImportError as e:
    print(f"Failed to import MATLAB Engine: {e}")
    sys.exit(1)

# Try to find shared sessions
print("\nLooking for shared MATLAB sessions...")
sessions = matlab.engine.find_matlab()
print(f"Found sessions: {sessions}")

if sessions:
    print(f"\nConnecting to shared session: {sessions[0]}")
    try:
        eng = matlab.engine.connect_matlab(sessions[0])
        print("Connected!")

        # Test basic operation
        result = eng.sqrt(4.0)
        print(f"Test: sqrt(4) = {result}")

        # Test our strategy functions if available
        try:
            strategies_dir = get_strategies_dir()
            if strategies_dir:
                eng.addpath(str(strategies_dir), nargout=0)
                print(f"Added tangled_strategies to path: {strategies_dir}")
            else:
                print("Warning: tangled_strategies directory not found")

            # Test evaluate_position
            import matlab
            state = matlab.double([0.0] * 15)  # All grey
            turn = matlab.double([1.0])
            value, policy = eng.evaluate_position(state, turn, nargout=2)
            print(f"evaluate_position test: value={value}")
        except Exception as e:
            print(f"Strategy function test failed: {e}")

        print("\nMATLAB connection successful!")
        # Don't quit - leave session connected

    except Exception as e:
        print(f"Failed to connect: {e}")
else:
    print("\nNo shared sessions found.")
    print("Please run 'matlab.engine.shareEngine' in MATLAB Command Window")
    print("Then run this script again.")

    print("\nAttempting to start new MATLAB session...")
    try:
        eng = matlab.engine.start_matlab()
        print("New session started!")
        result = eng.sqrt(4.0)
        print(f"Test: sqrt(4) = {result}")
        eng.quit()
        print("Session closed.")
    except Exception as e:
        print(f"Failed to start new session: {e}")
        print("\nThis may be due to missing MATLAB runtime dependencies.")
        print("Try running MATLAB and sharing the engine instead.")
