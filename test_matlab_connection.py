#!/usr/bin/env python3
"""Test MATLAB Engine connection."""

import os
import sys

# Add MATLAB to PATH before importing
matlab_bin = r"C:\Program Files\MATLAB\R2026a\bin\win64"
matlab_runtime = r"C:\Program Files\MATLAB\R2026a\runtime\win64"

if matlab_bin not in os.environ.get('PATH', ''):
    os.environ['PATH'] = matlab_bin + os.pathsep + os.environ.get('PATH', '')
if matlab_runtime not in os.environ.get('PATH', ''):
    os.environ['PATH'] = matlab_runtime + os.pathsep + os.environ.get('PATH', '')

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
            eng.addpath(r"C:\Users\murr2\MATLAB Drive\tangled_strategies", nargout=0)
            print("Added tangled_strategies to path")

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
