"""
MATLAB RL System Regression Tests

This module provides pytest integration for MATLAB RL tests, enabling them
to run as part of the standard test suite and CI/CD pipeline.

Tests are skipped gracefully if MATLAB is not installed.

Usage:
    pytest -v -m matlab                    # Run only MATLAB tests
    pytest -v                              # Run all tests (MATLAB skipped if unavailable)
    pytest -v --matlab-timeout=600         # Custom timeout (default: 300s)

Environment Variables:
    MATLAB_PATH: Override MATLAB executable path
    SKIP_MATLAB_TESTS: Set to "1" to skip MATLAB tests entirely
"""

import os
import shutil
import subprocess
import json
from pathlib import Path
from typing import Tuple, Optional

import pytest


# Custom pytest marker for MATLAB tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "matlab: marks tests requiring MATLAB installation"
    )


def find_matlab() -> Optional[str]:
    """Find MATLAB executable on the system."""
    # Import centralized config
    from snowdrop_tangled_agents.matlab.matlab_config import get_matlab_paths

    # Check environment variable first (for explicit override)
    matlab_path = os.environ.get('MATLAB_PATH')
    if matlab_path and Path(matlab_path).exists():
        return matlab_path

    # Use centralized detection
    matlab_root, matlab_bin, _ = get_matlab_paths()
    if matlab_root and matlab_bin:
        if os.name == 'nt':
            matlab_exe = matlab_bin.parent / 'matlab.exe'  # bin\matlab.exe
        else:
            matlab_exe = matlab_bin.parent / 'matlab'

        if matlab_exe.exists():
            return str(matlab_exe)

    # Try PATH as fallback
    matlab = shutil.which('matlab')
    if matlab:
        return matlab

    return None


def matlab_available() -> bool:
    """Check if MATLAB is available."""
    if os.environ.get('SKIP_MATLAB_TESTS') == '1':
        return False
    return find_matlab() is not None


def run_matlab_tests(
    test_command: str,
    timeout: int = 300,
    working_dir: Optional[Path] = None
) -> Tuple[int, str, str]:
    """
    Run MATLAB tests and return results.

    Args:
        test_command: MATLAB command to execute
        timeout: Timeout in seconds
        working_dir: Working directory for MATLAB

    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    matlab_exe = find_matlab()
    if not matlab_exe:
        raise RuntimeError("MATLAB not found")

    if working_dir is None:
        working_dir = Path(__file__).parent.parent / 'matlab' / 'rl'

    # Build command
    cmd = [matlab_exe, '-batch', test_command]

    # Run MATLAB
    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
        cwd=str(working_dir),
        text=True
    )

    return result.returncode, result.stdout, result.stderr


# Skip decorator for MATLAB tests
skip_if_no_matlab = pytest.mark.skipif(
    not matlab_available(),
    reason="MATLAB not installed or SKIP_MATLAB_TESTS=1"
)


@pytest.mark.matlab
@skip_if_no_matlab
class TestMatlabRLQuick:
    """Quick MATLAB RL tests (< 30 seconds)."""

    def test_environment_creation(self):
        """Test TangledEnvironment can be created."""
        returncode, stdout, stderr = run_matlab_tests(
            "env = TangledEnvironment(); "
            "assert(~isempty(env), 'Environment creation failed'); "
            "disp('PASSED: Environment creation')",
            timeout=60
        )
        # Check for PASSED in output (MATLAB may crash during shutdown but test passed)
        assert 'PASSED: Environment creation' in stdout, f"MATLAB test failed:\n{stdout}\n{stderr}"

    def test_action_mask(self):
        """Test action masking works correctly."""
        returncode, stdout, stderr = run_matlab_tests(
            "mask = getActionMask(repmat('-', 1, 15)); "
            "assert(sum(mask) == 30, 'All actions should be valid on empty board'); "
            "mask2 = getActionMask('GP-------------'); "
            "assert(sum(mask2) == 26, 'Should have 26 valid actions'); "
            "disp('PASSED: Action masking')",
            timeout=60
        )
        # Check for PASSED in output (MATLAB may crash during shutdown but test passed)
        assert 'PASSED: Action masking' in stdout, f"MATLAB test failed:\n{stdout}\n{stderr}"

    def test_ppo_agent_creation(self):
        """Test PPO agent can be created."""
        returncode, stdout, stderr = run_matlab_tests(
            "env = TangledEnvironment(); "
            "agent = createPPOAgent(env); "
            "assert(~isempty(agent), 'Agent creation failed'); "
            "disp('PASSED: PPO agent creation')",
            timeout=120
        )
        # Check for PASSED in output (MATLAB may crash during shutdown but test passed)
        assert 'PASSED: PPO agent creation' in stdout, f"MATLAB test failed:\n{stdout}\n{stderr}"


@pytest.mark.matlab
@skip_if_no_matlab
class TestMatlabRLFull:
    """Full MATLAB RL test suite."""

    def test_run_all_tests_quick(self):
        """Run the full quick test suite."""
        returncode, stdout, stderr = run_matlab_tests(
            "results = run_all_tests('quick'); "
            "disp(['TESTS_PASSED=' num2str(results.passed)]); "
            "disp(['TESTS_FAILED=' num2str(results.failed)]); "
            "exit(results.failed)",
            timeout=300
        )

        # Parse results from stdout
        passed = 0
        failed = 0
        for line in stdout.split('\n'):
            if line.startswith('TESTS_PASSED='):
                passed = int(line.split('=')[1])
            elif line.startswith('TESTS_FAILED='):
                failed = int(line.split('=')[1])

        assert returncode == 0, f"MATLAB tests failed ({failed} failures):\n{stdout}"
        assert failed == 0, f"{failed} MATLAB test(s) failed:\n{stdout}"
        assert passed > 0, f"No tests ran:\n{stdout}"

    def test_deployment_pipeline(self):
        """Run the deployment pipeline tests."""
        returncode, stdout, stderr = run_matlab_tests(
            "test_deployment",
            timeout=300
        )

        # Check for success message
        assert 'SUCCESS: All tests passed!' in stdout, \
            f"Deployment tests failed:\n{stdout}\n{stderr}"


@pytest.mark.matlab
@pytest.mark.slow
@skip_if_no_matlab
class TestMatlabRLTraining:
    """MATLAB RL training tests (slow, requires --runslow flag)."""

    def test_run_all_tests_full(self):
        """Run the full test suite including training tests."""
        returncode, stdout, stderr = run_matlab_tests(
            "results = run_all_tests('full'); "
            "disp(['TESTS_PASSED=' num2str(results.passed)]); "
            "disp(['TESTS_FAILED=' num2str(results.failed)]); "
            "exit(results.failed)",
            timeout=600
        )

        # Parse results
        failed = 0
        for line in stdout.split('\n'):
            if line.startswith('TESTS_FAILED='):
                failed = int(line.split('=')[1])

        assert returncode == 0, f"MATLAB full tests failed:\n{stdout}"
        assert failed == 0, f"{failed} MATLAB test(s) failed"


# Standalone runner for debugging
if __name__ == '__main__':
    print(f"MATLAB available: {matlab_available()}")
    if matlab_available():
        print(f"MATLAB path: {find_matlab()}")
        print("\nRunning quick test suite...")
        returncode, stdout, stderr = run_matlab_tests(
            "results = run_all_tests('quick'); exit(results.failed)",
            timeout=300
        )
        print(stdout)
        if stderr:
            print(f"STDERR:\n{stderr}")
        print(f"\nReturn code: {returncode}")
