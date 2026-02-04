"""
MATLAB Configuration and Detection.

Provides centralized MATLAB installation detection and path management.
All MATLAB-related modules should use this to find MATLAB installations.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def find_matlab_installation() -> Optional[Path]:
    """
    Find MATLAB installation on the system.

    Checks in priority order:
    1. MATLAB_ROOT environment variable
    2. Common installation directories (newest first)
    3. PATH for matlab executable

    Returns:
        Path to MATLAB root directory, or None if not found
    """
    # Check environment variable first
    matlab_root_env = os.environ.get('MATLAB_ROOT')
    if matlab_root_env:
        matlab_root = Path(matlab_root_env)
        if matlab_root.exists() and (matlab_root / 'bin').exists():
            logger.info(f"Found MATLAB via MATLAB_ROOT: {matlab_root}")
            return matlab_root

    # Check common installation locations (Windows)
    if os.name == 'nt':
        program_files = Path(r"C:\Program Files\MATLAB")
        if program_files.exists():
            # Look for version folders, prefer newer versions
            versions = [
                'R2026a', 'R2025b', 'R2025a', 'R2024b', 'R2024a',
                'R2023b', 'R2023a', 'R2022b', 'R2022a'
            ]
            for version in versions:
                candidate = program_files / version
                if candidate.exists() and (candidate / 'bin').exists():
                    logger.info(f"Found MATLAB installation: {candidate}")
                    return candidate

    # Check common installation locations (Unix/Mac)
    else:
        common_roots = [
            Path('/Applications/MATLAB'),  # macOS
            Path('/usr/local/MATLAB'),     # Linux
            Path.home() / 'MATLAB'          # User install
        ]
        for root in common_roots:
            if root.exists():
                # Look for version folders
                versions = sorted(
                    [d for d in root.iterdir() if d.is_dir() and d.name.startswith('R')],
                    reverse=True
                )
                if versions:
                    candidate = versions[0]
                    if (candidate / 'bin').exists():
                        logger.info(f"Found MATLAB installation: {candidate}")
                        return candidate

    # Try to find matlab executable in PATH
    import shutil
    matlab_exe = shutil.which('matlab')
    if matlab_exe:
        # Derive root from executable path
        # Typical: /path/to/MATLAB/R2025a/bin/matlab
        exe_path = Path(matlab_exe)
        if exe_path.name == 'matlab' or exe_path.name == 'matlab.exe':
            matlab_root = exe_path.parent.parent  # Go up from bin/
            if (matlab_root / 'bin').exists():
                logger.info(f"Found MATLAB via PATH: {matlab_root}")
                return matlab_root

    logger.warning("MATLAB installation not found")
    return None


def get_matlab_paths() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Get MATLAB binary and runtime paths.

    Returns:
        Tuple of (matlab_root, bin_path, runtime_path)
        Each element is None if MATLAB is not found.
    """
    matlab_root = find_matlab_installation()
    if not matlab_root:
        return None, None, None

    if os.name == 'nt':
        bin_path = matlab_root / 'bin' / 'win64'
        runtime_path = matlab_root / 'runtime' / 'win64'
    elif os.name == 'darwin':
        bin_path = matlab_root / 'bin' / 'maci64'
        runtime_path = matlab_root / 'runtime' / 'maci64'
    else:  # Linux
        bin_path = matlab_root / 'bin' / 'glnxa64'
        runtime_path = matlab_root / 'runtime' / 'glnxa64'

    # Verify paths exist
    if not bin_path.exists():
        logger.warning(f"MATLAB bin path does not exist: {bin_path}")
        bin_path = None
    if not runtime_path.exists():
        logger.warning(f"MATLAB runtime path does not exist: {runtime_path}")
        runtime_path = None

    return matlab_root, bin_path, runtime_path


def setup_matlab_path() -> bool:
    """
    Add MATLAB paths to system PATH environment variable.

    Returns:
        True if MATLAB paths were added, False otherwise
    """
    matlab_root, bin_path, runtime_path = get_matlab_paths()

    if not bin_path:
        logger.warning("Cannot setup MATLAB PATH: installation not found")
        return False

    # Add to PATH if not already present
    current_path = os.environ.get('PATH', '')
    paths_to_add = []

    if str(bin_path) not in current_path:
        paths_to_add.append(str(bin_path))

    if runtime_path and str(runtime_path) not in current_path:
        paths_to_add.append(str(runtime_path))

    if paths_to_add:
        new_path = os.pathsep.join(paths_to_add + [current_path])
        os.environ['PATH'] = new_path
        logger.info(f"Added MATLAB paths to PATH: {', '.join(paths_to_add)}")
        return True

    logger.debug("MATLAB paths already in PATH")
    return True


def get_matlab_drive() -> Optional[Path]:
    """
    Get MATLAB Drive path.

    Returns:
        Path to MATLAB Drive, or None if not found
    """
    # Common MATLAB Drive locations
    candidates = [
        Path.home() / "MATLAB Drive",
        Path.home() / "Documents" / "MATLAB Drive",
    ]

    for candidate in candidates:
        if candidate.exists():
            logger.info(f"Found MATLAB Drive: {candidate}")
            return candidate

    logger.debug("MATLAB Drive not found")
    return None


def get_strategies_dir() -> Optional[Path]:
    """
    Get tangled_strategies directory path.

    Returns:
        Path to strategies directory, or None if not found
    """
    matlab_drive = get_matlab_drive()
    if not matlab_drive:
        return None

    strategies_dir = matlab_drive / "tangled_strategies"
    if strategies_dir.exists():
        return strategies_dir

    logger.debug(f"Strategies directory not found: {strategies_dir}")
    return None


# Module-level cached values (computed once)
_MATLAB_ROOT: Optional[Path] = None
_MATLAB_BIN: Optional[Path] = None
_MATLAB_RUNTIME: Optional[Path] = None
_PATHS_INITIALIZED = False


def get_cached_matlab_paths() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """
    Get cached MATLAB paths (computed once per process).

    Returns:
        Tuple of (matlab_root, bin_path, runtime_path)
    """
    global _MATLAB_ROOT, _MATLAB_BIN, _MATLAB_RUNTIME, _PATHS_INITIALIZED

    if not _PATHS_INITIALIZED:
        _MATLAB_ROOT, _MATLAB_BIN, _MATLAB_RUNTIME = get_matlab_paths()
        _PATHS_INITIALIZED = True

    return _MATLAB_ROOT, _MATLAB_BIN, _MATLAB_RUNTIME


def get_parallel_config(engine) -> dict:
    """
    Get parallel computing configuration from MATLAB.

    Args:
        engine: Connected MATLAB engine instance

    Returns:
        Dictionary with parallel computing info:
        - workers: Number of configured workers
        - cluster_type: Type of parallel cluster
        - error: Error message if query failed
    """
    try:
        # Get default parallel cluster
        pc = engine.parcluster(nargout=1)
        num_workers = int(engine.eval(f"pc.NumWorkers", nargout=1))

        # Get cluster profile name
        try:
            cluster_profile = str(engine.eval("parallel.defaultClusterProfile", nargout=1))
        except Exception:
            cluster_profile = "unknown"

        return {
            'workers': num_workers,
            'cluster_type': cluster_profile,
            'available': True,
        }
    except Exception as e:
        return {
            'workers': 0,
            'cluster_type': 'none',
            'available': False,
            'error': str(e)
        }


def get_gpu_config(engine) -> dict:
    """
    Get GPU configuration from MATLAB.

    Args:
        engine: Connected MATLAB engine instance

    Returns:
        Dictionary with GPU info:
        - count: Number of GPUs
        - devices: List of GPU device info dicts
        - validated: Whether GPU validation passed
        - error: Error message if query failed
    """
    try:
        # Get GPU count
        gpu_count = int(engine.gpuDeviceCount(nargout=1))

        if gpu_count == 0:
            return {
                'count': 0,
                'devices': [],
                'available': False,
            }

        # Get GPU device information
        devices = []
        for i in range(1, gpu_count + 1):
            try:
                # Get device
                g = engine.gpuDevice(float(i), nargout=1)

                # Extract properties
                name = str(engine.eval(f"g.Name", nargout=1))
                compute_cap = str(engine.eval(f"g.ComputeCapability", nargout=1))
                total_mem = float(engine.eval(f"g.TotalMemory", nargout=1))
                available_mem = float(engine.eval(f"g.AvailableMemory", nargout=1))

                devices.append({
                    'index': i,
                    'name': name,
                    'compute_capability': compute_cap,
                    'total_memory_gb': total_mem / (1024**3),
                    'available_memory_gb': available_mem / (1024**3),
                })
            except Exception as e:
                logger.warning(f"Failed to get GPU {i} details: {e}")
                devices.append({
                    'index': i,
                    'name': f'GPU {i}',
                    'error': str(e)
                })

        # Try GPU validation (may fail if GPU not properly set up)
        validation_passed = False
        validation_output = None
        try:
            # Capture validateGPU output
            validation_output = engine.evalc("validateGPU", nargout=1)
            validation_passed = "no failures" in validation_output.lower()
        except Exception as e:
            validation_output = f"Validation failed: {e}"

        return {
            'count': gpu_count,
            'devices': devices,
            'available': True,
            'validated': validation_passed,
            'validation_output': validation_output,
        }

    except Exception as e:
        return {
            'count': 0,
            'devices': [],
            'available': False,
            'error': str(e)
        }
