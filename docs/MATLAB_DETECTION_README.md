# MATLAB Detection System

This project now includes automatic MATLAB installation detection that works across different platforms and versions.

## What Changed

### Before
MATLAB paths were hardcoded in multiple files:
- `bridge.py`: Hardcoded to `C:\Program Files\MATLAB\R2026a`
- `test_matlab_connection.py`: Hardcoded to `C:\Program Files\MATLAB\R2026a`
- `test_matlab_rl.py`: Checked multiple versions but not centralized

### After
Centralized MATLAB detection in `snowdrop_tangled_agents/matlab/matlab_config.py`:
- Automatically detects MATLAB installations
- Searches multiple version directories (R2026a → R2022a)
- Supports environment variable override
- Works on Windows, macOS, and Linux
- All MATLAB-related modules use this centralized detection

## How It Works

### Automatic Detection

The system searches for MATLAB in this priority order:

1. **Environment Variable**: `MATLAB_ROOT`
   ```bash
   set MATLAB_ROOT=C:\Program Files\MATLAB\R2025a
   ```

2. **Common Installation Directories** (newer versions first):
   - Windows: `C:\Program Files\MATLAB\R20XXx`
   - macOS: `/Applications/MATLAB/R20XXx`
   - Linux: `/usr/local/MATLAB/R20XXx` or `~/MATLAB/R20XXx`

3. **PATH Variable**: Searches for `matlab` executable

### Supported Versions

The detection checks for these MATLAB versions (in order):
- R2026a, R2025b, R2025a
- R2024b, R2024a
- R2023b, R2023a
- R2022b, R2022a

## Usage

### Basic Usage

The detection happens automatically when importing MATLAB modules:

```python
from snowdrop_tangled_agents.matlab import get_unified_bridge

bridge = get_unified_bridge()
backend = bridge.connect()  # Auto-detects MATLAB
```

### Manual Detection

You can also query MATLAB paths directly:

```python
from snowdrop_tangled_agents.matlab import (
    find_matlab_installation,
    get_matlab_paths,
    setup_matlab_path
)

# Find MATLAB root directory
matlab_root = find_matlab_installation()
print(f"MATLAB installed at: {matlab_root}")

# Get all paths
root, bin_path, runtime_path = get_matlab_paths()
print(f"Binary path: {bin_path}")
print(f"Runtime path: {runtime_path}")

# Add to system PATH
setup_matlab_path()
```

### Testing Detection

Run the detection test script:

```bash
python test_matlab_detection.py
```

This tests basic MATLAB installation detection and path configuration.

### Querying Capabilities

To query parallel computing and GPU capabilities (requires MATLAB Engine API):

```bash
python query_matlab_capabilities.py
```

This script will:
- Check parallel computing worker configuration
- Detect and validate GPUs
- Show GPU memory and compute capability
- Provide training recommendations

Expected output:
```
============================================================
MATLAB Capabilities Query
============================================================

Step 1: Checking MATLAB installation...
   [OK] MATLAB found: C:\Program Files\MATLAB\R2025b

Step 2: Importing MATLAB Engine API...
   [OK] MATLAB Engine API available

Step 3: Connecting to MATLAB Engine...
   Starting new MATLAB session (10-20 seconds)...
   [OK] New session started

======================================================================
Parallel Computing Configuration
======================================================================

Status: Available
Workers: 8
Cluster Profile: local

[INFO] Parallel training can use up to 8 workers

======================================================================
GPU Configuration
======================================================================

GPU Count: 1

──────────────────────────────────────────────────────────────────────
GPU 1: NVIDIA GeForce RTX 2070
──────────────────────────────────────────────────────────────────────
Compute Capability:  7.5
Total Memory:        8.00 GB
Available Memory:    7.85 GB
Deep Learning:       Supported (CC >= 3.5)

──────────────────────────────────────────────────────────────────────
GPU Validation
──────────────────────────────────────────────────────────────────────

[OK] GPU Validation: PASSED
     All GPU tests completed successfully

======================================================================
Training Recommendations
======================================================================

[OK] GPU-accelerated training is available!
     Configure your training to use GPU:
     - Set training option: UseGPU=true
     - Ensure batch size fits in GPU memory
     - Available GPU memory: 7.85 GB
```

## Configuration

### Override MATLAB Location

Set the `MATLAB_ROOT` environment variable to use a specific MATLAB installation:

**Windows (Command Prompt):**
```cmd
set MATLAB_ROOT=C:\Program Files\MATLAB\R2025a
python test_matlab_detection.py
```

**Windows (PowerShell):**
```powershell
$env:MATLAB_ROOT = "C:\Program Files\MATLAB\R2025a"
python test_matlab_detection.py
```

**Unix/macOS:**
```bash
export MATLAB_ROOT=/Applications/MATLAB/R2025a
python test_matlab_detection.py
```

### Permanent Configuration

Add to your environment permanently:

**Windows:**
1. System Properties → Advanced → Environment Variables
2. Add new variable: `MATLAB_ROOT` = `C:\Program Files\MATLAB\R2025a`

**Unix/macOS:**
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export MATLAB_ROOT=/Applications/MATLAB/R2025a
```

## Files Modified

### New Files
- `snowdrop_tangled_agents/matlab/matlab_config.py` - Centralized detection and capability queries
- `test_matlab_detection.py` - Basic detection test script
- `query_matlab_capabilities.py` - Full capability query (workers, GPU)
- `MATLAB_DETECTION_README.md` - This documentation

### Modified Files
- `snowdrop_tangled_agents/matlab/bridge.py` - Uses `matlab_config`
- `snowdrop_tangled_agents/matlab/__init__.py` - Exports config functions
- `test_matlab_connection.py` - Uses `matlab_config`
- `snowdrop_tangled_agents/tests/test_matlab_rl.py` - Uses `matlab_config`

## Troubleshooting

### MATLAB Not Found

If detection fails:

1. **Verify Installation**: Check that MATLAB is installed
   ```cmd
   dir "C:\Program Files\MATLAB"
   ```

2. **Check Version**: Ensure you have R2022a or newer

3. **Set Environment Variable**:
   ```cmd
   set MATLAB_ROOT=C:\Program Files\MATLAB\R2025a
   ```

4. **Check PATH**: Add MATLAB to PATH manually
   ```cmd
   set PATH=%PATH%;C:\Program Files\MATLAB\R2025a\bin\win64
   ```

### Multiple MATLAB Versions

If you have multiple MATLAB versions installed, the system will use the newest one by default. To use a specific version, set `MATLAB_ROOT`.

### MATLAB Drive Not Found

The MATLAB Drive location is optional. If you don't use MATLAB Drive or it's in a non-standard location, the system will skip it gracefully.

## Platform-Specific Notes

### Windows
- Default location: `C:\Program Files\MATLAB\R20XXx`
- Binary path: `bin\win64`
- Runtime path: `runtime\win64`

### macOS
- Default location: `/Applications/MATLAB/R20XXx`
- Binary path: `bin/maci64`
- Runtime path: `runtime/maci64`

### Linux
- Default location: `/usr/local/MATLAB/R20XXx` or `~/MATLAB/R20XXx`
- Binary path: `bin/glnxa64`
- Runtime path: `runtime/glnxa64`

## API Reference

### Detection Functions

#### `find_matlab_installation() -> Optional[Path]`
Finds MATLAB root directory. Returns `None` if not found.

#### `get_matlab_paths() -> Tuple[Optional[Path], Optional[Path], Optional[Path]]`
Returns `(matlab_root, bin_path, runtime_path)`.

#### `setup_matlab_path() -> bool`
Adds MATLAB paths to system PATH. Returns `True` on success.

#### `get_matlab_drive() -> Optional[Path]`
Finds MATLAB Drive directory. Returns `None` if not found.

#### `get_strategies_dir() -> Optional[Path]`
Finds `tangled_strategies` directory in MATLAB Drive. Returns `None` if not found.

### Capability Query Functions

These require a connected MATLAB Engine instance.

#### `get_parallel_config(engine) -> dict`
Query parallel computing configuration.

**Returns:**
```python
{
    'workers': 8,              # Number of workers
    'cluster_type': 'local',   # Cluster profile name
    'available': True,         # Whether available
    'error': '...'            # Error message if failed (optional)
}
```

#### `get_gpu_config(engine) -> dict`
Query GPU configuration and validation.

**Returns:**
```python
{
    'count': 1,                # Number of GPUs
    'available': True,         # Whether GPUs available
    'validated': True,         # Whether validation passed
    'devices': [               # List of GPU devices
        {
            'index': 1,
            'name': 'NVIDIA GeForce RTX 2070',
            'compute_capability': '7.5',
            'total_memory_gb': 8.0,
            'available_memory_gb': 7.85
        }
    ],
    'validation_output': '...',  # Full validation output
    'error': '...'              # Error message if failed (optional)
}
```

## Migration Guide

### Querying Capabilities in Code

You can query parallel and GPU configuration programmatically:

```python
from snowdrop_tangled_agents.matlab import (
    get_parallel_config,
    get_gpu_config
)
import matlab.engine

# Connect to MATLAB
eng = matlab.engine.start_matlab()

# Get parallel computing config
parallel = get_parallel_config(eng)
print(f"Workers: {parallel['workers']}")

# Get GPU config
gpu = get_gpu_config(eng)
if gpu['available']:
    print(f"GPUs: {gpu['count']}")
    for device in gpu['devices']:
        print(f"  {device['name']}: {device['compute_capability']}")

eng.quit()
```

### For Existing Code

If you have code that directly references MATLAB paths:

**Before:**
```python
import os
matlab_bin = r"C:\Program Files\MATLAB\R2026a\bin\win64"
os.environ['PATH'] = matlab_bin + os.pathsep + os.environ['PATH']
```

**After:**
```python
from snowdrop_tangled_agents.matlab import setup_matlab_path
setup_matlab_path()
```

### For Tests

**Before:**
```python
def find_matlab():
    common_paths = [
        r'C:\Program Files\MATLAB\R2026a\bin\matlab.exe',
        r'C:\Program Files\MATLAB\R2025a\bin\matlab.exe',
    ]
    for path in common_paths:
        if Path(path).exists():
            return path
```

**After:**
```python
from snowdrop_tangled_agents.matlab import find_matlab_installation

def find_matlab():
    matlab_root = find_matlab_installation()
    if matlab_root:
        return str(matlab_root / 'bin' / 'matlab.exe')  # Windows
```

## Benefits

1. **No Hardcoding**: Works with any MATLAB version R2022a+
2. **Cross-Platform**: Same code works on Windows, macOS, Linux
3. **Flexible**: Easy to override via environment variables
4. **Centralized**: One place to update if detection logic changes
5. **Automatic**: Just works out of the box for standard installations
6. **Fallback**: Graceful degradation if MATLAB not found

## Support

If you encounter issues with MATLAB detection, please include:
- Output of `test_matlab_detection.py`
- Your MATLAB version and installation path
- Operating system and version
- Whether you're using a standard or custom MATLAB installation
