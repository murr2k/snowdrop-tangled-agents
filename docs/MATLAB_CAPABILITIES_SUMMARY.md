# MATLAB Capabilities Query - Quick Start

## Overview

The enhanced MATLAB detection system now includes capability queries for:
- **Parallel Computing Workers** - How many workers available for parallel training
- **GPU Configuration** - GPU count, memory, compute capability
- **GPU Validation** - Full CUDA validation results

## Quick Start

### 1. Basic Detection (No MATLAB Engine Required)

```bash
python test_matlab_detection.py
```

This checks:
- MATLAB installation location
- Binary and runtime paths
- MATLAB Drive location
- Strategies directory

### 2. Full Capabilities Query (Requires MATLAB Engine API)

First, install MATLAB Engine API:

**Option A: From MATLAB Installation**
```bash
cd "C:\Program Files\MATLAB\R2025b\extern\engines\python"
python setup.py install
```

**Option B: From PyPI**
```bash
pip install matlabengine
```

Then run the capabilities query:

```bash
python query_matlab_capabilities.py
```

This reports:
- Number of parallel computing workers
- Cluster profile configuration
- GPU count and specifications
- GPU memory (total and available)
- GPU compute capability
- CUDA validation results
- Training recommendations

## Example Output

```
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

## Using in Code

```python
from snowdrop_tangled_agents.matlab import (
    get_parallel_config,
    get_gpu_config
)
import matlab.engine

# Connect to MATLAB
eng = matlab.engine.start_matlab("-nodesktop -nosplash")

# Query parallel computing
parallel = get_parallel_config(eng)
print(f"Workers available: {parallel['workers']}")

# Query GPU
gpu = get_gpu_config(eng)
if gpu['available'] and gpu['count'] > 0:
    device = gpu['devices'][0]
    print(f"GPU: {device['name']}")
    print(f"Memory: {device['total_memory_gb']:.1f} GB")
    print(f"Compute: {device['compute_capability']}")

    # Check if suitable for deep learning
    major = int(device['compute_capability'].split('.')[0])
    if major >= 3:
        print("✓ Suitable for deep learning")

eng.quit()
```

## Interpreting Results

### Parallel Computing

- **8+ workers**: Excellent for parallel training
- **4-7 workers**: Good for parallel training
- **1-3 workers**: Limited parallelism
- **0 workers / Not available**: Parallel Computing Toolbox not installed

### GPU Configuration

**Compute Capability:**
- **7.x**: Volta/Turing (RTX 20xx, Titan V) - Excellent
- **6.x**: Pascal (GTX 10xx, P100) - Very Good
- **5.x**: Maxwell (GTX 9xx, Titan X) - Good
- **3.5-4.x**: Kepler (K80, GTX 7xx) - Adequate
- **< 3.5**: Not supported for deep learning

**Memory:**
- **8+ GB**: Can train large networks
- **4-7 GB**: Suitable for medium networks
- **2-3 GB**: Limited to small networks
- **< 2 GB**: Very limited

### GPU Validation

**All tests PASSED**: GPU is ready for training
- CUDA drivers installed correctly
- GPU is accessible
- Can allocate memory and launch kernels

**Some tests FAILED**: GPU may have issues
- Driver version mismatch
- Compute mode restrictions
- Memory allocation problems

## Troubleshooting

### "MATLAB Engine API not installed"

Install using one of these methods:

1. **From MATLAB directory** (recommended):
   ```bash
   cd "C:\Program Files\MATLAB\R2025b\extern\engines\python"
   python setup.py install
   ```

2. **From PyPI**:
   ```bash
   pip install matlabengine
   ```

### "Parallel Computing Toolbox not available"

The Parallel Computing Toolbox may not be installed. Check in MATLAB:
```matlab
>> ver
```

If not listed, you'll need to install it through MATLAB Add-Ons.

### "No GPUs found" (but you have a GPU)

1. **Check CUDA drivers**: Ensure NVIDIA drivers are installed
2. **Run in MATLAB**: `gpuDeviceCount` should return > 0
3. **Check compute mode**: GPU may be restricted
4. **Verify CUDA toolkit**: Some MATLAB versions need specific CUDA versions

### GPU validation fails

1. **Update GPU drivers**: Get latest from NVIDIA
2. **Check MATLAB compatibility**: Some GPUs need newer MATLAB versions
3. **Verify CUDA version**: MATLAB version must match CUDA requirements
4. **Run `validateGPU` in MATLAB** for detailed diagnostics

## Configuration Tips

### Increase Parallel Workers

In MATLAB:
```matlab
>> pc = parcluster;
>> pc.NumWorkers = 8;  % Set to number of cores
>> saveProfile(pc);
```

### GPU Memory Management

For large models that exceed GPU memory:
```matlab
% In training options
options.MiniBatchSize = 32;  % Reduce if out of memory
options.SequenceLength = 'shortest';  % For RNN/LSTM
```

### Mixed CPU-GPU Training

If GPU memory is limited:
```matlab
% Use GPU for forward pass, CPU for data augmentation
options.UseGPU = true;
options.WorkerLoad = 0.8;  % Leave 20% for system
```

## Performance Estimates

Based on typical configurations:

| Configuration | Training Speed | Recommendation |
|--------------|----------------|----------------|
| 8 workers + RTX 2070 | ~10-15x faster | Excellent for RL training |
| 8 workers + CPU only | ~6-8x faster | Good, but limited by CPU |
| 1 worker + RTX 2070 | ~5-7x faster | GPU helps but no parallelism |
| 1 worker + CPU only | Baseline | Slow, not recommended |

## Next Steps

1. **Run capability query**: `python query_matlab_capabilities.py`
2. **Check GPU validation**: Ensure PASSED status
3. **Configure training**: Use GPU and parallel workers
4. **Monitor GPU usage**: During training, check `nvidia-smi`
5. **Optimize batch size**: Balance speed vs memory

## References

- MATLAB Documentation: [Parallel Computing Toolbox](https://www.mathworks.com/products/parallel-computing.html)
- MATLAB Documentation: [GPU Computing](https://www.mathworks.com/help/parallel-computing/gpu-computing.html)
- CUDA Compute Capability: [NVIDIA Documentation](https://developer.nvidia.com/cuda-gpus)
