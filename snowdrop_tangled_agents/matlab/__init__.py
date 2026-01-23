"""
MATLAB Integration for Tangled Game Strategy.

Provides bridge to MATLAB R2026a toolboxes for enhanced game analysis:
- Deep Learning Toolbox for value/policy neural networks
- Statistics and Machine Learning Toolbox for opponent modeling
- Database Toolbox for direct MATLAB-SQLite access
- MATLAB Compiler SDK for Python-callable packages

Architecture supports three backend modes:
1. Compiled packages (fastest, no license required, just MATLAB Runtime)
2. MATLAB Engine API (full functionality, requires license)
3. Pure Python heuristics (always available)

Usage:
    from snowdrop_tangled_agents.matlab import get_unified_bridge

    bridge = get_unified_bridge()
    backend = bridge.connect()  # 'compiled', 'engine', or 'heuristic'

    value, policy = bridge.evaluate_position(state, is_our_turn=True)
"""

from .bridge import MatlabBridge, get_bridge
from .compiled_bridge import CompiledMatlabBridge, get_compiled_bridge, packages_available
from .unified_bridge import UnifiedMatlabBridge, get_unified_bridge
from .matlab_strategy import MatlabEnhancedStrategy, HybridSolverStrategy
from .training import (
    TrainingOrchestrator,
    get_training_orchestrator,
    print_training_status,
)

__all__ = [
    # Original bridge
    'MatlabBridge',
    'get_bridge',
    # Compiled bridge
    'CompiledMatlabBridge',
    'get_compiled_bridge',
    'packages_available',
    # Unified bridge (recommended)
    'UnifiedMatlabBridge',
    'get_unified_bridge',
    # Strategy
    'MatlabEnhancedStrategy',
    'HybridSolverStrategy',
    # Training
    'TrainingOrchestrator',
    'get_training_orchestrator',
    'print_training_status',
]
