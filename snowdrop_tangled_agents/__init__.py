# Strategy (no SDK dependency)
from snowdrop_tangled_agents.strategy.petersen_strategy import PetersenStrategy

__all__ = ['PetersenStrategy']

# SDK-dependent imports (optional)
try:
    from snowdrop_tangled_agents.agents.random_agent import RandomRandyAgent
    from snowdrop_tangled_agents.agents.petersen_agent import PetersenAgent
    from snowdrop_tangled_agents.agents import random_agent
    from snowdrop_tangled_agents.agents import petersen_agent
    from snowdrop_tangled_agents.utils.utilities import import_agent
    __all__.extend([
        'RandomRandyAgent',
        'PetersenAgent',
        'random_agent',
        'petersen_agent',
        'import_agent'
    ])
except ImportError:
    pass  # SDK not installed, skip agent imports

# Web bridge (optional, needs playwright)
try:
    from snowdrop_tangled_agents.web.tangled_bridge import TangledBridge
    __all__.append('TangledBridge')
except ImportError:
    pass  # Playwright not installed
