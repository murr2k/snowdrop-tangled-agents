"""
Unified bridge with automatic backend selection.

Provides a single interface for MATLAB functionality with fallback chain:
1. Compiled MATLAB packages (fastest, no license required)
2. MATLAB Engine API (full functionality, requires license)
3. Pure Python heuristics (always available)

Usage:
    from snowdrop_tangled_agents.matlab import get_unified_bridge

    bridge = get_unified_bridge()
    backend = bridge.connect()  # Returns 'compiled', 'engine', or 'heuristic'

    value, policy = bridge.evaluate_position(state, is_our_turn=True)
"""

import logging
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

from .bridge import MatlabBridge, get_bridge
from .compiled_bridge import CompiledMatlabBridge, get_compiled_bridge

logger = logging.getLogger(__name__)

# Edge category definitions (0-indexed)
MY_EDGES = [9, 10, 11]      # Edges connected to our vertex (5)
OPP_EDGES = [5, 12, 13]     # Edges connected to opponent vertex (7)
HUB_EDGES = [2, 6, 10, 12]  # Edges connected to hub vertex (6)


class UnifiedMatlabBridge:
    """
    Unified interface with automatic backend selection.

    Fallback chain:
    1. Compiled MATLAB packages (fastest, no license)
    2. MATLAB Engine API (full functionality)
    3. Pure Python heuristics (always available)

    MATLAB Engine startup modes:
    - If a shared MATLAB session exists, connects to it (instant)
    - Otherwise, starts a new MATLAB instance:
      - Headless mode (default): No GUI, faster startup (~10-15s)
      - Desktop mode: Full GUI for debugging (~20-30s)

    No manual `matlab.engine.shareEngine` required!
    """

    def __init__(
        self,
        prefer_existing_session: bool = True,
        headless: bool = True
    ):
        """
        Initialize unified bridge.

        Args:
            prefer_existing_session: If True, try to connect to a shared
                MATLAB session before starting a new one. Set False to
                always start a fresh instance.
            headless: If True, start MATLAB without desktop GUI (faster).
                Only applies when starting a new instance.
        """
        self.compiled: Optional[CompiledMatlabBridge] = None
        self.engine: Optional[MatlabBridge] = None
        self.backend: Optional[str] = None
        self._model_dir: Optional[Path] = None
        self._prefer_existing = prefer_existing_session
        self._headless = headless

    def connect(self, model_dir: Optional[Path] = None) -> str:
        """
        Connect to best available backend.

        Connection process:
        1. Try compiled MATLAB packages (no license required)
        2. Try MATLAB Engine API:
           - If prefer_existing=True, look for shared sessions first
           - Start new MATLAB instance if needed (headless by default)
        3. Fall back to pure Python heuristics

        Args:
            model_dir: Directory containing trained models

        Returns:
            Backend name: 'compiled', 'engine', or 'heuristic'
        """
        if model_dir:
            self._model_dir = Path(model_dir)
        else:
            self._model_dir = Path.home() / ".tangled" / "models"

        # Try compiled packages first (fastest, no license)
        self.compiled = get_compiled_bridge()
        if self.compiled and self.compiled.initialize(self._model_dir):
            self.backend = 'compiled'
            logger.info("Using compiled MATLAB packages")
            return 'compiled'

        # Try MATLAB Engine (full functionality)
        # Create bridge with our configuration
        self.engine = MatlabBridge(
            prefer_existing=self._prefer_existing,
            headless=self._headless
        )
        if self.engine.connect():
            self.backend = 'engine'
            if self.engine._started_new:
                mode = "headless" if self._headless else "desktop"
                logger.info(f"Using new MATLAB Engine instance ({mode} mode)")
            else:
                logger.info("Using existing shared MATLAB session")
            return 'engine'

        # Fallback to heuristics
        self.backend = 'heuristic'
        logger.info("Using pure Python heuristics (MATLAB unavailable)")
        return 'heuristic'

    def is_available(self) -> bool:
        """Check if any backend is available."""
        return self.backend is not None

    def get_backend(self) -> Optional[str]:
        """Get current backend name."""
        return self.backend

    def evaluate_position(
        self,
        state: str,
        is_our_turn: bool = True
    ) -> Tuple[float, Dict[Tuple[int, str], float]]:
        """
        Evaluate position using best available backend.

        Args:
            state: 15-char board state ('G', 'P', '-')
            is_our_turn: True if it's our turn

        Returns:
            (value, policy_dict) where:
            - value: Expected outcome in [-1, 1]
            - policy_dict: {(edge, color): probability} for available actions
        """
        if self.backend == 'compiled' and self.compiled:
            return self.compiled.evaluate_position(state, is_our_turn)

        elif self.backend == 'engine' and self.engine:
            return self.engine.evaluate_position_rl(state, is_our_turn)

        else:
            return self._heuristic_eval(state, is_our_turn)

    def classify_opponent(
        self,
        opponent_name: Optional[str] = None,
        features: Optional[List[float]] = None
    ) -> Tuple[int, float]:
        """
        Classify opponent play style.

        Args:
            opponent_name: Opponent name (for database lookup)
            features: Pre-computed 20-element feature vector

        Returns:
            (style, confidence) where:
            - style: 1=aggressive, 2=defensive, 3=balanced, 0=unknown
            - confidence: Classification confidence [0, 1]
        """
        if features is None:
            # Would need to extract features from database
            return 0, 0.0

        if self.backend == 'compiled' and self.compiled:
            return self.compiled.classify_opponent(features)

        elif self.backend == 'engine' and self.engine:
            try:
                result = self.engine.call_function(
                    'classify_opponent', '', opponent_name or '',
                    nargout=2
                )
                return int(result[0]), float(result[1])
            except Exception:
                pass

        # Heuristic classification
        return self._heuristic_classify(features)

    def adapt_priors(
        self,
        state: str,
        opponent_features: Optional[List[float]] = None,
        base_priors: Optional[Dict[Tuple[int, str], float]] = None,
        style: Optional[int] = None
    ) -> Dict[Tuple[int, str], float]:
        """
        Adapt action priors based on opponent model.

        Args:
            state: 15-char board state
            opponent_features: 20-element opponent feature vector
            base_priors: Base action probabilities (or None for uniform)
            style: Opponent style override

        Returns:
            Adapted action probabilities as {(edge, color): prob}
        """
        # Default to uniform priors
        if base_priors is None:
            base_priors = self._uniform_priors(state)

        if opponent_features is None:
            return base_priors

        # Convert dict to list for MATLAB
        priors_list = self._priors_to_list(base_priors)

        if self.backend == 'compiled' and self.compiled:
            adapted_list = self.compiled.adapt_priors(
                state, opponent_features, priors_list, style
            )
            return self._list_to_priors(adapted_list, state)

        elif self.backend == 'engine' and self.engine:
            try:
                import matlab
                state_vec = matlab.double([
                    1.0 if c == 'G' else (-1.0 if c == 'P' else 0.0)
                    for c in state
                ])
                result = self.engine.call_function(
                    'adapt_to_opponent',
                    state_vec,
                    matlab.double(opponent_features),
                    matlab.double(priors_list),
                    nargout=1
                )
                adapted_list = list(result)
                return self._list_to_priors(adapted_list, state)
            except Exception:
                pass

        # Heuristic adaptation
        return self._heuristic_adapt(state, opponent_features, base_priors, style)

    # ========== Heuristic Fallbacks ==========

    def _heuristic_eval(
        self,
        state: str,
        is_our_turn: bool
    ) -> Tuple[float, Dict[Tuple[int, str], float]]:
        """
        Pure Python heuristic position evaluation.

        Uses edge categories and board control metrics.
        """
        # Count controlled edges by category
        my_green = sum(1 for i in MY_EDGES if state[i] == 'G')
        my_purple = sum(1 for i in MY_EDGES if state[i] == 'P')
        opp_green = sum(1 for i in OPP_EDGES if state[i] == 'G')
        opp_purple = sum(1 for i in OPP_EDGES if state[i] == 'P')
        hub_green = sum(1 for i in HUB_EDGES if state[i] == 'G')
        hub_purple = sum(1 for i in HUB_EDGES if state[i] == 'P')

        # Simple value estimate
        # +1 for each MY edge we control (green)
        # -1 for each MY edge opponent controls (purple)
        # +0.5 for each OPP edge we attack (purple)
        # -0.5 for each OPP edge opponent defends (green)
        # +0.3 for hub control
        value = (
            (my_green - my_purple) * 0.3 +
            (opp_purple - opp_green) * 0.2 +
            (hub_green - hub_purple) * 0.15
        )

        # Clamp to [-1, 1]
        value = max(-1.0, min(1.0, value))

        # Uniform policy over available moves
        policy = self._uniform_priors(state)

        return value, policy

    def _heuristic_classify(
        self,
        features: List[float]
    ) -> Tuple[int, float]:
        """
        Heuristic opponent classification based on features.

        Features 17, 18, 19 are key:
        - 17: Opening aggression
        - 18: Response to our edges (purple rate)
        - 19: Hub control priority
        """
        if len(features) < 20:
            return 0, 0.0

        opening_agg = features[16]  # 0-indexed
        response_rate = features[17]
        hub_priority = features[18]

        # Simple rule-based classification
        if opening_agg > 0.35:
            return 1, 0.7  # Aggressive
        elif hub_priority > 0.35:
            return 3, 0.6  # Hub-focused
        elif response_rate > 0.55:
            return 2, 0.6  # Defensive
        else:
            return 2, 0.4  # Default to balanced/defensive

    def _heuristic_adapt(
        self,
        state: str,
        features: List[float],
        base_priors: Dict[Tuple[int, str], float],
        style: Optional[int]
    ) -> Dict[Tuple[int, str], float]:
        """
        Heuristic prior adaptation based on opponent style.
        """
        if style is None and len(features) >= 20:
            style, _ = self._heuristic_classify(features)

        priors = dict(base_priors)

        # Apply style-based adjustments
        if style == 1:  # Against aggressive
            # Prioritize defense
            for edge in MY_EDGES:
                if (edge, 'G') in priors:
                    priors[(edge, 'G')] *= 1.5

        elif style == 2:  # Against defensive
            # Early attack
            for edge in OPP_EDGES:
                if (edge, 'P') in priors:
                    priors[(edge, 'P')] *= 1.4

        elif style == 3:  # Against hub-focused
            # Compete for hub
            for edge in HUB_EDGES:
                if (edge, 'G') in priors:
                    priors[(edge, 'G')] *= 1.3

        # Renormalize
        total = sum(priors.values())
        if total > 0:
            priors = {k: v / total for k, v in priors.items()}

        return priors

    def _uniform_priors(
        self,
        state: str
    ) -> Dict[Tuple[int, str], float]:
        """Generate uniform priors over available actions."""
        priors = {}
        for i in range(15):
            if state[i] == '-':
                priors[(i, 'G')] = 1.0
                priors[(i, 'P')] = 1.0

        # Normalize
        total = sum(priors.values())
        if total > 0:
            priors = {k: v / total for k, v in priors.items()}

        return priors

    def _priors_to_list(
        self,
        priors: Dict[Tuple[int, str], float]
    ) -> List[float]:
        """Convert priors dict to 30-element list."""
        result = [0.0] * 30
        for (edge, color), prob in priors.items():
            idx = edge * 2 + (0 if color == 'G' else 1)
            result[idx] = prob
        return result

    def _list_to_priors(
        self,
        priors_list: List[float],
        state: str
    ) -> Dict[Tuple[int, str], float]:
        """Convert 30-element list to priors dict (masked to valid moves)."""
        priors = {}
        for i in range(15):
            if state[i] == '-':
                priors[(i, 'G')] = priors_list[i * 2]
                priors[(i, 'P')] = priors_list[i * 2 + 1]

        # Renormalize
        total = sum(priors.values())
        if total > 0:
            priors = {k: v / total for k, v in priors.items()}

        return priors


# Module-level singleton
_unified_bridge: Optional[UnifiedMatlabBridge] = None


def get_unified_bridge(
    prefer_existing_session: bool = True,
    headless: bool = True,
    force_new: bool = False
) -> UnifiedMatlabBridge:
    """
    Get unified bridge instance.

    Args:
        prefer_existing_session: Try to connect to shared MATLAB sessions
            before starting a new instance.
        headless: Start MATLAB without desktop GUI (faster startup).
        force_new: Force creation of a new bridge instance (resets singleton).

    Returns:
        UnifiedMatlabBridge instance

    Example:
        # Default: prefer existing sessions, headless if starting new
        bridge = get_unified_bridge()
        backend = bridge.connect()

        # Always start fresh MATLAB instance with desktop for debugging
        bridge = get_unified_bridge(
            prefer_existing_session=False,
            headless=False,
            force_new=True
        )
    """
    global _unified_bridge
    if _unified_bridge is None or force_new:
        _unified_bridge = UnifiedMatlabBridge(
            prefer_existing_session=prefer_existing_session,
            headless=headless
        )
    return _unified_bridge
