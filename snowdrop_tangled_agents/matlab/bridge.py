"""
Python-MATLAB Bridge for Tangled Game.

Provides connection to MATLAB Engine API for calling MATLAB toolbox functions.
Supports:
- Reinforcement Learning position evaluation
- Simulated annealing optimization
- Opponent pattern identification
"""

import logging
import os
from typing import Optional
from pathlib import Path

from .matlab_config import (
    get_cached_matlab_paths,
    setup_matlab_path,
    get_matlab_drive,
    get_strategies_dir
)

logger = logging.getLogger(__name__)

# Setup MATLAB paths automatically
setup_matlab_path()

# Get MATLAB installation paths
_MATLAB_ROOT, _MATLAB_BIN, _MATLAB_RUNTIME = get_cached_matlab_paths()

# MATLAB shared directory
MATLAB_DRIVE = get_matlab_drive()
STRATEGIES_DIR = get_strategies_dir()

# Singleton bridge instance
_bridge_instance: Optional['MatlabBridge'] = None


class MatlabBridge:
    """
    Bridge to MATLAB Engine for Tangled game analysis.

    Uses MATLAB Engine API for Python to call MATLAB functions.
    Falls back gracefully if MATLAB is not available.

    Connection modes:
    1. Connect to existing shared session (fastest if MATLAB already open)
    2. Start new headless MATLAB instance (no manual setup required)
    3. Start new MATLAB with desktop (for debugging)
    """

    def __init__(self, prefer_existing: bool = True, headless: bool = True):
        """
        Initialize MATLAB bridge.

        Args:
            prefer_existing: If True, try to connect to shared session first.
                           If False, always start a new instance.
            headless: If True, start MATLAB without desktop UI (faster).
                     Only applies when starting new instance.
        """
        self.engine = None
        self.connected = False
        self._setup_attempted = False
        self.prefer_existing = prefer_existing
        self.headless = headless
        self._started_new = False  # Track if we started a new instance

    def connect(self) -> bool:
        """
        Connect to MATLAB Engine.

        Connection priority:
        1. If prefer_existing=True, try shared sessions first
        2. Start new MATLAB instance (headless by default for faster startup)

        Returns True if connection successful, False otherwise.
        """
        if self.connected:
            return True

        if self._setup_attempted:
            return self.connected

        self._setup_attempted = True

        try:
            import matlab.engine
            logger.info("Initializing MATLAB Engine...")

            # Try to connect to shared session first (if preferred)
            if self.prefer_existing:
                sessions = matlab.engine.find_matlab()
                if sessions:
                    logger.info(f"Found existing MATLAB sessions: {sessions}")
                    try:
                        self.engine = matlab.engine.connect_matlab(sessions[0])
                        logger.info(f"Connected to existing session: {sessions[0]}")
                        self._started_new = False
                    except Exception as e:
                        logger.warning(f"Failed to connect to existing session: {e}")
                        sessions = []  # Fall through to start new

                if not sessions:
                    logger.info("No shared sessions found, starting new instance...")

            # Start new MATLAB instance
            if self.engine is None:
                # Build startup options for faster launch
                startup_opts = []
                if self.headless:
                    # -nodesktop: No GUI (faster startup)
                    # -nosplash: Skip splash screen
                    # -noFigureWindows: Suppress figure windows
                    startup_opts = ["-nodesktop", "-nosplash"]
                    logger.info("Starting MATLAB in headless mode (no desktop)...")
                else:
                    startup_opts = ["-nosplash"]
                    logger.info("Starting MATLAB with desktop...")

                # Start MATLAB with options
                option_str = " ".join(startup_opts) if startup_opts else ""
                self.engine = matlab.engine.start_matlab(option_str)
                self._started_new = True
                logger.info("New MATLAB instance started successfully")

            # Add strategies directory to path
            if STRATEGIES_DIR and STRATEGIES_DIR.exists():
                self.engine.addpath(str(STRATEGIES_DIR), nargout=0)
                logger.info(f"Added {STRATEGIES_DIR} to MATLAB path")

            self.connected = True
            logger.info("MATLAB Engine ready")
            return True

        except ImportError:
            logger.warning("MATLAB Engine API not installed. Run: pip install matlabengine")
            return False
        except Exception as e:
            logger.warning(f"Could not connect to MATLAB: {e}")
            return False

    def disconnect(self):
        """Disconnect from MATLAB Engine."""
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None
        self.connected = False
        logger.info("MATLAB Engine disconnected")

    def is_available(self) -> bool:
        """Check if MATLAB is available."""
        return self.connected and self.engine is not None

    def evaluate_position_rl(
        self,
        state: str,
        is_our_turn: bool
    ) -> tuple[float, dict]:
        """
        Evaluate position using Reinforcement Learning value network.

        Args:
            state: 15-char board state ('G', 'P', '-')
            is_our_turn: True if it's our turn

        Returns:
            (value, policy_dict) where:
            - value: Expected outcome in [-1, 1]
            - policy_dict: {(edge, color): probability} for available actions
        """
        if not self.is_available():
            return 0.0, {}

        try:
            # Convert state to MATLAB format
            import matlab
            state_vec = matlab.double([
                1.0 if c == 'G' else (-1.0 if c == 'P' else 0.0)
                for c in state
            ])
            turn_flag = matlab.double([1.0 if is_our_turn else -1.0])

            # Call MATLAB function
            result = self.engine.evaluate_position(state_vec, turn_flag, nargout=2)

            # Handle MATLAB return types
            value = float(result[0]) if not hasattr(result[0], '__iter__') else float(list(result[0])[0])
            policy_raw = result[1]

            # Convert MATLAB array to Python list
            if hasattr(policy_raw, '_data'):
                # matlab.double object
                policy_list = list(policy_raw._data)
            elif hasattr(policy_raw, '__iter__'):
                policy_list = [float(x) for x in policy_raw]
            else:
                policy_list = []

            # Convert policy to Python dict
            policy = {}
            for i in range(15):
                if state[i] == '-' and len(policy_list) >= 30:
                    # MATLAB uses 1-based indexing, policy is [G1,P1,G2,P2,...]
                    policy[(i, 'G')] = policy_list[i * 2]
                    policy[(i, 'P')] = policy_list[i * 2 + 1]

            return value, policy

        except Exception as e:
            logger.warning(f"MATLAB RL evaluation failed: {e}")
            return 0.0, {}

    def run_simulated_annealing(
        self,
        state: str,
        num_samples: int = 100
    ) -> tuple[float, float, list]:
        """
        Run simulated annealing to deeply evaluate position.

        Uses Global Optimization Toolbox for multi-start optimization.

        Args:
            state: 15-char board state
            num_samples: Number of SA runs for confidence estimation

        Returns:
            (mean_value, confidence, best_moves) where:
            - mean_value: Average outcome across samples
            - confidence: Standard deviation (lower = more confident)
            - best_moves: List of (edge, color) recommendations
        """
        if not self.is_available():
            return 0.0, 1.0, []

        try:
            import matlab
            state_vec = matlab.double([
                1.0 if c == 'G' else (-1.0 if c == 'P' else 0.0)
                for c in state
            ])

            result = self.engine.sa_evaluate(
                state_vec,
                float(num_samples),
                nargout=3
            )

            mean_val = float(result[0])
            confidence = float(result[1])
            moves_raw = result[2]

            # Parse best moves
            best_moves = []
            if moves_raw:
                for m in moves_raw:
                    edge = int(m[0]) - 1  # MATLAB 1-indexed
                    color = 'G' if m[1] > 0 else 'P'
                    best_moves.append((edge, color))

            return mean_val, confidence, best_moves

        except Exception as e:
            logger.warning(f"MATLAB SA evaluation failed: {e}")
            return 0.0, 1.0, []

    def identify_opponent_strategy(
        self,
        game_traces: list
    ) -> dict:
        """
        Identify opponent strategy patterns using System Identification.

        Args:
            game_traces: List of game histories, each containing moves

        Returns:
            Dict with:
            - edge_preferences: {edge: weight} for opponent's edge preferences
            - color_bias: Overall green vs purple preference
            - response_patterns: Common opponent responses to our moves
        """
        if not self.is_available():
            return {}

        try:
            import matlab

            # Convert traces to MATLAB format
            # Each trace: [(edge, color, score), ...]
            traces_matlab = []
            for trace in game_traces:
                trace_vec = []
                for edge, color, score in trace:
                    trace_vec.extend([float(edge), 1.0 if color == 'G' else -1.0, score])
                traces_matlab.append(matlab.double(trace_vec))

            result = self.engine.identify_opponent(
                matlab.double(traces_matlab),
                nargout=1
            )

            # Parse result - edge_prefs is a 1x15 array, convert to dict with edge indices
            edge_prefs_raw = result.get('edge_prefs', [])
            if hasattr(edge_prefs_raw, '_data'):
                edge_prefs_list = list(edge_prefs_raw._data)
            elif hasattr(edge_prefs_raw, '__iter__'):
                edge_prefs_list = [float(x) for x in edge_prefs_raw]
            else:
                edge_prefs_list = []

            # Convert to {edge_index: preference} dict
            edge_preferences = {i: edge_prefs_list[i] for i in range(len(edge_prefs_list))}

            return {
                'edge_preferences': edge_preferences,
                'color_bias': float(result.get('color_bias', 0.0)),
                'response_patterns': result.get('responses', {})
            }

        except Exception as e:
            logger.warning(f"MATLAB opponent identification failed: {e}")
            return {}

    def call_function(self, func_name: str, *args, nargout: int = 1):
        """
        Call arbitrary MATLAB function.

        Args:
            func_name: Name of MATLAB function
            *args: Arguments to pass
            nargout: Number of output arguments

        Returns:
            Function result(s)
        """
        if not self.is_available():
            raise RuntimeError("MATLAB not connected")

        func = getattr(self.engine, func_name)
        return func(*args, nargout=nargout)


def get_bridge() -> MatlabBridge:
    """Get singleton MATLAB bridge instance."""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = MatlabBridge()
    return _bridge_instance
