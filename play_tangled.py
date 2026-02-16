#!/usr/bin/env python3
"""
Play Tangled on tangled-game.com using configurable strategies.

Port of the working browser JS bot (V28.2) to Python/Playwright.
Uses the same robust DOM interaction patterns:
- Nearest-vertex edge matching
- Direct MouseEvent dispatch
- Text-based turn/game-over detection

Strategies:
- hybrid_solver: D-Wave inspired minimax+MCTS+learning (DEFAULT)
- alphaq_explorer: Explore/exploit vs AlphaQ Up with closed learning loop
- melissa_killer: Cycles E12P/E13P against Melissa (40% win rate)
- amara_killer: Uses E14P against Amara
- amara_explorer: Cycles all 30 openings
- mcts: Monte Carlo Tree Search
- heuristic: Fast parameterized strategy with learning

Usage:
    python play_tangled.py                                    # Play 5 games vs Melissa (hybrid_solver)
    python play_tangled.py --strategy alphaq_explorer         # Use AlphaQ Explorer strategy
    python play_tangled.py --opponent alphaq                  # Play vs AlphaQ Up
    python play_tangled.py --games 10                         # Play 10 games
    python play_tangled.py --mcts-iterations 100000           # Use 100K iterations (faster, lower quality)
    python play_tangled.py --mcts-iterations 1000000          # Use 1M iterations (slower, diminishing returns)
    python play_tangled.py --mcts-time 30                     # Limit thinking time to 30 seconds per move
"""

import argparse
import atexit
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import coloredlogs
from dotenv import load_dotenv

load_dotenv()


class StuckOpponentError(Exception):
    """Raised when opponent doesn't play within timeout (likely stuck/crashed)."""
    pass


# Process tracking for safe cleanup (supports multiple concurrent sessions)
PROCESS_TRACKING_DIR = Path.home() / ".tangled" / "active_processes"


def register_process(run_id: int = None, planned_games: int = None,
                     strategy: str = None, opponent: str = None):
    """Register this process as an active game runner."""
    PROCESS_TRACKING_DIR.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    info = {
        "pid": pid,
        "started": datetime.now().isoformat(),
        "run_id": run_id,
        "planned_games": planned_games,
        "strategy": strategy,
        "opponent": opponent,
    }
    with open(PROCESS_TRACKING_DIR / f"{pid}.json", 'w') as f:
        json.dump(info, f)


def unregister_process():
    """Remove this process from tracking."""
    try:
        pid_file = PROCESS_TRACKING_DIR / f"{os.getpid()}.json"
        if pid_file.exists():
            pid_file.unlink()
    except Exception:
        pass


def get_active_processes() -> list:
    """Get info about all currently active game processes."""
    results = []
    if not PROCESS_TRACKING_DIR.exists():
        return results
    for pid_file in PROCESS_TRACKING_DIR.glob("*.json"):
        try:
            with open(pid_file, 'r') as f:
                info = json.load(f)
            pid = info.get("pid")
            if pid:
                try:
                    os.kill(pid, 0)  # Check if process exists
                    results.append(info)
                except OSError:
                    # Process no longer running, clean up stale file
                    pid_file.unlink()
        except Exception:
            pass
    return results


def get_active_process() -> dict:
    """Get info about an active game process (backward compat). Returns first found."""
    processes = get_active_processes()
    return processes[0] if processes else None

# Global reference for cleanup on signals
_active_player = None


def _cleanup_on_exit():
    """Cleanup handler for atexit."""
    global _active_player
    if _active_player:
        logging.getLogger(__name__).info("Cleaning up browser on exit...")
        try:
            _active_player.stop()
        except Exception:
            pass
        _active_player = None


def _signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT for graceful cleanup."""
    global _active_player
    sig_name = signal.Signals(signum).name
    logging.getLogger(__name__).info(f"Received {sig_name}, cleaning up...")
    if _active_player:
        try:
            _active_player.stop()
        except Exception:
            pass
        _active_player = None
    sys.exit(128 + signum)


# Register cleanup handlers
atexit.register(_cleanup_on_exit)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT, _signal_handler)

from snowdrop_tangled_agents.strategy.petersen_strategy import PetersenStrategy
from snowdrop_tangled_agents.strategy.mcts_strategy import MCTSStrategy, HybridStrategy, evaluate_terminal_state
from snowdrop_tangled_agents.strategy.oracle_route_strategy import OracleRouteStrategy
from snowdrop_tangled_agents.strategy.terminal_explorer_strategy import TerminalExplorerStrategy
from snowdrop_tangled_agents.stats import get_collector, queries as stats_queries, GameMetricsTracker
from snowdrop_tangled_agents.stats import get_publisher, StatsPublisher
from snowdrop_tangled_agents.stats.session_stats import get_session_stats, get_run_stats

# Optional MATLAB integration
try:
    from snowdrop_tangled_agents.matlab import (
        MatlabEnhancedStrategy,
        HybridSolverStrategy,
        get_unified_bridge,
        print_training_status,
    )
    from snowdrop_tangled_agents.matlab.matlab_strategy import AmaraExplorerStrategy, AmaraKillerStrategy, MelissaKillerStrategy, AlphaQExplorerStrategy
    MATLAB_IMPORTS_AVAILABLE = True
except ImportError:
    MATLAB_IMPORTS_AVAILABLE = False
    HybridSolverStrategy = None
    AmaraExplorerStrategy = None
    AmaraKillerStrategy = None
    MelissaKillerStrategy = None
    AlphaQExplorerStrategy = None
    print_training_status = None

# Check actual MATLAB Engine availability (not just Python wrappers)
MATLAB_AVAILABLE = False
MATLAB_ENGINE_AVAILABLE = False
MATLAB_SESSIONS = []
MATLAB_UNAVAILABLE_REASON = None

if MATLAB_IMPORTS_AVAILABLE:
    try:
        import matlab.engine
        MATLAB_ENGINE_AVAILABLE = True
        # Check for shared sessions first
        try:
            MATLAB_SESSIONS = list(matlab.engine.find_matlab())
        except Exception:
            MATLAB_SESSIONS = []
        # Engine API available = MATLAB available (bridge will start one if no shared session)
        MATLAB_AVAILABLE = True
    except ImportError:
        MATLAB_UNAVAILABLE_REASON = "MATLAB Engine API not installed. Run: pip install matlabengine"
        # Check for compiled packages as fallback
        try:
            from snowdrop_tangled_agents.matlab.compiled_bridge import packages_available
            pkgs = packages_available()
            if any(pkgs.values()):
                MATLAB_AVAILABLE = True
                MATLAB_UNAVAILABLE_REASON = None
        except Exception:
            pass
else:
    MATLAB_UNAVAILABLE_REASON = "MATLAB integration module not installed"

# Optional RL strategy
try:
    from snowdrop_tangled_agents.strategy.rl_strategy import RLStrategy, EnsembleStrategy
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    EnsembleStrategy = None

# Optional MATLAB MCTS strategy
try:
    from snowdrop_tangled_agents.strategy.matlab_mcts_strategy import MatlabMCTSStrategy, MCTSParams
    MATLAB_MCTS_AVAILABLE = True
except ImportError:
    MATLAB_MCTS_AVAILABLE = False
    MatlabMCTSStrategy = None
    MCTSParams = None


# Strategy metadata for validation
# Loss rates based on historical game data
STRATEGY_INFO = {
    "heuristic": {
        "requires_matlab": False,
        "loss_rate": 0.70,  # 70.4% loss rate
        "description": "Fast heuristic-based moves",
    },
    "mcts": {
        "requires_matlab": False,
        "loss_rate": 0.61,  # 60.6% loss rate
        "description": "Pure Monte Carlo Tree Search",
    },
    "hybrid": {
        "requires_matlab": False,
        "loss_rate": 0.37,  # 37.4% loss rate - BEST without MATLAB
        "description": "Opening book + MCTS + exhaustive endgame",
    },
    "matlab": {
        "requires_matlab": True,
        "loss_rate": 0.40,  # Estimated
        "description": "MATLAB-enhanced with neural network priors",
    },
    "rl": {
        "requires_matlab": False,
        "loss_rate": 1.00,  # 100% loss rate (only 4 games)
        "description": "Trained PPO reinforcement learning agent",
    },
    "ensemble": {
        "requires_matlab": False,
        "loss_rate": 0.50,  # Estimated
        "description": "RL + Monte Carlo rollouts ensemble",
    },
    "matlab_mcts": {
        "requires_matlab": True,
        "loss_rate": 0.67,  # 66.7% loss rate
        "description": "MATLAB MCTS engine with high compute",
    },
    "hybrid_solver": {
        "requires_matlab": True,
        "loss_rate": 0.45,  # 45.5% loss rate
        "description": "D-Wave inspired minimax + MCTS + Tabu Search",
    },
}


def verify_matlab_readiness() -> bool:
    """
    Verify MATLAB is ready for game execution.

    Connects to MATLAB session, cleans up any stale parallel pools,
    and verifies MATLAB is responsive before starting the browser.

    Returns True if ready, False otherwise.
    """
    import matlab.engine
    from snowdrop_tangled_agents.matlab import get_bridge

    print("\n" + "=" * 60)
    print("MATLAB READINESS CHECK")
    print("=" * 60)

    try:
        # Connect to MATLAB
        print(f"  Connecting to MATLAB session...")
        bridge = get_bridge()
        if not bridge.connect():
            print(f"  [FAIL] Failed to connect to MATLAB")
            return False

        print(f"  [OK] Connected to MATLAB")

        # Clean up any stale parallel pools
        print(f"  Cleaning up stale parallel pools...")
        try:
            bridge.engine.eval(
                "pool = gcp('nocreate'); if ~isempty(pool), delete(pool); fprintf('  [OK] Deleted stale pool with %d workers\\n', pool.NumWorkers); else fprintf('  [OK] No stale pools found\\n'); end",
                nargout=0
            )
        except Exception as e:
            print(f"  [FAIL] Pool cleanup failed: {e}")
            return False

        # Verify MATLAB is responsive with a simple test
        print(f"  Verifying MATLAB responsiveness...")
        try:
            result = bridge.engine.eval("2 + 2")
            if result != 4:
                print(f"  [FAIL] MATLAB returned unexpected result: {result}")
                return False
            print(f"  [OK] MATLAB is responsive")
        except Exception as e:
            print(f"  [FAIL] MATLAB test failed: {e}")
            return False

        print("=" * 60)
        print("MATLAB IS READY")
        print("=" * 60 + "\n")
        return True

    except Exception as e:
        print(f"  [FAIL] Readiness check failed: {e}")
        print("=" * 60 + "\n")
        return False


def check_matlab_availability() -> bool:
    """
    Check and report MATLAB availability status.

    If MATLAB is not available, prompts user to confirm proceeding without it.
    Returns True if OK to proceed, False to abort.
    """
    print("\n" + "=" * 60)
    print("MATLAB AVAILABILITY CHECK")
    print("=" * 60)

    if MATLAB_AVAILABLE:
        if MATLAB_SESSIONS:
            print(f"  Status: AVAILABLE")
            print(f"  Sessions: {', '.join(MATLAB_SESSIONS)}")
        elif MATLAB_ENGINE_AVAILABLE:
            print(f"  Status: AVAILABLE (will start engine on demand)")
        else:
            print(f"  Status: AVAILABLE (compiled packages)")
        print(f"  MATLAB strategies (hybrid_solver, matlab_mcts) are enabled")
        print("=" * 60 + "\n")
        return True

    # MATLAB not available - explain why and ask user
    print(f"  Status: NOT AVAILABLE")
    print(f"  Reason: {MATLAB_UNAVAILABLE_REASON}")
    print("")
    print("  Without MATLAB, the following strategies will NOT work:")
    print("    - hybrid_solver (D-Wave inspired, best MATLAB strategy)")
    print("    - matlab_mcts (MATLAB MCTS engine)")
    print("    - matlab (MATLAB-enhanced with neural networks)")
    print("")
    print("  Available non-MATLAB strategies:")
    print("    - hybrid: 37% loss rate (RECOMMENDED)")
    print("    - mcts: 61% loss rate")
    print("    - heuristic: 70% loss rate")
    print("=" * 60)

    while True:
        try:
            response = input("\nProceed without MATLAB? [y/N]: ").strip().lower()
            if response in ('y', 'yes'):
                print("Continuing without MATLAB support.\n")
                return True
            elif response in ('n', 'no', ''):
                print("\nTo enable MATLAB:")
                print("  1. Start MATLAB")
                print("  2. Run: matlab.engine.shareEngine")
                print("  3. Restart this script")
                print("\nAborting.")
                return False
            else:
                print("Please enter 'y' or 'n'.")
        except (EOFError, KeyboardInterrupt):
            print("\nAborting.")
            return False


def validate_strategy_selection(strategy: str) -> str:
    """
    Validate strategy selection and prompt for alternatives if needed.

    Returns the validated (possibly changed) strategy name.
    """
    info = STRATEGY_INFO.get(strategy)
    if not info:
        return strategy

    # Check if MATLAB is required but unavailable
    if info["requires_matlab"] and not MATLAB_AVAILABLE:
        print(f"\n*** Strategy '{strategy}' requires MATLAB which is not available ***")
        print(f"Description: {info['description']}")
        print("\nAvailable alternatives (sorted by performance):")

        # Get non-MATLAB strategies sorted by loss rate
        alternatives = [
            (name, data) for name, data in STRATEGY_INFO.items()
            if not data["requires_matlab"]
        ]
        alternatives.sort(key=lambda x: x[1]["loss_rate"])

        for i, (name, data) in enumerate(alternatives, 1):
            loss_pct = data["loss_rate"] * 100
            marker = " (recommended)" if data["loss_rate"] < 0.50 else ""
            warning = " [HIGH LOSS RATE]" if data["loss_rate"] >= 0.50 else ""
            print(f"  {i}. {name}: {loss_pct:.0f}% loss rate - {data['description']}{marker}{warning}")

        print(f"  {len(alternatives) + 1}. Continue with '{strategy}' anyway (will use fallback)")

        while True:
            try:
                choice = input("\nSelect strategy [1]: ").strip()
                if not choice:
                    choice = "1"
                choice_num = int(choice)
                if 1 <= choice_num <= len(alternatives):
                    strategy = alternatives[choice_num - 1][0]
                    print(f"Selected: {strategy}")
                    break
                elif choice_num == len(alternatives) + 1:
                    print(f"Continuing with {strategy} (will use fallback strategy)")
                    break
                else:
                    print("Invalid choice, try again.")
            except ValueError:
                print("Please enter a number.")
            except (EOFError, KeyboardInterrupt):
                print("\nUsing default: hybrid")
                strategy = "hybrid"
                break

    # Check if strategy has high loss rate
    if info["loss_rate"] >= 0.50:
        loss_pct = info["loss_rate"] * 100
        print(f"\n*** Warning: Strategy '{strategy}' has a {loss_pct:.0f}% historical loss rate ***")
        print(f"Description: {info['description']}")

        # Suggest better alternatives
        better = [(n, d) for n, d in STRATEGY_INFO.items()
                  if d["loss_rate"] < 0.50 and (not d["requires_matlab"] or MATLAB_AVAILABLE)]
        if better:
            better.sort(key=lambda x: x[1]["loss_rate"])
            print("\nBetter alternatives available:")
            for name, data in better[:3]:
                print(f"  - {name}: {data['loss_rate']*100:.0f}% loss rate")

        while True:
            try:
                confirm = input(f"\nContinue with '{strategy}'? [y/N]: ").strip().lower()
                if confirm in ('y', 'yes'):
                    print(f"Confirmed: using {strategy}")
                    break
                elif confirm in ('n', 'no', ''):
                    # Suggest the best alternative
                    if better:
                        strategy = better[0][0]
                        print(f"Switched to: {strategy}")
                    break
                else:
                    print("Please enter 'y' or 'n'.")
            except (EOFError, KeyboardInterrupt):
                if better:
                    strategy = better[0][0]
                    print(f"\nSwitched to: {strategy}")
                break

    return strategy


# Vertex coordinates - from working JS bot (tangled-bot-v28.txt)
# These match the actual website SVG layout
VERTEX_COORDS = {
    0: (451, 416),
    1: (163, 80),
    2: (19, 304),
    3: (307, 500),
    4: (595, 500),
    5: (80, 248),   # RED (us)
    6: (451, 80),   # HUB
    7: (820, 248),  # BLUE (opponent)
    8: (739, 80),
    9: (883, 304),
}

# Petersen graph edges - matches actual website SVG structure
# Inner vertices form a pentagram (star), not a simple pentagon
EDGES = [
    (0, 2), (0, 3), (0, 6), (1, 3), (1, 4),
    (1, 7), (2, 4), (2, 8), (3, 9), (4, 5),
    (5, 6), (5, 9), (6, 7), (7, 8), (8, 9),
]


class WebPlayer:
    """
    Plays Tangled on tangled-game.com using Playwright.
    Uses the same DOM interaction patterns as the working JS bot.
    """

    BASE_URL = "https://tangled-game.com"
    OPPONENTS = {
        "randy": "Random Randy",
        "amara": "AlphaZero Amara",
        "melissa": "MCTS Melissa",
        "andy": "AlphaZero Andy",
        "alphaq": "AlphaQ Up",
    }

    def __init__(
        self,
        headless: bool = False,
        slow_mo: int = 100,
        strategy_type: str = "heuristic",
        mcts_time: float = float('inf'),
        mcts_iterations: int = 500_000,
        use_nn: bool = True,
        adapt_opponent: bool = True,
        opening_mode: Optional[str] = None,
        route_mode: Optional[str] = None,
        routes_file: Optional[str] = None,
        random_turns: Optional[set] = None,
        novel_branch: bool = False,
        seat: int = 1,
    ):
        self.username = os.getenv("TANGLED_USERNAME")
        self.password = os.getenv("TANGLED_PASSWORD")
        self.headless = headless
        self.slow_mo = slow_mo
        self.strategy_type = strategy_type
        self.seat = seat

        # Store options for MATLAB strategy
        self._use_nn = use_nn
        self._adapt_opponent = adapt_opponent
        self._opening_mode = opening_mode

        # Oracle route options
        self._oracle_route_mode = route_mode or 'fixed'
        self._oracle_routes_file = routes_file

        # Terminal explorer options
        self._random_turns = random_turns
        self._novel_branch = novel_branch

        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None

        # Strategy initialization based on type
        params_path = Path.home() / ".tangled" / "petersen_params.json"
        params_path.parent.mkdir(parents=True, exist_ok=True)
        self.params_path = str(params_path)

        if strategy_type == "mcts":
            self.strategy = MCTSStrategy(
                time_limit=mcts_time,
                max_iterations=mcts_iterations
            )
        elif strategy_type == "hybrid":
            self.strategy = HybridStrategy(
                mcts_time_limit=mcts_time,
                mcts_iterations=mcts_iterations
            )
        elif strategy_type == "matlab":
            if not MATLAB_AVAILABLE:
                self.logger.warning("MATLAB strategy unavailable, falling back to hybrid")
                self.strategy = HybridStrategy(
                    mcts_time_limit=mcts_time,
                    mcts_iterations=mcts_iterations
                )
            else:
                self.strategy = MatlabEnhancedStrategy(
                    mcts_time_limit=mcts_time,
                    mcts_iterations=mcts_iterations,
                    use_nn_priors=getattr(self, '_use_nn', True),
                    use_opponent_adaptation=getattr(self, '_adapt_opponent', True),
                )
                # Note: initialize() called in play_game() with opponent name
        elif strategy_type == "rl":
            if not RL_AVAILABLE:
                self.logger.warning("RL strategy unavailable, falling back to petersen")
                self.strategy = PetersenStrategy(params_path=self.params_path)
            else:
                self.strategy = RLStrategy()
        elif strategy_type == "ensemble":
            if not RL_AVAILABLE or EnsembleStrategy is None:
                self.logger.warning("Ensemble strategy unavailable, falling back to petersen")
                self.strategy = PetersenStrategy(params_path=self.params_path)
            else:
                self.strategy = EnsembleStrategy(
                    num_workers=22,
                    rollouts_per_action=50,
                    top_k=5,
                )
        elif strategy_type == "matlab_mcts":
            if not MATLAB_MCTS_AVAILABLE or MatlabMCTSStrategy is None:
                self.logger.warning("MATLAB MCTS strategy unavailable, falling back to python mcts")
                self.strategy = MCTSStrategy(time_limit=float('inf'), max_iterations=500000)
            else:
                # Create params with high compute (match Melissa's ~20-25s)
                mcts_params_path = Path.home() / ".tangled" / "matlab_mcts_params.json"
                self.strategy = MatlabMCTSStrategy(
                    params_path=str(mcts_params_path),
                    fallback_to_python=True,
                )
        elif strategy_type == "hybrid_solver":
            if not MATLAB_AVAILABLE or HybridSolverStrategy is None:
                self.logger.warning("Hybrid solver unavailable, falling back to python mcts")
                self.strategy = MCTSStrategy(time_limit=float('inf'), max_iterations=mcts_iterations)
            else:
                self.strategy = HybridSolverStrategy(
                    time_limit=mcts_time,
                    minimax_depth=4,
                    mcts_iterations=mcts_iterations,
                    player=1,
                )
        elif strategy_type == "amara_explorer":
            if not MATLAB_AVAILABLE or AmaraExplorerStrategy is None:
                self.strategy = MCTSStrategy(time_limit=float('inf'), max_iterations=mcts_iterations)
            else:
                self.strategy = AmaraExplorerStrategy(
                    time_limit=mcts_time,
                    minimax_depth=4,
                    mcts_iterations=mcts_iterations,
                    player=1,
                )
        elif strategy_type == "amara_killer":
            if not MATLAB_AVAILABLE or AmaraKillerStrategy is None:
                self.strategy = MCTSStrategy(time_limit=float('inf'), max_iterations=mcts_iterations)
            else:
                self.strategy = AmaraKillerStrategy(
                    time_limit=mcts_time,
                    minimax_depth=4,
                    mcts_iterations=mcts_iterations,
                    player=1,
                    opening_mode='best',  # Always use E14P (highest win score)
                )
        elif strategy_type == "melissa_killer":
            if not MATLAB_AVAILABLE or MelissaKillerStrategy is None:
                self.strategy = MCTSStrategy(time_limit=float('inf'), max_iterations=mcts_iterations)
            else:
                self.strategy = MelissaKillerStrategy(
                    time_limit=mcts_time,
                    minimax_depth=4,
                    mcts_iterations=mcts_iterations,
                    player=1,
                    opening_mode='cycle',  # Cycle through E9G, E12P, E13P
                )
        elif strategy_type == "alphaq_explorer":
            if not MATLAB_AVAILABLE or AlphaQExplorerStrategy is None:
                self.strategy = MCTSStrategy(time_limit=float('inf'), max_iterations=mcts_iterations)
            else:
                opening_mode = getattr(self, '_opening_mode', None)
                if opening_mode == 'round_robin':
                    force_opening = None
                elif opening_mode == 'thompson':
                    force_opening = None
                else:
                    # Default: forced E7G (historical best)
                    force_opening = 'E7G'
                self.strategy = AlphaQExplorerStrategy(
                    time_limit=mcts_time,
                    minimax_depth=4,
                    mcts_iterations=mcts_iterations,
                    player=1,
                    force_opening=force_opening,
                    opening_mode=opening_mode or 'forced',
                )
        elif strategy_type == "oracle_route":
            routes_file = getattr(self, '_oracle_routes_file', None) or "oracle_routes.json"
            oracle_routes_path = Path(__file__).parent / "oracle-solver" / "output" / routes_file
            fallback = MCTSStrategy(time_limit=float('inf'), max_iterations=mcts_iterations)
            route_index = getattr(self, '_oracle_route_index', 6)  # Route 7 (best confidence)
            route_mode = getattr(self, '_oracle_route_mode', 'fixed')
            self.strategy = OracleRouteStrategy(
                routes_path=str(oracle_routes_path),
                fallback_strategy=fallback,
                route_index=route_index,
                route_mode=route_mode,
            )
        elif strategy_type == "terminal_explorer":
            fallback = MCTSStrategy(time_limit=float('inf'), max_iterations=mcts_iterations)
            self.strategy = TerminalExplorerStrategy(
                fallback_strategy=fallback,
                randomize_midgame=True,
                random_move_turns=self._random_turns,
                novel_branch=self._novel_branch,
            )
        else:  # "heuristic" (default)
            self.strategy = PetersenStrategy(params_path=self.params_path)

        self.score_history = []
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using strategy: {strategy_type}")

        # Stats collection
        self.stats_collector = get_collector()
        self.current_game_id = None
        self.mcts_time = mcts_time

        # Opponent modeling for online learning
        self.opponent_model = None
        self._opponent_model_updates = 0

    def _ensure_opponent_model(self, opponent: str = "melissa"):
        """Lazily initialize opponent model for online learning."""
        if self.opponent_model is None:
            try:
                from snowdrop_tangled_agents.stats import get_opponent_model
                self.opponent_model = get_opponent_model(opponent)
                self.logger.info(f"Loaded opponent model: {self.opponent_model.total_games} games, "
                               f"{self.opponent_model.total_moves} moves")
            except Exception as e:
                self.logger.warning(f"Could not load opponent model: {e}")

    def _update_opponent_model(self):
        """Update opponent model with moves from completed game (online learning)."""
        if self.opponent_model is None:
            return

        try:
            # Reload from database to get latest game's moves
            self.opponent_model.load_from_database()
            self._opponent_model_updates += 1

            # Re-export to .mat for MATLAB (every game for now)
            # Use PID-suffixed path to avoid concurrent write conflicts
            from pathlib import Path as _Path
            mat_dir = _Path(__file__).parent / 'snowdrop_tangled_agents' / 'matlab' / 'rl' / 'data'
            mat_path = mat_dir / f'opponent_model_{os.getpid()}.mat'
            self.opponent_model.save_mat(mat_path)
            self.logger.debug(f"Opponent model updated: {self.opponent_model.total_moves} moves "
                            f"(update #{self._opponent_model_updates})")
        except Exception as e:
            self.logger.warning(f"Could not update opponent model: {e}")

    def start(self):
        global _active_player
        from playwright.sync_api import sync_playwright
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
        )
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        _active_player = self  # Register for cleanup on signals
        self.logger.info("Browser started")

    def stop(self):
        global _active_player
        if self.context:
            try:
                self.context.close()
            except Exception:
                pass
            self.context = None
        if self.browser:
            try:
                self.browser.close()
            except Exception:
                pass
            self.browser = None
        if self.playwright:
            try:
                self.playwright.stop()
            except Exception:
                pass
            self.playwright = None
        if _active_player is self:
            _active_player = None
        self.logger.info("Browser stopped")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.stop()
        return False  # Don't suppress exceptions

    def login(self):
        self.logger.info(f"Navigating to {self.BASE_URL}")
        self.page.goto(self.BASE_URL)
        self.page.wait_for_load_state("networkidle")
        time.sleep(2)

        # Click login button
        try:
            login_btn = self.page.locator("text=/log.?in/i").first
            if login_btn.is_visible(timeout=3000):
                login_btn.click()
                time.sleep(2)
                self.page.wait_for_load_state("networkidle")
        except:
            pass

        # Wait for Auth0 form
        time.sleep(3)

        # Fill credentials
        try:
            self.page.locator("input[name='username']").fill(self.username)
            self.page.locator("input[name='password']").fill(self.password)
            # Click the Continue button (not Google sign-in)
            self.page.locator("button[name='action']").click()
            self.logger.info("Submitted login form")
        except Exception as e:
            self.logger.warning(f"Login form error: {e}")

        time.sleep(3)
        self.page.wait_for_load_state("networkidle")
        self.logger.info(f"Logged in, URL: {self.page.url}")
        return True

    def start_game(self, opponent: str = "melissa"):
        """Start a new game. Navigate fresh to avoid stale state."""
        opponent_name = self.OPPONENTS.get(opponent.lower(), opponent)
        self.logger.info(f"Starting game against {opponent_name}")

        # Always navigate to /play fresh to clear any previous game state
        self.page.goto(f"{self.BASE_URL}/play")
        self.page.wait_for_load_state("networkidle")
        time.sleep(1)

        # Select player seat
        try:
            if self.seat == 2:
                self.page.locator("text=/Player 2.*Blue/i").first.click(timeout=3000)
                self.logger.info("Selected Player 2 (Blue)")
            else:
                self.page.locator("text=/Player 1.*Red/i").first.click(timeout=3000)
                self.logger.info("Selected Player 1 (Red)")
            time.sleep(0.5)
        except:
            pass

        # Select Petersen graph
        try:
            select = self.page.locator("select").first
            options = self.page.locator("select option").all()
            for i, opt in enumerate(options):
                if "petersen" in opt.inner_text().lower():
                    select.select_option(index=i)
                    self.logger.info(f"Selected Petersen graph")
                    break
            time.sleep(0.5)
        except Exception as e:
            self.logger.warning(f"Graph selection: {e}")

        # Select opponent
        try:
            self.page.locator(f"text=/{opponent_name}/i").first.click(timeout=3000)
            self.logger.info(f"Selected opponent: {opponent_name}")
            time.sleep(0.5)
        except Exception as e:
            self.logger.warning(f"Opponent selection: {e}")

        # Click Start Game
        try:
            self.page.locator("text=/start game/i").first.click(timeout=3000)
            self.logger.info("Clicked Start Game")
            time.sleep(2)
        except Exception as e:
            self.logger.warning(f"Start Game: {e}")

        # Wait for game board
        try:
            self.page.wait_for_selector("svg line", timeout=30000)
            self.logger.info("Game board ready")
            return True
        except:
            self.logger.error("Game board not found")
            return False

    def read_score(self) -> float:
        """Read score from page text."""
        try:
            text = self.page.inner_text("body")
            import re
            match = re.search(r"Score:\s*([-\d.]+)", text)
            if match:
                return float(match.group(1))
        except:
            pass
        return 0.0

    def is_our_turn(self) -> bool:
        """Check if it's our turn."""
        try:
            text = self.page.inner_text("body").lower()
            # Debug: log turn detection occasionally
            if hasattr(self, '_turn_check_count'):
                self._turn_check_count += 1
            else:
                self._turn_check_count = 1
            if self._turn_check_count % 50 == 1:
                # Extract relevant portion for logging
                turn_text = ""
                for phrase in ["your turn", "opponent", "waiting", "game over", "winner"]:
                    if phrase in text:
                        turn_text += f"[{phrase}] "
                if turn_text:
                    self.logger.debug(f"Turn status: {turn_text.strip()}")
            # Explicit turn indicators
            if "your turn" in text:
                return True
            # Seat-aware player turn detection
            our_player = f"player {self.seat}"
            their_player = f"player {3 - self.seat}"
            if our_player in text and "turn" in text and their_player not in text:
                return True
            # Explicit NOT our turn indicators
            if "opponent" in text and "turn" in text:
                return False
            if their_player in text and "turn" in text:
                return False
            if "waiting" in text:
                return False
        except:
            pass
        return False

    def is_game_over(self) -> bool:
        """Check if game is over."""
        try:
            text = self.page.inner_text("body")
            # Only explicit game over indicators (case-sensitive like JS bot)
            if "Game Over" in text:
                return True
            if "Winner:" in text:
                return True
            # Check if all edges are played
            state = self.read_board()
            if state.count('-') == 0:
                return True
        except:
            pass
        return False

    def get_outcome(self) -> str:
        """Get game outcome."""
        try:
            text = self.page.inner_text("body")
            our_player = f"Player {self.seat}"
            their_player = f"Player {3 - self.seat}"
            if f"Winner: {our_player}" in text:
                return "win"
            if f"Winner: {their_player}" in text:
                return "loss"
            if "Draw" in text:
                return "draw"
            # Fallback to score
            score = self.read_score()
            if score > 0.5:
                return "win"
            if score < -0.5:
                return "loss"
        except:
            pass
        return "draw"

    def read_board(self) -> str:
        """Read board state by extracting vertex coordinates dynamically from SVG."""
        # Dynamically discover vertices from line endpoints, then match to edge list
        js_code = """
        () => {
            // Collect all unique endpoints from SVG lines
            const points = [];
            document.querySelectorAll('line').forEach(l => {
                points.push({x: +l.getAttribute('x1'), y: +l.getAttribute('y1')});
                points.push({x: +l.getAttribute('x2'), y: +l.getAttribute('y2')});
            });

            // Cluster points to find unique vertices (within 5px tolerance)
            const vertices = [];
            for (const p of points) {
                let found = false;
                for (const v of vertices) {
                    const d = Math.sqrt((p.x - v.x)**2 + (p.y - v.y)**2);
                    if (d < 5) { found = true; break; }
                }
                if (!found) vertices.push({x: p.x, y: p.y});
            }

            if (vertices.length !== 10) {
                // Fallback: return all grey if vertex count wrong
                return '-'.repeat(15);
            }

            const cx = vertices.reduce((s,v) => s + v.x, 0) / 10;
            const cy = vertices.reduce((s,v) => s + v.y, 0) / 10;
            const angle = (v) => Math.atan2(v.y - cy, v.x - cx);

            // Angular distance that handles wrap-around
            const angleDist = (a1, a2) => {
                let d = a1 - a2;
                while (d > Math.PI) d -= 2 * Math.PI;
                while (d < -Math.PI) d += 2 * Math.PI;
                return Math.abs(d);
            };

            // Separate into outer (further from center) and inner (closer)
            const dists = vertices.map(v => ({v, d: Math.sqrt((v.x-cx)**2 + (v.y-cy)**2), ang: angle(v)}));
            dists.sort((a,b) => b.d - a.d);
            const outer = dists.slice(0, 5);
            const inner = dists.slice(5, 10);

            // Sort each group by angle
            outer.sort((a,b) => a.ang - b.ang);
            inner.sort((a,b) => a.ang - b.ang);

            // Rotate to align vertices: outer left = 5, inner top = 0
            const rotateToAngle = (arr, targetAngle) => {
                let minIdx = 0, minDist = Infinity;
                for (let i = 0; i < arr.length; i++) {
                    const d = angleDist(arr[i].ang, targetAngle);
                    if (d < minDist) { minDist = d; minIdx = i; }
                }
                return [...arr.slice(minIdx), ...arr.slice(0, minIdx)];
            };

            const outerSorted = rotateToAngle(outer, Math.PI);  // Left = vertex 5
            const innerSorted = rotateToAngle(inner, -Math.PI/2);  // Top = vertex 0

            // Build VTX map: inner 0-4, outer 5-9
            const VTX = {};
            for (let i = 0; i < 5; i++) VTX[i] = innerSorted[i].v;
            for (let i = 0; i < 5; i++) VTX[5 + i] = outerSorted[i].v;

            // Edge list (must match strategy)
            const EDGES = [[0,2],[0,3],[0,6],[1,3],[1,4],[1,7],[2,4],[2,8],[3,9],[4,5],[5,6],[5,9],[6,7],[7,8],[8,9]];

            function nearest(x, y) {
                let best = -1, bestD = 1e9;
                for (const k in VTX) {
                    const dx = x - VTX[k].x, dy = y - VTX[k].y;
                    const d = dx*dx + dy*dy;
                    if (d < bestD) { bestD = d; best = +k; }
                }
                return best;
            }

            const state = new Array(15).fill('-');
            document.querySelectorAll('line').forEach(l => {
                const x1 = +l.getAttribute('x1'), y1 = +l.getAttribute('y1');
                const x2 = +l.getAttribute('x2'), y2 = +l.getAttribute('y2');
                const v1 = nearest(x1, y1), v2 = nearest(x2, y2);
                const e = EDGES.findIndex(p => p[0] === Math.min(v1,v2) && p[1] === Math.max(v1,v2));
                if (e < 0) return;
                const stroke = l.getAttribute('stroke') || '';
                if (/green|#10b981|16,\\s*185/i.test(stroke)) state[e] = 'G';
                else if (/purple|#a855f7|168,\\s*85/i.test(stroke)) state[e] = 'P';
                // Only treat as grey (available) if it matches known grey patterns
                // #9ca3af and other non-standard colors mean edge is not available
                else if (/grey|gray|#e5e7eb|229,\\s*231/i.test(stroke)) state[e] = '-';
                else state[e] = 'P';  // Unknown color = treat as colored (unavailable)
            });
            return state.join('');
        }
        """
        try:
            return self.page.evaluate(js_code)
        except Exception as e:
            self.logger.warning(f"read_board error: {e}")
            return "-" * 15

    def read_vertex_colors(self) -> str:
        """Read vertex colors from the game board circles.

        Returns a 10-character string where each character represents a vertex (0-9):
        - 'R' = red (player 1 owns this vertex)
        - 'B' = blue (player 2 owns this vertex)
        - '-' = neutral (no owner yet)

        Note: The game always initializes vertex 5 as red (player 1) and vertex 7 as blue (player 2).
        """
        js_code = """
        () => {
            // First build vertex map from line endpoints (same as read_board)
            const points = [];
            document.querySelectorAll('line').forEach(l => {
                points.push({x: +l.getAttribute('x1'), y: +l.getAttribute('y1')});
                points.push({x: +l.getAttribute('x2'), y: +l.getAttribute('y2')});
            });

            const vertices = [];
            for (const p of points) {
                let found = false;
                for (const v of vertices) {
                    const d = Math.sqrt((p.x - v.x)**2 + (p.y - v.y)**2);
                    if (d < 5) { found = true; break; }
                }
                if (!found) vertices.push({x: p.x, y: p.y});
            }

            if (vertices.length !== 10) return '-----R-B--';  // Default: V5=R, V7=B

            const cx = vertices.reduce((s,v) => s + v.x, 0) / 10;
            const cy = vertices.reduce((s,v) => s + v.y, 0) / 10;
            const angle = (v) => Math.atan2(v.y - cy, v.x - cx);

            const angleDist = (a1, a2) => {
                let d = a1 - a2;
                while (d > Math.PI) d -= 2 * Math.PI;
                while (d < -Math.PI) d += 2 * Math.PI;
                return Math.abs(d);
            };

            const dists = vertices.map(v => ({v, d: Math.sqrt((v.x-cx)**2 + (v.y-cy)**2), ang: angle(v)}));
            dists.sort((a,b) => b.d - a.d);
            const outer = dists.slice(0, 5);
            const inner = dists.slice(5, 10);

            outer.sort((a,b) => a.ang - b.ang);
            inner.sort((a,b) => a.ang - b.ang);

            const rotateToAngle = (arr, targetAngle) => {
                let minIdx = 0, minDist = Infinity;
                for (let i = 0; i < arr.length; i++) {
                    const d = angleDist(arr[i].ang, targetAngle);
                    if (d < minDist) { minDist = d; minIdx = i; }
                }
                return [...arr.slice(minIdx), ...arr.slice(0, minIdx)];
            };

            const outerSorted = rotateToAngle(outer, Math.PI);
            const innerSorted = rotateToAngle(inner, -Math.PI/2);

            const VTX = {};
            for (let i = 0; i < 5; i++) VTX[i] = innerSorted[i].v;
            for (let i = 0; i < 5; i++) VTX[5 + i] = outerSorted[i].v;

            // Initialize with defaults: V5=R (red/player1), V7=B (blue/player2)
            const vertexColors = ['-','-','-','-','-','R','-','B','-','-'];

            // Read circle colors and match to vertices
            document.querySelectorAll('circle').forEach(c => {
                const circleX = +c.getAttribute('cx');
                const circleY = +c.getAttribute('cy');

                // Find nearest vertex
                let bestV = -1, bestD = 1e9;
                for (const k in VTX) {
                    const dx = circleX - VTX[k].x, dy = circleY - VTX[k].y;
                    const d = dx*dx + dy*dy;
                    if (d < bestD) { bestD = d; bestV = +k; }
                }

                if (bestV >= 0 && bestD < 400) {  // Within 20px
                    const fill = (c.getAttribute('fill') || '').toLowerCase();
                    const style = (c.getAttribute('style') || '').toLowerCase();
                    const fillStr = fill + ' ' + style;

                    // Detect red variants (hex, rgb, named)
                    if (/red|#ef4444|#dc2626|#f87171|#b91c1c|#fee2e2|rgb\\s*\\(\\s*2[0-5]\\d|rgb\\s*\\(\\s*239/i.test(fillStr)) {
                        vertexColors[bestV] = 'R';
                    // Detect blue variants (hex, rgb, named)
                    } else if (/blue|#3b82f6|#2563eb|#60a5fa|#1d4ed8|#dbeafe|rgb\\s*\\(\\s*59|rgb\\s*\\(\\s*37/i.test(fillStr)) {
                        vertexColors[bestV] = 'B';
                    // Detect black/neutral (reset to neutral if explicitly black)
                    } else if (/black|#000|#1[0-9a-f]{5}|rgb\\s*\\(\\s*0/i.test(fillStr)) {
                        vertexColors[bestV] = '-';
                    }
                }
            });
            return vertexColors.join('');
        }
        """
        try:
            result = self.page.evaluate(js_code)
            self.logger.debug(f"Vertex colors read: {result}")
            return result if result else '-----R-B--'
        except Exception as e:
            self.logger.warning(f"read_vertex_colors error: {e}")
            return '-----R-B--'  # Default: V5=R, V7=B

    def debug_vertex_fills(self) -> list:
        """Debug: return raw fill colors from all circles on the game board."""
        js_code = """
        () => {
            const results = [];
            document.querySelectorAll('circle').forEach((c, i) => {
                results.push({
                    index: i,
                    cx: c.getAttribute('cx'),
                    cy: c.getAttribute('cy'),
                    r: c.getAttribute('r'),
                    fill: c.getAttribute('fill'),
                    stroke: c.getAttribute('stroke'),
                    style: c.getAttribute('style'),
                    className: c.getAttribute('class')
                });
            });
            return results;
        }
        """
        try:
            return self.page.evaluate(js_code)
        except Exception as e:
            self.logger.warning(f"debug_vertex_fills error: {e}")
            return []

    def debug_lines(self):
        """Debug: print all line coordinates and their dynamically discovered vertices."""
        js_code = """
        () => {
            // Collect all unique endpoints from SVG lines
            const points = [];
            document.querySelectorAll('line').forEach(l => {
                points.push({x: +l.getAttribute('x1'), y: +l.getAttribute('y1')});
                points.push({x: +l.getAttribute('x2'), y: +l.getAttribute('y2')});
            });

            // Cluster points to find unique vertices (within 5px tolerance)
            const vertices = [];
            for (const p of points) {
                let found = false;
                for (const v of vertices) {
                    const d = Math.sqrt((p.x - v.x)**2 + (p.y - v.y)**2);
                    if (d < 5) { found = true; break; }
                }
                if (!found) vertices.push({x: p.x, y: p.y});
            }

            if (vertices.length !== 10) {
                return [{type: 'error', msg: 'Expected 10 vertices, found ' + vertices.length}];
            }

            const cx = vertices.reduce((s,v) => s + v.x, 0) / 10;
            const cy = vertices.reduce((s,v) => s + v.y, 0) / 10;
            const angle = (v) => Math.atan2(v.y - cy, v.x - cx);

            // Angular distance that handles wrap-around
            const angleDist = (a1, a2) => {
                let d = a1 - a2;
                while (d > Math.PI) d -= 2 * Math.PI;
                while (d < -Math.PI) d += 2 * Math.PI;
                return Math.abs(d);
            };

            // Separate into outer (further from center) and inner (closer)
            const dists = vertices.map(v => ({v, d: Math.sqrt((v.x-cx)**2 + (v.y-cy)**2), ang: angle(v)}));
            dists.sort((a,b) => b.d - a.d);
            const outer = dists.slice(0, 5);
            const inner = dists.slice(5, 10);

            // Sort each group by angle
            outer.sort((a,b) => a.ang - b.ang);
            inner.sort((a,b) => a.ang - b.ang);

            // Rotate outer so vertex closest to angle=PI (left side) is first -> becomes index 5
            const rotateToAngle = (arr, targetAngle) => {
                let minIdx = 0, minDist = Infinity;
                for (let i = 0; i < arr.length; i++) {
                    const d = angleDist(arr[i].ang, targetAngle);
                    if (d < minDist) { minDist = d; minIdx = i; }
                }
                return [...arr.slice(minIdx), ...arr.slice(0, minIdx)];
            };

            const outerSorted = rotateToAngle(outer, Math.PI);  // Left = vertex 5
            const innerSorted = rotateToAngle(inner, -Math.PI/2);  // Top = vertex 0

            // Build VTX map: inner 0-4, outer 5-9
            const VTX = {};
            for (let i = 0; i < 5; i++) VTX[i] = innerSorted[i].v;
            for (let i = 0; i < 5; i++) VTX[5 + i] = outerSorted[i].v;

            const EDGES = [[0,2],[0,3],[0,6],[1,3],[1,4],[1,7],[2,4],[2,8],[3,9],[4,5],[5,6],[5,9],[6,7],[7,8],[8,9]];

            function nearest(x, y) {
                let best = -1, bestD = 1e9;
                for (const k in VTX) {
                    const dx = x - VTX[k].x, dy = y - VTX[k].y;
                    const d = dx*dx + dy*dy;
                    if (d < bestD) { bestD = d; best = +k; }
                }
                return {v: best, d: Math.sqrt(bestD)};
            }

            const results = [];

            // Output center
            results.push({type: 'center', cx: cx.toFixed(1), cy: cy.toFixed(1)});

            // Output discovered vertices with their angles and distances
            results.push({type: 'vertices', count: 10,
                inner: innerSorted.map((v,i) => ({id: i, x: v.v.x.toFixed(0), y: v.v.y.toFixed(0), ang: (v.ang * 180/Math.PI).toFixed(1), dist: v.d.toFixed(0)})),
                outer: outerSorted.map((v,i) => ({id: 5+i, x: v.v.x.toFixed(0), y: v.v.y.toFixed(0), ang: (v.ang * 180/Math.PI).toFixed(1), dist: v.d.toFixed(0)}))
            });

            // Output edge mappings
            document.querySelectorAll('line').forEach((l, i) => {
                const x1 = +l.getAttribute('x1'), y1 = +l.getAttribute('y1');
                const x2 = +l.getAttribute('x2'), y2 = +l.getAttribute('y2');
                const n1 = nearest(x1, y1), n2 = nearest(x2, y2);
                const stroke = l.getAttribute('stroke') || '';
                const edge_v1 = Math.min(n1.v, n2.v), edge_v2 = Math.max(n1.v, n2.v);
                const edge_idx = EDGES.findIndex(p => p[0] === edge_v1 && p[1] === edge_v2);
                results.push({type: 'line', i, x1, y1, x2, y2, v1: n1.v, d1: n1.d.toFixed(0), v2: n2.v, d2: n2.d.toFixed(0), edge_idx, stroke: stroke.slice(0, 20)});
            });
            return results;
        }
        """
        try:
            results = self.page.evaluate(js_code)
            for r in results:
                if r.get('type') == 'error':
                    self.logger.error(f"Vertex discovery error: {r['msg']}")
                elif r.get('type') == 'center':
                    self.logger.info(f"Graph center: ({r['cx']}, {r['cy']})")
                elif r.get('type') == 'vertices':
                    self.logger.info(f"Inner vertices (0-4):")
                    for v in r['inner']:
                        self.logger.info(f"  V{v['id']}: ({v['x']}, {v['y']}) ang={v['ang']}° dist={v['dist']}")
                    self.logger.info(f"Outer vertices (5-9):")
                    for v in r['outer']:
                        self.logger.info(f"  V{v['id']}: ({v['x']}, {v['y']}) ang={v['ang']}° dist={v['dist']}")
                elif r.get('type') == 'line':
                    edge_str = f"E{r['edge_idx']}" if r['edge_idx'] >= 0 else "UNMAPPED"
                    color_char = 'G' if 'green' in r['stroke'].lower() or '#10b981' in r['stroke'].lower() else ('P' if 'purple' in r['stroke'].lower() or '#a855f7' in r['stroke'].lower() else '-')
                    self.logger.info(f"  Line {r['i']}: ({r['x1']},{r['y1']})->({r['x2']},{r['y2']}) = V{r['v1']}-V{r['v2']} -> {edge_str} [{color_char}]")
        except Exception as e:
            self.logger.error(f"debug_lines error: {e}")

    def debug_edge_strokes(self) -> dict:
        """Return raw stroke values for all edges - useful for debugging color detection."""
        js_code = """
        () => {
            const points = [];
            document.querySelectorAll('line').forEach(l => {
                points.push({x: +l.getAttribute('x1'), y: +l.getAttribute('y1')});
                points.push({x: +l.getAttribute('x2'), y: +l.getAttribute('y2')});
            });
            const vertices = [];
            for (const p of points) {
                let found = false;
                for (const v of vertices) {
                    if (Math.sqrt((p.x - v.x)**2 + (p.y - v.y)**2) < 5) { found = true; break; }
                }
                if (!found) vertices.push({x: p.x, y: p.y});
            }
            if (vertices.length !== 10) return {error: 'vertex_count', count: vertices.length};

            const cx = vertices.reduce((s,v) => s + v.x, 0) / 10;
            const cy = vertices.reduce((s,v) => s + v.y, 0) / 10;
            const angle = (v) => Math.atan2(v.y - cy, v.x - cx);
            const angleDist = (a1, a2) => {
                let d = a1 - a2;
                while (d > Math.PI) d -= 2 * Math.PI;
                while (d < -Math.PI) d += 2 * Math.PI;
                return Math.abs(d);
            };
            const dists = vertices.map(v => ({v, d: Math.sqrt((v.x-cx)**2 + (v.y-cy)**2), ang: angle(v)}));
            dists.sort((a,b) => b.d - a.d);
            const outer = dists.slice(0, 5);
            const inner = dists.slice(5, 10);
            outer.sort((a,b) => a.ang - b.ang);
            inner.sort((a,b) => a.ang - b.ang);
            const rotateToAngle = (arr, targetAngle) => {
                let minIdx = 0, minDist = Infinity;
                for (let i = 0; i < arr.length; i++) {
                    const d = angleDist(arr[i].ang, targetAngle);
                    if (d < minDist) { minDist = d; minIdx = i; }
                }
                return [...arr.slice(minIdx), ...arr.slice(0, minIdx)];
            };
            const outerSorted = rotateToAngle(outer, Math.PI);
            const innerSorted = rotateToAngle(inner, -Math.PI/2);
            const VTX = {};
            for (let i = 0; i < 5; i++) VTX[i] = innerSorted[i].v;
            for (let i = 0; i < 5; i++) VTX[5 + i] = outerSorted[i].v;
            const EDGES = [[0,2],[0,3],[0,6],[1,3],[1,4],[1,7],[2,4],[2,8],[3,9],[4,5],[5,6],[5,9],[6,7],[7,8],[8,9]];
            function nearest(x, y) {
                let best = -1, bestD = 1e9;
                for (const k in VTX) {
                    const dx = x - VTX[k].x, dy = y - VTX[k].y;
                    const d = dx*dx + dy*dy;
                    if (d < bestD) { bestD = d; best = +k; }
                }
                return best;
            }
            const result = {};
            document.querySelectorAll('line').forEach(l => {
                const x1 = +l.getAttribute('x1'), y1 = +l.getAttribute('y1');
                const x2 = +l.getAttribute('x2'), y2 = +l.getAttribute('y2');
                const v1 = nearest(x1, y1), v2 = nearest(x2, y2);
                const e = EDGES.findIndex(p => p[0] === Math.min(v1,v2) && p[1] === Math.max(v1,v2));
                if (e >= 0) {
                    result['E' + e] = l.getAttribute('stroke') || 'none';
                }
            });
            return result;
        }
        """
        try:
            return self.page.evaluate(js_code)
        except Exception as e:
            self.logger.warning(f"debug_edge_strokes error: {e}")
            return {}

    def execute_move(self, edge: int, color: str) -> bool:
        """Execute a move using dynamic vertex discovery and MouseEvent dispatch."""
        color_name = "Green" if color == 'G' else "Purple"
        v1, v2 = EDGES[edge]

        # JavaScript that dynamically discovers vertices then clicks the edge
        js_code = f"""
        () => {{
            // Collect all unique endpoints from SVG lines
            const points = [];
            document.querySelectorAll('line').forEach(l => {{
                points.push({{x: +l.getAttribute('x1'), y: +l.getAttribute('y1')}});
                points.push({{x: +l.getAttribute('x2'), y: +l.getAttribute('y2')}});
            }});

            // Cluster points to find unique vertices (within 5px tolerance)
            const vertices = [];
            for (const p of points) {{
                let found = false;
                for (const v of vertices) {{
                    const d = Math.sqrt((p.x - v.x)**2 + (p.y - v.y)**2);
                    if (d < 5) {{ found = true; break; }}
                }}
                if (!found) vertices.push({{x: p.x, y: p.y}});
            }}

            if (vertices.length !== 10) return 'vertex_error:' + vertices.length;

            const cx = vertices.reduce((s,v) => s + v.x, 0) / 10;
            const cy = vertices.reduce((s,v) => s + v.y, 0) / 10;
            const angle = (v) => Math.atan2(v.y - cy, v.x - cx);

            // Angular distance that handles wrap-around
            const angleDist = (a1, a2) => {{
                let d = a1 - a2;
                while (d > Math.PI) d -= 2 * Math.PI;
                while (d < -Math.PI) d += 2 * Math.PI;
                return Math.abs(d);
            }};

            // Separate into outer (further from center) and inner (closer)
            const dists = vertices.map(v => ({{v, d: Math.sqrt((v.x-cx)**2 + (v.y-cy)**2), ang: angle(v)}}));
            dists.sort((a,b) => b.d - a.d);
            const outer = dists.slice(0, 5);
            const inner = dists.slice(5, 10);

            // Sort each group by angle
            outer.sort((a,b) => a.ang - b.ang);
            inner.sort((a,b) => a.ang - b.ang);

            // Rotate to align vertices: outer left = 5, inner top = 0
            const rotateToAngle = (arr, targetAngle) => {{
                let minIdx = 0, minDist = Infinity;
                for (let i = 0; i < arr.length; i++) {{
                    const d = angleDist(arr[i].ang, targetAngle);
                    if (d < minDist) {{ minDist = d; minIdx = i; }}
                }}
                return [...arr.slice(minIdx), ...arr.slice(0, minIdx)];
            }};

            const outerSorted = rotateToAngle(outer, Math.PI);  // Left = vertex 5
            const innerSorted = rotateToAngle(inner, -Math.PI/2);  // Top = vertex 0

            // Build VTX map: inner 0-4, outer 5-9
            const VTX = {{}};
            for (let i = 0; i < 5; i++) VTX[i] = innerSorted[i].v;
            for (let i = 0; i < 5; i++) VTX[5 + i] = outerSorted[i].v;

            function nearest(x, y) {{
                let best = -1, bestD = 1e9;
                for (const k in VTX) {{
                    const dx = x - VTX[k].x, dy = y - VTX[k].y;
                    const d = dx*dx + dy*dy;
                    if (d < bestD) {{ bestD = d; best = +k; }}
                }}
                return best;
            }}

            // Find the line for edge {v1}-{v2}
            const line = [...document.querySelectorAll('line')].find(l => {{
                const x1 = +l.getAttribute('x1'), y1 = +l.getAttribute('y1');
                const x2 = +l.getAttribute('x2'), y2 = +l.getAttribute('y2');
                const a = nearest(x1, y1), b = nearest(x2, y2);
                return (a === {v1} && b === {v2}) || (a === {v2} && b === {v1});
            }});

            if (!line) {{
                // Debug: report what vertices we found for each line
                const debug = [...document.querySelectorAll('line')].map(l => {{
                    const x1 = +l.getAttribute('x1'), y1 = +l.getAttribute('y1');
                    const x2 = +l.getAttribute('x2'), y2 = +l.getAttribute('y2');
                    return nearest(x1, y1) + '-' + nearest(x2, y2);
                }}).join(',');
                return 'no_line:' + debug;
            }}

            // Check if grey (available)
            const stroke = line.getAttribute('stroke') || '';
            if (!/grey|gray|#e5e7eb|229,\\s*231/i.test(stroke)) return 'not_grey:' + stroke;

            // Click the line using synthetic event (works with React)
            const r = line.getBoundingClientRect();
            line.dispatchEvent(new MouseEvent('click', {{
                bubbles: true,
                cancelable: true,
                view: window,
                clientX: r.left + r.width / 2,
                clientY: r.top + r.height / 2
            }}));

            return 'clicked';
        }}
        """

        try:
            result = self.page.evaluate(js_code)
            if result != 'clicked':
                self.logger.warning(f"Edge click failed: {result}")
                return False
        except Exception as e:
            self.logger.warning(f"Edge click error: {e}")
            return False

        time.sleep(0.5)  # Give dialog time to appear

        # Click color button with retry (dialog may take time to appear)
        # Try multiple text patterns for the color button
        color_patterns = ['Green', 'green', 'FM', 'Ferromagnetic'] if color == 'G' else ['Purple', 'purple', 'AFM', 'Antiferromagnetic']

        js_color = f"""
        () => {{
            const patterns = {color_patterns};
            const allBtns = [...document.querySelectorAll('button')];
            const btn = allBtns.find(b => {{
                const txt = b.textContent || '';
                return patterns.some(p => txt.includes(p));
            }});
            if (btn) {{ btn.click(); return 'clicked'; }}
            // Return debug info about available buttons
            const btnTexts = allBtns.map(b => (b.textContent || '').slice(0, 30)).join('|');
            return 'no_button:' + btnTexts;
        }}
        """

        # Retry up to 10 times with longer wait
        for attempt in range(10):
            try:
                result = self.page.evaluate(js_color)
                if result == 'clicked':
                    time.sleep(0.3)
                    return True
                if attempt == 0 and result.startswith('no_button:'):
                    btns = result[10:]
                    if btns:
                        self.logger.debug(f"Available buttons: {btns[:100]}")
            except Exception as e:
                self.logger.warning(f"Color button error: {e}")
            time.sleep(0.3)

        self.logger.warning(f"Color button failed after retries")
        return False

    def wait_for_turn(self, timeout: float = 15.0) -> bool:
        """Wait for our turn or game over."""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_game_over():
                return False
            if self.is_our_turn():
                return True
            time.sleep(0.3)
        return True  # Timeout - try anyway

    def wait_for_opponent_to_play(self, timeout: float = 30.0, stuck_timeout: float = 180.0, initial_grey_count: int = None) -> bool:
        """Wait for opponent to take their turn.

        Uses two signals for reliability:
        1. Turn indicator: wait for it to switch to opponent, then back to us
        2. Board state: wait for grey count to decrease (opponent colored an edge)

        Args:
            timeout: Normal timeout for slow opponent thinking (warns but continues)
            stuck_timeout: Hard timeout - raises StuckOpponentError if exceeded (default 3 min)
            initial_grey_count: Grey edges before opponent's turn (for state-based detection)

        Returns False if game is over, True when opponent has played.
        Raises StuckOpponentError if stuck_timeout exceeded.
        """
        start = time.time()

        # Phase 1: Wait for turn to switch to opponent (max 5 seconds)
        phase1_timeout = min(5.0, timeout / 4)
        while time.time() - start < phase1_timeout:
            if self.is_game_over():
                return False
            if not self.is_our_turn():
                # Turn switched to opponent, now wait for them to play
                self.logger.debug("Phase 1: Turn switched to opponent")
                break
            time.sleep(0.1)

        warned = False
        # Phase 2: Wait for opponent to finish (turn back to us OR grey count changed)
        while time.time() - start < stuck_timeout:
            if self.is_game_over():
                return False

            # Primary signal: turn indicator
            if self.is_our_turn():
                self.logger.debug("Phase 2: Turn returned to us")
                return True

            # Secondary signal: board state changed (grey count decreased)
            if initial_grey_count is not None:
                current_state = self.read_board()
                current_grey = current_state.count('-')
                if current_grey < initial_grey_count:
                    self.logger.debug(f"Phase 2: Board changed (grey {initial_grey_count}->{current_grey})")
                    # Wait a moment for turn indicator to catch up
                    time.sleep(0.5)
                    return True

            # Warn once after normal timeout
            elapsed = time.time() - start
            if not warned and elapsed > timeout:
                self.logger.warning(f"Opponent slow after {timeout}s, waiting up to {stuck_timeout}s...")
                warned = True

            time.sleep(0.3)

        # Stuck timeout exceeded - opponent is likely crashed/stuck
        elapsed = time.time() - start
        self.logger.error(f"Opponent stuck! No response after {elapsed:.0f}s")
        raise StuckOpponentError(f"Opponent did not play within {stuck_timeout}s")

    def _get_dashboard_stats(self) -> dict:
        """Get all stats needed for dashboard display.

        Returns a dict with all fields needed by publish_state, computed from
        the current run stats. This ensures every publish sends complete data.
        """
        run_id = getattr(self, '_run_id', None)
        current_game = getattr(self, '_current_game_number', 1)
        stats = get_run_stats(run_id) if run_id else get_session_stats()

        # Compute recent_5 and score_trend from games list
        recent_5 = ''
        score_trend = None
        if stats and stats.games:
            completed = [g for g in stats.games if g.result is not None]
            if completed:
                recent = completed[-5:] if len(completed) >= 5 else completed
                recent_5 = ''.join(
                    'W' if g.result == 'win' else 'L' if g.result == 'loss' else 'D'
                    for g in recent
                )
                # Score trend: compare first half to second half averages
                if len(completed) >= 4:
                    scores = [g.final_score for g in completed if g.final_score is not None]
                    if len(scores) >= 4:
                        first_half = scores[:len(scores)//2]
                        second_half = scores[len(scores)//2:]
                        score_trend = sum(second_half)/len(second_half) - sum(first_half)/len(first_half)

        return {
            'current_game': current_game,
            'completed_games': stats.completed_games if stats else 0,
            'strategy': getattr(self, 'strategy_type', None),
            'opponent': getattr(self, 'opponent', None),
            'wins': stats.wins if stats else 0,
            'draws': stats.draws if stats else 0,
            'losses': stats.losses if stats else 0,
            'avg_score': stats.avg_score if stats else None,
            'median_score': stats.median_score if stats else None,
            'min_score': stats.min_score if stats else None,
            'max_score': stats.max_score if stats else None,
            'std_score': stats.score_std if stats else None,
            'recent_5': recent_5,
            'score_trend': score_trend,
            'avg_entropy': stats.avg_entropy if stats else None,
            'avg_top3_hit': stats.avg_top3_hit if stats else None,
            'avg_pred_accuracy': stats.avg_prediction_accuracy if stats else None,
        }

    def play_game(self, opponent: str = "melissa") -> dict:
        """Play one complete game."""
        self.score_history = []
        self.opponent = opponent  # Store for dashboard display

        # Initialize opponent model for online learning
        self._ensure_opponent_model(opponent)

        # Initialize metrics tracker for opponent model visibility
        self.metrics_tracker = GameMetricsTracker(self.opponent_model)
        self.metrics_tracker.record_snapshot()  # Capture model state at game start

        # Initialize MATLAB strategy with opponent model
        if hasattr(self.strategy, 'initialize'):
            self.strategy.initialize(opponent=opponent)

        # Start stats tracking for this game (with model metrics and run info)
        run_id = getattr(self, '_run_id', None)
        game_number = getattr(self, '_current_game_number', None)
        self.current_game_id = self.stats_collector.start_game(
            opponent=opponent,
            graph="petersen",
            strategy=self.strategy_type,
            mcts_time=self.mcts_time,
            model_metrics=self.metrics_tracker.get_game_metrics(),
            run_id=run_id,
            game_number=game_number
        )

        if not self.start_game(opponent):
            return {"result": None, "error": "Could not start game"}

        # Debug: show line coordinates
        self.debug_lines()

        # Send pre-game initialization to dashboard (all black except V5=red, V7=blue)
        publisher = get_publisher()
        if publisher.is_configured():
            try:
                initial_board = self.read_board()
                initial_vertices = '-----R-B--'  # V5=red (player 1), V7=blue (player 2)
                self.logger.info(f"DEBUG pre-game init: board={initial_board}, vertices={initial_vertices}")
                # Get all dashboard stats
                ds = self._get_dashboard_stats()
                publisher.publish_state(
                    board_state=initial_board,
                    vertex_state=initial_vertices,
                    **ds,
                )
            except Exception as e:
                self.logger.debug(f"Pre-game init publish failed: {e}")

        # Wait for initial turn (AlphaQ Up can be slow to present)
        initial_turn_timeout = 60.0
        initial_turn_start = time.time()
        while not self.is_our_turn() and not self.is_game_over():
            if time.time() - initial_turn_start > initial_turn_timeout:
                self.logger.warning(
                    f"Initial turn not detected after {initial_turn_timeout:.0f}s — proceeding anyway"
                )
                break
            time.sleep(0.3)

        move_count = 0
        loop_iterations = 0
        max_iterations = 30  # Safety limit: Petersen has 15 edges, ~8 moves max per player
        prev_score = self.read_score()
        failed_edges = set()  # Track edges that failed to click
        max_retries = 3  # Max retries per move attempt

        while not self.is_game_over():
            if not self.is_our_turn():
                time.sleep(0.3)
                continue

            # Wait for board state to stabilize after turn change
            # The turn indicator text updates faster than the SVG board state
            # Read board twice with a gap - only proceed when state is stable
            time.sleep(0.3)
            state_check1 = self.read_board()
            time.sleep(0.4)
            state_check2 = self.read_board()

            if state_check1 != state_check2:
                # Board still changing, wait more and re-read
                self.logger.debug(f"Board state unstable, waiting for sync")
                time.sleep(0.5)
                state_check2 = self.read_board()

            # Only count iterations where it's actually our turn (move attempts)
            loop_iterations += 1
            if loop_iterations > max_iterations:
                self.logger.error(f"Safety limit reached: {loop_iterations} move attempts (likely infinite loop)")
                break

            state = state_check2  # Use the stable state
            score = self.read_score()

            # Check if all edges played
            if state.count('-') == 0:
                self.logger.info("All edges played")
                break

            # Get available edges (excluding failed ones)
            available = [i for i, c in enumerate(state) if c == '-' and i not in failed_edges]
            if not available:
                # If all edges failed, clear failed set and try again
                if failed_edges:
                    self.logger.warning("All edges failed, clearing failed list")
                    failed_edges.clear()
                    available = [i for i, c in enumerate(state) if c == '-']
                if not available:
                    break

            # Calculate move (track our thinking time)
            our_start_time = time.time()
            result = self.strategy.calculate_move(state, score, self.score_history)
            our_think_time = time.time() - our_start_time

            if result is None:
                break

            # Handle both (edge, color) and (edge, color, solver_stats) returns
            solver_stats = {}
            if len(result) == 3:
                edge, color, solver_stats = result
            else:
                edge, color = result

            # Safety check: verify strategy returned a valid edge
            # (Turn-based game means opponent can't play during our calculation,
            # but strategy might return invalid edge due to bug or stale internal state)
            current_state = self.read_board()
            state = current_state  # Update state for next iteration
            available = [i for i, c in enumerate(current_state) if c == '-' and i not in failed_edges]

            # Verify edge is actually available
            if current_state[edge] != '-' or edge in failed_edges:
                if available:
                    self.logger.warning(f"E{edge} not available - strategy returned invalid edge")
                    # Re-read board to ensure we have latest state
                    time.sleep(0.5)  # Brief wait for DOM to stabilize
                    fresh_state = self.read_board()
                    fresh_available = [i for i, c in enumerate(fresh_state) if c == '-' and i not in failed_edges]
                    self.logger.info(f"Fresh state: {sum(1 for c in fresh_state if c == '-')} grey edges")
                    # Recalculate move with updated state instead of picking blindly
                    recalc_result = self.strategy.calculate_move(fresh_state, score, self.score_history)
                    if recalc_result is not None:
                        if len(recalc_result) == 3:
                            edge, color, solver_stats = recalc_result
                        else:
                            edge, color = recalc_result
                        # Verify recalculated edge is valid against fresh state
                        if fresh_state[edge] != '-' or edge in failed_edges:
                            # Strategy returned invalid edge, pick first available
                            self.logger.warning(f"Recalculated E{edge} also unavailable, using fallback")
                            edge = fresh_available[0] if fresh_available else available[0]
                            # Use heuristic for color: Green on our edges, Purple on opponent's
                            color = 'G' if edge in [9, 10, 11] else ('P' if edge in [5, 12, 13] else 'G')
                    else:
                        edge = fresh_available[0] if fresh_available else available[0]
                        color = 'G' if edge in [9, 10, 11] else ('P' if edge in [5, 12, 13] else 'G')
                else:
                    self.logger.warning("No available edges after rechecking")
                    break

            self.logger.info(f"Move: E{edge} {'Green' if color == 'G' else 'Purple'}")

            # Try to execute with retries
            success = False
            for attempt in range(max_retries):
                state_before = self.read_board()
                if self.execute_move(edge, color):
                    # Verify the board state actually changed
                    time.sleep(0.3)
                    state_after = self.read_board()
                    if state_after[edge] != '-':
                        success = True
                        break
                    else:
                        # Click appeared to succeed but board didn't change
                        self.logger.warning(f"Move E{edge} click succeeded but board unchanged (attempt {attempt+1})")
                        # Debug: show raw stroke values
                        strokes = self.debug_edge_strokes()
                        self.logger.debug(f"Raw strokes: E{edge}={strokes.get(f'E{edge}', 'N/A')}")
                        self.logger.debug(f"State before: {state_before}, State after: {state_after}")
                else:
                    self.logger.warning(f"Move E{edge} attempt {attempt+1}/{max_retries} failed")
                time.sleep(0.5)

            if success:
                move_count += 1
                failed_edges.discard(edge)  # Remove from failed if it worked
                time.sleep(0.5)
                new_score = self.read_score()
                self.score_history.append((edge, color, new_score))
                self.logger.info(f"Move {move_count}: E{edge} {color} -> Score: {new_score:.4f}")

                # Add timing to solver stats
                solver_stats['wall_clock_time'] = our_think_time

                # Record move to stats database with solver statistics
                current_board = self.read_board()
                self.stats_collector.record_move(
                    game_id=self.current_game_id,
                    move_number=move_count,
                    player="us",
                    edge=edge,
                    color=color,
                    score_after=new_score,
                    score_before=prev_score,
                    state_after=current_board,
                    thinking_time=our_think_time,
                    solver_stats=solver_stats
                )
                prev_score = new_score

                # Publish move to live dashboard
                publisher = get_publisher()
                if publisher.is_configured():
                    try:
                        # Debug: log raw circle fills
                        fills = self.debug_vertex_fills()
                        self.logger.info(f"DEBUG vertex fills: {[(f.get('cx','?'), f.get('cy','?'), f.get('fill','?')) for f in fills[:12]]}")

                        vertex_colors = self.read_vertex_colors()
                        edges_colored = sum(1 for c in current_board if c in 'GP')
                        self.logger.info(f"DEBUG publishing: edges={edges_colored}, board={current_board}, vertices={vertex_colors}")

                        # Get all dashboard stats
                        ds = self._get_dashboard_stats()
                        publisher.publish_state(
                            move_edge=edge,
                            move_color=color,
                            move_score=new_score,
                            move_thinking_time=our_think_time,
                            move_player="us",
                            board_state=current_board,
                            vertex_state=vertex_colors,
                            **ds,
                        )
                    except Exception as e:
                        self.logger.debug(f"Move publish failed: {e}")

                # Record move for learning (if strategy supports it)
                if hasattr(self.strategy, 'record_move'):
                    self.strategy.record_move(edge, color, new_score)
            else:
                self.logger.error(f"Move E{edge} failed after {max_retries} attempts, marking as failed")
                failed_edges.add(edge)
                continue  # Try another edge without waiting for opponent

            # Save board state after our move (for opponent detection)
            our_post_move_state = self.read_board()
            our_grey_count = our_post_move_state.count('-')

            # Wait for opponent to play (using both turn indicator and board state)
            opponent_start_time = time.time()
            try:
                if not self.wait_for_opponent_to_play(timeout=30.0, stuck_timeout=240.0, initial_grey_count=our_grey_count):
                    break
            except StuckOpponentError:
                # Opponent is stuck - abandon this game and signal restart needed
                self.logger.error("Opponent stuck! Abandoning game and restarting...")
                self.stats_collector.abandon_game(self.current_game_id, "stuck_opponent")
                return {
                    "result": "abandoned",
                    "error": "stuck_opponent",
                    "final_score": self.read_score(),
                    "moves": move_count,
                    "score_history": self.score_history.copy(),
                }
            opponent_think_time = time.time() - opponent_start_time

            # Record opponent's move timing (we detect their move by board state change)
            state_after_opponent = self.read_board()
            opponent_grey_count = state_after_opponent.count('-')
            opponent_score = self.read_score()

            # Find which edge opponent played (compare OUR post-move state with post-opponent state)
            opponent_edge = None
            opponent_color = None
            for i in range(15):
                if our_post_move_state[i] == '-' and state_after_opponent[i] != '-':
                    opponent_edge = i
                    opponent_color = state_after_opponent[i]
                    break

            # Debug: Log opponent detection details
            if opponent_edge is not None:
                self.logger.debug(f"Opponent detected: E{opponent_edge}{opponent_color} "
                                  f"(grey: {our_grey_count}->{opponent_grey_count})")
                self.stats_collector.record_move(
                    game_id=self.current_game_id,
                    move_number=move_count,
                    player="opponent",
                    edge=opponent_edge,
                    color=opponent_color,
                    score_after=opponent_score,
                    score_before=new_score,
                    state_after=state_after_opponent,
                    thinking_time=opponent_think_time,
                    solver_stats={'opponent_think_time': opponent_think_time}
                )

                # Record to opponent_history table for pattern analysis
                our_prev_edge = None
                our_prev_color = None
                if self.score_history:
                    last_move = self.score_history[-1]
                    our_prev_edge = last_move[0]
                    our_prev_color = last_move[1]

                self.stats_collector.record_opponent_move(
                    opponent_name=self.opponent,
                    game_id=self.current_game_id,
                    move_number=move_count,
                    board_state_before=our_post_move_state,
                    edge=opponent_edge,
                    color=opponent_color,
                    score_before=new_score,
                    score_after=opponent_score,
                    our_prev_edge=our_prev_edge,
                    our_prev_color=our_prev_color
                )

                # Publish opponent move to live dashboard
                publisher = get_publisher()
                if publisher.is_configured():
                    try:
                        # Debug: log raw circle fills
                        fills = self.debug_vertex_fills()
                        self.logger.info(f"DEBUG opp vertex fills: {[(f.get('cx','?'), f.get('cy','?'), f.get('fill','?')) for f in fills[:12]]}")

                        vertex_colors = self.read_vertex_colors()
                        edges_colored = sum(1 for c in state_after_opponent if c in 'GP')
                        self.logger.info(f"DEBUG opp publishing: edges={edges_colored}, board={state_after_opponent}, vertices={vertex_colors}")

                        # Get all dashboard stats
                        ds = self._get_dashboard_stats()
                        publisher.publish_state(
                            move_edge=opponent_edge,
                            move_color=opponent_color,
                            move_score=opponent_score,
                            move_thinking_time=opponent_think_time,
                            move_player="opponent",
                            board_state=state_after_opponent,
                            vertex_state=vertex_colors,
                            **ds,
                        )
                    except Exception as e:
                        self.logger.debug(f"Opponent move publish failed: {e}")
                # Record prediction accuracy for opponent model learning visibility
                if self.metrics_tracker and self.score_history:
                    last_edge, last_color, _ = self.score_history[-1]
                    self.metrics_tracker.record_prediction(
                        our_move=(last_edge, last_color),
                        actual_opp_move=(opponent_edge, opponent_color),
                        grey_count=our_grey_count
                    )
            elif our_grey_count != opponent_grey_count:
                # Board changed but we didn't detect which edge - log warning
                self.logger.warning(f"Opponent move missed! Grey count changed {our_grey_count}->{opponent_grey_count}")
                self.logger.warning(f"  Our state:  {our_post_move_state}")
                self.logger.warning(f"  Opp state:  {state_after_opponent}")
            else:
                # No change detected - might be game over or timing issue
                self.logger.debug(f"No opponent move detected (grey count unchanged: {our_grey_count})")

        # Game over - wait for result to display
        time.sleep(2)
        final_score = self.read_score()
        result = self.get_outcome()
        terminal_state = self.read_board()

        # Extract opening info from strategy if available
        opening_edge = None
        opening_color = None
        opening_mode = None
        mcts_iters_setting = None
        if hasattr(self.strategy, 'current_game_opening') and self.strategy.current_game_opening:
            opening_edge, opening_color = self.strategy.current_game_opening
        if hasattr(self.strategy, 'opening_mode'):
            opening_mode = self.strategy.opening_mode
        if hasattr(self.strategy, 'mcts_iterations'):
            mcts_iters_setting = self.strategy.mcts_iterations
        elif hasattr(self.strategy, 'solver') and hasattr(self.strategy.solver, 'mcts_iterations'):
            mcts_iters_setting = self.strategy.solver.mcts_iterations

        # Record game end to stats database (with prediction metrics and opening info)
        self.stats_collector.end_game(
            game_id=self.current_game_id,
            result=result,
            final_score=final_score,
            model_metrics=self.metrics_tracker.get_game_metrics() if self.metrics_tracker else None,
            opening_edge=opening_edge,
            opening_color=opening_color,
            opening_mode=opening_mode,
            mcts_iterations_setting=mcts_iters_setting,
        )

        # Online learning: update opponent model and re-export
        self._update_opponent_model()

        # Calibration: compare our terminal evaluation to website score
        if terminal_state.count('-') == 0:  # All edges colored
            try:
                predicted_score = evaluate_terminal_state(terminal_state)
                self.stats_collector.record_calibration(
                    game_id=self.current_game_id,
                    terminal_state=terminal_state,
                    website_score=final_score,
                    predicted_score=predicted_score
                )
                self.logger.info(f"Calibration: predicted={predicted_score:+.4f}, actual={final_score:+.4f}, error={predicted_score - final_score:+.4f}")
            except Exception as e:
                self.logger.warning(f"Calibration error: {e}")

        # Log detailed game summary
        self.logger.info(f"=" * 40)
        self.logger.info(f"GAME OVER: {result.upper()}")
        self.logger.info(f"Final Score: {final_score:.4f}")
        self.logger.info(f"Total Moves: {move_count}")
        self.logger.info(f"Score History:")
        for i, (edge, color, score) in enumerate(self.score_history):
            color_name = "Green" if color == 'G' else "Purple"
            self.logger.info(f"  Move {i+1}: E{edge} {color_name} -> {score:.4f}")
        self.logger.info(f"=" * 40)

        # Update strategy learning (for any strategy that supports it)
        if self.strategy_type == "heuristic":
            self.strategy.update_from_trial(result, self.score_history, final_score)
            self.strategy.save_params(self.params_path)
        elif hasattr(self.strategy, 'end_game'):
            # HybridStrategy and others with learning support
            self.strategy.end_game(result, final_score)

        # Log MCTS stats if applicable
        if hasattr(self.strategy, 'get_stats'):
            stats = self.strategy.get_stats()
            if 'iterations' in stats and 'time' in stats:
                self.logger.info(f"MCTS Stats: {stats['iterations']} iterations in {stats['time']:.2f}s")
            elif 'games' in stats:
                self.logger.info(f"Strategy Stats: {stats['games']} games, {stats.get('wins', 0)}W/{stats.get('losses', 0)}L/{stats.get('draws', 0)}D")
            # Log game stats if available
            if 'game_stats' in stats:
                gs = stats['game_stats']
                self.logger.info(f"Session Stats: {gs['wins']}W / {gs['losses']}L / {gs['draws']}D")

        # Publish game result to dashboard immediately (don't wait for next game)
        publisher = get_publisher()
        if publisher.is_configured():
            try:
                publisher.game_completed()  # Finalize turn time tracking
                ds = self._get_dashboard_stats()
                publisher.publish_state(
                    board_state=terminal_state,
                    **ds,
                )
            except Exception as e:
                self.logger.debug(f"Game-end publish failed: {e}")

        return {
            "result": result,
            "final_score": final_score,
            "moves": move_count,
            "score_history": self.score_history.copy(),
        }


def main():
    parser = argparse.ArgumentParser(description="Play Tangled on tangled-game.com")
    parser.add_argument("--opponent", "-o", choices=["randy", "amara", "melissa", "andy", "alphaq"], default="melissa")
    parser.add_argument("--games", "-n", type=int, default=5)
    parser.add_argument("--run", "-r", type=int, help="Start/resume a run of N planned games (enables run tracking)")
    parser.add_argument("--slow-mo", type=int, default=100)
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode (no visible window)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--keep-open", "-k", type=int, default=5,
                        help="Seconds to keep browser open after last game (0 to close immediately)")
    parser.add_argument("--strategy", "-s", choices=["heuristic", "mcts", "hybrid", "matlab", "rl", "ensemble", "matlab_mcts", "hybrid_solver", "amara_explorer", "amara_killer", "melissa_killer", "alphaq_explorer", "oracle_route", "terminal_explorer"], default="hybrid_solver",
                        help="Strategy to use: hybrid_solver (DEFAULT: D-Wave inspired minimax+MCTS+learning), alphaq_explorer (explore/exploit vs AlphaQ Up with closed learning loop), amara_killer (uses E14P against Amara), melissa_killer (cycles E12P/E13P against Melissa - 40%% win rate), amara_explorer (cycles all 30 openings), hybrid (MCTS with opening), mcts (Monte Carlo, 30s/move), heuristic (fast), matlab (MATLAB-enhanced), rl (trained PPO), ensemble (RL + MC rollouts), matlab_mcts (MATLAB MCTS)")
    parser.add_argument("--mcts-time", type=float, default=float('inf'),
                        help="MCTS time limit per move in seconds (default unlimited)")
    parser.add_argument("--mcts-iterations", type=int, default=500000,
                        help="Maximum MCTS iterations per move (default 500K for optimal quality)")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics summary and exit (no games played)")
    parser.add_argument("--calibration", action="store_true",
                        help="Show adjudicator calibration report and exit (no games played)")
    parser.add_argument("--process-status", action="store_true",
                        help="Show active game process status and exit")
    parser.add_argument("--kill-active", action="store_true",
                        help="Kill the active game process and exit")

    # MATLAB training commands
    parser.add_argument("--training-status", action="store_true",
                        help="Show MATLAB training system status and exit")
    parser.add_argument("--train-value-network", action="store_true",
                        help="Train value network from game history (requires MATLAB)")
    parser.add_argument("--train-policy-network", action="store_true",
                        help="Train policy network from game history (requires MATLAB)")
    parser.add_argument("--cluster-opponents", action="store_true",
                        help="Cluster opponents by play style (requires MATLAB)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs for neural networks (default: 100)")
    parser.add_argument("--clusters", type=int, default=3,
                        help="Number of opponent clusters (default: 3)")

    # Neural network and opponent adaptation options
    parser.add_argument("--opening-mode", choices=["thompson", "forced", "round_robin"],
                        default=None,
                        help="Opening selection mode for alphaq_explorer: "
                             "thompson (default, Beta sampling), "
                             "forced (always E7G), "
                             "round_robin (cycle all 30 openings for systematic exploration)")
    parser.add_argument("--use-nn", action="store_true",
                        help="Use neural network priors with matlab strategy")
    parser.add_argument("--adapt-opponent", action="store_true",
                        help="Adapt play style to opponent with matlab strategy")
    parser.add_argument("--route-mode", choices=["fixed", "cycle"], default="fixed",
                        help="Oracle route selection mode: fixed (always same route) or "
                             "cycle (round-robin through all routes for terminal state discovery)")
    parser.add_argument("--routes-file", type=str, default=None,
                        help="Oracle routes JSON filename in oracle-solver/output/ "
                             "(default: oracle_routes.json, use website_oracle_routes.json for "
                             "website-calibrated routes)")
    parser.add_argument("--random-turns", type=str, default=None,
                        help="Comma-separated our-move indices for random play (e.g., 0,2,4). "
                             "Default for terminal_explorer: 0 (opening only)")
    parser.add_argument("--novel-branch", action="store_true",
                        help="Enable novel branch forcing: avoid repeating historical moves")
    parser.add_argument("--seat", type=int, choices=[1, 2], default=1,
                        help="Player seat: 1=Red/first (default), 2=Blue/second")

    args = parser.parse_args()

    # If --stats flag, just show stats and exit
    if args.stats:
        stats_queries.print_summary()
        return

    # If --calibration flag, show calibration report and exit
    if args.calibration:
        stats_queries.print_calibration_report()
        return

    # If --process-status flag, show active process info and exit
    if args.process_status:
        processes = get_active_processes()
        if processes:
            print(f"Active game processes ({len(processes)}):")
            for p in processes:
                print(f"  PID {p['pid']}: {p.get('strategy', '?')} vs {p.get('opponent', '?')}, "
                      f"run {p.get('run_id')}, {p.get('planned_games')} planned, "
                      f"started {p['started']}")
        else:
            print("No active game processes")
        return

    # If --kill-active flag, kill active processes and exit
    if args.kill_active:
        processes = get_active_processes()
        if processes:
            for p in processes:
                pid = p['pid']
                print(f"Killing PID {pid} ({p.get('strategy', '?')} vs {p.get('opponent', '?')})...")
                try:
                    os.kill(pid, signal.SIGTERM)
                    print(f"  Sent SIGTERM to PID {pid}")
                    pid_file = PROCESS_TRACKING_DIR / f"{pid}.json"
                    if pid_file.exists():
                        pid_file.unlink()
                except OSError as e:
                    print(f"  Failed to kill PID {pid}: {e}")
        else:
            print("No active game processes to kill")
        return

    # If --training-status flag, show training system status and exit
    if args.training_status:
        if print_training_status:
            print_training_status()
        else:
            print("MATLAB integration not available")
        return

    # If --train-value-network flag, train and exit
    if args.train_value_network:
        if not MATLAB_AVAILABLE:
            print("MATLAB integration not available. Install with: poetry install -E matlab")
            return

        from snowdrop_tangled_agents.matlab.training import get_training_orchestrator
        orchestrator = get_training_orchestrator()

        if not orchestrator.is_available():
            print("No training backend available. Install MATLAB or compiled packages.")
            return

        print(f"Training value network with {args.epochs} epochs...")
        try:
            metrics = orchestrator.train_value_network(epochs=args.epochs, verbose=True)
            print(f"\nTraining complete!")
            print(f"  Training samples: {metrics.get('training_samples', 0)}")
            print(f"  Validation loss:  {metrics.get('validation_loss', 0):.4f}")
            print(f"  Model saved to:   {metrics.get('model_path', 'N/A')}")
        except Exception as e:
            print(f"Training failed: {e}")
        return

    # If --train-policy-network flag, train and exit
    if args.train_policy_network:
        if not MATLAB_AVAILABLE:
            print("MATLAB integration not available. Install with: poetry install -E matlab")
            return

        from snowdrop_tangled_agents.matlab.training import get_training_orchestrator
        orchestrator = get_training_orchestrator()

        if not orchestrator.is_available():
            print("No training backend available. Install MATLAB or compiled packages.")
            return

        print(f"Training policy network with {args.epochs} epochs...")
        try:
            metrics = orchestrator.train_policy_network(epochs=args.epochs, verbose=True)
            print(f"\nTraining complete!")
            print(f"  Training samples:       {metrics.get('training_samples', 0)}")
            print(f"  Validation accuracy:    {metrics.get('validation_accuracy', 0):.2%}")
            print(f"  Model saved to:         {metrics.get('model_path', 'N/A')}")
        except Exception as e:
            print(f"Training failed: {e}")
        return

    # If --cluster-opponents flag, cluster and exit
    if args.cluster_opponents:
        if not MATLAB_AVAILABLE:
            print("MATLAB integration not available. Install with: poetry install -E matlab")
            return

        from snowdrop_tangled_agents.matlab.training import get_training_orchestrator
        orchestrator = get_training_orchestrator()

        if not orchestrator.is_available():
            print("No training backend available. Install MATLAB or compiled packages.")
            return

        print(f"Clustering opponents into {args.clusters} clusters...")
        try:
            result = orchestrator.cluster_opponents(k=args.clusters, verbose=True)
            print(f"\nClustering complete!")
            print(f"  Opponents:  {result.get('num_opponents', 0)}")
            print(f"  Clusters:   {args.clusters}")

            # Show cluster distribution
            labels = result.get('labels', [])
            for c in range(1, args.clusters + 1):
                count = sum(1 for l in labels if l == c)
                print(f"    Cluster {c}: {count} opponents")
        except Exception as e:
            print(f"Clustering failed: {e}")
        return

    level = "DEBUG" if args.debug else "INFO"
    coloredlogs.install(level=level, fmt="%(asctime)s %(levelname)s %(message)s")

    # Show other running sessions (informational, non-blocking)
    other_processes = get_active_processes()
    if other_processes:
        print(f"Note: {len(other_processes)} other session(s) running:")
        for p in other_processes:
            print(f"  PID {p['pid']}: {p.get('strategy', '?')} vs {p.get('opponent', '?')}")

    # Check MATLAB availability and get user confirmation if unavailable
    if not check_matlab_availability():
        return

    # Verify MATLAB readiness (connect, cleanup stale pools, verify responsive)
    if MATLAB_AVAILABLE and MATLAB_SESSIONS:
        if not verify_matlab_readiness():
            print("\nMATLAB is not ready. Please restart MATLAB and try again.")
            return

    # Validate strategy selection (prompt for alternatives if MATLAB unavailable or high loss rate)
    args.strategy = validate_strategy_selection(args.strategy)

    # Initialize stats collector outside context manager for run tracking
    from snowdrop_tangled_agents.stats import get_collector
    stats_collector = get_collector()

    # Determine run info - always create a run for tracking
    planned_games = args.run if args.run else args.games
    if args.run:
        # Resume existing run or create new one
        run_id, start_game_number = stats_collector.get_or_create_run(
            planned_games=args.run,
            strategy=args.strategy,
            opponent=args.opponent
        )
        run_info = stats_collector.get_run(run_id)
        total_planned = run_info['planned_games']
        print(f"Run {run_id}: {run_info['completed_games']}/{total_planned} completed")
    else:
        # Always create a new run, even for simple --games N
        run_id = stats_collector.start_run(
            planned_games=planned_games,
            strategy=args.strategy,
            opponent=args.opponent
        )
        start_game_number = 1
        total_planned = planned_games
        print(f"Run {run_id}: {total_planned} games planned")

    # Register this process for tracking
    register_process(run_id=run_id, planned_games=total_planned,
                     strategy=args.strategy, opponent=args.opponent)
    atexit.register(unregister_process)

    # Initialize live stats publisher (if configured)
    publisher = get_publisher()
    if publisher.is_configured():
        publisher.set_session_info(run_id=run_id, planned_games=total_planned)
        status = publisher.get_status()
        logging.getLogger(__name__).info(
            f"Live stats publishing enabled (url={status['url']}, "
            f"connected={status['connected']}, connecting={status['connecting']})"
        )
    else:
        logging.getLogger(__name__).debug("Live stats publishing not configured (set TANGLED_DASHBOARD_URL and TANGLED_DASHBOARD_API_KEY)")

    results = []
    current_game_number = start_game_number
    restart_needed = False

    # Main game loop with browser restart support
    while True:
        # Calculate remaining games
        if run_id:
            run_info = stats_collector.get_run(run_id)
            games_remaining = run_info['planned_games'] - run_info['completed_games']
            if games_remaining <= 0:
                print(f"\nRun {run_id} complete! ({run_info['planned_games']}/{run_info['planned_games']})")
                break
        else:
            games_remaining = total_planned - len(results)
            if games_remaining <= 0:
                break

        # Create player and play games
        # Parse random-turns into a set of ints
        random_turns = None
        if args.random_turns:
            random_turns = {int(x.strip()) for x in args.random_turns.split(',')}

        with WebPlayer(
            headless=args.headless,
            slow_mo=args.slow_mo,
            strategy_type=args.strategy,
            mcts_time=args.mcts_time,
            mcts_iterations=args.mcts_iterations,
            use_nn=args.use_nn,
            adapt_opponent=args.adapt_opponent,
            opening_mode=getattr(args, 'opening_mode', None),
            route_mode=getattr(args, 'route_mode', None),
            routes_file=getattr(args, 'routes_file', None),
            random_turns=random_turns,
            novel_branch=args.novel_branch,
            seat=args.seat,
        ) as player:
            player.login()

            # Store run info on player for use in play_game
            player._run_id = run_id
            player._current_game_number = current_game_number

            restart_needed = False

            while games_remaining > 0:
                # Update run info for display
                if run_id:
                    run_info = stats_collector.get_run(run_id)
                    games_remaining = run_info['planned_games'] - run_info['completed_games']
                    display_num = run_info['completed_games'] + 1
                    display_total = run_info['planned_games']
                else:
                    display_num = len(results) + 1
                    display_total = total_planned

                if games_remaining <= 0:
                    break

                print(f"\n=== Game {display_num}/{display_total} ===")

                # Update ETA: record game start timestamp for avg calculation
                games_completed_so_far = display_num - 1
                publisher = get_publisher()
                if publisher.is_configured():
                    publisher.game_started(
                        game_number=display_num,
                        games_completed=games_completed_so_far,
                    )

                result = player.play_game(args.opponent)
                results.append(result)

                # Note: Dashboard publish happens immediately at game end in play_game()
                # No need for duplicate publish here

                # Check for stuck opponent
                if result.get("error") == "stuck_opponent":
                    print("\n*** OPPONENT STUCK - Restarting browser... ***\n")
                    restart_needed = True

                    # Check if this was the final game
                    if run_id:
                        run_info = stats_collector.get_run(run_id)
                        if run_info['completed_games'] >= run_info['planned_games']:
                            print(f"Run {run_id} was on final game - starting new run")
                            # Start a new run with same parameters
                            run_id = stats_collector.start_run(
                                planned_games=args.run,
                                strategy=args.strategy,
                                opponent=args.opponent
                            )
                            current_game_number = 1
                            print(f"Started new run {run_id}")
                    break  # Exit inner loop to restart browser

                if run_id:
                    player._current_game_number += 1
                    current_game_number = player._current_game_number

                games_remaining -= 1

                if games_remaining > 0:
                    time.sleep(2)

        # If no restart needed, we're done
        if not restart_needed:
            break

        # Brief pause before restart
        print("Restarting in 5 seconds...")
        time.sleep(5)

    # Detailed Summary
    print("\n" + "=" * 50)
    print("SESSION SUMMARY")
    print("=" * 50)

    for i, r in enumerate(results):
        result = r.get("result", "unknown")
        score = r.get("final_score", 0) or 0
        moves = r.get("moves", 0)
        history = r.get("score_history", [])

        result_symbol = {"win": "WIN", "loss": "LOSS", "draw": "DRAW", "abandoned": "ABANDONED"}.get(result, "?")
        print(f"\nGame {i+1}: {result_symbol} (Score: {score:.4f}, Moves: {moves})")

        if history:
            scores_str = " -> ".join(f"{s:.2f}" for _, _, s in history)
            print(f"  Progression: {scores_str}")

    print("\n" + "-" * 50)
    wins = sum(1 for r in results if r.get("result") == "win")
    losses = sum(1 for r in results if r.get("result") == "loss")
    draws = sum(1 for r in results if r.get("result") == "draw")
    abandoned = sum(1 for r in results if r.get("result") == "abandoned")
    print(f"TOTAL: {wins}W / {losses}L / {draws}D / {abandoned}A")
    print("=" * 50)

    # Show database statistics
    stats_queries.print_summary()

    # Keep browser open is no longer applicable since we exit context manager
    # The browser closes when we leave the 'with' block


if __name__ == "__main__":
    main()
