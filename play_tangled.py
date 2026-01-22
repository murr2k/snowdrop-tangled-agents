#!/usr/bin/env python3
"""
Play Tangled on tangled-game.com using configurable strategies.

Port of the working browser JS bot (V28.2) to Python/Playwright.
Uses the same robust DOM interaction patterns:
- Nearest-vertex edge matching
- Direct MouseEvent dispatch
- Text-based turn/game-over detection

Strategies:
- heuristic: Fast parameterized strategy with learning
- mcts: Monte Carlo Tree Search (fight fire with fire!)
- hybrid: MCTS with heuristic opening book

Usage:
    python play_tangled.py                        # Play 5 games vs Melissa (heuristic)
    python play_tangled.py --strategy mcts        # Use MCTS strategy
    python play_tangled.py --strategy hybrid      # Use hybrid strategy
    python play_tangled.py --opponent randy       # Play vs Randy
    python play_tangled.py --games 10             # Play 10 games
"""

import argparse
import atexit
import logging
import os
import signal
import sys
import time
from pathlib import Path

import coloredlogs
from dotenv import load_dotenv

load_dotenv()

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
from snowdrop_tangled_agents.stats import get_collector, queries as stats_queries

# Optional MATLAB integration
try:
    from snowdrop_tangled_agents.matlab import (
        MatlabEnhancedStrategy,
        get_unified_bridge,
        print_training_status,
    )
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    print_training_status = None

# Optional RL strategy
try:
    from snowdrop_tangled_agents.strategy.rl_strategy import RLStrategy, EnsembleStrategy
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    EnsembleStrategy = None


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
    }

    def __init__(
        self,
        headless: bool = False,
        slow_mo: int = 100,
        strategy_type: str = "heuristic",
        mcts_time: float = 2.0,
        mcts_iterations: int = 5000,
        use_nn: bool = True,
        adapt_opponent: bool = True,
    ):
        self.username = os.getenv("TANGLED_USERNAME")
        self.password = os.getenv("TANGLED_PASSWORD")
        self.headless = headless
        self.slow_mo = slow_mo
        self.strategy_type = strategy_type

        # Store options for MATLAB strategy
        self._use_nn = use_nn
        self._adapt_opponent = adapt_opponent

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
        else:  # "heuristic" (default)
            self.strategy = PetersenStrategy(params_path=self.params_path)

        self.score_history = []
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using strategy: {strategy_type}")

        # Stats collection
        self.stats_collector = get_collector()
        self.current_game_id = None
        self.mcts_time = mcts_time

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

        # Select Player 1 (Red)
        try:
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
            self.page.wait_for_selector("svg line", timeout=10000)
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
            if "player 1" in text and "turn" in text and "player 2" not in text:
                return True
            # Explicit NOT our turn indicators
            if "opponent" in text and "turn" in text:
                return False
            if "player 2" in text and "turn" in text:
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
            if "Winner: Player 1" in text:
                return "win"
            if "Winner: Player 2" in text:
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
            });
            return state.join('');
        }
        """
        try:
            return self.page.evaluate(js_code)
        except Exception as e:
            self.logger.warning(f"read_board error: {e}")
            return "-" * 15

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

            // Click the line
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

    def play_game(self, opponent: str = "melissa") -> dict:
        """Play one complete game."""
        self.score_history = []

        # Initialize MATLAB strategy with opponent model
        if self.strategy_type == "matlab" and hasattr(self.strategy, 'initialize'):
            self.strategy.initialize(opponent=opponent)

        # Start stats tracking for this game
        self.current_game_id = self.stats_collector.start_game(
            opponent=opponent,
            graph="petersen",
            strategy=self.strategy_type,
            mcts_time=self.mcts_time
        )

        if not self.start_game(opponent):
            return {"result": None, "error": "Could not start game"}

        # Debug: show line coordinates
        self.debug_lines()

        # Wait for initial turn
        while not self.is_our_turn() and not self.is_game_over():
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

            # Only count iterations where it's actually our turn (move attempts)
            loop_iterations += 1
            if loop_iterations > max_iterations:
                self.logger.error(f"Safety limit reached: {loop_iterations} move attempts (likely infinite loop)")
                break

            state = self.read_board()
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

            # Calculate move
            result = self.strategy.calculate_move(state, score, self.score_history)
            if result is None:
                break

            edge, color = result

            # Re-read board state after MCTS calculation (opponent may have played)
            current_state = self.read_board()
            available = [i for i, c in enumerate(current_state) if c == '-' and i not in failed_edges]

            # Verify edge is still available after calculation time
            if current_state[edge] != '-' or edge in failed_edges:
                if available:
                    self.logger.warning(f"E{edge} no longer available (opponent played during calculation), picking from: {available[:3]}...")
                    edge = available[0]
                    color = 'G'
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

                # Record move to stats database
                self.stats_collector.record_move(
                    game_id=self.current_game_id,
                    move_number=move_count,
                    player="us",
                    edge=edge,
                    color=color,
                    score_after=new_score,
                    score_before=prev_score,
                    state_after=self.read_board()
                )
                prev_score = new_score

                # Record move for learning (if strategy supports it)
                if hasattr(self.strategy, 'record_move'):
                    self.strategy.record_move(edge, color, new_score)
            else:
                self.logger.error(f"Move E{edge} failed after {max_retries} attempts, marking as failed")
                failed_edges.add(edge)
                continue  # Try another edge without waiting for opponent

            # Wait for opponent
            if not self.wait_for_turn(timeout=15.0):
                break

        # Game over - wait for result to display
        time.sleep(2)
        final_score = self.read_score()
        result = self.get_outcome()
        terminal_state = self.read_board()

        # Record game end to stats database
        self.stats_collector.end_game(
            game_id=self.current_game_id,
            result=result,
            final_score=final_score
        )

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
            self.logger.info(f"MCTS Stats: {stats['iterations']} iterations in {stats['time']:.2f}s")
            # Log game stats if available
            if 'game_stats' in stats:
                gs = stats['game_stats']
                self.logger.info(f"Session Stats: {gs['wins']}W / {gs['losses']}L / {gs['draws']}D")

        return {
            "result": result,
            "final_score": final_score,
            "moves": move_count,
            "score_history": self.score_history.copy(),
        }


def main():
    parser = argparse.ArgumentParser(description="Play Tangled on tangled-game.com")
    parser.add_argument("--opponent", "-o", choices=["randy", "amara", "melissa"], default="melissa")
    parser.add_argument("--games", "-n", type=int, default=5)
    parser.add_argument("--slow-mo", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--keep-open", "-k", type=int, default=5,
                        help="Seconds to keep browser open after last game (0 to close immediately)")
    parser.add_argument("--strategy", "-s", choices=["heuristic", "mcts", "hybrid", "matlab", "rl", "ensemble"], default="heuristic",
                        help="Strategy to use: heuristic (fast), mcts (Monte Carlo), hybrid (MCTS with opening), matlab (MATLAB-enhanced), rl (trained PPO), ensemble (RL + MC rollouts)")
    parser.add_argument("--mcts-time", type=float, default=2.0,
                        help="MCTS time limit per move in seconds")
    parser.add_argument("--mcts-iterations", type=int, default=5000,
                        help="Maximum MCTS iterations per move")
    parser.add_argument("--stats", action="store_true",
                        help="Show statistics summary and exit (no games played)")
    parser.add_argument("--calibration", action="store_true",
                        help="Show adjudicator calibration report and exit (no games played)")

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
    parser.add_argument("--use-nn", action="store_true",
                        help="Use neural network priors with matlab strategy")
    parser.add_argument("--adapt-opponent", action="store_true",
                        help="Adapt play style to opponent with matlab strategy")

    args = parser.parse_args()

    # If --stats flag, just show stats and exit
    if args.stats:
        stats_queries.print_summary()
        return

    # If --calibration flag, show calibration report and exit
    if args.calibration:
        stats_queries.print_calibration_report()
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

    # Use context manager for automatic cleanup on exit/kill/interrupt
    with WebPlayer(
        headless=False,
        slow_mo=args.slow_mo,
        strategy_type=args.strategy,
        mcts_time=args.mcts_time,
        mcts_iterations=args.mcts_iterations,
        use_nn=args.use_nn,
        adapt_opponent=args.adapt_opponent,
    ) as player:
        player.login()

        results = []
        for i in range(args.games):
            print(f"\n=== Game {i+1}/{args.games} ===")
            result = player.play_game(args.opponent)
            results.append(result)

            if i < args.games - 1:
                time.sleep(2)

        # Detailed Summary
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)

        for i, r in enumerate(results):
            result = r.get("result", "unknown")
            score = r.get("final_score", 0)
            moves = r.get("moves", 0)
            history = r.get("score_history", [])

            result_symbol = {"win": "WIN", "loss": "LOSS", "draw": "DRAW"}.get(result, "?")
            print(f"\nGame {i+1}: {result_symbol} (Score: {score:.4f}, Moves: {moves})")

            if history:
                scores_str = " -> ".join(f"{s:.2f}" for _, _, s in history)
                print(f"  Progression: {scores_str}")

        print("\n" + "-" * 50)
        wins = sum(1 for r in results if r.get("result") == "win")
        losses = sum(1 for r in results if r.get("result") == "loss")
        draws = sum(1 for r in results if r.get("result") == "draw")
        print(f"TOTAL: {wins}W / {losses}L / {draws}D")
        print("=" * 50)

        # Show database statistics
        stats_queries.print_summary()

        # Keep browser open to see results
        if args.keep_open > 0:
            print(f"\nBrowser closing in {args.keep_open} seconds...")
            time.sleep(args.keep_open)


if __name__ == "__main__":
    main()
