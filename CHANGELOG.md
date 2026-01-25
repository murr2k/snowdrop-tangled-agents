# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Run Tracking & Resume System** (Migration v8)
  - New `runs` table tracks planned game batches with progress
  - Games store `run_id` and `game_number` for batch tracking
  - `--run N` flag to start/resume a run of N planned games
  - Automatic resume: interrupted runs continue from where they left off
  - Session stats shows run progress: `run = 2`, `games = 5/750`
  - `get_or_create_run()`, `get_active_run()`, `abandon_game()` collector methods

- **Stuck Opponent Detection & Recovery**
  - 3-minute timeout when waiting for opponent to play
  - `StuckOpponentError` exception for timeout handling
  - Automatic browser restart on stuck opponent
  - Games marked as "abandoned" with reason tracking
  - If final game of run is stuck, automatically starts new run

- **Process Tracking for Safe Management**
  - Active process info stored in `~/.tangled/active_process.json`
  - Prevents accidentally starting multiple game sessions
  - `--process-status` flag shows active game process info
  - `--kill-active` flag safely terminates running game process
  - Automatic cleanup on process exit via atexit handler

- **Policy Versioning & Model Metrics Tracking** (Migration v7)
  - `policy_id` field tracks code version via git tag or commit hash
  - Opponent model metrics: `model_entropy`, `model_games_learned`, `model_moves_learned`
  - Prediction accuracy metrics: `model_top3_hit`, `prediction_accuracy`
  - `GameMetricsTracker` class for per-game prediction tracking
  - `get_policy_id()` auto-detects version from git tag/commit
  - `backfill_policy_id.py` tool for historical game backfill
  - Integrated into `play_tangled.py` game loop for automatic metrics capture

- **Live Session Statistics** (`snowdrop_tangled_agents/stats/session_stats.py`)
  - Real-time session statistics with automatic session boundary detection
  - Watch mode: `--watch` refreshes stats periodically (press `q` to exit)
  - Configurable refresh interval: `--interval` (default 60s)
  - Configurable session gap: `--gap` (default 30 minutes)
  - Trend analysis: score trend, win rate trend, recent 5 results
  - Estimated end time based on play rate
  - UTC storage with local timezone display
  - Cleanup tool for stale in-progress games: `--cleanup --force`
  - JSON output: `--json`

- **D-Wave Inspired Hybrid Solver** (`snowdrop_tangled_agents/matlab/rl/HybridTangledSolver.m`)
  - Integrated solver combining Alpha-Beta Minimax, MCTS, and Tabu Search
  - Automatic strategy selection based on game phase:
    - Late game (≤3 grey edges): Pure minimax with guaranteed optimal play
    - Mid game (4-8 grey edges): Hybrid minimax + MCTS
    - Early game (>8 grey edges): MCTS + Tabu refinement
  - D-Wave MST2-inspired Tabu Search (`TabuSearch.m`) with multistart optimization
  - Alpha-Beta Pruning (`AlphaBetaSearch.m`) with transposition tables
  - Time-budgeted search with configurable allocation (35% minimax, 55% MCTS, 10% tabu)
  - Python integration via `HybridSolverStrategy` in `matlab_strategy.py`
  - **REINFORCE-style learning** from game outcomes (added to Python strategy):
    - Temporal credit assignment with discount factor (γ=0.9)
    - Edge value adjustments learned from wins/losses/draws
    - Persistent storage of learned adjustments (`~/.tangled/hybrid_solver_adjustments.json`)
    - Configurable learning rate (default 0.03)
  - CLI: `python play_tangled.py --strategy hybrid_solver`

- **Expanded Lookup Table (19M Entries)** (`snowdrop_tangled_agents/matlab/rl/ExpandedLUT.m`)
  - Pre-computed exact minimax values for 0-3 grey edge states
  - Coverage breakdown:
    - 0 grey (terminal): 32,768 states
    - 1 grey: 491,520 states
    - 2 grey: 3,440,640 states
    - 3 grey: 14,909,440 states
    - **Total: 18,874,368 exact values**
  - Guarantees optimal play for final 4 moves of every game
  - Generation scripts: `generate_expanded_lut.m`, `generate_expanded_lut_parallel.m`, `extend_lut_three_grey_parallel.m`
  - 3-grey extension generated in ~12.6 minutes using 6 parallel workers

- **Hybrid Solver Test Suite** (`snowdrop_tangled_agents/matlab/rl/test_hybrid_solver.m`)
  - 27 unit tests covering ExpandedLUT, TabuSearch, AlphaBetaSearch, HybridTangledSolver
  - Integration tests for full game simulation and solver consistency
  - LUT generation file verification tests

- **MATLAB MCTS Strategy** (`snowdrop_tangled_agents/strategy/matlab_mcts_strategy.py`)
  - High-compute MCTS using MATLAB TangledMCTS engine (5000 iterations, 20s time limit)
  - `MCTSParams` dataclass for tunable parameters with JSON persistence
  - Opening book for first 3 moves (E9→E11→E10 Green sequence)
  - Adaptive exploration based on score momentum (boost when losing, reduce when winning)
  - Fallback chain: MATLAB MCTS → Python MCTS → heuristic
  - Integration with stats collection for edge bonus learning
  - CLI: `python play_tangled.py --strategy matlab_mcts`

- **TangledMCTS MATLAB Engine** (`snowdrop_tangled_agents/matlab/rl/TangledMCTS.m`)
  - Full MCTS with UCB1 selection and progressive bias
  - Domain-specific heuristic rollout policy
  - Calibrated terminal evaluation with priority system (MY_EDGES → OPP_EDGES → HUB_EDGES)
  - Compute effort diagnostics using MATLAB's `cputime` and `memory` functions
  - Session-level statistics tracking (total CPU time, total iterations)
  - `getComputeEffort()` method for detailed performance metrics

- **MCTSNode Class** (`snowdrop_tangled_agents/matlab/rl/MCTSNode.m`)
  - Tree node implementation with containers.Map for children
  - UCB1 selection with configurable exploration and prior weight
  - Action masking for valid moves only
  - Visit counting and value backpropagation

- **MATLAB MCTS Documentation** (`docs/MATLAB_MCTS_STRATEGY.md`)
  - Theory of operation for the MATLAB MCTS strategy
  - Petersen graph edge classification (MY_EDGES, OPP_EDGES, HUB_EDGES)
  - Terminal evaluation scoring values and priority system
  - Critical discoveries: E12 G and E2 G cause score collapse
  - Compute diagnostics and performance profiling guide
  - Turn-based state reading analysis and fix
  - Roadmap for future improvements

- **MATLAB RL System** (`snowdrop_tangled_agents/matlab/rl/`)
  - `TangledEnvironment.m`: RL environment with 50-element observation space, 30 discrete actions
  - `createPPOAgent.m`: PPO agent with actor-critic networks (16K+ parameters each)
  - `trainParallel.m`: Parallel self-play training with worker pools
  - `collectEpisode.m`: Experience collection with action masking
  - `SQLiteExperienceBuffer.m`: Persistent replay buffer with base64 serialization
  - `SimulatedOpponent.m`: Configurable opponent for self-play (random, heuristic, copy)
  - `getActionMask.m`: Valid action filtering (15 edges × 2 colors)

- **Phase 5: Continuous Deployment Pipeline** (`snowdrop_tangled_agents/matlab/rl/`)
  - `ModelRegistry.m`: SQLite-backed model version management
  - `tangled_agent_inference.m`: Compiled inference with hot-reload (60s refresh)
  - `autoDeploy.m`: Automatic deployment when win rate improves
  - `build_rl_package.m`: MATLAB Compiler SDK build script
  - `rl_bridge.py`: Python bridge with fallback chain (compiled → engine → heuristic)

- **CI/CD Integration** (`.github/workflows/test.yml`)
  - GitHub Actions workflow for Python tests (3.11, 3.12, 3.13)
  - Self-hosted runner support for MATLAB tests
  - Manual trigger option for MATLAB test suite
  - pytest markers: `matlab` (requires MATLAB), `slow` (training tests)

- **MATLAB Test Suite** (`snowdrop_tangled_agents/matlab/rl/run_all_tests.m`)
  - 23 regression tests covering Phases 0-4
  - Isolated test directories to protect production database
  - GPU detection with graceful CPU fallback
  - pytest wrapper: `snowdrop_tangled_agents/tests/test_matlab_rl.py`

- **MATLAB Toolbox Integration** (`snowdrop_tangled_agents/matlab/`)
  - Deep Learning Toolbox integration for neural network position evaluation
  - Statistics and ML Toolbox for opponent clustering and style classification
  - Database Toolbox for direct MATLAB-SQLite access in training pipelines
  - MATLAB Compiler SDK support for Python-callable compiled packages
  - `unified_bridge.py`: Automatic backend selection with fallback chain
    - Compiled packages (fastest, MATLAB Runtime only)
    - MATLAB Engine API (full functionality, requires license)
    - Pure Python heuristics (always available)
  - `compiled_bridge.py`: Bridge to pre-compiled MATLAB packages
  - `training.py`: Training orchestration from Python
  - `matlab_strategy.py`: MCTS enhanced with neural network priors
  - Value network architecture: 50-input features → FC(128-64-32) → tanh output
  - Opponent modeling with 20-element feature vectors and K-means clustering
  - CLI: `--strategy matlab --use-nn --adapt-opponent`

- **Database Schema Migrations** (`snowdrop_tangled_agents/stats/migrations.py`)
  - Automatic schema versioning and migration runner
  - v2: `models` table for trained network metadata
  - v3: `opponents` table for opponent profiles and clustering
  - v4: `training_data` table for ML training samples
  - v5: `opponent_history` table for move-level opponent tracking

- **Compiled MATLAB Packages** (for deployment without MATLAB license)
  - `tangled_value_network`: Neural network inference
  - `tangled_opponent_model`: Opponent classification and prior adaptation
  - `tangled_training`: Model training and opponent clustering
  - Requires MATLAB Runtime R2026a (free download)

- **Adjudicator Calibration System** (`snowdrop_tangled_agents/stats/`)
  - Compares our terminal state evaluation to actual tangled-game.com scores
  - New `calibration` table in SQLite database
  - Automatic data collection at game end when all edges are colored
  - `record_calibration()` method in StatsCollector
  - Analysis queries: `get_calibration_summary()`, `get_calibration_details()`
  - `print_calibration_report()` for detailed analysis output
  - CLI access: `python play_tangled.py --calibration`
  - Error distribution tracking (exact, close, moderate, large)
  - Systematic bias detection and interpretation

- **MCTS Strategy Engine** (`snowdrop_tangled_agents/strategy/mcts_strategy.py`)
  - Monte Carlo Tree Search with UCB1 selection for deep lookahead
  - Progressive Bias: heuristic priors guide early exploration, decay with visits
  - Action prioritization: good moves expanded first based on domain knowledge
  - Domain-specific rollout policy using Tangled heuristics
  - Terminal state evaluation using official `SimulatedAnnealingAdjudicator`
  - LRU cache for efficient repeated state evaluations
  - Edge classifications derived from 50+ game empirical analysis:
    - GOOD_PURPLE_EDGES: E0, E1, E3, E5, E12, E13
    - BAD_PURPLE_EDGES: E2, E4, E6, E7, E8, E14

- **Hybrid Strategy** (`snowdrop_tangled_agents/strategy/mcts_strategy.py`)
  - Combines heuristic opening, MCTS midgame, and exhaustive endgame
  - Opening sequence: E9→E10→E11 Green, E5→E12→E13 Purple
  - Adaptive time allocation: 3x more time for critical late-game moves
  - Exhaustive minimax for positions with ≤2 edges remaining
  - REINFORCE-style learning from game outcomes
  - Edge adjustment tracking across games

- **SQLite Statistics Collection** (`snowdrop_tangled_agents/stats/`)
  - `collector.py`: StatsCollector class for game/move recording
  - `queries.py`: Analysis functions for pattern discovery
  - Database schema: games table + moves table with full indexing
  - Automatic integration with play_tangled.py
  - CLI access: `python play_tangled.py --stats`
  - Analysis queries:
    - `get_edge_effectiveness()`: Edge/color performance ranking
    - `get_winning_patterns()`: Move patterns leading to wins
    - `get_score_progression()`: Score trajectory by game result
    - `get_opening_sequences()`: Common openings and outcomes
    - `get_critical_positions()`: Large score swing analysis
    - `get_opponent_patterns()`: Opponent behavior analysis

- **Strategy CLI Options** (`play_tangled.py`)
  - `--strategy {heuristic,mcts,hybrid}`: Select strategy type
  - `--mcts-time SECONDS`: MCTS time limit per move
  - `--mcts-iterations N`: Maximum MCTS iterations
  - `--stats`: Show statistics summary and exit

- **Petersen Strategy Engine** (`snowdrop_tangled_agents/strategy/petersen_strategy.py`)
  - Parameterized strategy calculator for Petersen graph games
  - Edge priority scoring based on vertex ownership (MY_VERTEX=5, OPP_VERTEX=7, HUB_VERTEX=6)
  - Configurable opening sequence override for first N moves
  - Adaptive color selection based on score thresholds and strategy mode
  - Momentum tracking from recent score history
  - Opponent pattern analysis to detect valued edges
  - REINFORCE-style learning from game outcomes with discounted returns
  - Parameter persistence to JSON for learning across sessions
  - Game statistics tracking (wins/losses/draws)

- **Petersen Agent** (`snowdrop_tangled_agents/agents/petersen_agent.py`)
  - SDK-compatible wrapper implementing `GameAgentBase`
  - Translates SDK game state to strategy state string format
  - Supports external score injection for web play
  - Move history tracking for learning updates

- **Web Player** (`play_tangled.py`)
  - Playwright-based automation for tangled-game.com
  - Dynamic vertex discovery from SVG line endpoints
  - Angle-based vertex alignment (outer pentagon, inner pentagram)
  - Robust edge-to-line mapping using nearest-vertex matching
  - Color button detection with multiple text pattern matching
  - Turn detection with explicit state checking
  - Automatic browser cleanup on exit/signal/exception
  - Game outcome recording with full score history
  - Command-line interface with configurable opponent and game count

- **Strategy Module** (`snowdrop_tangled_agents/strategy/__init__.py`)
  - Package exports for PetersenStrategy class

- **Documentation**
  - `CLAUDE.md` - Project guidance for Claude Code
  - `docs/THEORY_OF_OPERATION.md` - Comprehensive system documentation (moved from root)
    - Added Adjudicator Calibration section
    - Added Mermaid diagrams for data flow and gameplay transaction flow
  - `docs/MATLAB_INTEGRATION.md` - Complete MATLAB integration guide
  - `docs/tangled-bot-v28.txt` - Reference JavaScript bot implementation
  - `docs/THE_MATHEMATICS_OF_TANGLED_GAME.md` - Added Section 9: HybridSolverStrategy
    - D-Wave inspired hybrid search architecture
    - Alpha-Beta minimax with transposition tables
    - MCTS with Tabu Search refinement
    - 19M-entry expanded lookup table theory
    - REINFORCE-style adaptive learning with temporal credit assignment
    - Database integration and persistence mechanisms
    - Performance analysis and research extensions for graduate students

### Added

- **Game Analytics & Visualization** (`snowdrop_tangled_agents/tools/plot_progress.py`)
  - Progress tracking: win rate and score trends over time with rolling averages
  - Edge effectiveness analysis: score delta and win rate by edge/color
  - Opening sequence analysis: identify winning opening patterns
  - Timestamped output: `plots/{type}_{YYYYMMDD}_{HHMMSS}.png`
  - CLI: `python -m snowdrop_tangled_agents.tools.plot_progress --all`
  - See `docs/GAME_ANALYTICS.md` for full documentation

### Changed

- **Unified Dashboard Messaging** (`snowdrop_tangled_agents/stats/websocket_publisher.py`)
  - Consolidated `publish_move()`, `publish_stats()`, and `publish_session_stats()` into single `publish_state()` function
  - Dashboard now receives exactly ONE message type (`full_state`) containing complete state
  - Every message includes: move info, board state, vertex colors, session stats, score stats, trends, model metrics, ETA
  - Removes partial message handling complexity from both publisher and dashboard
  - Dashboard always renders full UI regardless of message content

- **Run-Specific Dashboard Stats** (`play_tangled.py`, `snowdrop_tangled_agents/stats/session_stats.py`)
  - Added `get_run_stats(run_id)` function for run-specific statistics (not time-based session)
  - Dashboard now publishes immediately when game ends (after edge 15), not waiting for next game loop
  - All dashboard messages now use run-specific stats when a run_id is active
  - Removed redundant post-game publish block in main loop (single publisher at game end)
  - Fixes issue where dashboard showed stats from previous runs mixed with current run

- **Default Strategy Changed** (`play_tangled.py`)
  - Changed default strategy from `heuristic` to `hybrid_solver`
  - HybridSolverStrategy with REINFORCE learning is now the recommended approach
  - Combines MATLAB's powerful search (minimax + MCTS + Tabu) with adaptive learning
  - Requires MATLAB Engine connection; falls back to Python MCTS if unavailable

- **SimulatedOpponent Real MCTS Mode** (`snowdrop_tangled_agents/matlab/rl/SimulatedOpponent.m`)
  - Added real MCTS opponent using `TangledMCTS` engine for realistic training
  - Lazy-initialized MCTS engine to avoid overhead when not needed
  - Renamed previous MCTS approximation to `fast_mcts` style
  - Increased default iterations from 100 to 500 for stronger play
  - Added `MCTSTimeLimit` parameter (default 1.0s)
  - Available styles: `random`, `heuristic`, `mcts` (real), `fast_mcts`, `petersen`, `defensive`, `aggressive`

- **TangledEnvironment Auto-Correction** (`snowdrop_tangled_agents/matlab/rl/TangledEnvironment.m`)
  - Enabled `AutoCorrectInvalidActions` by default to improve RL learning
  - Reduced invalid action penalty from -0.5 to -0.1 when action is remapped
  - Kept larger -0.5 penalty when auto-correct is disabled
  - Added `info.OriginalAction` tracking for debugging remapped moves

- **Curriculum Training Worker Detection** (`snowdrop_tangled_agents/matlab/rl/train_curriculum_ensemble.m`)
  - Added automatic detection of maximum available parallel workers
  - Graceful degradation when requested workers exceed cluster capacity
  - Changed `AutoCorrect` from false to true in `trainLevel()`, `trainSelfPlayEnsemble()`, and `evaluateAgent()`
  - Improved logging to show when worker count is reduced

- **Opponent Model Data** (`snowdrop_tangled_agents/matlab/rl/data/opponent_model.mat`)
  - Updated with latest learned patterns from online gameplay
  - Contains response-conditional and phase-conditional probability matrices

- Moved `THEORY_OF_OPERATION.md` from project root to `docs/` directory
- Updated `README.md` with Development Progress section documenting all implementation steps
- Extended `StatsCollector` with model and opponent management methods
- Added migration support to stats collector initialization

- Updated `snowdrop_tangled_agents/__init__.py` to export PetersenAgent
- Updated `snowdrop_tangled_agents/agents/__init__.py` to include PetersenAgent
- Updated `pyproject.toml` with new dependencies (playwright, python-dotenv, coloredlogs)
- Updated `pyproject.toml` with pytest markers (`matlab`, `slow`)
- Updated `docs/TEST_SUITE.md` with MATLAB RL test documentation and CI/CD workflow

### Fixed

- Terminal state evaluation accuracy (`mcts_strategy.py:evaluate_terminal_state`)
  - Replaced incorrect brute-force spin enumeration with official `SimulatedAnnealingAdjudicator`
  - Calibration improved from ~3-4 point errors to <0.02 point errors
  - Now matches tangled-game.com scores exactly (within stochastic variance)
  - Added LRU caching for efficient repeated evaluations

- Edge mapping between strategy edge indices and website SVG lines
  - Implemented consistent dynamic vertex discovery algorithm
  - Fixed angle wrap-around handling for vertex rotation
  - Aligned inner pentagram and outer pentagon vertex numbering

- Color button detection reliability
  - Increased dialog appearance wait time
  - Added multiple button text patterns (Green/FM/Ferromagnetic)
  - Extended retry logic with longer delays

- Turn detection accuracy
  - Made detection more conservative with explicit checks only
  - Added negative indicators for opponent's turn
  - Removed aggressive fallback assumptions

- Browser session cleanup
  - Added signal handlers for SIGTERM/SIGINT
  - Implemented atexit cleanup handler
  - Added context manager support for automatic cleanup

- MATLAB RL training bugs (`trainParallel.m`)
  - Fixed brace indexing error: SQLiteExperienceBuffer returns matrices, not cell arrays
  - Fixed integer/double type mismatch with explicit `double()` conversions
  - Fixed System ID Toolbox conflict: use `forward(getModel(getCritic(agent)), ...)` instead of `predict()`

- MATLAB test reliability
  - Tests check for "PASSED" message in stdout instead of return code
  - Handles MATLAB prerelease shutdown crashes gracefully

- **MATLAB MCTS Terminal Evaluation** (`TangledMCTS.m:evaluateTerminal`)
  - Fixed E12 G scoring: Was +0.2 (OPP + HUB double-count), now -0.8 penalty
    - E12 connects hub (V6) to opponent vertex (V7)
    - Green/ferromagnetic coupling helps opponent in quantum adjudication
  - Fixed E2 G scoring: Was +0.3 (hub control), now -0.5 penalty
    - Game data showed E2 G at move 4 caused -2.6 point collapses
  - Added priority system to avoid double-counting overlapping edge categories
  - Updated rollout policy: Hub edges 25% green (was 70%)

- **Turn-Based State Reading** (`play_tangled.py`)
  - Fixed stale DOM state issue causing invalid move attempts
  - Root cause: `is_our_turn()` returned True before DOM fully updated with opponent's move
  - Added 0.5s delay after turn change detection before reading board state
  - Clarified that game is strictly turn-based (no true race conditions)

## [0.0.5] - 2026-01-20

### Changed

- Preparation for version 0.0.5 release

## [0.0.4] - 2026-01-20

### Changed

- Preparation for version 0.0.4 release

## [0.0.3] - 2026-01-20

### Changed

- Updated dependencies
- Preparation for version 0.0.3 release

## [0.0.2] - 2026-01-20

### Changed

- Preparation for version 0.0.2 release

## [0.0.1] - 2026-01-20

### Added

- Initial commit with base agent framework
- Random Randy agent implementation
- Local tournament runner with parallel execution
- Support for multiple X-Prize graphs (2, 11, 12, 18, 19, 20)
- Simulated annealing and Schrodinger equation adjudicators

[Unreleased]: https://github.com/user/snowdrop-tangled-agents/compare/v0.0.5...HEAD
[0.0.5]: https://github.com/user/snowdrop-tangled-agents/compare/v0.0.4...v0.0.5
[0.0.4]: https://github.com/user/snowdrop-tangled-agents/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/user/snowdrop-tangled-agents/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/user/snowdrop-tangled-agents/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/user/snowdrop-tangled-agents/releases/tag/v0.0.1
