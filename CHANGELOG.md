# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

### Changed

- Moved `THEORY_OF_OPERATION.md` from project root to `docs/` directory
- Updated `README.md` with Development Progress section documenting all implementation steps
- Extended `StatsCollector` with model and opponent management methods
- Added migration support to stats collector initialization

- Updated `snowdrop_tangled_agents/__init__.py` to export PetersenAgent
- Updated `snowdrop_tangled_agents/agents/__init__.py` to include PetersenAgent
- Updated `pyproject.toml` with new dependencies (playwright, python-dotenv, coloredlogs)

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
