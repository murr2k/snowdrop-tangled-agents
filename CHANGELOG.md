# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
  - `THEORY_OF_OPERATION.md` - Comprehensive system documentation
  - `docs/tangled-bot-v28.txt` - Reference JavaScript bot implementation

### Changed

- Updated `snowdrop_tangled_agents/__init__.py` to export PetersenAgent
- Updated `snowdrop_tangled_agents/agents/__init__.py` to include PetersenAgent
- Updated `pyproject.toml` with new dependencies (playwright, python-dotenv, coloredlogs)

### Fixed

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
