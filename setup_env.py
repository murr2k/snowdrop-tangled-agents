#!/usr/bin/env python3
"""
Cross-platform setup script for snowdrop-tangled-agents.

This script automates the installation of dependencies and configuration
for Windows, macOS, and Linux systems.

Usage:
    python setup_env.py [--no-browsers] [--dev]

Options:
    --no-browsers   Skip Playwright browser installation
    --dev           Install development dependencies (pytest, etc.)

Requires Poetry (https://python-poetry.org/).  If not found, the script
will attempt to install it via pipx or pip.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
import argparse


def print_header(msg: str):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def print_step(msg: str):
    """Print a step message."""
    print(f"[*] {msg}")


def print_success(msg: str):
    """Print a success message."""
    print(f"[+] {msg}")


def print_error(msg: str):
    """Print an error message."""
    print(f"[!] ERROR: {msg}", file=sys.stderr)


def print_warning(msg: str):
    """Print a warning message."""
    print(f"[~] WARNING: {msg}")


def check_python_version():
    """Check that Python version meets requirements."""
    print_step("Checking Python version...")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 11):
        print_error(f"Python 3.11+ required, found {version.major}.{version.minor}")
        sys.exit(1)

    if version.major == 3 and version.minor >= 14:
        print_error(f"Python <3.14 required, found {version.major}.{version.minor}")
        sys.exit(1)

    print_success(f"Python {version.major}.{version.minor}.{version.micro} OK")


def check_poetry():
    """Check if Poetry is installed."""
    return shutil.which("poetry") is not None


def install_with_poetry(dev: bool = False):
    """Install dependencies using Poetry."""
    print_step("Installing dependencies with Poetry...")

    cmd = ["poetry", "install"]
    if not dev:
        cmd.append("--without=dev")

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print_error("Poetry install failed")
        return False

    print_success("Dependencies installed with Poetry")
    return True


def install_poetry():
    """Install Poetry using pipx (or pip as fallback)."""
    print_step("Poetry not found. Installing Poetry...")

    # Try pipx first (recommended)
    pipx = shutil.which("pipx")
    if pipx:
        result = subprocess.run(["pipx", "install", "poetry"], capture_output=False)
        if result.returncode == 0:
            print_success("Poetry installed via pipx")
            return True

    # Fallback: pip install
    print_step("pipx not available, installing Poetry via pip...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "poetry"],
        capture_output=False
    )
    if result.returncode == 0:
        print_success("Poetry installed via pip")
        return True

    print_error(
        "Could not install Poetry automatically.\n"
        "  Install manually: https://python-poetry.org/docs/#installation\n"
        "  Or run: pipx install poetry"
    )
    return False


def install_playwright_browsers():
    """Install Playwright browsers."""
    print_step("Installing Playwright browsers (this may take a few minutes)...")

    # Try with playwright command first
    playwright_cmd = shutil.which("playwright")
    if playwright_cmd:
        result = subprocess.run(["playwright", "install", "chromium"], capture_output=False)
    else:
        # Fall back to python -m playwright
        result = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium"],
            capture_output=False
        )

    if result.returncode != 0:
        print_error("Playwright browser installation failed")
        print_warning("You can install browsers manually with: playwright install chromium")
        return False

    print_success("Playwright browsers installed")
    return True


def setup_database_directory():
    """Create the database directory."""
    print_step("Setting up database directory...")

    db_dir = Path.home() / ".tangled"
    db_dir.mkdir(parents=True, exist_ok=True)

    print_success(f"Database directory ready: {db_dir}")
    return True


def create_env_template():
    """Create a .env template file if it doesn't exist."""
    print_step("Checking for .env file...")

    env_file = Path(".env")
    env_example = Path(".env.example")

    if env_file.exists():
        print_success(".env file already exists")
        return True

    template = """# Tangled Game Configuration
# Copy this file to .env and fill in your values.

# tangled-game.com credentials (required for web play)
# Sign up at https://tangled-game.com to create an account.
TANGLED_USERNAME=your_email@example.com
TANGLED_PASSWORD=your_password

# Live stats dashboard (optional)
# Deploy your own from: https://github.com/murr2k/tangled-workspace/tree/main/tangled-stats-dashboard
# TANGLED_DASHBOARD_URL=wss://your-dashboard.fly.dev/ws/publish
# TANGLED_DASHBOARD_API_KEY=your_api_key
"""

    env_example.write_text(template)
    print_success(f"Created {env_example} - copy to .env and add your credentials")
    return True


def verify_installation():
    """Verify the installation works."""
    print_step("Verifying installation...")

    try:
        # Test critical imports
        print_step("Testing imports...")
        import snowdrop_tangled_game_engine
        import snowdrop_adjudicators
        import playwright
        import coloredlogs
        import dotenv
        import websocket

        print_success("  ✓ snowdrop-tangled-game-engine")
        print_success("  ✓ snowdrop-adjudicators")
        print_success("  ✓ playwright")
        print_success("  ✓ coloredlogs")
        print_success("  ✓ python-dotenv")
        print_success("  ✓ websocket-client")

        # Test our package
        print_step("Testing package imports...")
        from snowdrop_tangled_agents.strategy.mcts_strategy import evaluate_terminal_state

        # Quick sanity check
        score = evaluate_terminal_state("G" * 15)

        print_success("All imports successful")
        print_success(f"Terminal evaluation test: {score:.4f}")
        return True

    except ImportError as e:
        print_error(f"Import failed: {e}")
        print_error("Try running: pip install coloredlogs playwright python-dotenv websocket-client")
        return False
    except Exception as e:
        print_error(f"Verification failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Setup snowdrop-tangled-agents development environment"
    )
    parser.add_argument(
        "--no-browsers",
        action="store_true",
        help="Skip Playwright browser installation"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install development dependencies"
    )
    args = parser.parse_args()

    print_header("Snowdrop Tangled Agents Setup")

    # Check Python version
    check_python_version()

    # Ensure Poetry is available
    if not check_poetry():
        if not install_poetry():
            sys.exit(1)
    else:
        print_success("Poetry found")

    # Install dependencies with Poetry (single source of truth: pyproject.toml + poetry.lock)
    if not install_with_poetry(dev=args.dev):
        print_error("Poetry install failed")
        sys.exit(1)

    # Install Playwright browsers
    if not args.no_browsers:
        install_playwright_browsers()
    else:
        print_warning("Skipping Playwright browser installation")

    # Setup database directory
    setup_database_directory()

    # Create .env template
    create_env_template()

    # Verify installation
    print_header("Verifying Installation")
    if verify_installation():
        print_header("Setup Complete!")
        print("""
Next steps:
  1. Copy .env.example to .env and add your tangled-game.com credentials
     (sign up at https://tangled-game.com)
  2. Install MATLAB Engine API (see README → MATLAB Setup)
  3. Verify MATLAB: poetry run python test_matlab_detection.py
  4. Run a test game:
       poetry run python play_tangled.py --strategy alphaq_explorer \\
         --opponent alphaq --games 1 --mcts-iterations 5000
  5. View statistics: poetry run python play_tangled.py --stats

For development:
  - Run tests: poetry run pytest
  - Run tournament: poetry run python snowdrop_tangled_agents/playing_games/run_local_parallel_tournament.py
""")
    else:
        print_header("Setup Incomplete")
        print("Some verification steps failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
