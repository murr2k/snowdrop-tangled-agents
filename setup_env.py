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


def install_with_pip(dev: bool = False):
    """Install dependencies using pip."""
    print_step("Installing dependencies with pip...")

    # First, ensure pip and build tools are up to date
    print_step("Upgrading pip and build tools...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        capture_output=True  # Suppress output for this step
    )

    # Core dependencies (must match pyproject.toml)
    packages = [
        "snowdrop-tangled-game-engine>=1.1.0",
        "snowdrop-adjudicators>=0.1.0",
        "scipy>=1.16.2",
        "coloredlogs>=15.0.1",
        "playwright>=1.40.0",
        "python-dotenv>=1.0.0",
        "websocket-client>=1.6.0",
    ]

    if dev:
        packages.extend([
            "pytest>=8.4.1",
            "pytest-cov>=7.0.0",
        ])

    # Install packages
    print_step(f"Installing {len(packages)} core packages...")
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + packages
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print_error("pip install failed")
        return False

    # Install the package itself in editable mode
    print_step("Installing snowdrop-tangled-agents in editable mode...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", "."],
        capture_output=False
    )

    if result.returncode != 0:
        print_warning("Editable install failed (non-critical)")

    print_success("Dependencies installed with pip")
    return True


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
# Copy this file to .env and fill in your credentials

# tangled-game.com login credentials (required for web play)
TANGLED_EMAIL=your_email@example.com
TANGLED_PASSWORD=your_password

# Optional: anthropic API key for Claude-based agents
# ANTHROPIC_API_KEY=your_api_key
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
    parser.add_argument(
        "--pip",
        action="store_true",
        help="Force pip installation (skip Poetry)"
    )
    args = parser.parse_args()

    print_header("Snowdrop Tangled Agents Setup")

    # Check Python version
    check_python_version()

    # Install dependencies
    if not args.pip and check_poetry():
        print_success("Poetry found")
        if not install_with_poetry(dev=args.dev):
            print_warning("Poetry failed, falling back to pip")
            if not install_with_pip(dev=args.dev):
                sys.exit(1)
    else:
        if not args.pip:
            print_warning("Poetry not found, using pip")
        if not install_with_pip(dev=args.dev):
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
  2. Run a test game: python play_tangled.py --games 1
  3. View statistics: python play_tangled.py --stats
  4. View calibration: python play_tangled.py --calibration

For development:
  - Run tests: pytest
  - Run tournament: python snowdrop_tangled_agents/playing_games/run_local_parallel_tournament.py
""")
    else:
        print_header("Setup Incomplete")
        print("Some verification steps failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
