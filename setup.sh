#!/bin/bash
# Setup script for Unix-like systems (macOS, Linux)
# Usage: ./setup.sh [--dev] [--no-browsers]

set -e

echo "=========================================="
echo "  Snowdrop Tangled Agents Setup"
echo "=========================================="

# Check for Python 3.11+
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found. Please install Python 3.11 or later."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python $PYTHON_VERSION"

# Run the Python setup script
python3 setup_env.py "$@"
