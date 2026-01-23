"""
Version and policy ID utilities.

Provides functions to identify the current code version for tracking
which policy generated game results.
"""

import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_policy_id() -> str:
    """Get current policy ID from git tag or commit.

    Priority:
    1. Git tag if HEAD is tagged (e.g., 'v0.6.0-bayesian-oracle')
    2. Short commit hash (e.g., '961779c')
    3. 'unknown' if not in a git repo

    Returns:
        Policy identifier string
    """
    try:
        # Get the repo root (where .git is)
        repo_root = Path(__file__).parent.parent.parent

        # Check if HEAD has a tag
        result = subprocess.run(
            ['git', 'describe', '--tags', '--exact-match'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_root
        )
        if result.returncode == 0:
            return result.stdout.strip()

        # Fall back to short commit
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=repo_root
        )
        if result.returncode == 0:
            return result.stdout.strip()

    except Exception:
        pass

    return 'unknown'


@lru_cache(maxsize=1)
def get_full_version_info() -> dict:
    """Get detailed version information.

    Returns:
        Dict with commit, tag, branch, and dirty status
    """
    info = {
        'policy_id': get_policy_id(),
        'commit': None,
        'tag': None,
        'branch': None,
        'dirty': False,
    }

    try:
        repo_root = Path(__file__).parent.parent.parent

        # Full commit hash
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=5, cwd=repo_root
        )
        if result.returncode == 0:
            info['commit'] = result.stdout.strip()

        # Current branch
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True, text=True, timeout=5, cwd=repo_root
        )
        if result.returncode == 0:
            info['branch'] = result.stdout.strip()

        # Check for tag
        result = subprocess.run(
            ['git', 'describe', '--tags', '--exact-match'],
            capture_output=True, text=True, timeout=5, cwd=repo_root
        )
        if result.returncode == 0:
            info['tag'] = result.stdout.strip()

        # Check dirty status
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, timeout=5, cwd=repo_root
        )
        if result.returncode == 0:
            info['dirty'] = len(result.stdout.strip()) > 0

    except Exception:
        pass

    return info
