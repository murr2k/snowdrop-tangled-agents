# Policy ID Tracking Plan

## Goal

Track which version of the player code generated each game result, enabling analysis like "did v0.6.0 perform better than v0.5.0?"

## Motivation

Currently the database records `strategy` (e.g., "hybrid_solver") but not which version of that strategy's code was running. When we make code changes (not just training data changes), we need to distinguish results from different implementations.

| Field | Tracks | Changes when... |
|-------|--------|-----------------|
| `strategy` | Algorithm name | Different strategy selected |
| `policy_id` | Code version | Code changes (git commit/tag) |
| `model_version` | Trained weights | Model retrained (future) |

## Schema Change

```sql
-- Migration v7
ALTER TABLE games ADD COLUMN policy_id TEXT;
CREATE INDEX idx_games_policy_id ON games(policy_id);
```

## Policy ID Format

**Priority order:**
1. Git tag (if HEAD is tagged): `v0.6.0-bayesian-oracle`
2. Short commit hash: `961779c`
3. Fallback: `unknown`

**Examples:**
- `v0.6.0-bayesian-oracle` - tagged release
- `961779c` - untagged commit
- `961779c-dirty` - uncommitted changes (optional)

## Implementation

### 1. New utility: `snowdrop_tangled_agents/utils/version.py`

```python
import subprocess
from functools import lru_cache

@lru_cache(maxsize=1)
def get_policy_id() -> str:
    """Get current policy ID from git tag or commit.

    Returns:
        Git tag if HEAD is tagged, otherwise short commit hash.
        Falls back to 'unknown' if not in a git repo.
    """
    try:
        # Check if HEAD has a tag
        result = subprocess.run(
            ['git', 'describe', '--tags', '--exact-match'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()

        # Fall back to short commit
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return 'unknown'
```

### 2. Update `StatsCollector.start_game()`

```python
def start_game(self, opponent: str, strategy: str = None,
               policy_id: str = None, **kwargs) -> str:
    """Start recording a new game.

    Args:
        opponent: Opponent name
        strategy: Strategy name (e.g., 'hybrid_solver')
        policy_id: Code version identifier (auto-detected if None)
    """
    if policy_id is None:
        from snowdrop_tangled_agents.utils.version import get_policy_id
        policy_id = get_policy_id()

    # ... existing code, add policy_id to INSERT
```

### 3. Update `play_tangled.py`

```python
# At startup, fetch once
from snowdrop_tangled_agents.utils.version import get_policy_id
policy_id = get_policy_id()
logger.info(f"Policy ID: {policy_id}")

# Pass to start_game
self.current_game_id = self.stats_collector.start_game(
    opponent=opponent,
    strategy=self.strategy_name,
    policy_id=policy_id,
    # ...
)
```

## Migration Safety

- Migration only runs when collector is initialized
- `ALTER TABLE ADD COLUMN` is safe for existing data
- Existing rows get `NULL` policy_id (acceptable for historical data)
- No downtime required

## Rollout Steps

1. **Wait** for current 500-game run to complete
2. **Implement** version.py utility
3. **Add** migration v7 to migrations.py
4. **Update** StatsCollector.start_game()
5. **Update** play_tangled.py to pass policy_id
6. **Test** with a few games
7. **Optionally backfill** existing games:
   - Games before `f142fe5`: `pre-opponent-modeling`
   - Games from `f142fe5` to `961779c`: `v0.6.0-bayesian-oracle` (or leave NULL)

## Analysis Queries

Once implemented, we can query by policy:

```sql
-- Win rate by policy version
SELECT
    policy_id,
    COUNT(*) as games,
    SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) as wins,
    ROUND(100.0 * SUM(CASE WHEN result = 'win' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_pct
FROM games
WHERE result IS NOT NULL
GROUP BY policy_id
ORDER BY MIN(timestamp);
```

```python
# In queries.py
def get_performance_by_policy(db_path=None) -> list[dict]:
    """Compare win rates across policy versions."""
    # ...
```

## Future Considerations

- **Model versioning**: When we have trained models, add `model_id` column
- **Dirty detection**: Optionally append `-dirty` if uncommitted changes exist
- **Config hashing**: For strategies with tunable parameters, hash the config
