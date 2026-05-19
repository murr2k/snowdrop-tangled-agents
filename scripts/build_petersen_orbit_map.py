"""Track 1 — Petersen orbit map under Aut(G) fixing player nodes {5, 7}.

The Tangled score s = influence[p1] - influence[p2] is invariant under any
automorphism of the Petersen graph that fixes the ordered pair (p1=5, p2=7).
This script:

1. Enumerates Aut(Petersen) (S_5, order 120).
2. Filters to the subgroup Stab((5,7)) — automorphisms fixing both vertices.
3. Computes the induced action of each stabilizer element on the 15-edge set.
4. Partitions all 32,768 terminal states into orbits via union-find.
5. Maps the 1,452 observed website-scored boards onto orbits.
6. Reports orbit count, observation coverage, unexplored orbits, and per-orbit
   physical features (G count, 5-cycle frustration, max website score).

Output: data/petersen_orbits.json with full orbit metadata, ready for Track 2
(switchback solver) and Track 3 (LUT harvest target ordering).

Usage:
    poetry run python scripts/build_petersen_orbit_map.py
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from snowdrop_tangled_game_engine import GraphProperties

DB = Path.home() / ".tangled" / "game_stats.db"
OUT_JSON = Path("data/petersen_orbits.json")
N_STATES = 1 << 15  # 32768


def build_petersen_graph() -> tuple[nx.Graph, list[tuple[int, int]], int, int]:
    g_data = GraphProperties().graph_database[5]
    edges = list(g_data["edge_list"])
    p1 = g_data["player1_node"]
    p2 = g_data["player2_node"]
    G = nx.Graph()
    G.add_nodes_from(range(g_data["num_nodes"]))
    G.add_edges_from(edges)
    return G, edges, p1, p2


def enumerate_automorphisms(G: nx.Graph) -> list[dict[int, int]]:
    """Return Aut(G) as a list of vertex-permutation dicts."""
    matcher = GraphMatcher(G, G)
    autos: list[dict[int, int]] = []
    for iso in matcher.isomorphisms_iter():
        autos.append(dict(iso))
    return autos


def stabilizer(autos: list[dict[int, int]], fixed: list[int]) -> list[dict[int, int]]:
    return [s for s in autos if all(s[v] == v for v in fixed)]


def edge_permutation(sigma: dict[int, int], edges: list[tuple[int, int]]) -> tuple[int, ...]:
    """Return tuple π such that edge j maps to edge π[j] under σ."""
    edge_set = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    perm = [-1] * len(edges)
    for j, (a, b) in enumerate(edges):
        sa, sb = sigma[a], sigma[b]
        key = (min(sa, sb), max(sa, sb))
        if key not in edge_set:
            raise RuntimeError(f"σ does not preserve edge {(a, b)} -> {(sa, sb)}")
        perm[j] = edge_set[key]
    return tuple(perm)


def apply_edge_perm(state: int, perm: tuple[int, ...]) -> int:
    """Apply the edge permutation to a state.

    A state encodes edge j's color in bit j (1 = G, 0 = P). Under permutation
    π, the new state has bit π[j] set iff the old state had bit j set.
    """
    out = 0
    s = state
    for j in range(15):
        if s & 1:
            out |= 1 << perm[j]
        s >>= 1
    return out


class UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def state_g_count(state: int) -> int:
    return bin(state).count("1")


def petersen_5cycles(edges: list[tuple[int, int]]) -> list[tuple[int, ...]]:
    G = nx.Graph()
    G.add_edges_from(edges)
    edge_idx = {tuple(sorted(e)): i for i, e in enumerate(edges)}
    cyclesets: set[tuple[int, ...]] = set()
    for cycle in nx.simple_cycles(G, length_bound=5):
        if len(cycle) != 5:
            continue
        c_edges = tuple(
            sorted(edge_idx[(min(cycle[k], cycle[(k+1) % 5]), max(cycle[k], cycle[(k+1) % 5]))]
                   for k in range(5))
        )
        cyclesets.add(c_edges)
    return sorted(cyclesets)


def state_5cycle_frustration(state: int, cycles: list[tuple[int, ...]]) -> int:
    """Count 5-cycles with an odd number of P-edges (frustrated)."""
    n = 0
    for c in cycles:
        afm = sum(1 for e in c if not (state >> e) & 1)  # bit=0 means P
        if afm % 2 == 1:
            n += 1
    return n


def board_to_index(board: str) -> int:
    return sum(1 << j for j, c in enumerate(board) if c == "G")


def index_to_board(idx: int) -> str:
    return "".join("G" if (idx >> j) & 1 else "P" for j in range(15))


def load_observed_terminals(db_path: Path) -> dict[int, tuple[float, int]]:
    """Return {state_index: (mean_website_score, num_observations)}."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT terminal_state, AVG(website_score), COUNT(*) "
        "FROM calibration WHERE terminal_state IS NOT NULL "
        "GROUP BY terminal_state"
    ).fetchall()
    conn.close()
    out: dict[int, tuple[float, int]] = {}
    for state_str, mean_score, n in rows:
        if not state_str or len(state_str) != 15 or any(c not in "GP" for c in state_str):
            continue
        out[board_to_index(state_str)] = (float(mean_score), int(n))
    return out


def main() -> None:
    print("=" * 78)
    print("Track 1 — Petersen orbit map under Aut(G) fixing {5, 7}")
    print("=" * 78)

    G, edges, p1, p2 = build_petersen_graph()
    print(f"\nPetersen graph: {G.number_of_nodes()} vertices, "
          f"{G.number_of_edges()} edges, player nodes ({p1}, {p2})")

    print("\nEnumerating Aut(Petersen)...")
    t0 = time.time()
    autos = enumerate_automorphisms(G)
    print(f"  |Aut(G)| = {len(autos)}  (computed in {time.time()-t0:.2f}s)")
    if len(autos) != 120:
        print(f"  WARNING: expected 120 (=|S_5|), got {len(autos)}")

    stab = stabilizer(autos, [p1, p2])
    print(f"  |Stab({p1}, {p2})| = {len(stab)}")
    print(f"  Index in Aut(G): {len(autos) // len(stab)} (= number of vertex orbits)")

    edge_perms = [edge_permutation(s, edges) for s in stab]
    print(f"\nEdge permutations: {len(edge_perms)} unique-by-vertex-action")
    distinct_eperms = set(edge_perms)
    print(f"  Distinct edge actions: {len(distinct_eperms)}")
    for ep in sorted(distinct_eperms):
        print(f"    {ep}")

    print("\nComputing orbits of {G,P}^15 via union-find...")
    t0 = time.time()
    uf = UnionFind(N_STATES)
    for state in range(N_STATES):
        for perm in distinct_eperms:
            uf.union(state, apply_edge_perm(state, perm))
    print(f"  done in {time.time()-t0:.1f}s")

    roots: dict[int, list[int]] = defaultdict(list)
    for s in range(N_STATES):
        roots[uf.find(s)].append(s)
    orbit_sizes = sorted((len(v) for v in roots.values()), reverse=True)
    n_orbits = len(roots)
    print(f"\nTotal orbits: {n_orbits}")
    print(f"  Orbit-size distribution (descending, first 30): {orbit_sizes[:30]}")
    print(f"  Largest orbit: {orbit_sizes[0]}")
    print(f"  Smallest orbit: {orbit_sizes[-1]}")
    sum_sizes = sum(orbit_sizes)
    assert sum_sizes == N_STATES, f"orbit size sum {sum_sizes} != {N_STATES}"

    # Burnside check: |orbits| should equal (1/|G|) * sum over g of |Fix(g)|.
    # Here |G| = number of distinct edge actions; Fix(g) is the number of
    # states fixed by g.
    g_size = len(distinct_eperms)
    fix_sum = 0
    for perm in distinct_eperms:
        cnt = 0
        for state in range(N_STATES):
            if apply_edge_perm(state, perm) == state:
                cnt += 1
        fix_sum += cnt
    burnside = fix_sum / g_size
    print(f"\nBurnside check: (1/{g_size}) * Σ|Fix| = {burnside:.4f} (expected {n_orbits})")
    assert abs(burnside - n_orbits) < 1e-6

    print("\nLoading observed terminals from calibration table...")
    observed = load_observed_terminals(DB)
    print(f"  {len(observed)} distinct boards observed on website")

    print("\nComputing 5-cycle list (for frustration metric)...")
    cycles = petersen_5cycles(edges)
    print(f"  {len(cycles)} five-cycles")
    assert len(cycles) == 12, f"expected 12 five-cycles, got {len(cycles)}"

    print("\nBuilding per-orbit records...")
    # Pick canonical representative = lexicographically-smallest state in orbit.
    orbit_records = []
    for root, states in roots.items():
        rep = min(states)
        obs_in_orbit = [(s, *observed[s]) for s in states if s in observed]
        if obs_in_orbit:
            scores = [o[1] for o in obs_in_orbit]
            total_obs = sum(o[2] for o in obs_in_orbit)
            mean_score = float(np.mean(scores))
            max_score = float(np.max(scores))
            min_score = float(np.min(scores))
            n_states_obs = len(obs_in_orbit)
        else:
            scores, total_obs, mean_score, max_score, min_score, n_states_obs = [], 0, None, None, None, 0
        orbit_records.append({
            "rep_state": rep,
            "rep_board": index_to_board(rep),
            "orbit_size": len(states),
            "g_count": state_g_count(rep),
            "frustration": state_5cycle_frustration(rep, cycles),
            "n_states_observed": n_states_obs,
            "total_games": total_obs,
            "mean_website_score": mean_score,
            "max_website_score": max_score,
            "min_website_score": min_score,
            "all_states": states,
        })

    # Sort: most observation first (just for readability)
    orbit_records.sort(key=lambda r: (-r["total_games"], r["rep_state"]))

    explored_orbits = sum(1 for r in orbit_records if r["total_games"] > 0)
    unexplored = sum(1 for r in orbit_records if r["total_games"] == 0)
    print(f"\n--- Orbit coverage ---")
    print(f"  total orbits: {n_orbits}")
    print(f"  observed at least once: {explored_orbits} ({100*explored_orbits/n_orbits:.1f}%)")
    print(f"  unexplored: {unexplored} ({100*unexplored/n_orbits:.1f}%)")
    if explored_orbits:
        games_dist = [r["total_games"] for r in orbit_records if r["total_games"] > 0]
        print(f"  games-per-orbit: mean={np.mean(games_dist):.1f}, "
              f"median={np.median(games_dist):.1f}, max={max(games_dist)}")

    print("\n--- Best observed orbits (top 15 by mean score) ---")
    obs_only = [r for r in orbit_records if r["total_games"] > 0]
    obs_only.sort(key=lambda r: -r["mean_website_score"])
    print(f"  {'mean_score':>10s}  {'max_score':>9s}  {'min_score':>9s}  {'G':>2s}  {'frust':>5s}  {'games':>5s}  {'orbit_size':>10s}  board")
    for r in obs_only[:15]:
        print(f"  {r['mean_website_score']:>+10.4f}  {r['max_website_score']:>+9.4f}  "
              f"{r['min_website_score']:>+9.4f}  {r['g_count']:>2d}  {r['frustration']:>5d}  "
              f"{r['total_games']:>5d}  {r['orbit_size']:>10d}  {r['rep_board']}")

    print("\n--- Frustration distribution across all orbits ---")
    from collections import Counter
    frust_counts = Counter(r["frustration"] for r in orbit_records)
    for f in sorted(frust_counts):
        print(f"  frust={f}: {frust_counts[f]} orbits "
              f"({sum(r['orbit_size'] for r in orbit_records if r['frustration']==f)} states)")

    print("\n--- G-count distribution across all orbits ---")
    g_counts = Counter(r["g_count"] for r in orbit_records)
    for g in sorted(g_counts):
        print(f"  G={g}: {g_counts[g]} orbits "
              f"({sum(r['orbit_size'] for r in orbit_records if r['g_count']==g)} states)")

    # Save
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "graph_id": 5,
        "graph_name": "petersen",
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "edge_list": edges,
        "p1_node": p1,
        "p2_node": p2,
        "aut_size": len(autos),
        "stabilizer_size": len(stab),
        "distinct_edge_actions": len(distinct_eperms),
        "edge_perms": [list(p) for p in sorted(distinct_eperms)],
        "n_states": N_STATES,
        "n_orbits": n_orbits,
        "explored_orbits": explored_orbits,
        "unexplored_orbits": unexplored,
        "orbits": orbit_records,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nWrote {OUT_JSON}")
    print(f"File size: {OUT_JSON.stat().st_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
