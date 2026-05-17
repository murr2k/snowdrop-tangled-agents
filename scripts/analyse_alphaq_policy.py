"""
Investigation 2: AlphaQ Policy Analysis

Reads the AlphaQ move corpus from the local game DB and computes the
information-theoretic structure of AlphaQ's policy. Output is a decision
input for whether classical exploit of AlphaQ is plausible.

Per the project plan, this script:

  1. Extracts (state_before, alphaq_move) pairs from the moves table
  2. Computes per-state response entropy H(pi_AlphaQ(.|s))
  3. Computes mutual information I(pi_AlphaQ; S) with Miller-Madow bias
     correction, both globally and stratified by grey count
  4. Identifies high-entropy states with sufficient observations
     (exploit candidates)
  5. Identifies low-observation states where AlphaQ has changed its move
     (decision-boundary candidates)
  6. Generates plots: response entropy histogram, MI vs grey count,
     per-edge response distribution
  7. Writes findings to docs/INVESTIGATION_2_RESULTS.md

Usage:
    poetry run python scripts/analyse_alphaq_policy.py
"""

import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# UTF-8 output so any special chars don't crash on Windows consoles
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = Path.home() / ".tangled" / "game_stats.db"
PLOT_DIR = PROJECT_ROOT / "plots"
REPORT_PATH = PROJECT_ROOT / "docs" / "INVESTIGATION_2_RESULTS.md"

INITIAL_STATE = "-" * 15  # all-grey board (Petersen graph has 15 edges)

# Thresholds (per project plan)
EXPLOIT_MIN_OBSERVATIONS = 10
EXPLOIT_MIN_ENTROPY_BITS = 0.5
DECISION_BOUNDARY_MAX_OBSERVATIONS = 6


# -------------------------------------------------------------------------
# Data extraction
# -------------------------------------------------------------------------
def extract_alphaq_decisions(conn: sqlite3.Connection) -> list[dict]:
    """
    Return a list of dicts: each is one AlphaQ decision observation.

    Fields:
      state_before : 15-char board state AlphaQ saw before moving
      edge         : edge index (0-14) AlphaQ played
      color        : 'G' or 'P'
      grey_before  : number of grey edges before AlphaQ's move
      game_id      : provenance
      move_number  : provenance

    AlphaQ's state_before is the state_after of the row immediately
    preceding it within the same game (or INITIAL_STATE if it moved first).
    """
    rows = conn.execute("""
        SELECT g.id, m.rowid, m.move_number, m.player, m.edge, m.color, m.state_after
        FROM moves m
        JOIN games g ON g.id = m.game_id
        WHERE g.opponent = 'alphaq'
          AND g.result IS NOT NULL
          AND m.edge IS NOT NULL
          AND m.state_after IS NOT NULL
        ORDER BY g.id, m.rowid
    """).fetchall()

    decisions = []
    current_game = None
    prev_state = INITIAL_STATE

    for game_id, rowid, mv_num, player, edge, color, state_after in rows:
        if game_id != current_game:
            current_game = game_id
            prev_state = INITIAL_STATE

        if player == "opponent":
            grey_before = prev_state.count("-")
            decisions.append({
                "state_before": prev_state,
                "edge": int(edge),
                "color": color,
                "grey_before": grey_before,
                "game_id": game_id,
                "move_number": mv_num,
            })
        prev_state = state_after

    return decisions


# -------------------------------------------------------------------------
# Aggregation
# -------------------------------------------------------------------------
def aggregate_by_state(decisions: list[dict]) -> dict[str, Counter]:
    """state_before -> Counter of (edge, color) action tuples."""
    by_state: dict[str, Counter] = defaultdict(Counter)
    for d in decisions:
        by_state[d["state_before"]][(d["edge"], d["color"])] += 1
    return by_state


# -------------------------------------------------------------------------
# Information theory
# -------------------------------------------------------------------------
def entropy_bits(counts: Counter) -> float:
    """Shannon entropy in bits from an unnormalised count Counter."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


def miller_madow_correction(k: int, n: int) -> float:
    """Bias correction term for plug-in entropy: + (K - 1) / (2 N) in nats; convert to bits."""
    if n <= 0:
        return 0.0
    return (k - 1) / (2.0 * n) / math.log(2.0)


def mutual_information(by_state: dict[str, Counter]) -> dict:
    """
    Compute I(A; S) where S is state_before and A is (edge, color).
    Uses plug-in estimator and Miller-Madow bias correction.
    """
    state_totals: Counter = Counter()
    action_totals: Counter = Counter()
    joint_total = 0

    for state, action_counts in by_state.items():
        n_state = sum(action_counts.values())
        state_totals[state] = n_state
        joint_total += n_state
        for action, c in action_counts.items():
            action_totals[action] += c

    if joint_total == 0:
        return {"plugin_bits": 0.0, "mm_bits": 0.0, "n_samples": 0}

    h_s = entropy_bits(state_totals)
    h_a = entropy_bits(action_totals)

    h_joint = 0.0
    n_joint_cells = 0
    for state, action_counts in by_state.items():
        for c in action_counts.values():
            if c > 0:
                p = c / joint_total
                h_joint -= p * math.log2(p)
                n_joint_cells += 1

    mi_plugin = h_s + h_a - h_joint

    mi_mm = mi_plugin - (
        miller_madow_correction(len(state_totals), joint_total)
        + miller_madow_correction(len(action_totals), joint_total)
        - miller_madow_correction(n_joint_cells, joint_total)
    )

    return {
        "plugin_bits": mi_plugin,
        "mm_bits": mi_mm,
        "h_states_bits": h_s,
        "h_actions_bits": h_a,
        "n_samples": joint_total,
        "n_distinct_states": len(state_totals),
        "n_distinct_actions": len(action_totals),
        "n_joint_cells": n_joint_cells,
    }


# -------------------------------------------------------------------------
# Candidate identification
# -------------------------------------------------------------------------
def per_state_summary(by_state: dict[str, Counter]) -> list[dict]:
    """One row per state with n, entropy, distinct responses, top response."""
    summary = []
    for state, counts in by_state.items():
        n = sum(counts.values())
        h = entropy_bits(counts)
        n_distinct = len(counts)
        top_action, top_count = counts.most_common(1)[0]
        summary.append({
            "state": state,
            "grey": state.count("-"),
            "n": n,
            "entropy_bits": h,
            "n_distinct_responses": n_distinct,
            "top_action": top_action,
            "top_fraction": top_count / n,
        })
    return summary


def find_exploit_candidates(summary: list[dict]) -> list[dict]:
    """High-entropy, sufficiently-observed states."""
    cands = [
        s for s in summary
        if s["n"] >= EXPLOIT_MIN_OBSERVATIONS
        and s["entropy_bits"] >= EXPLOIT_MIN_ENTROPY_BITS
    ]
    return sorted(cands, key=lambda s: (-s["entropy_bits"], -s["n"]))


def find_decision_boundary_candidates(summary: list[dict]) -> list[dict]:
    """Low-observation states where AlphaQ's response varied."""
    cands = [
        s for s in summary
        if s["n"] <= DECISION_BOUNDARY_MAX_OBSERVATIONS
        and s["n"] >= 2
        and s["n_distinct_responses"] >= 2
    ]
    return sorted(cands, key=lambda s: (-s["n_distinct_responses"], -s["n"]))


# -------------------------------------------------------------------------
# Plots
# -------------------------------------------------------------------------
def plot_entropy_histogram(summary: list[dict], out_path: Path) -> None:
    eligible = [s for s in summary if s["n"] >= 3]
    if not eligible:
        return
    entropies = [s["entropy_bits"] for s in eligible]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(entropies, bins=30, color="#4c72b0", edgecolor="black", alpha=0.8)
    ax.axvline(EXPLOIT_MIN_ENTROPY_BITS, color="red", linestyle="--",
               label=f"exploit threshold ({EXPLOIT_MIN_ENTROPY_BITS} bits)")
    ax.set_xlabel("Per-state response entropy H(pi|s) [bits]")
    ax.set_ylabel("Number of states (n>=3 observations)")
    ax.set_title(f"AlphaQ per-state response entropy ({len(eligible)} states)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_entropy_vs_grey(summary: list[dict], out_path: Path) -> None:
    """For each grey count, plot mean and 5/95 percentile of per-state entropy."""
    eligible = [s for s in summary if s["n"] >= 3]
    if not eligible:
        return
    by_grey: dict[int, list[float]] = defaultdict(list)
    for s in eligible:
        by_grey[s["grey"]].append(s["entropy_bits"])
    greys = sorted(by_grey.keys())
    means = [np.mean(by_grey[g]) for g in greys]
    p05 = [np.percentile(by_grey[g], 5) for g in greys]
    p95 = [np.percentile(by_grey[g], 95) for g in greys]
    n_states = [len(by_grey[g]) for g in greys]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(greys, p05, p95, alpha=0.25, label="5-95 percentile")
    ax.plot(greys, means, marker="o", color="#4c72b0", label="mean entropy")
    ax.set_xlabel("Grey edges remaining (game phase)")
    ax.set_ylabel("Per-state entropy [bits]")
    ax.set_title("AlphaQ response entropy vs game phase")
    for x, n in zip(greys, n_states):
        ax.annotate(str(n), (x, 0), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8, color="grey")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_per_edge_color_distribution(decisions: list[dict], out_path: Path) -> None:
    edge_color = np.zeros((15, 2), dtype=int)  # cols: G, P
    for d in decisions:
        e = d["edge"]
        c = 0 if d["color"] == "G" else 1
        edge_color[e, c] += 1
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(15)
    ax.bar(x - 0.18, edge_color[:, 0], width=0.36, color="#2ca02c", label="Green")
    ax.bar(x + 0.18, edge_color[:, 1], width=0.36, color="#7a3e9d", label="Purple")
    ax.set_xticks(x)
    ax.set_xticklabels([f"E{i}" for i in range(15)], fontsize=8)
    ax.set_xlabel("Edge")
    ax.set_ylabel("AlphaQ play count")
    ax.set_title("AlphaQ aggregate edge-and-color preference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# -------------------------------------------------------------------------
# Report writer
# -------------------------------------------------------------------------
def write_report(decisions, by_state, summary, mi, exploit, boundary,
                 mi_by_grey, deterministic_stats) -> None:
    n_games = len({d["game_id"] for d in decisions})
    eligible = [s for s in summary if s["n"] >= 3]
    n_eligible = len(eligible)
    entropies = np.array([s["entropy_bits"] for s in eligible]) if eligible else np.array([])

    lines: list[str] = []
    push = lines.append

    push("# Investigation 2 Results: AlphaQ Policy Analysis")
    push("")
    push("**Source corpus:** local `~/.tangled/game_stats.db` (read-only)")
    push("**Method:** information-theoretic analysis of the empirical "
         "AlphaQ move distribution, conditional on observed board state.")
    push("")
    push("---")
    push("")
    push("## Corpus summary")
    push("")
    push("| Metric | Value |")
    push("|--------|-------|")
    push(f"| AlphaQ games (completed) | {n_games} |")
    push(f"| AlphaQ decisions observed | {len(decisions)} |")
    push(f"| Distinct states observed | {len(by_state)} |")
    push(f"| States with n >= 3 observations | {n_eligible} |")
    push("")
    push("---")
    push("")
    push("## Per-state response entropy")
    push("")
    if eligible:
        push("| Statistic | Value (bits) |")
        push("|-----------|--------------|")
        push(f"| Mean | {entropies.mean():.4f} |")
        push(f"| Median | {np.median(entropies):.4f} |")
        push(f"| Std dev | {entropies.std():.4f} |")
        push(f"| 5th percentile | {np.percentile(entropies, 5):.4f} |")
        push(f"| 95th percentile | {np.percentile(entropies, 95):.4f} |")
        push(f"| Max | {entropies.max():.4f} |")
        push("")
        push(f"**Deterministic states (H = 0):** "
             f"{deterministic_stats['n_zero']} of {n_eligible} "
             f"({100 * deterministic_stats['frac_zero']:.1f}%)")
        push("")
        push(f"**High-entropy states (H >= {EXPLOIT_MIN_ENTROPY_BITS} bits):** "
             f"{deterministic_stats['n_high']} of {n_eligible} "
             f"({100 * deterministic_stats['frac_high']:.1f}%)")
        push("")
        push("![Entropy histogram](../plots/investigation2_entropy_histogram.png)")
        push("")
        push("![Entropy vs grey count](../plots/investigation2_entropy_vs_grey.png)")
    else:
        push("_No states with n >= 3 observations._")
    push("")
    push("---")
    push("")
    push("## Mutual information I(pi_AlphaQ; S)")
    push("")
    push("| Quantity | Value |")
    push("|----------|-------|")
    push(f"| H(states) [bits] | {mi['h_states_bits']:.4f} |")
    push(f"| H(actions) [bits] | {mi['h_actions_bits']:.4f} |")
    push(f"| MI plug-in [bits] | {mi['plugin_bits']:.4f} |")
    push(f"| MI Miller-Madow corrected [bits] | {mi['mm_bits']:.4f} |")
    push(f"| Distinct (state, action) cells | {mi['n_joint_cells']} |")
    push("")
    push("**Interpretation:** higher MI means AlphaQ's move is more "
         "informative about the board state (more state-dependent). MI is "
         "biased upward by sparsity; the Miller-Madow correction subtracts "
         "the leading bias term.")
    push("")
    push("### MI stratified by grey count")
    push("")
    push("| Grey edges | N observations | N states | MI plug-in (bits) | MI MM-corrected (bits) | Mean entropy (bits) |")
    push("|------------|----------------|----------|-------------------|------------------------|---------------------|")
    for g in sorted(mi_by_grey.keys()):
        row = mi_by_grey[g]
        push(f"| {g} | {row['n_samples']} | {row['n_distinct_states']} | "
             f"{row['plugin_bits']:.4f} | {row['mm_bits']:.4f} | "
             f"{row['mean_entropy']:.4f} |")
    push("")
    push("---")
    push("")
    push("## Exploit candidates")
    push("")
    push(f"States with n >= {EXPLOIT_MIN_OBSERVATIONS} observations and "
         f"H >= {EXPLOIT_MIN_ENTROPY_BITS} bits of response entropy. "
         "These are positions where AlphaQ has been observed enough times "
         "to estimate its response distribution and where that distribution "
         "is not strongly peaked on a single action.")
    push("")
    if exploit:
        push(f"**{len(exploit)} exploit candidate state(s) found.**")
        push("")
        push("| Rank | State (E0..E14) | Grey | N obs | H (bits) | Distinct responses | Top response | Top fraction |")
        push("|------|-----------------|------|-------|----------|--------------------|--------------|--------------|")
        for i, s in enumerate(exploit[:25], 1):
            edge, color = s["top_action"]
            push(f"| {i} | `{s['state']}` | {s['grey']} | {s['n']} | "
                 f"{s['entropy_bits']:.3f} | {s['n_distinct_responses']} | "
                 f"E{edge}{color} | {s['top_fraction']:.2f} |")
        if len(exploit) > 25:
            push("")
            push(f"_(showing top 25 of {len(exploit)})_")
    else:
        push("**No exploit candidates found.** AlphaQ's response distribution "
             "is sharply peaked on a single action in every sufficiently-"
             "observed state.")
    push("")
    push("---")
    push("")
    push("## Decision-boundary candidates")
    push("")
    push(f"States with 2 <= n <= {DECISION_BOUNDARY_MAX_OBSERVATIONS} "
         "observations where AlphaQ's response has differed across "
         "observations. These are low-confidence regions where AlphaQ "
         "may be near a policy decision boundary.")
    push("")
    if boundary:
        push(f"**{len(boundary)} decision-boundary candidate state(s) found.**")
        push("")
        push("| Rank | State | Grey | N obs | Distinct responses |")
        push("|------|-------|------|-------|--------------------|")
        for i, s in enumerate(boundary[:25], 1):
            push(f"| {i} | `{s['state']}` | {s['grey']} | {s['n']} | "
                 f"{s['n_distinct_responses']} |")
        if len(boundary) > 25:
            push("")
            push(f"_(showing top 25 of {len(boundary)})_")
    else:
        push("**No decision-boundary candidates found.** Every "
             "low-observation state with multiple visits had a consistent "
             "AlphaQ response.")
    push("")
    push("---")
    push("")
    push("## Per-edge AlphaQ preferences")
    push("")
    push("![Per-edge color distribution](../plots/investigation2_per_edge.png)")
    push("")
    push("---")
    push("")
    push("## Decision-gate verdict")
    push("")
    verdict = derive_verdict(deterministic_stats, exploit, boundary)
    push(verdict)
    push("")
    push("---")
    push("")
    push("## Generated artefacts")
    push("")
    push("- `plots/investigation2_entropy_histogram.png`")
    push("- `plots/investigation2_entropy_vs_grey.png`")
    push("- `plots/investigation2_per_edge.png`")
    push("- `docs/INVESTIGATION_2_RESULTS.md` (this file)")
    push("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {REPORT_PATH}")


def derive_verdict(det_stats, exploit, boundary) -> str:
    """Map findings to one of the three decision-gate verdicts."""
    frac_deterministic = det_stats["frac_zero"]
    n_exploit = len(exploit)
    n_boundary = len(boundary)

    if frac_deterministic > 0.85 and n_exploit == 0:
        return (
            "**PESSIMISTIC** — AlphaQ is near-deterministic on the observed "
            f"basin ({100 * frac_deterministic:.1f}% of sufficiently-observed "
            "states have zero response entropy) and no high-entropy exploit "
            "candidates were found. This is consistent with AlphaQ being at "
            "or near a Nash policy within its reachable basin. The plan "
            "should still proceed through Phases 2-4 to validate this with "
            "an expected-value solver, but the prior on classical exploit "
            "is low. If Phase 4 produces zero wins, commit to tensor-network "
            "simulation (Investigation 5)."
        )
    elif n_exploit >= 5 or n_boundary >= 20:
        return (
            "**OPTIMISTIC** — AlphaQ shows pockets of high-entropy response "
            f"({n_exploit} exploit candidates, {n_boundary} decision-boundary "
            "candidates). These are concrete positions where AlphaQ's "
            "policy is not strongly peaked. Phase 2 (predictive model) and "
            "Phase 4 (expected-value solver) should target these states as "
            "the primary search frontier."
        )
    else:
        return (
            "**MIXED** — AlphaQ is mostly deterministic on observed states "
            f"({100 * frac_deterministic:.1f}% zero-entropy) with limited but "
            f"non-zero exploit candidates ({n_exploit} states, {n_boundary} "
            "decision-boundary states). Proceed with Phases 2-4 as planned. "
            "The predictive model accuracy in Phase 2 is the critical signal: "
            "if top-1 prediction accuracy is high (>= 0.80) the exploit window "
            "is narrow; if it is moderate (0.40-0.70) the off-distribution "
            "attack surface is larger."
        )


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main() -> int:
    if not DB_PATH.exists():
        print(f"ERROR: DB not found: {DB_PATH}", file=sys.stderr)
        return 1
    PLOT_DIR.mkdir(exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))
    try:
        print("Extracting AlphaQ decisions...", flush=True)
        decisions = extract_alphaq_decisions(conn)
        print(f"  {len(decisions)} decisions across "
              f"{len({d['game_id'] for d in decisions})} games", flush=True)

        if not decisions:
            print("No AlphaQ decisions found. Nothing to analyse.")
            return 1

        print("Aggregating by state...", flush=True)
        by_state = aggregate_by_state(decisions)
        print(f"  {len(by_state)} distinct states", flush=True)

        print("Computing per-state summary...", flush=True)
        summary = per_state_summary(by_state)
        eligible = [s for s in summary if s["n"] >= 3]
        n_zero = sum(1 for s in eligible if s["entropy_bits"] == 0.0)
        n_high = sum(1 for s in eligible
                     if s["entropy_bits"] >= EXPLOIT_MIN_ENTROPY_BITS)
        deterministic_stats = {
            "n_zero": n_zero,
            "n_high": n_high,
            "frac_zero": n_zero / len(eligible) if eligible else 0.0,
            "frac_high": n_high / len(eligible) if eligible else 0.0,
        }

        print("Computing global MI...", flush=True)
        mi = mutual_information(by_state)
        print(f"  MI plug-in: {mi['plugin_bits']:.4f} bits  "
              f"MM-corrected: {mi['mm_bits']:.4f} bits", flush=True)

        print("Stratifying MI by grey count...", flush=True)
        mi_by_grey = {}
        decisions_by_grey: dict[int, list[dict]] = defaultdict(list)
        for d in decisions:
            decisions_by_grey[d["grey_before"]].append(d)
        for g, ds in decisions_by_grey.items():
            bg = aggregate_by_state(ds)
            mi_g = mutual_information(bg)
            entropies = [entropy_bits(c) for c in bg.values() if sum(c.values()) >= 3]
            mi_g["mean_entropy"] = float(np.mean(entropies)) if entropies else 0.0
            mi_by_grey[g] = mi_g

        print("Identifying candidates...", flush=True)
        exploit = find_exploit_candidates(summary)
        boundary = find_decision_boundary_candidates(summary)
        print(f"  {len(exploit)} exploit candidates, "
              f"{len(boundary)} decision-boundary candidates", flush=True)

        print("Generating plots...", flush=True)
        plot_entropy_histogram(summary, PLOT_DIR / "investigation2_entropy_histogram.png")
        plot_entropy_vs_grey(summary, PLOT_DIR / "investigation2_entropy_vs_grey.png")
        plot_per_edge_color_distribution(decisions, PLOT_DIR / "investigation2_per_edge.png")

        print("Writing report...", flush=True)
        write_report(decisions, by_state, summary, mi, exploit, boundary,
                     mi_by_grey, deterministic_stats)

        print()
        print("=" * 60)
        print(derive_verdict(deterministic_stats, exploit, boundary)
              .split(" — ")[0])
        print("=" * 60)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
