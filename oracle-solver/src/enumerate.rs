use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use log::{debug, info};
use rayon::prelude::*;

use crate::graph::NUM_EDGES;
use crate::oracle::Oracle;
use crate::types::{BoardState, Color, Move, TerminalInfo};

/// Configuration for the game tree enumeration.
pub struct EnumerationConfig {
    /// Oracle confidence threshold for deterministic branching.
    pub confidence_threshold: f32,
    /// Max branches to explore when oracle confidence is below threshold.
    pub max_branch_on_uncertain: usize,
    /// Max branches for oracle gap (unseen states).
    pub max_branch_on_gap: usize,
    /// Minimum cumulative path confidence to continue exploring.
    pub min_path_confidence: f32,
}

impl Default for EnumerationConfig {
    fn default() -> Self {
        EnumerationConfig {
            confidence_threshold: 0.9,
            max_branch_on_uncertain: 3,
            max_branch_on_gap: 3,
            min_path_confidence: 0.5,
        }
    }
}

/// Results of the game tree enumeration.
pub struct EnumerationResult {
    /// All reachable terminal states, keyed by terminal index.
    pub terminals: HashMap<u16, TerminalInfo>,
    pub total_nodes_visited: u64,
    pub oracle_hits: u64,
    pub oracle_misses: u64,
}

impl EnumerationResult {
    fn new() -> Self {
        EnumerationResult {
            terminals: HashMap::new(),
            total_nodes_visited: 0,
            oracle_hits: 0,
            oracle_misses: 0,
        }
    }

    fn merge(&mut self, other: EnumerationResult) {
        for (idx, info) in other.terminals {
            let entry = self.terminals.entry(idx).or_insert_with(|| TerminalInfo {
                state: info.state,
                best_route: Vec::new(),
                min_path_confidence: 0.0,
                oracle_gaps: u32::MAX,
                route_count: 0,
            });
            entry.route_count += info.route_count;
            // Keep the route with highest min confidence and fewest gaps
            if info.min_path_confidence > entry.min_path_confidence
                || (info.min_path_confidence == entry.min_path_confidence
                    && info.oracle_gaps < entry.oracle_gaps)
            {
                entry.best_route = info.best_route;
                entry.min_path_confidence = info.min_path_confidence;
                entry.oracle_gaps = info.oracle_gaps;
            }
        }
        self.total_nodes_visited += other.total_nodes_visited;
        self.oracle_hits += other.oracle_hits;
        self.oracle_misses += other.oracle_misses;
    }
}

/// Enumerate all reachable terminal states from all 30 possible openings.
pub fn enumerate_all_openings(
    oracle: &Oracle,
    config: &EnumerationConfig,
) -> EnumerationResult {
    let openings: Vec<Move> = (0..NUM_EDGES as u8)
        .flat_map(|edge| {
            [Color::Green, Color::Purple]
                .into_iter()
                .map(move |color| Move { edge, color })
        })
        .collect();

    let progress = AtomicU64::new(0);

    let results: Vec<EnumerationResult> = openings
        .par_iter()
        .map(|opening| {
            let result = enumerate_opening(*opening, oracle, config);
            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            info!(
                "[{}/30] Opening E{}{}: {} terminals, {} nodes",
                done,
                opening.edge,
                opening.color.to_char(),
                result.terminals.len(),
                result.total_nodes_visited
            );
            result
        })
        .collect();

    let mut combined = EnumerationResult::new();
    for r in results {
        combined.merge(r);
    }

    info!(
        "Enumeration complete: {} unique terminals, {} nodes visited, {} oracle hits, {} oracle misses",
        combined.terminals.len(),
        combined.total_nodes_visited,
        combined.oracle_hits,
        combined.oracle_misses
    );

    combined
}

/// Enumerate all reachable terminal states from a single opening.
fn enumerate_opening(
    opening: Move,
    oracle: &Oracle,
    config: &EnumerationConfig,
) -> EnumerationResult {
    let mut result = EnumerationResult::new();

    // Apply opening move (our first move)
    let initial = BoardState::new().set(opening.edge, opening.color);
    let path = vec![(opening, true, None)]; // (move, is_ours, oracle_confidence)

    // After our opening, it's opponent's turn
    dfs(
        initial,
        false, // opponent's turn next
        path,
        1.0, // full confidence at start
        0,   // no oracle gaps yet
        oracle,
        config,
        &mut result,
    );

    result
}

/// Depth-first search through the game tree.
fn dfs(
    state: BoardState,
    is_our_turn: bool,
    path: Vec<(Move, bool, Option<f32>)>,
    min_confidence: f32,
    oracle_gaps: u32,
    oracle: &Oracle,
    config: &EnumerationConfig,
    result: &mut EnumerationResult,
) {
    result.total_nodes_visited += 1;

    // Terminal state: record it
    if state.is_terminal() {
        let idx = state.to_terminal_index();
        let entry = result.terminals.entry(idx).or_insert_with(|| TerminalInfo {
            state,
            best_route: Vec::new(),
            min_path_confidence: 0.0,
            oracle_gaps: u32::MAX,
            route_count: 0,
        });
        entry.route_count += 1;
        // Keep best route (highest confidence, fewest gaps)
        if min_confidence > entry.min_path_confidence
            || (min_confidence == entry.min_path_confidence && oracle_gaps < entry.oracle_gaps)
        {
            entry.best_route = path.clone();
            entry.min_path_confidence = min_confidence;
            entry.oracle_gaps = oracle_gaps;
        }
        return;
    }

    let grey = state.grey_edges();

    if is_our_turn {
        // Branch over all legal moves (each grey edge x 2 colors)
        for &edge in &grey {
            for color in [Color::Green, Color::Purple] {
                let next = state.set(edge, color);
                let mut new_path = path.clone();
                new_path.push((Move { edge, color }, true, None));
                dfs(next, false, new_path, min_confidence, oracle_gaps, oracle, config, result);
            }
        }
    } else {
        // Opponent's turn: consult oracle
        match oracle.lookup(state) {
            Some(entry) if entry.confidence >= config.confidence_threshold => {
                // High confidence: follow single deterministic response
                result.oracle_hits += 1;
                let mv = entry.primary;
                // Verify the move is actually legal (edge must be grey)
                if state.get(mv.edge).is_some() {
                    return; // Invalid oracle prediction, prune
                }
                let next = state.set(mv.edge, mv.color);
                let mut new_path = path.clone();
                new_path.push((mv, false, Some(entry.confidence)));
                let new_conf = min_confidence.min(entry.confidence);
                dfs(next, true, new_path, new_conf, oracle_gaps, oracle, config, result);
            }
            Some(entry) => {
                // Low confidence: branch over top responses
                result.oracle_hits += 1;
                let mut all_responses: Vec<(Move, f32)> = vec![(entry.primary, entry.confidence)];
                all_responses.extend_from_slice(&entry.alternatives);
                all_responses.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                for (mv, conf) in all_responses.iter().take(config.max_branch_on_uncertain) {
                    if state.get(mv.edge).is_some() {
                        continue; // Not a legal move
                    }
                    let next = state.set(mv.edge, mv.color);
                    let mut new_path = path.clone();
                    new_path.push((*mv, false, Some(*conf)));
                    let new_conf = min_confidence.min(*conf);
                    if new_conf < config.min_path_confidence {
                        continue; // Confidence too low, prune
                    }
                    dfs(next, true, new_path, new_conf, oracle_gaps, oracle, config, result);
                }
            }
            None => {
                // Oracle gap: no data for this state
                result.oracle_misses += 1;
                let new_gaps = oracle_gaps + 1;

                // Use global fallback distribution
                let fallback = oracle.fallback_moves(config.max_branch_on_gap * 2);
                let mut used = 0;
                for (mv, _prob) in fallback {
                    if used >= config.max_branch_on_gap {
                        break;
                    }
                    if state.get(mv.edge).is_some() {
                        continue; // Not legal
                    }
                    let next = state.set(mv.edge, mv.color);
                    let mut new_path = path.clone();
                    new_path.push((*mv, false, None));
                    // Halve confidence for gap-based moves
                    let new_conf = min_confidence * 0.5;
                    if new_conf < config.min_path_confidence {
                        continue;
                    }
                    dfs(next, true, new_path, new_conf, new_gaps, oracle, config, result);
                    used += 1;
                }
            }
        }
    }
}
