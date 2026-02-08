use serde::Serialize;

use crate::enumerate::EnumerationResult;
use crate::graph::EPSILON;
use crate::oracle::Oracle;
use crate::scoring::Lut;
use crate::types::{AnnotatedMove, Outcome, Route};

/// Top-level JSON output structure.
#[derive(Serialize)]
pub struct SolverOutput {
    pub oracle_stats: OracleStats,
    pub enumeration_stats: EnumerationStats,
    pub winning_routes: Vec<Route>,
    pub near_miss_routes: Vec<Route>,
}

#[derive(Serialize)]
pub struct OracleStats {
    pub total_states_observed: usize,
    pub deterministic_states: usize,
    pub deterministic_pct: f32,
    pub avg_confidence: f32,
}

#[derive(Serialize)]
pub struct EnumerationStats {
    pub total_terminal_states: usize,
    pub win_candidates: usize,
    pub near_misses: usize,
    pub draws: usize,
    pub losses: usize,
    pub total_nodes_visited: u64,
    pub oracle_hits: u64,
    pub oracle_misses: u64,
}

/// Build the final output from enumeration results.
pub fn build_output(
    oracle: &Oracle,
    result: &EnumerationResult,
    lut: &Lut,
) -> SolverOutput {
    let oracle_stats = OracleStats {
        total_states_observed: oracle.total_states,
        deterministic_states: oracle.deterministic_states,
        deterministic_pct: if oracle.total_states > 0 {
            oracle.deterministic_states as f32 / oracle.total_states as f32 * 100.0
        } else {
            0.0
        },
        avg_confidence: oracle.avg_confidence,
    };

    let mut winning_routes: Vec<Route> = Vec::new();
    let mut near_miss_routes: Vec<Route> = Vec::new();
    let mut win_count = 0usize;
    let mut near_miss_count = 0usize;
    let mut draw_count = 0usize;
    let mut loss_count = 0usize;

    for (&terminal_index, info) in &result.terminals {
        let score = lut.score(terminal_index);
        let outcome = lut.classify(terminal_index);

        match outcome {
            Outcome::Win => win_count += 1,
            Outcome::Draw => {
                if score > 0.0 {
                    near_miss_count += 1;
                } else {
                    draw_count += 1;
                }
            }
            Outcome::Loss => loss_count += 1,
        }

        // Build route from path
        let moves: Vec<AnnotatedMove> = info
            .best_route
            .iter()
            .enumerate()
            .map(|(i, (mv, is_ours, conf))| AnnotatedMove {
                turn: (i + 1) as u8,
                player: if *is_ours { "us".to_string() } else { "opponent".to_string() },
                edge: mv.edge,
                color: mv.color,
                oracle_confidence: *conf,
            })
            .collect();

        let route = Route {
            terminal_state: info.state.to_string(),
            terminal_index,
            lut_score: score,
            margin_over_epsilon: score - EPSILON,
            path_min_confidence: info.min_path_confidence,
            oracle_gaps: info.oracle_gaps,
            moves,
        };

        match outcome {
            Outcome::Win => winning_routes.push(route),
            Outcome::Draw if score > 0.0 => near_miss_routes.push(route),
            _ => {} // Don't output loss routes
        }
    }

    // Sort: highest score first, then highest confidence
    winning_routes.sort_by(|a, b| {
        b.lut_score
            .partial_cmp(&a.lut_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.path_min_confidence
                    .partial_cmp(&a.path_min_confidence)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(a.oracle_gaps.cmp(&b.oracle_gaps))
    });

    near_miss_routes.sort_by(|a, b| {
        b.lut_score
            .partial_cmp(&a.lut_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Limit near-miss output to top 100
    near_miss_routes.truncate(100);

    let enumeration_stats = EnumerationStats {
        total_terminal_states: result.terminals.len(),
        win_candidates: win_count,
        near_misses: near_miss_count,
        draws: draw_count,
        losses: loss_count,
        total_nodes_visited: result.total_nodes_visited,
        oracle_hits: result.oracle_hits,
        oracle_misses: result.oracle_misses,
    };

    SolverOutput {
        oracle_stats,
        enumeration_stats,
        winning_routes,
        near_miss_routes,
    }
}
