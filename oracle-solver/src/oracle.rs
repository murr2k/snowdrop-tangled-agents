use std::collections::HashMap;
use std::path::Path;

use log::{info, warn};
use rusqlite::Connection;

use crate::types::{BoardState, Color, Move, OracleEntry};

/// Oracle: predicts AlphaQ's response for any board state based on historical data.
pub struct Oracle {
    responses: HashMap<BoardState, OracleEntry>,
    /// Global fallback distribution for states not in the database.
    /// Maps move_index (0-29) to probability.
    pub fallback_distribution: Vec<(Move, f32)>,
    // Statistics
    pub total_states: usize,
    pub deterministic_states: usize,
    pub avg_confidence: f32,
}

impl Oracle {
    /// Build oracle from the game_stats.db SQLite database.
    pub fn from_database(db_path: &Path, opponent: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let conn = Connection::open(db_path)?;

        // Check if opponent_history table exists (migration v5+)
        let has_opponent_history: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM sqlite_master WHERE type='table' AND name='opponent_history'",
            [],
            |row| row.get(0),
        )?;

        let mut state_moves: HashMap<BoardState, HashMap<Move, u32>> = HashMap::new();
        let mut global_move_counts: HashMap<Move, u32> = HashMap::new();

        if has_opponent_history {
            info!("Reading from opponent_history table...");
            let mut stmt = conn.prepare(
                "SELECT board_state_before, edge, color, COUNT(*) as cnt
                 FROM opponent_history
                 WHERE opponent_name = ?1
                   AND board_state_before IS NOT NULL
                 GROUP BY board_state_before, edge, color
                 ORDER BY board_state_before, cnt DESC"
            )?;

            let rows = stmt.query_map([opponent], |row| {
                let state_str: String = row.get(0)?;
                let edge: i32 = row.get(1)?;
                let color_str: String = row.get(2)?;
                let count: i32 = row.get(3)?;
                Ok((state_str, edge, color_str, count))
            })?;

            for row in rows {
                let (state_str, edge, color_str, count) = row?;
                let color = match color_str.as_str() {
                    "G" => Color::Green,
                    "P" => Color::Purple,
                    _ => continue,
                };
                if edge < 0 || edge > 14 {
                    continue;
                }
                let board_state = match BoardState::from_string(&state_str) {
                    Some(s) => s,
                    None => continue,
                };
                let mv = Move { edge: edge as u8, color };
                *state_moves.entry(board_state).or_default().entry(mv).or_insert(0) += count as u32;
                *global_move_counts.entry(mv).or_insert(0) += count as u32;
            }
        }

        // Fallback: if opponent_history is empty or missing, reconstruct from moves table
        if state_moves.is_empty() {
            info!("opponent_history empty, reconstructing from moves table...");
            Self::build_from_moves_table(&conn, opponent, &mut state_moves, &mut global_move_counts)?;
        }

        // Build oracle entries
        let mut responses: HashMap<BoardState, OracleEntry> = HashMap::new();
        let mut total_confidence = 0.0f64;
        let mut deterministic_count = 0usize;

        for (board_state, move_counts) in &state_moves {
            let total: u32 = move_counts.values().sum();
            if total == 0 {
                continue;
            }

            // Sort by count descending
            let mut sorted: Vec<(Move, u32)> = move_counts.iter().map(|(&m, &c)| (m, c)).collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));

            let primary = sorted[0].0;
            let confidence = sorted[0].1 as f32 / total as f32;

            let alternatives: Vec<(Move, f32)> = sorted[1..]
                .iter()
                .map(|&(m, c)| (m, c as f32 / total as f32))
                .collect();

            if confidence >= 0.9 {
                deterministic_count += 1;
            }
            total_confidence += confidence as f64;

            responses.insert(*board_state, OracleEntry {
                primary,
                confidence,
                total_observations: total,
                alternatives,
            });
        }

        let total_states = responses.len();
        let avg_confidence = if total_states > 0 {
            (total_confidence / total_states as f64) as f32
        } else {
            0.0
        };

        // Build global fallback distribution
        let global_total: u32 = global_move_counts.values().sum();
        let mut fallback_distribution: Vec<(Move, f32)> = global_move_counts
            .into_iter()
            .map(|(m, c)| (m, c as f32 / global_total as f32))
            .collect();
        fallback_distribution.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        info!("Oracle built: {} states, {} deterministic ({:.1}%), avg confidence {:.1}%",
            total_states, deterministic_count,
            if total_states > 0 { deterministic_count as f32 / total_states as f32 * 100.0 } else { 0.0 },
            avg_confidence * 100.0);

        Ok(Oracle {
            responses,
            fallback_distribution,
            total_states,
            deterministic_states: deterministic_count,
            avg_confidence,
        })
    }

    /// Reconstruct oracle data from the moves table (fallback when opponent_history is empty).
    fn build_from_moves_table(
        conn: &Connection,
        opponent: &str,
        state_moves: &mut HashMap<BoardState, HashMap<Move, u32>>,
        global_move_counts: &mut HashMap<Move, u32>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Get all games against this opponent
        let mut game_stmt = conn.prepare(
            "SELECT id FROM games WHERE opponent = ?1"
        )?;
        let game_ids: Vec<String> = game_stmt
            .query_map([opponent], |row| row.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();

        info!("Reconstructing oracle from {} games via moves table", game_ids.len());

        for game_id in &game_ids {
            // Get all moves for this game, ordered
            let mut move_stmt = conn.prepare(
                "SELECT move_number, player, edge, color, state_after
                 FROM moves
                 WHERE game_id = ?1
                 ORDER BY move_number ASC"
            )?;

            let moves: Vec<(i32, String, i32, String, Option<String>)> = move_stmt
                .query_map([game_id], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                })?
                .filter_map(|r| r.ok())
                .collect();

            // Track board state progression
            let mut prev_state_after: Option<String> = None;

            for (_, player, edge, color_str, state_after) in &moves {
                if player == "opponent" {
                    // Board state before opponent's move = previous move's state_after
                    if let Some(ref prev) = prev_state_after {
                        if let Some(board_state) = BoardState::from_string(prev) {
                            let color = match color_str.as_str() {
                                "G" => Color::Green,
                                "P" => Color::Purple,
                                _ => continue,
                            };
                            if *edge >= 0 && *edge <= 14 {
                                let mv = Move { edge: *edge as u8, color };
                                *state_moves.entry(board_state).or_default().entry(mv).or_insert(0) += 1;
                                *global_move_counts.entry(mv).or_insert(0) += 1;
                            }
                        }
                    }
                }
                prev_state_after = state_after.clone();
            }
        }

        Ok(())
    }

    /// Look up the oracle's prediction for a given board state.
    pub fn lookup(&self, state: BoardState) -> Option<&OracleEntry> {
        self.responses.get(&state)
    }

    /// Get the top N fallback moves for states not in the oracle.
    pub fn fallback_moves(&self, n: usize) -> &[(Move, f32)] {
        &self.fallback_distribution[..n.min(self.fallback_distribution.len())]
    }
}
