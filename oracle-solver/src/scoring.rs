use std::path::Path;

use log::info;

use crate::graph::EPSILON;
use crate::types::Outcome;

/// Terminal state lookup table: 32,768 f32 scores indexed by terminal state bit-pack.
pub struct Lut {
    scores: Vec<f32>,
}

impl Lut {
    /// Load LUT from a raw binary file (32768 x f32 little-endian).
    pub fn from_binary(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let bytes = std::fs::read(path)?;
        let expected = 32768 * 4;
        if bytes.len() != expected {
            return Err(format!(
                "LUT file size mismatch: expected {} bytes, got {}",
                expected, bytes.len()
            ).into());
        }

        let scores: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        let min = scores.iter().copied().fold(f32::INFINITY, f32::min);
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let wins = scores.iter().filter(|&&s| s > EPSILON).count();
        let losses = scores.iter().filter(|&&s| s < -EPSILON).count();
        let draws = scores.len() - wins - losses;

        info!("LUT loaded: {} entries, score range [{:.3}, {:.3}]", scores.len(), min, max);
        info!("  Wins (>{:.4}): {} ({:.1}%)", EPSILON, wins, wins as f32 / scores.len() as f32 * 100.0);
        info!("  Draws: {} ({:.1}%)", draws, draws as f32 / scores.len() as f32 * 100.0);
        info!("  Losses (<{:.4}): {} ({:.1}%)", -EPSILON, losses, losses as f32 / scores.len() as f32 * 100.0);

        Ok(Lut { scores })
    }

    /// Get the score for a terminal state index.
    pub fn score(&self, terminal_index: u16) -> f32 {
        self.scores[terminal_index as usize]
    }

    /// Classify a terminal state.
    pub fn classify(&self, terminal_index: u16) -> Outcome {
        let score = self.score(terminal_index);
        if score > EPSILON {
            Outcome::Win
        } else if score < -EPSILON {
            Outcome::Loss
        } else {
            Outcome::Draw
        }
    }
}
