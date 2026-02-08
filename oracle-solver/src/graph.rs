/// Petersen graph constants and utilities.

pub const NUM_EDGES: usize = 15;
pub const NUM_VERTICES: usize = 10;

/// Draw threshold from D-Wave ground truth adjudication.
/// Confirmed by Geordie Rose: epsilon = 0.0005 for Petersen graph.
pub const EPSILON: f32 = 0.0005;

/// Petersen graph edge list, indexed 0-14.
/// Matches the canonical ordering in petersen_strategy.py.
pub const PETERSEN_EDGES: [(u8, u8); NUM_EDGES] = [
    (0, 2), // E0
    (0, 3), // E1
    (0, 6), // E2
    (1, 3), // E3
    (1, 4), // E4
    (1, 7), // E5
    (2, 4), // E6
    (2, 8), // E7
    (3, 9), // E8
    (4, 5), // E9
    (5, 6), // E10
    (5, 9), // E11
    (6, 7), // E12
    (7, 8), // E13
    (8, 9), // E14
];

/// Player vertex assignments (from petersen_strategy.py).
pub const MY_VERTEX: u8 = 5;
pub const OPP_VERTEX: u8 = 7;
pub const HUB_VERTEX: u8 = 6;

/// Edges touching each key vertex.
pub const MY_EDGES: [u8; 3] = [9, 10, 11];
pub const OPP_EDGES: [u8; 3] = [5, 12, 13];
pub const HUB_EDGES: [u8; 3] = [2, 10, 12];

/// Total possible moves: 15 edges x 2 colors.
pub const NUM_MOVES: usize = 30;

/// Convert (edge, color_idx) to move index (0-29).
/// color_idx: 0 = Green, 1 = Purple.
pub fn move_to_index(edge: u8, color_idx: u8) -> usize {
    (edge as usize) * 2 + (color_idx as usize)
}
