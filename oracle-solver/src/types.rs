use serde::Serialize;
use std::fmt;

/// Edge color in the Tangled game.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Color {
    Green,  // FM (ferromagnetic, J=-1)
    Purple, // AFM (antiferromagnetic, J=+1)
}

impl Color {
    pub fn from_char(c: char) -> Option<Color> {
        match c {
            'G' => Some(Color::Green),
            'P' => Some(Color::Purple),
            _ => None,
        }
    }

    pub fn to_char(self) -> char {
        match self {
            Color::Green => 'G',
            Color::Purple => 'P',
        }
    }
}

impl Serialize for Color {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(match self {
            Color::Green => "G",
            Color::Purple => "P",
        })
    }
}

/// A single move: color an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct Move {
    pub edge: u8,
    pub color: Color,
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "E{}{}", self.edge, self.color.to_char())
    }
}

/// Compact board state: 2 bits per edge, 30 bits total in a u32.
/// Encoding per edge: 00 = grey (uncolored), 01 = Green, 10 = Purple.
/// Bits [2*i .. 2*i+1] encode edge i.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoardState(pub u32);

impl BoardState {
    /// All edges grey.
    pub fn new() -> Self {
        BoardState(0)
    }

    /// Get the color of an edge (None = grey).
    pub fn get(&self, edge: u8) -> Option<Color> {
        let shift = (edge as u32) * 2;
        match (self.0 >> shift) & 0b11 {
            0b01 => Some(Color::Green),
            0b10 => Some(Color::Purple),
            _ => None, // 00 = grey
        }
    }

    /// Return a new state with the given edge colored.
    pub fn set(&self, edge: u8, color: Color) -> BoardState {
        let shift = (edge as u32) * 2;
        let mask = !(0b11u32 << shift);
        let bits = match color {
            Color::Green => 0b01u32,
            Color::Purple => 0b10u32,
        };
        BoardState((self.0 & mask) | (bits << shift))
    }

    /// List of grey (uncolored) edge indices.
    pub fn grey_edges(&self) -> Vec<u8> {
        let mut edges = Vec::new();
        for i in 0..15u8 {
            if self.get(i).is_none() {
                edges.push(i);
            }
        }
        edges
    }

    /// Number of grey edges.
    pub fn grey_count(&self) -> u8 {
        let mut count = 0u8;
        for i in 0..15u8 {
            if self.get(i).is_none() {
                count += 1;
            }
        }
        count
    }

    /// True if all 15 edges are colored.
    pub fn is_terminal(&self) -> bool {
        self.grey_count() == 0
    }

    /// Convert terminal state to LUT index (u16).
    /// G=1, P=0, bit i = edge i. Only valid when is_terminal() == true.
    pub fn to_terminal_index(&self) -> u16 {
        let mut idx = 0u16;
        for i in 0..15u8 {
            if self.get(i) == Some(Color::Green) {
                idx |= 1 << i;
            }
        }
        idx
    }

    /// Convert to 15-character string ("-"/"G"/"P").
    pub fn to_string(&self) -> String {
        let mut s = String::with_capacity(15);
        for i in 0..15u8 {
            s.push(match self.get(i) {
                Some(Color::Green) => 'G',
                Some(Color::Purple) => 'P',
                None => '-',
            });
        }
        s
    }

    /// Parse from 15-character string.
    pub fn from_string(s: &str) -> Option<BoardState> {
        if s.len() != 15 {
            return None;
        }
        let mut state = BoardState::new();
        for (i, c) in s.chars().enumerate() {
            match c {
                'G' => state = state.set(i as u8, Color::Green),
                'P' => state = state.set(i as u8, Color::Purple),
                '-' => {} // grey, already default
                _ => return None,
            }
        }
        Some(state)
    }
}

impl fmt::Debug for BoardState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BoardState({})", self.to_string())
    }
}

impl fmt::Display for BoardState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// Oracle entry: predicted opponent response for a given board state.
#[derive(Clone, Debug)]
pub struct OracleEntry {
    pub primary: Move,
    pub confidence: f32,
    pub total_observations: u32,
    pub alternatives: Vec<(Move, f32)>, // (move, probability)
}

/// An annotated move in a route (includes oracle confidence for opponent moves).
#[derive(Clone, Debug, Serialize)]
pub struct AnnotatedMove {
    pub turn: u8,
    pub player: String,
    pub edge: u8,
    pub color: Color,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oracle_confidence: Option<f32>,
}

/// A complete route from opening to terminal state.
#[derive(Clone, Debug, Serialize)]
pub struct Route {
    pub terminal_state: String,
    pub terminal_index: u16,
    pub lut_score: f32,
    pub margin_over_epsilon: f32,
    pub path_min_confidence: f32,
    pub oracle_gaps: u32,
    pub moves: Vec<AnnotatedMove>,
}

/// Terminal state info collected during enumeration.
#[derive(Clone, Debug)]
pub struct TerminalInfo {
    pub state: BoardState,
    /// Best route (highest min-confidence) reaching this terminal.
    pub best_route: Vec<(Move, bool, Option<f32>)>, // (move, is_our_turn, oracle_confidence)
    pub min_path_confidence: f32,
    pub oracle_gaps: u32,
    pub route_count: u32,
}

/// Game outcome classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Outcome {
    Win,
    Draw,
    Loss,
}
