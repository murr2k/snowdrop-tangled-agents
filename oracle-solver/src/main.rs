mod enumerate;
mod graph;
mod oracle;
mod output;
mod scoring;
mod types;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use log::{error, info};

use enumerate::{EnumerationConfig, enumerate_all_openings};
use oracle::Oracle;
use output::build_output;
use scoring::Lut;

#[derive(Parser)]
#[command(name = "oracle-solver")]
#[command(about = "AlphaQ Oracle Solver - Find winning routes against deterministic opponents in Tangled")]
struct Cli {
    /// Path to game_stats.db
    #[arg(long, default_value_os_t = default_db_path())]
    db_path: PathBuf,

    /// Path to terminal_scores.bin (32768 x f32 LE)
    #[arg(long, default_value = "data/terminal_scores.bin")]
    lut_path: PathBuf,

    /// Opponent name to build oracle for
    #[arg(long, default_value = "alphaq")]
    opponent: String,

    /// Oracle confidence threshold for deterministic branching
    #[arg(long, default_value = "0.9")]
    confidence: f32,

    /// Max branches on uncertain oracle states
    #[arg(long, default_value = "3")]
    max_branch: usize,

    /// Max branches on oracle gaps (unseen states)
    #[arg(long, default_value = "3")]
    max_gap_branch: usize,

    /// Minimum cumulative path confidence to continue exploring
    #[arg(long, default_value = "0.5")]
    min_path_confidence: f32,

    /// Output JSON path
    #[arg(long, short, default_value = "output/oracle_routes.json")]
    output: PathBuf,

    /// Number of rayon threads (0 = auto)
    #[arg(long, default_value = "0")]
    threads: usize,
}

fn default_db_path() -> PathBuf {
    dirs_next().unwrap_or_else(|| PathBuf::from("."))
        .join("game_stats.db")
}

fn dirs_next() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE")
            .ok()
            .map(|p| PathBuf::from(p).join(".tangled"))
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::env::var("HOME")
            .ok()
            .map(|p| PathBuf::from(p).join(".tangled"))
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    if cli.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cli.threads)
            .build_global()
            .unwrap();
    }

    let total_start = Instant::now();

    // Phase 1: Build Oracle
    info!("=== Phase 1: Building Oracle ===");
    let oracle_start = Instant::now();
    let oracle = match Oracle::from_database(&cli.db_path, &cli.opponent) {
        Ok(o) => o,
        Err(e) => {
            error!("Failed to build oracle: {}", e);
            error!("DB path: {:?}", cli.db_path);
            std::process::exit(1);
        }
    };
    info!("Oracle built in {:.1}s", oracle_start.elapsed().as_secs_f32());

    if oracle.total_states == 0 {
        error!("No oracle data found for opponent '{}'. Check the database.", cli.opponent);
        std::process::exit(1);
    }

    // Phase 2: Load LUT
    info!("=== Phase 2: Loading Terminal LUT ===");
    let lut = match Lut::from_binary(&cli.lut_path) {
        Ok(l) => l,
        Err(e) => {
            error!("Failed to load LUT: {}", e);
            error!("LUT path: {:?}", cli.lut_path);
            error!("Export LUT with: python -c \"import scipy.io, numpy as np; d = scipy.io.loadmat('snowdrop_tangled_agents/matlab/rl/data/terminal_scores.mat'); d['terminal_scores'].astype(np.float32).tofile('oracle-solver/data/terminal_scores.bin')\"");
            std::process::exit(1);
        }
    };

    // Phase 3: Enumerate Game Tree
    info!("=== Phase 3: Enumerating Game Tree ===");
    let enum_start = Instant::now();
    let config = EnumerationConfig {
        confidence_threshold: cli.confidence,
        max_branch_on_uncertain: cli.max_branch,
        max_branch_on_gap: cli.max_gap_branch,
        min_path_confidence: cli.min_path_confidence,
    };
    let result = enumerate_all_openings(&oracle, &config);
    info!("Enumeration completed in {:.1}s", enum_start.elapsed().as_secs_f32());

    // Phase 4: Score and Output
    info!("=== Phase 4: Scoring and Output ===");
    let output = build_output(&oracle, &result, &lut);

    // Print summary
    info!("========================================");
    info!("           RESULTS SUMMARY");
    info!("========================================");
    info!("Oracle: {} states, {:.1}% deterministic, {:.1}% avg confidence",
        output.oracle_stats.total_states_observed,
        output.oracle_stats.deterministic_pct,
        output.oracle_stats.avg_confidence * 100.0);
    info!("Terminals found: {}", output.enumeration_stats.total_terminal_states);
    info!("  WIN candidates: {}", output.enumeration_stats.win_candidates);
    info!("  Near-misses (0 < score <= epsilon): {}", output.enumeration_stats.near_misses);
    info!("  Draws: {}", output.enumeration_stats.draws);
    info!("  Losses: {}", output.enumeration_stats.losses);
    info!("Nodes visited: {}", output.enumeration_stats.total_nodes_visited);
    info!("Oracle hits/misses: {}/{}", output.enumeration_stats.oracle_hits, output.enumeration_stats.oracle_misses);

    if !output.winning_routes.is_empty() {
        info!("========================================");
        info!("  *** WINNING ROUTES FOUND! ***");
        info!("========================================");
        for (i, route) in output.winning_routes.iter().enumerate().take(10) {
            info!("  Route {}: {} (score={:.6}, margin={:.6}, confidence={:.2}, gaps={})",
                i + 1, route.terminal_state, route.lut_score,
                route.margin_over_epsilon, route.path_min_confidence, route.oracle_gaps);
            let our_moves: Vec<String> = route.moves.iter()
                .filter(|m| m.player == "us")
                .map(|m| format!("E{}{}", m.edge, match m.color { crate::types::Color::Green => "G", crate::types::Color::Purple => "P" }))
                .collect();
            info!("    Our moves: {}", our_moves.join(" -> "));
        }
    } else {
        info!("No winning routes found.");
        if !output.near_miss_routes.is_empty() {
            info!("Top near-misses (positive score, within epsilon of win):");
            for (i, route) in output.near_miss_routes.iter().enumerate().take(5) {
                info!("  {}: {} (score={:.6}, margin={:.6})",
                    i + 1, route.terminal_state, route.lut_score, route.margin_over_epsilon);
            }
        }
    }

    // Write JSON output
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let json = serde_json::to_string_pretty(&output).expect("JSON serialization failed");
    std::fs::write(&cli.output, &json).expect("Failed to write output file");
    info!("Output written to: {:?}", cli.output);

    info!("Total time: {:.1}s", total_start.elapsed().as_secs_f32());
}
