//! Server configuration.

use std::path::PathBuf;

use clap::Parser;

/// HVAT Axum Server - Hyperspectral image streaming server.
#[derive(Parser, Debug, Clone)]
#[command(name = "hvat-server")]
#[command(about = "HVAT API server for hyperspectral image streaming")]
#[command(version)]
pub struct CliArgs {
    /// Port to listen on
    #[arg(short, long, default_value = "3000", env = "HVAT_PORT")]
    pub port: u16,

    /// Root directory containing images to serve.
    /// This becomes the "project" that clients connect to.
    #[arg(short, long, default_value = "./data", env = "HVAT_DATA_DIR")]
    pub data_dir: PathBuf,

    /// Directory for pyramid cache
    #[arg(long, default_value = "./.cache/pyramids", env = "HVAT_CACHE_DIR")]
    pub cache_dir: PathBuf,

    /// Maximum total memory for pyramid cache in MB
    #[arg(long, default_value = "2048", env = "HVAT_MAX_CACHE_MB")]
    pub max_cache_mb: u64,

    /// Maximum memory per user session in MB
    #[arg(long, default_value = "500", env = "HVAT_MAX_USER_MB")]
    pub max_user_mb: u64,

    /// Maximum concurrent streams per user
    #[arg(long, default_value = "4", env = "HVAT_MAX_STREAMS")]
    pub max_streams: usize,

    /// Number of rows to send per WebSocket message
    #[arg(long, default_value = "128", env = "HVAT_CHUNK_ROWS")]
    pub chunk_rows: u32,

    /// Project name (display name for this server/data directory).
    /// Defaults to the data directory name.
    #[arg(short = 'n', long, env = "HVAT_PROJECT_NAME")]
    pub name: Option<String>,
}

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Port to listen on
    pub port: u16,

    /// Root directory for images (the "project")
    pub data_dir: PathBuf,

    /// Directory for pyramid cache
    pub cache_dir: PathBuf,

    /// Maximum total memory for pyramid cache in bytes
    pub max_cache_memory: u64,

    /// Maximum memory per user session in bytes
    pub max_user_memory: u64,

    /// Maximum concurrent streams per user
    pub max_user_streams: usize,

    /// Number of rows to send per WebSocket message
    pub stream_chunk_rows: u32,

    /// Project name (display name)
    pub project_name: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 3000,
            data_dir: PathBuf::from("./data"),
            cache_dir: PathBuf::from("./.cache/pyramids"),
            max_cache_memory: 2 * 1024 * 1024 * 1024, // 2GB
            max_user_memory: 500 * 1024 * 1024,       // 500MB
            max_user_streams: 4,
            stream_chunk_rows: 128,
            project_name: "data".to_string(),
        }
    }
}

impl ServerConfig {
    /// Create configuration from CLI arguments.
    pub fn from_cli(args: CliArgs) -> Self {
        // Derive project name from data_dir if not specified
        let project_name = args.name.unwrap_or_else(|| {
            args.data_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("project")
                .to_string()
        });

        Self {
            port: args.port,
            data_dir: args.data_dir,
            cache_dir: args.cache_dir,
            max_cache_memory: args.max_cache_mb * 1024 * 1024,
            max_user_memory: args.max_user_mb * 1024 * 1024,
            max_user_streams: args.max_streams,
            stream_chunk_rows: args.chunk_rows,
            project_name,
        }
    }

    /// Load configuration from environment variables (legacy method).
    #[allow(dead_code)]
    pub fn from_env() -> Self {
        // Parse CLI args which also reads env vars via clap
        Self::from_cli(CliArgs::parse())
    }
}
