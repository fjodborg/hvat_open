use std::ops::Deref;
use std::path::PathBuf;

use clap::{Args, Parser, value_parser};

/// Shared base CLI arguments used by backend implementations.
#[derive(Args, Debug, Clone)]
pub struct BaseCliArgs {
    /// Port to listen on.
    #[arg(short, long, default_value = "3000", env = "HVAT_PORT")]
    pub port: u16,

    /// Root directory containing images to serve.
    #[arg(short, long, default_value = "./data", env = "HVAT_DATA_DIR")]
    pub data_dir: PathBuf,

    /// Directory for pyramid cache.
    #[arg(long, default_value = "./.cache/pyramids", env = "HVAT_CACHE_DIR")]
    pub cache_dir: PathBuf,

    /// Maximum total memory for pyramid cache in MB.
    #[arg(long, default_value = "2048", env = "HVAT_MAX_CACHE_MB")]
    pub max_cache_mb: u64,

    /// Maximum concurrent streams per connection.
    #[arg(long, default_value = "4", env = "HVAT_MAX_STREAMS")]
    pub max_streams: usize,

    /// Maximum total WebSocket connections.
    #[arg(long, default_value = "100", env = "HVAT_MAX_CONNECTIONS")]
    pub max_connections: usize,

    /// WebSocket ping interval in seconds (0 to disable).
    #[arg(long, default_value = "30", env = "HVAT_PING_INTERVAL")]
    pub ping_interval: u64,

    /// WebSocket connection timeout in seconds (no pong response).
    #[arg(long, default_value = "90", env = "HVAT_CONNECTION_TIMEOUT")]
    pub connection_timeout: u64,

    /// Number of rows to send per WebSocket message.
    #[arg(
        long,
        default_value = "128",
        env = "HVAT_CHUNK_ROWS",
        value_parser = value_parser!(u32).range(1..)
    )]
    pub chunk_rows: u32,

    /// Project name (display name for this server/data directory).
    /// Defaults to the data directory name.
    #[arg(short = 'n', long, env = "HVAT_PROJECT_NAME")]
    pub name: Option<String>,
}

/// HVAT simple server - hyperspectral image streaming backend.
#[derive(Parser, Debug, Clone)]
#[command(name = "hvat-server")]
#[command(about = "HVAT API server for hyperspectral image streaming")]
#[command(version)]
pub struct CliArgs {
    #[command(flatten)]
    pub base: BaseCliArgs,
}

impl Deref for CliArgs {
    type Target = BaseCliArgs;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

/// Parse CLI arguments.
#[must_use]
pub fn parse_cli_args() -> CliArgs {
    CliArgs::parse()
}

/// Server configuration.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Port to listen on.
    pub port: u16,
    /// Root directory for images.
    pub data_dir: PathBuf,
    /// Directory for pyramid cache.
    pub cache_dir: PathBuf,
    /// Maximum total memory for pyramid cache in bytes.
    pub max_cache_memory: u64,
    /// Maximum concurrent streams per connection.
    pub max_user_streams: usize,
    /// Maximum total WebSocket connections.
    pub max_connections: usize,
    /// WebSocket ping interval in seconds.
    pub ping_interval_secs: u64,
    /// WebSocket connection timeout in seconds.
    pub connection_timeout_secs: u64,
    /// Number of rows to send per WebSocket message.
    pub stream_chunk_rows: u32,
    /// Project name (display name).
    pub project_name: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 3000,
            data_dir: PathBuf::from("./data"),
            cache_dir: PathBuf::from("./.cache/pyramids"),
            max_cache_memory: 2 * 1024 * 1024 * 1024,
            max_user_streams: 4,
            max_connections: 100,
            ping_interval_secs: 30,
            connection_timeout_secs: 90,
            stream_chunk_rows: 128,
            project_name: "data".to_string(),
        }
    }
}

impl ServerConfig {
    /// Create configuration from CLI arguments.
    #[must_use]
    pub fn from_cli(args: CliArgs) -> Self {
        Self::from_base_cli(args.base)
    }

    /// Create configuration from shared base CLI arguments.
    #[must_use]
    pub fn from_base_cli(args: BaseCliArgs) -> Self {
        let project_name = args.name.unwrap_or_else(|| {
            args.data_dir
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("project")
                .to_string()
        });

        Self {
            port: args.port,
            data_dir: args.data_dir,
            cache_dir: args.cache_dir,
            max_cache_memory: args.max_cache_mb * 1024 * 1024,
            max_user_streams: args.max_streams,
            max_connections: args.max_connections,
            ping_interval_secs: args.ping_interval,
            connection_timeout_secs: args.connection_timeout,
            // Defensive clamp: CLI parser already enforces this, but keep config robust.
            stream_chunk_rows: args.chunk_rows.max(1),
            project_name,
        }
    }
}
