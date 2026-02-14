//! Server configuration.

use std::path::PathBuf;

use clap::Parser;

use crate::sam::{ExecutionProvider, SamVariant};

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

    /// Maximum total WebSocket connections
    #[arg(long, default_value = "100", env = "HVAT_MAX_CONNECTIONS")]
    pub max_connections: usize,

    /// WebSocket ping interval in seconds (0 to disable)
    #[arg(long, default_value = "30", env = "HVAT_PING_INTERVAL")]
    pub ping_interval: u64,

    /// WebSocket connection timeout in seconds (no pong response)
    #[arg(long, default_value = "90", env = "HVAT_CONNECTION_TIMEOUT")]
    pub connection_timeout: u64,

    /// Number of rows to send per WebSocket message
    #[arg(long, default_value = "128", env = "HVAT_CHUNK_ROWS")]
    pub chunk_rows: u32,

    /// Maximum concurrent pyramid pre-generation jobs at startup
    #[arg(long, default_value = "2", env = "HVAT_PYRAMID_CONCURRENCY")]
    pub pyramid_concurrency: usize,

    /// Project name (display name for this server/data directory).
    /// Defaults to the data directory name.
    #[arg(short = 'n', long, env = "HVAT_PROJECT_NAME")]
    pub name: Option<String>,

    // --- SAM Configuration ---
    /// Enable SAM (Segment Anything Model) integration.
    #[arg(long, env = "HVAT_SAM_ENABLED", default_value = "false")]
    pub sam_enabled: bool,

    /// SAM model directory. Defaults to ./.cache/models
    #[arg(long, env = "HVAT_SAM_MODEL_DIR")]
    pub sam_model_dir: Option<PathBuf>,

    /// SAM model variant: tiny, small, base-plus, large
    #[arg(long, env = "HVAT_SAM_VARIANT", default_value = "tiny")]
    pub sam_variant: String,

    /// SAM execution provider: auto, cpu, cuda, rocm, directml, coreml
    #[arg(long, env = "HVAT_SAM_PROVIDER", default_value = "auto")]
    pub sam_provider: String,

    /// Maximum cached image embeddings for SAM
    #[arg(long, env = "HVAT_SAM_CACHE_SIZE", default_value = "50")]
    pub sam_cache_size: usize,
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

    /// Maximum total WebSocket connections
    pub max_connections: usize,

    /// WebSocket ping interval in seconds (0 to disable keepalive)
    pub ping_interval_secs: u64,

    /// WebSocket connection timeout in seconds (no pong response)
    pub connection_timeout_secs: u64,

    /// Number of rows to send per WebSocket message
    pub stream_chunk_rows: u32,

    /// Maximum concurrent pyramid pre-generation jobs at startup
    pub pyramid_concurrency: usize,

    /// Project name (display name)
    pub project_name: String,

    // --- SAM Configuration ---
    /// Enable SAM integration
    pub sam_enabled: bool,

    /// SAM model directory
    pub sam_model_dir: PathBuf,

    /// SAM model variant
    pub sam_variant: SamVariant,

    /// SAM execution provider
    pub sam_provider: ExecutionProvider,

    /// Maximum cached embeddings
    pub sam_cache_size: usize,
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
            max_connections: 100,
            ping_interval_secs: 30,
            connection_timeout_secs: 90,
            stream_chunk_rows: 128,
            pyramid_concurrency: 2,
            project_name: "data".to_string(),
            // SAM defaults
            sam_enabled: false,
            sam_model_dir: crate::sam::models::default_model_dir(),
            sam_variant: SamVariant::default(),
            sam_provider: ExecutionProvider::default(),
            sam_cache_size: 50,
        }
    }
}

impl ServerConfig {
    /// Create configuration from CLI arguments.
    ///
    /// # Panics
    ///
    /// Panics if SAM-specific flags are used without `--sam-enabled`.
    pub fn from_cli(args: CliArgs) -> Self {
        // Validate SAM configuration - error if SAM flags used without --sam-enabled
        Self::validate_sam_flags(&args);

        // Derive project name from data_dir if not specified
        let project_name = args.name.unwrap_or_else(|| {
            args.data_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("project")
                .to_string()
        });

        // Parse SAM variant
        let sam_variant = args.sam_variant.parse().unwrap_or_default();

        // Parse SAM provider (auto = detect best available)
        let sam_provider = if args.sam_provider == "auto" {
            crate::sam::engine::detect_best_provider()
        } else {
            args.sam_provider.parse().unwrap_or_default()
        };

        // SAM model directory
        let sam_model_dir = args
            .sam_model_dir
            .unwrap_or_else(crate::sam::models::default_model_dir);

        Self {
            port: args.port,
            data_dir: args.data_dir,
            cache_dir: args.cache_dir,
            max_cache_memory: args.max_cache_mb * 1024 * 1024,
            max_user_memory: args.max_user_mb * 1024 * 1024,
            max_user_streams: args.max_streams,
            max_connections: args.max_connections,
            ping_interval_secs: args.ping_interval,
            connection_timeout_secs: args.connection_timeout,
            stream_chunk_rows: args.chunk_rows,
            pyramid_concurrency: args.pyramid_concurrency.max(1),
            project_name,
            // SAM
            sam_enabled: args.sam_enabled,
            sam_model_dir,
            sam_variant,
            sam_provider,
            sam_cache_size: args.sam_cache_size,
        }
    }

    /// Validate SAM-related CLI flags.
    ///
    /// Panics if SAM-specific flags are used without `--sam-enabled`.
    fn validate_sam_flags(args: &CliArgs) {
        if args.sam_enabled {
            return; // All good, SAM is enabled
        }

        // Check if any SAM-specific flags were explicitly set (not defaults)
        let has_custom_variant = args.sam_variant != "tiny";
        let has_custom_provider = args.sam_provider != "auto";
        let has_custom_model_dir = args.sam_model_dir.is_some();
        let has_custom_cache_size = args.sam_cache_size != 50;

        let mut issues = Vec::new();
        if has_custom_variant {
            issues.push(format!("--sam-variant={}", args.sam_variant));
        }
        if has_custom_provider {
            issues.push(format!("--sam-provider={}", args.sam_provider));
        }
        if has_custom_model_dir {
            issues.push(format!(
                "--sam-model-dir={}",
                args.sam_model_dir.as_ref().unwrap().display()
            ));
        }
        if has_custom_cache_size {
            issues.push(format!("--sam-cache-size={}", args.sam_cache_size));
        }

        if !issues.is_empty() {
            panic!(
                "\n\nERROR: SAM flags specified without --sam-enabled!\n\n\
                 The following SAM flags were set:\n  {}\n\n\
                 But --sam-enabled was not specified (defaults to false).\n\n\
                 To enable SAM, add --sam-enabled to your command:\n  \
                 cargo run -p hvat_axum -- --sam-enabled {}\n\n",
                issues.join("\n  "),
                issues.join(" ")
            );
        }
    }

    /// Load configuration from environment variables.
    pub fn from_env() -> Self {
        // Parse CLI args which also reads env vars via clap
        Self::from_cli(CliArgs::parse())
    }
}
