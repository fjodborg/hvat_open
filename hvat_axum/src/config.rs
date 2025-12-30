//! Server configuration.

use std::path::PathBuf;

/// Server configuration, loaded from environment variables.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Port to listen on (default: 3000)
    pub port: u16,

    /// Root directory for image projects
    pub data_dir: PathBuf,

    /// Directory for pyramid cache
    pub cache_dir: PathBuf,

    /// Maximum total memory for pyramid cache in bytes (default: 2GB)
    pub max_cache_memory: u64,

    /// Maximum memory per user session in bytes (default: 500MB)
    pub max_user_memory: u64,

    /// Maximum concurrent streams per user (default: 4)
    pub max_user_streams: usize,

    /// Number of rows to send per WebSocket message (default: 128)
    pub stream_chunk_rows: u32,
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
        }
    }
}

impl ServerConfig {
    /// Load configuration from environment variables.
    ///
    /// Environment variables:
    /// - `HVAT_PORT` - Server port (default: 3000)
    /// - `HVAT_DATA_DIR` - Root data directory (default: ./data)
    /// - `HVAT_CACHE_DIR` - Pyramid cache directory (default: ./.cache/pyramids)
    /// - `HVAT_MAX_CACHE_MB` - Max cache memory in MB (default: 2048)
    /// - `HVAT_MAX_USER_MB` - Max per-user memory in MB (default: 500)
    /// - `HVAT_MAX_STREAMS` - Max streams per user (default: 4)
    /// - `HVAT_CHUNK_ROWS` - Rows per WebSocket message (default: 128)
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(port) = std::env::var("HVAT_PORT") {
            if let Ok(p) = port.parse() {
                config.port = p;
            }
        }

        if let Ok(dir) = std::env::var("HVAT_DATA_DIR") {
            config.data_dir = PathBuf::from(dir);
        }

        if let Ok(dir) = std::env::var("HVAT_CACHE_DIR") {
            config.cache_dir = PathBuf::from(dir);
        }

        if let Ok(mb) = std::env::var("HVAT_MAX_CACHE_MB") {
            if let Ok(m) = mb.parse::<u64>() {
                config.max_cache_memory = m * 1024 * 1024;
            }
        }

        if let Ok(mb) = std::env::var("HVAT_MAX_USER_MB") {
            if let Ok(m) = mb.parse::<u64>() {
                config.max_user_memory = m * 1024 * 1024;
            }
        }

        if let Ok(streams) = std::env::var("HVAT_MAX_STREAMS") {
            if let Ok(s) = streams.parse() {
                config.max_user_streams = s;
            }
        }

        if let Ok(rows) = std::env::var("HVAT_CHUNK_ROWS") {
            if let Ok(r) = rows.parse() {
                config.stream_chunk_rows = r;
            }
        }

        config
    }
}
