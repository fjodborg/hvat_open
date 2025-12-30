//! Application state shared across handlers.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::config::ServerConfig;
use crate::loaders::{ImageLoaderRegistry, NpyLoader, StandardImageLoader};
use crate::pyramid::{FilesystemStorage, PyramidBuilder, PyramidStorage};

/// Shared application state.
pub struct AppState {
    /// Server configuration
    pub config: ServerConfig,

    /// Registry of image loaders
    pub loaders: ImageLoaderRegistry,

    /// Pyramid storage backend
    pub pyramid_storage: Arc<dyn PyramidStorage>,

    /// Pyramid builder
    pub pyramid_builder: PyramidBuilder,

    /// Active user sessions (user_id -> session state)
    pub sessions: RwLock<HashMap<String, SessionState>>,
}

impl AppState {
    /// Create new application state.
    pub fn new(config: ServerConfig) -> Self {
        // Register image loaders
        let mut loaders = ImageLoaderRegistry::new();
        loaders.register(Arc::new(StandardImageLoader::new()));
        loaders.register(Arc::new(NpyLoader::new()));

        // Create pyramid storage
        let pyramid_storage: Arc<dyn PyramidStorage> =
            Arc::new(FilesystemStorage::new(config.cache_dir.clone()));

        // Create pyramid builder
        let pyramid_builder = PyramidBuilder::new(pyramid_storage.clone());

        Self {
            config,
            loaders,
            pyramid_storage,
            pyramid_builder,
            sessions: RwLock::new(HashMap::new()),
        }
    }
}

/// Per-user session state for resource management.
#[derive(Debug)]
pub struct SessionState {
    /// Currently active stream count
    pub active_streams: usize,

    /// Memory currently used by this session (bytes)
    pub memory_used: u64,

    /// Last activity timestamp
    pub last_activity: std::time::Instant,
}

impl Default for SessionState {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionState {
    pub fn new() -> Self {
        Self {
            active_streams: 0,
            memory_used: 0,
            last_activity: std::time::Instant::now(),
        }
    }
}
