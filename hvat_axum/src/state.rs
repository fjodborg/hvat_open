//! Application state shared across handlers.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use crate::config::ServerConfig;
use crate::inference::{ModelRegistry, SamInferenceAdapter};
use crate::loaders::{ImageLoaderRegistry, NpyLoader, StandardImageLoader};
use crate::pyramid::{FilesystemStorage, PyramidBuilder, PyramidStorage};
use crate::sam::{EmbeddingCache, OnnxSamEngine, SamBackend};

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

    /// Active pyramid build tasks (image_hash -> task handle)
    ///
    /// Used to track and potentially cancel in-progress pyramid builds.
    /// Tasks are removed when they complete (success or failure).
    pub pyramid_tasks: RwLock<HashMap<String, JoinHandle<()>>>,

    /// SAM engine for AI-assisted segmentation (None if SAM is disabled)
    pub sam_engine: Option<Arc<dyn SamBackend>>,

    /// Cache for SAM image embeddings
    pub embedding_cache: Arc<EmbeddingCache>,

    /// Model registry for inference backends (Protocol v2)
    pub model_registry: Option<Arc<ModelRegistry>>,

    /// Active WebSocket connection count (atomic for lock-free access)
    pub active_connections: AtomicU64,

    /// Connection ID counter for generating unique IDs
    connection_id_counter: AtomicU64,
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

        // Create embedding cache
        let embedding_cache = Arc::new(EmbeddingCache::new(config.sam_cache_size));

        // Create SAM engine if enabled
        let sam_engine: Option<Arc<dyn SamBackend>> = if config.sam_enabled {
            match OnnxSamEngine::new(
                &config.sam_model_dir,
                config.sam_variant,
                config.sam_provider,
            ) {
                Ok(engine) => {
                    log::info!("SAM engine initialized successfully");
                    Some(Arc::new(engine))
                }
                Err(e) => {
                    // User explicitly enabled SAM but initialization failed.
                    // This is a configuration error - fail loudly!
                    panic!(
                        "\n\nERROR: SAM initialization failed!\n\n\
                         You specified --sam-enabled but SAM could not be initialized:\n\
                         {}\n\n\
                         This is usually because the ONNX model files are missing.\n\
                         Expected model directory: {}\n\
                         Expected files:\n\
                           - {}\n\
                           - {}\n\n\
                         To fix this, either:\n\
                         1. Download the models manually from:\n\
                            https://huggingface.co/vietanhdev/segment-anything-2-onnx-models\n\
                         2. Remove --sam-enabled to run without SAM\n\n",
                        e,
                        config.sam_model_dir.display(),
                        config.sam_variant.encoder_filename(),
                        config.sam_variant.decoder_filename(),
                    );
                }
            }
        } else {
            log::info!("SAM is disabled in configuration");
            None
        };

        // Create model registry and register SAM if enabled
        let model_registry = if let Some(ref sam_backend) = sam_engine {
            let mut registry = ModelRegistry::new();
            let adapter = SamInferenceAdapter::new(
                sam_backend.clone(),
                embedding_cache.clone(),
                "sam-base",
                format!("Segment Anything ({})", config.sam_variant.name()),
            );
            registry.register(adapter);
            log::info!("Registered SAM model in inference registry");
            Some(Arc::new(registry))
        } else {
            None
        };

        Self {
            config,
            loaders,
            pyramid_storage,
            pyramid_builder,
            sessions: RwLock::new(HashMap::new()),
            pyramid_tasks: RwLock::new(HashMap::new()),
            sam_engine,
            embedding_cache,
            model_registry,
            active_connections: AtomicU64::new(0),
            connection_id_counter: AtomicU64::new(0),
        }
    }

    /// Try to acquire a connection slot.
    ///
    /// Returns `Some(connection_id)` if under the limit, `None` if at capacity.
    pub fn try_acquire_connection(&self) -> Option<u64> {
        let max_connections = self.config.max_connections as u64;
        let current = self.active_connections.fetch_add(1, Ordering::SeqCst);

        if current >= max_connections {
            // Over limit, rollback
            self.active_connections.fetch_sub(1, Ordering::SeqCst);
            None
        } else {
            // Generate unique connection ID
            let id = self.connection_id_counter.fetch_add(1, Ordering::SeqCst);
            Some(id)
        }
    }

    /// Release a connection slot.
    pub fn release_connection(&self) {
        self.active_connections.fetch_sub(1, Ordering::SeqCst);
    }

    /// Get current active connection count.
    pub fn connection_count(&self) -> u64 {
        self.active_connections.load(Ordering::SeqCst)
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
