//! Application state shared across handlers.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use crate::config::ServerConfig;
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

    /// SAM engine for AI-assisted segmentation (None if SAM is disabled)
    pub sam_engine: Option<Arc<dyn SamBackend>>,

    /// Cache for SAM image embeddings
    pub embedding_cache: Arc<EmbeddingCache>,
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

        Self {
            config,
            loaders,
            pyramid_storage,
            pyramid_builder,
            sessions: RwLock::new(HashMap::new()),
            sam_engine,
            embedding_cache,
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
