//! Application state shared across handlers.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;

use crate::band_cache::PackedLayerCache;
use crate::config::ServerConfig;
use crate::inference::{ModelRegistry, SamInferenceAdapter};
use crate::loaders::ImageLoaderRegistry;
use crate::pyramid::{FilesystemStorage, PyramidBuilder, PyramidStorage};
use crate::sam::{EmbeddingCache, OnnxSamEngine, SamBackend};

#[derive(Debug, Clone, Serialize)]
pub struct EndpointMetricsSnapshot {
    pub requests: u64,
    pub errors: u64,
    pub avg_latency_ms: u64,
    pub max_latency_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RestMetricsSnapshot {
    pub image_bands: EndpointMetricsSnapshot,
    pub sam_infer: EndpointMetricsSnapshot,
    pub sam_warm: EndpointMetricsSnapshot,
}

#[derive(Debug, Default)]
struct EndpointMetrics {
    requests: AtomicU64,
    errors: AtomicU64,
    total_latency_ms: AtomicU64,
    max_latency_ms: AtomicU64,
}

impl EndpointMetrics {
    fn record(&self, latency_ms: u64, ok: bool) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.total_latency_ms
            .fetch_add(latency_ms, Ordering::Relaxed);
        if !ok {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
        update_max(&self.max_latency_ms, latency_ms);
    }

    fn snapshot(&self) -> EndpointMetricsSnapshot {
        let requests = self.requests.load(Ordering::Relaxed);
        let total_latency_ms = self.total_latency_ms.load(Ordering::Relaxed);
        let avg_latency_ms = if requests == 0 {
            0
        } else {
            total_latency_ms / requests
        };
        EndpointMetricsSnapshot {
            requests,
            errors: self.errors.load(Ordering::Relaxed),
            avg_latency_ms,
            max_latency_ms: self.max_latency_ms.load(Ordering::Relaxed),
        }
    }
}

fn update_max(max: &AtomicU64, value: u64) {
    let mut current = max.load(Ordering::Relaxed);
    while value > current {
        match max.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

#[derive(Debug, Default)]
pub struct RestRouteMetrics {
    image_bands: EndpointMetrics,
    sam_infer: EndpointMetrics,
    sam_warm: EndpointMetrics,
}

impl RestRouteMetrics {
    pub fn record_image_bands(&self, latency_ms: u64, ok: bool) {
        self.image_bands.record(latency_ms, ok);
    }

    pub fn record_sam_infer(&self, latency_ms: u64, ok: bool) {
        self.sam_infer.record(latency_ms, ok);
    }

    pub fn record_sam_warm(&self, latency_ms: u64, ok: bool) {
        self.sam_warm.record(latency_ms, ok);
    }

    #[must_use]
    pub fn snapshot(&self) -> RestMetricsSnapshot {
        RestMetricsSnapshot {
            image_bands: self.image_bands.snapshot(),
            sam_infer: self.sam_infer.snapshot(),
            sam_warm: self.sam_warm.snapshot(),
        }
    }
}

/// Errors that can occur when initializing application state.
#[derive(Debug, thiserror::Error)]
pub enum StateInitError {
    #[error(
        "SAM backend '{backend_name}' initialization failed: {source}. model directory: {model_dir}, expected files: {expected_files}"
    )]
    SamInit {
        #[source]
        source: anyhow::Error,
        backend_name: &'static str,
        model_dir: PathBuf,
        expected_files: String,
    },
}

type SamBackendFactoryFn = fn(&ServerConfig) -> anyhow::Result<Arc<dyn SamBackend>>;
type SamModelNameFn = fn(&ServerConfig) -> String;
type SamExpectedFilesFn = fn(&ServerConfig) -> Vec<String>;

/// Describes how SAM should be wired into the inference registry.
///
/// Keeping this explicit allows sibling backends (for example SAM2 and SAM3)
/// to share state plumbing while owning their model registration details.
#[derive(Clone, Copy)]
pub struct SamBackendWiring {
    pub backend_name: &'static str,
    pub model_id: &'static str,
    pub model_name: SamModelNameFn,
    pub expected_files: SamExpectedFilesFn,
    pub create_backend: SamBackendFactoryFn,
}

impl SamBackendWiring {
    pub fn sam2_default() -> Self {
        Self {
            backend_name: "sam2-onnx",
            model_id: "sam-base",
            model_name: sam2_model_name,
            expected_files: sam2_expected_files,
            create_backend: create_sam2_backend,
        }
    }
}

fn sam2_model_name(config: &ServerConfig) -> String {
    format!("Segment Anything ({})", config.sam_variant.name())
}

fn sam2_expected_files(config: &ServerConfig) -> Vec<String> {
    vec![
        config.sam_variant.encoder_filename().to_string(),
        config.sam_variant.decoder_filename().to_string(),
    ]
}

fn create_sam2_backend(config: &ServerConfig) -> anyhow::Result<Arc<dyn SamBackend>> {
    let engine = OnnxSamEngine::new(
        &config.sam_model_dir,
        config.sam_variant,
        config.sam_provider,
    )?;
    Ok(Arc::new(engine))
}

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

    /// Model registry for inference backends (Protocol v2)
    pub model_registry: Option<Arc<ModelRegistry>>,

    /// Active connection count retained for shared helper/example compatibility.
    pub active_connections: AtomicU64,

    /// Connection ID counter for generating unique IDs
    connection_id_counter: AtomicU64,

    /// REST route telemetry counters for dashboarding.
    pub rest_metrics: RestRouteMetrics,

    /// Packed RGBA layer cache — avoids re-decoding image files on every band request.
    pub band_cache: PackedLayerCache,
}

impl AppState {
    /// Create new application state.
    pub fn new(config: ServerConfig) -> std::result::Result<Self, StateInitError> {
        Self::new_with_sam_wiring(config, SamBackendWiring::sam2_default())
    }

    /// Create application state with explicit SAM backend wiring.
    pub fn new_with_sam_wiring(
        config: ServerConfig,
        sam_wiring: SamBackendWiring,
    ) -> std::result::Result<Self, StateInitError> {
        // Register image loaders
        let loaders = ImageLoaderRegistry::with_defaults();

        // Create pyramid storage
        let pyramid_storage: Arc<dyn PyramidStorage> =
            Arc::new(FilesystemStorage::new(config.cache_dir.clone()));

        // Create pyramid builder
        let pyramid_builder = PyramidBuilder::new(pyramid_storage.clone());

        // Create model registry and register SAM if enabled.
        let model_registry = if config.sam_enabled {
            let sam_backend = match (sam_wiring.create_backend)(&config) {
                Ok(engine) => {
                    log::info!(
                        "SAM engine '{}' initialized successfully",
                        sam_wiring.backend_name
                    );
                    engine
                }
                Err(e) => {
                    let expected_files = (sam_wiring.expected_files)(&config);
                    let expected_files = if expected_files.is_empty() {
                        "<unspecified>".to_string()
                    } else {
                        expected_files.join(", ")
                    };
                    return Err(StateInitError::SamInit {
                        source: e,
                        backend_name: sam_wiring.backend_name,
                        model_dir: config.sam_model_dir.clone(),
                        expected_files,
                    });
                }
            };

            let embedding_cache = Arc::new(EmbeddingCache::new(config.sam_cache_size));
            let mut registry = ModelRegistry::new();
            let adapter = SamInferenceAdapter::new(
                sam_backend,
                embedding_cache,
                sam_wiring.model_id,
                (sam_wiring.model_name)(&config),
            );
            registry.register(adapter);
            log::info!("Registered SAM model in inference registry");
            Some(Arc::new(registry))
        } else {
            log::info!("SAM is disabled in configuration");
            None
        };

        Ok(Self {
            band_cache: PackedLayerCache::new(config.band_cache_size),
            config,
            loaders,
            pyramid_storage,
            pyramid_builder,
            sessions: RwLock::new(HashMap::new()),
            pyramid_tasks: RwLock::new(HashMap::new()),
            model_registry,
            active_connections: AtomicU64::new(0),
            connection_id_counter: AtomicU64::new(0),
            rest_metrics: RestRouteMetrics::default(),
        })
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

impl crate::helper::http::projects::ProjectRoutesState for AppState {
    fn project_name(&self) -> &str {
        &self.config.project_name
    }

    fn data_dir(&self) -> &std::path::Path {
        &self.config.data_dir
    }

    fn loaders(&self) -> &ImageLoaderRegistry {
        &self.loaders
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

#[cfg(test)]
#[path = "state.test.rs"]
mod tests;
