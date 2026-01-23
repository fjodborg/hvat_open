//! Registry of available inference backends.

use super::InferenceBackend;
use hvat_common::ModelCapability;
use std::collections::HashMap;
use std::sync::Arc;

/// Registry of available inference backends.
///
/// Allows dynamic registration of model backends and provides
/// capability discovery for the protocol layer.
pub struct ModelRegistry {
    backends: HashMap<String, Arc<dyn InferenceBackend>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    /// Register an inference backend.
    pub fn register(&mut self, backend: impl InferenceBackend + 'static) {
        let id = backend.model_id().to_string();
        log::info!("Registering inference backend: {}", id);
        self.backends.insert(id, Arc::new(backend));
    }

    /// Get a backend by model ID.
    pub fn get(&self, model_id: &str) -> Option<Arc<dyn InferenceBackend>> {
        self.backends.get(model_id).cloned()
    }

    /// Get all registered model IDs.
    pub fn model_ids(&self) -> Vec<&str> {
        self.backends.keys().map(|s| s.as_str()).collect()
    }

    /// Build capabilities list for protocol.
    pub fn capabilities(&self) -> Vec<ModelCapability> {
        self.backends.values().map(|b| b.capability()).collect()
    }

    /// Check if any models are registered.
    pub fn is_empty(&self) -> bool {
        self.backends.is_empty()
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}
