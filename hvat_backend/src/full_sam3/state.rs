//! SAM3 backend application state wiring.
//!
//! This module keeps SAM3-specific model wiring separate from the shared
//! state/runtime plumbing in `full_sam::state`.

use std::sync::Arc;

use crate::full_sam::state::SamBackendWiring;
use crate::full_sam3::ServerConfig;
use crate::full_sam3::sam::{Sam3CompatOnnxEngine, SamBackend, models};

pub use crate::full_sam::state::{AppState, SessionState, StateInitError};

pub const SAM3_COMPAT_BACKEND_NAME: &str = "sam3-compat-onnx";
pub const SAM3_COMPAT_MODEL_ID: &str = "sam3-compat-base";

/// Build application state for the SAM3 backend entrypoint.
///
/// Current implementation uses a SAM2-compatible ONNX engine while SAM3 runtime
/// support is developed, but keeps wiring isolated so SAM3 can diverge cleanly.
pub fn build_app_state(config: ServerConfig) -> std::result::Result<AppState, StateInitError> {
    AppState::new_with_sam_wiring(config, sam3_compat_wiring())
}

/// SAM wiring used by the `full_sam3` executable today.
pub fn sam3_compat_wiring() -> SamBackendWiring {
    SamBackendWiring {
        backend_name: SAM3_COMPAT_BACKEND_NAME,
        model_id: SAM3_COMPAT_MODEL_ID,
        model_name: sam3_compat_model_name,
        expected_files: sam3_compat_expected_files,
        create_backend: create_sam3_compat_backend,
    }
}

fn sam3_compat_model_name(config: &ServerConfig) -> String {
    format!("Segment Anything 3 (compat: {})", config.sam_variant.name())
}

fn sam3_compat_expected_files(config: &ServerConfig) -> Vec<String> {
    models::expected_onnx_files(config.sam_variant).into()
}

fn create_sam3_compat_backend(config: &ServerConfig) -> anyhow::Result<Arc<dyn SamBackend>> {
    let engine = Sam3CompatOnnxEngine::new(
        &config.sam_model_dir,
        config.sam_variant,
        config.sam_provider,
    )?;
    Ok(Arc::new(engine))
}

#[cfg(test)]
#[path = "state.test.rs"]
mod tests;
