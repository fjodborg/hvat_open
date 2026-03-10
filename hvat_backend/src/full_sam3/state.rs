//! SAM3 backend application state wiring.
//!
//! This module keeps SAM3-specific model wiring separate from the shared
//! state/runtime plumbing in `full_sam::state`.

use std::sync::Arc;

use crate::full_sam::sam::{OnnxSamEngine, SamBackend};
use crate::full_sam::state::SamBackendWiring;
use crate::full_sam3::ServerConfig;

pub use crate::full_sam::state::{AppState, SessionState, StateInitError};

/// Build application state for the SAM3 backend entrypoint.
///
/// Current implementation uses a SAM2-compatible ONNX engine while SAM3 runtime
/// support is developed, but keeps wiring isolated so SAM3 can diverge cleanly.
pub fn build_app_state(config: ServerConfig) -> std::result::Result<AppState, StateInitError> {
    AppState::new_with_sam_wiring(
        config,
        SamBackendWiring {
            backend_name: "sam3-compat-onnx",
            model_id: "sam-base",
            model_name: sam3_compat_model_name,
            expected_files: sam3_compat_expected_files,
            create_backend: create_sam3_compat_backend,
        },
    )
}

fn sam3_compat_model_name(config: &ServerConfig) -> String {
    format!("Segment Anything (compat: {})", config.sam_variant.name())
}

fn sam3_compat_expected_files(config: &ServerConfig) -> Vec<String> {
    vec![
        config.sam_variant.encoder_filename().to_string(),
        config.sam_variant.decoder_filename().to_string(),
    ]
}

fn create_sam3_compat_backend(config: &ServerConfig) -> anyhow::Result<Arc<dyn SamBackend>> {
    let engine = OnnxSamEngine::new(
        &config.sam_model_dir,
        config.sam_variant,
        config.sam_provider,
    )?;
    Ok(Arc::new(engine))
}
