//! SAM3 backend application state wiring.
//!
//! This module keeps SAM3-specific model wiring separate from the shared
//! state/runtime plumbing in `full_sam::state`.

use std::sync::Arc;

use crate::full_sam::config::ServerConfig as FullSamServerConfig;
use crate::full_sam::inference::{ModelRegistry, SamInferenceAdapter};
use crate::full_sam::sam::EmbeddingCache;
use crate::full_sam::state::SamBackendWiring;
use crate::full_sam3::ServerConfig;
use crate::full_sam3::sam::{Sam3CompatOnnxEngine, Sam3RuntimeModelSpec, SamBackend, models};

pub use crate::full_sam::state::{AppState, SessionState, StateInitError};

pub const SAM3_BACKEND_NAME: &str = "sam3-runtime";
pub const SAM3_COMPAT_BACKEND_NAME: &str = "sam3-compat-onnx";
pub const SAM3_COMPAT_MODEL_ID: &str = "sam3-compat-base";

/// Build application state for the SAM3 backend entrypoint.
///
/// The runtime remains compat-onnx by default, but this path can now accept a
/// native SAM3 runtime spec and report explicit initialization errors while
/// native engine support is still being implemented.
pub fn build_app_state(config: ServerConfig) -> std::result::Result<AppState, StateInitError> {
    let mut base_config = config.to_full_sam_config();
    // Prevent AppState::new from initializing the SAM2 runtime automatically.
    base_config.sam_enabled = false;

    let mut state = AppState::new(base_config)?;

    if !config.sam_enabled {
        return Ok(state);
    }

    let spec = resolve_model_spec(&config)?;
    let backend = create_backend_for_spec(&config, &spec).map_err(|source| {
        let expected = models::expected_model_files(&spec);
        StateInitError::SamInit {
            source,
            backend_name: models::backend_name(&spec),
            model_dir: config.sam_model_dir.clone(),
            expected_files: format_expected_files(&expected),
        }
    })?;

    let mut registry = ModelRegistry::new();
    let adapter = SamInferenceAdapter::new(
        backend,
        Arc::new(EmbeddingCache::new(config.sam_cache_size)),
        SAM3_COMPAT_MODEL_ID,
        sam3_model_name_from_spec(&spec),
    );
    registry.register(adapter);
    state.model_registry = Some(Arc::new(registry));
    Ok(state)
}

fn resolve_model_spec(
    config: &ServerConfig,
) -> std::result::Result<Sam3RuntimeModelSpec, StateInitError> {
    models::model_spec_from_config(config).map_err(|source| {
        let expected = sam3_expected_files_from_config(config);
        StateInitError::SamInit {
            source: source.into(),
            backend_name: SAM3_BACKEND_NAME,
            model_dir: config.sam_model_dir.clone(),
            expected_files: format_expected_files(&expected),
        }
    })
}

fn create_backend_for_spec(
    config: &ServerConfig,
    spec: &Sam3RuntimeModelSpec,
) -> anyhow::Result<Arc<dyn SamBackend>> {
    match spec {
        Sam3RuntimeModelSpec::CompatOnnx { variant } => {
            let engine =
                Sam3CompatOnnxEngine::new(&config.sam_model_dir, *variant, config.sam_provider)?;
            Ok(Arc::new(engine))
        }
        Sam3RuntimeModelSpec::NativeArtifacts {
            checkpoint_path,
            config_path,
        } => anyhow::bail!(
            "SAM3 native runtime is not implemented yet (checkpoint: {}, config: {})",
            checkpoint_path.display(),
            config_path.display()
        ),
    }
}

fn sam3_model_name_from_spec(spec: &Sam3RuntimeModelSpec) -> String {
    match spec {
        Sam3RuntimeModelSpec::CompatOnnx { variant } => {
            format!("Segment Anything 3 (compat: {})", variant.name())
        }
        Sam3RuntimeModelSpec::NativeArtifacts {
            checkpoint_path, ..
        } => {
            let label = checkpoint_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("checkpoint");
            format!("Segment Anything 3 (native: {label})")
        }
    }
}

fn sam3_expected_files_from_config(config: &ServerConfig) -> Vec<String> {
    match models::model_spec_from_config(config) {
        Ok(spec) => models::expected_model_files(&spec),
        Err(err) => vec![format!("<invalid sam3 runtime config: {err}>")],
    }
}

fn format_expected_files(files: &[String]) -> String {
    if files.is_empty() {
        "<unspecified>".to_string()
    } else {
        files.join(", ")
    }
}

/// Legacy wiring helper retained for existing capability-ID tests.
///
/// This helper still uses full_sam config callbacks and compat-onnx behavior.
pub fn sam3_compat_wiring() -> SamBackendWiring {
    SamBackendWiring {
        backend_name: SAM3_COMPAT_BACKEND_NAME,
        model_id: SAM3_COMPAT_MODEL_ID,
        model_name: sam3_compat_model_name,
        expected_files: sam3_compat_expected_files,
        create_backend: create_sam3_compat_backend,
    }
}

fn sam3_compat_model_name(config: &FullSamServerConfig) -> String {
    format!("Segment Anything 3 (compat: {})", config.sam_variant.name())
}

fn sam3_compat_expected_files(config: &FullSamServerConfig) -> Vec<String> {
    vec![
        config.sam_variant.encoder_filename().to_string(),
        config.sam_variant.decoder_filename().to_string(),
    ]
}

fn create_sam3_compat_backend(config: &FullSamServerConfig) -> anyhow::Result<Arc<dyn SamBackend>> {
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
