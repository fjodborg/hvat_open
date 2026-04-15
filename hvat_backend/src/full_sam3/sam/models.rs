//! SAM3 runtime model specification helpers.

use std::path::{Path, PathBuf};

use crate::full_sam::sam::SamVariant;
use crate::full_sam3::config::{Sam3RuntimeKind, ServerConfig};

/// Runtime model specification for the SAM3 backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sam3RuntimeModelSpec {
    /// Compatibility runtime backed by SAM2 ONNX encoder/decoder artifacts.
    CompatOnnx { variant: SamVariant },
    /// Native SAM3 runtime backed by explicit checkpoint and config paths.
    NativeArtifacts {
        checkpoint_path: PathBuf,
        config_path: PathBuf,
    },
}

/// Errors while resolving a runtime model spec from configuration.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum Sam3ModelSpecError {
    #[error("missing --sam3-checkpoint for native runtime")]
    MissingNativeCheckpoint,
    #[error("missing --sam3-config for native runtime")]
    MissingNativeConfig,
    #[error("SAM3 native checkpoint not found: {path}")]
    NativeCheckpointNotFound { path: PathBuf },
    #[error("SAM3 native checkpoint must be a file: {path}")]
    NativeCheckpointNotFile { path: PathBuf },
    #[error("SAM3 native checkpoint must not be empty: {path}")]
    NativeCheckpointEmpty { path: PathBuf },
    #[error("failed to read SAM3 native checkpoint metadata ({path}): {reason}")]
    NativeCheckpointMetadataReadFailed { path: PathBuf, reason: String },
    #[error("SAM3 native config not found: {path}")]
    NativeConfigNotFound { path: PathBuf },
    #[error("SAM3 native config must be a file: {path}")]
    NativeConfigNotFile { path: PathBuf },
    #[error("SAM3 native config must be valid JSON ({path}): {reason}")]
    InvalidNativeConfigJson { path: PathBuf, reason: String },
}

/// Build a SAM3 runtime model spec from validated server configuration.
pub fn model_spec_from_config(
    config: &ServerConfig,
) -> std::result::Result<Sam3RuntimeModelSpec, Sam3ModelSpecError> {
    match config.sam3_runtime {
        Sam3RuntimeKind::CompatOnnx => Ok(Sam3RuntimeModelSpec::CompatOnnx {
            variant: config.sam_variant,
        }),
        Sam3RuntimeKind::Native => {
            let checkpoint_path = config
                .sam3_checkpoint
                .clone()
                .ok_or(Sam3ModelSpecError::MissingNativeCheckpoint)?;
            let config_path = config
                .sam3_config
                .clone()
                .ok_or(Sam3ModelSpecError::MissingNativeConfig)?;
            ensure_native_checkpoint_file(&checkpoint_path)?;
            ensure_native_config_file(&config_path)?;
            Ok(Sam3RuntimeModelSpec::NativeArtifacts {
                checkpoint_path,
                config_path,
            })
        }
    }
}

/// Expected model artifact descriptions for initialization diagnostics.
pub fn expected_model_files(spec: &Sam3RuntimeModelSpec) -> Vec<String> {
    match spec {
        Sam3RuntimeModelSpec::CompatOnnx { variant } => vec![
            variant.encoder_filename().to_string(),
            variant.decoder_filename().to_string(),
        ],
        Sam3RuntimeModelSpec::NativeArtifacts {
            checkpoint_path,
            config_path,
        } => vec![
            checkpoint_path.display().to_string(),
            config_path.display().to_string(),
        ],
    }
}

/// Backend identifier for diagnostics and startup logging.
pub fn backend_name(spec: &Sam3RuntimeModelSpec) -> &'static str {
    match spec {
        Sam3RuntimeModelSpec::CompatOnnx { .. } => "sam3-compat-onnx",
        Sam3RuntimeModelSpec::NativeArtifacts { .. } => "sam3-native",
    }
}

fn ensure_native_checkpoint_file(path: &Path) -> std::result::Result<(), Sam3ModelSpecError> {
    if !path.exists() {
        return Err(Sam3ModelSpecError::NativeCheckpointNotFound {
            path: path.to_path_buf(),
        });
    }

    if !path.is_file() {
        return Err(Sam3ModelSpecError::NativeCheckpointNotFile {
            path: path.to_path_buf(),
        });
    }

    let metadata = std::fs::metadata(path).map_err(|source| {
        Sam3ModelSpecError::NativeCheckpointMetadataReadFailed {
            path: path.to_path_buf(),
            reason: source.to_string(),
        }
    })?;
    if metadata.len() == 0 {
        return Err(Sam3ModelSpecError::NativeCheckpointEmpty {
            path: path.to_path_buf(),
        });
    }

    Ok(())
}

fn ensure_native_config_file(path: &Path) -> std::result::Result<(), Sam3ModelSpecError> {
    if !path.exists() {
        return Err(Sam3ModelSpecError::NativeConfigNotFound {
            path: path.to_path_buf(),
        });
    }

    if !path.is_file() {
        return Err(Sam3ModelSpecError::NativeConfigNotFile {
            path: path.to_path_buf(),
        });
    }

    let config_data = std::fs::read_to_string(path).map_err(|source| {
        Sam3ModelSpecError::InvalidNativeConfigJson {
            path: path.to_path_buf(),
            reason: format!("failed to read file: {source}"),
        }
    })?;
    serde_json::from_str::<serde_json::Value>(&config_data).map_err(|source| {
        Sam3ModelSpecError::InvalidNativeConfigJson {
            path: path.to_path_buf(),
            reason: source.to_string(),
        }
    })?;

    Ok(())
}

#[cfg(test)]
#[path = "models.test.rs"]
mod tests;
