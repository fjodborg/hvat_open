//! SAM model variant definitions and download utilities.
//!
//! Pre-exported ONNX models are available from Hugging Face:
//! <https://huggingface.co/vietanhdev/segment-anything-2-onnx-models>

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::io::AsyncWriteExt;

/// SAM 2 model variants.
///
/// Each variant has different size/speed/quality tradeoffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SamVariant {
    /// Tiny variant (~134 MB encoder, fastest).
    #[default]
    Tiny,
    /// Small variant (~163 MB encoder, better quality).
    Small,
    /// Base Plus variant (~340 MB encoder).
    BasePlus,
    /// Large variant (~889 MB encoder, best quality).
    Large,
}

impl SamVariant {
    /// Parse from string (for CLI/env config).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "tiny" | "t" => Some(Self::Tiny),
            "small" | "s" => Some(Self::Small),
            "base_plus" | "base-plus" | "baseplus" | "b+" => Some(Self::BasePlus),
            "large" | "l" => Some(Self::Large),
            _ => None,
        }
    }

    /// Get the encoder ONNX filename.
    pub fn encoder_filename(&self) -> &'static str {
        match self {
            Self::Tiny => "sam2_hiera_tiny.encoder.onnx",
            Self::Small => "sam2_hiera_small.encoder.onnx",
            Self::BasePlus => "sam2_hiera_base_plus.encoder.onnx",
            Self::Large => "sam2_hiera_large.encoder.onnx",
        }
    }

    /// Get the decoder ONNX filename.
    pub fn decoder_filename(&self) -> &'static str {
        match self {
            Self::Tiny => "sam2_hiera_tiny.decoder.onnx",
            Self::Small => "sam2_hiera_small.decoder.onnx",
            Self::BasePlus => "sam2_hiera_base_plus.decoder.onnx",
            Self::Large => "sam2_hiera_large.decoder.onnx",
        }
    }

    /// Get the Hugging Face download URL for the encoder.
    pub fn encoder_url(&self) -> String {
        let base = "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main";
        format!("{}/{}", base, self.encoder_filename())
    }

    /// Get the Hugging Face download URL for the decoder.
    pub fn decoder_url(&self) -> String {
        let base = "https://huggingface.co/vietanhdev/segment-anything-2-onnx-models/resolve/main";
        format!("{}/{}", base, self.decoder_filename())
    }

    /// Get approximate encoder size in MB.
    pub fn encoder_size_mb(&self) -> u32 {
        match self {
            Self::Tiny => 134,
            Self::Small => 163,
            Self::BasePlus => 340,
            Self::Large => 889,
        }
    }

    /// Get approximate decoder size in MB (same for all variants).
    pub fn decoder_size_mb(&self) -> u32 {
        21 // All decoders are ~20.6 MB
    }

    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Tiny => "Tiny",
            Self::Small => "Small",
            Self::BasePlus => "Base+",
            Self::Large => "Large",
        }
    }
}

/// Get the default model directory.
///
/// Returns `./.cache/models` (local to the current working directory).
pub fn default_model_dir() -> PathBuf {
    PathBuf::from(".cache/models")
}

/// Check if models exist for a variant.
pub async fn models_exist(model_dir: &Path, variant: SamVariant) -> bool {
    let encoder_path = model_dir.join(variant.encoder_filename());
    let decoder_path = model_dir.join(variant.decoder_filename());

    encoder_path.exists() && decoder_path.exists()
}

/// Ensure models are downloaded for a variant.
///
/// Downloads from Hugging Face if not present locally.
pub async fn ensure_models(model_dir: &Path, variant: SamVariant) -> Result<()> {
    // Create model directory if needed
    fs::create_dir_all(model_dir)
        .await
        .context("Failed to create model directory")?;

    let encoder_path = model_dir.join(variant.encoder_filename());
    let decoder_path = model_dir.join(variant.decoder_filename());

    // Download encoder if missing
    if !encoder_path.exists() {
        log::info!(
            "Downloading SAM {} encoder (~{} MB)...",
            variant.name(),
            variant.encoder_size_mb()
        );
        download_file(&variant.encoder_url(), &encoder_path).await?;
        log::info!("Encoder downloaded: {}", encoder_path.display());
    }

    // Download decoder if missing
    if !decoder_path.exists() {
        log::info!(
            "Downloading SAM {} decoder (~{} MB)...",
            variant.name(),
            variant.decoder_size_mb()
        );
        download_file(&variant.decoder_url(), &decoder_path).await?;
        log::info!("Decoder downloaded: {}", decoder_path.display());
    }

    Ok(())
}

/// Download a file from URL to local path.
async fn download_file(url: &str, path: &Path) -> Result<()> {
    // Use reqwest for HTTP download
    let response = reqwest::get(url)
        .await
        .context("Failed to start download")?
        .error_for_status()
        .context("Download failed with HTTP error")?;

    let bytes = response.bytes().await.context("Failed to read response")?;

    // Write to temp file first, then rename (atomic)
    let temp_path = path.with_extension("tmp");
    let mut file = fs::File::create(&temp_path)
        .await
        .context("Failed to create temp file")?;

    file.write_all(&bytes)
        .await
        .context("Failed to write file")?;
    file.flush().await?;

    fs::rename(&temp_path, path)
        .await
        .context("Failed to rename temp file")?;

    Ok(())
}

/// Get model file paths for a variant.
pub fn model_paths(model_dir: &Path, variant: SamVariant) -> (PathBuf, PathBuf) {
    (
        model_dir.join(variant.encoder_filename()),
        model_dir.join(variant.decoder_filename()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_parsing() {
        assert_eq!(SamVariant::from_str("tiny"), Some(SamVariant::Tiny));
        assert_eq!(SamVariant::from_str("SMALL"), Some(SamVariant::Small));
        assert_eq!(
            SamVariant::from_str("base-plus"),
            Some(SamVariant::BasePlus)
        );
        assert_eq!(SamVariant::from_str("large"), Some(SamVariant::Large));
        assert_eq!(SamVariant::from_str("invalid"), None);
    }

    #[test]
    fn test_urls() {
        let tiny = SamVariant::Tiny;
        assert!(tiny.encoder_url().contains("huggingface.co"));
        assert!(tiny.encoder_url().contains("tiny.encoder.onnx"));
        assert!(tiny.decoder_url().contains("tiny.decoder.onnx"));
    }
}
