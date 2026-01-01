//! SAM backend trait and types for AI-assisted segmentation.
//!
//! Defines the interface for SAM inference backends, enabling
//! different implementations (ONNX, future alternatives).

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Execution provider for ONNX Runtime.
///
/// Determines which hardware accelerator to use for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionProvider {
    /// CPU inference (always available, slowest).
    #[default]
    Cpu,
    /// NVIDIA CUDA acceleration.
    Cuda,
    /// AMD ROCm acceleration.
    Rocm,
    /// Windows DirectML acceleration.
    DirectML,
    /// Apple CoreML acceleration.
    CoreML,
}

impl ExecutionProvider {
    /// Parse from string (for CLI/env config).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "cuda" => Some(Self::Cuda),
            "rocm" => Some(Self::Rocm),
            "directml" => Some(Self::DirectML),
            "coreml" => Some(Self::CoreML),
            _ => None,
        }
    }

    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Cuda => "CUDA",
            Self::Rocm => "ROCm",
            Self::DirectML => "DirectML",
            Self::CoreML => "CoreML",
        }
    }
}

/// A point prompt for SAM segmentation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamPoint {
    /// X coordinate in image space (pixels).
    pub x: f32,
    /// Y coordinate in image space (pixels).
    pub y: f32,
    /// Label: 1 = foreground (include), 0 = background (exclude).
    pub label: i32,
}

impl SamPoint {
    /// Create a foreground point (object to segment).
    pub fn foreground(x: f32, y: f32) -> Self {
        Self { x, y, label: 1 }
    }

    /// Create a background point (area to exclude).
    pub fn background(x: f32, y: f32) -> Self {
        Self { x, y, label: 0 }
    }
}

/// Output from the SAM encoder.
///
/// SAM 2 encoder produces three outputs needed for decoding:
/// - Main image embedding
/// - Two sets of high-resolution features
#[derive(Debug, Clone)]
pub struct EncoderOutput {
    /// Main image embedding (256, 64, 64).
    pub image_embed: Vec<f32>,
    /// High resolution features 0 (32, 256, 256).
    pub high_res_feats_0: Vec<f32>,
    /// High resolution features 1 (64, 128, 128).
    pub high_res_feats_1: Vec<f32>,
}

/// Result of SAM mask prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamMaskResult {
    /// Predicted masks as binary data (one per prediction).
    /// Each mask is width * height bytes (0 or 255).
    pub masks: Vec<Vec<u8>>,
    /// Vectorized polygon contours for each mask.
    /// Each polygon is a list of [x, y] points.
    pub polygons: Vec<Vec<[f32; 2]>>,
    /// IoU (Intersection over Union) confidence scores for each mask.
    pub iou_scores: Vec<f32>,
    /// Original image dimensions used for mask generation.
    pub width: u32,
    pub height: u32,
}

/// Backend trait for SAM inference.
///
/// Implementations provide the actual model inference logic.
/// The trait is designed for the encoder-decoder architecture:
/// - `encode_image`: Run the heavy encoder once per image (cached)
/// - `decode_mask`: Run the light decoder for each prompt (fast)
#[async_trait]
pub trait SamBackend: Send + Sync {
    /// Get the name of this backend implementation.
    fn name(&self) -> &'static str;

    /// Get the execution provider being used.
    fn provider(&self) -> ExecutionProvider;

    /// Compute image embedding (expensive operation, should be cached).
    ///
    /// # Arguments
    /// * `image_rgb` - RGB image data (width * height * 3 bytes)
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    ///
    /// # Returns
    /// Encoder outputs: (image_embed, high_res_feats_0, high_res_feats_1)
    async fn encode_image(
        &self,
        image_rgb: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<EncoderOutput>;

    /// Decode mask from embedding and prompts (fast operation).
    ///
    /// # Arguments
    /// * `encoder_output` - Pre-computed encoder output from `encode_image`
    /// * `original_width` - Original image width (for coordinate scaling)
    /// * `original_height` - Original image height (for coordinate scaling)
    /// * `points` - Point prompts (foreground/background clicks)
    /// * `box_prompt` - Optional bounding box prompt [x1, y1, x2, y2]
    ///
    /// # Returns
    /// Mask prediction result with binary masks, polygons, and scores.
    async fn decode_mask(
        &self,
        encoder_output: &EncoderOutput,
        original_width: u32,
        original_height: u32,
        points: &[SamPoint],
        box_prompt: Option<[f32; 4]>,
    ) -> anyhow::Result<SamMaskResult>;

    /// Check if this backend is available on the current system.
    ///
    /// This should check if the required runtime libraries are present.
    fn is_available(&self) -> bool;
}
