//! Generic inference backend trait for any ML model.

use async_trait::async_trait;
use hvat_common::ModelCapability;
use serde_json::Value;

/// Image context for inference operations.
///
/// Note: For efficiency, the WebSocket handler should cache loaded image data
/// per connection to avoid reloading for each `infer` call. The `rgb_data` field
/// may be empty for `infer` calls on models that use cached embeddings (like SAM),
/// but must be populated for `prepare` calls.
pub struct ImageContext {
    pub image_id: String,
    /// RGB pixel data. May be empty if model uses cached embeddings and this is an `infer` call.
    pub rgb_data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Result of an inference operation.
pub struct InferenceResult {
    pub model_id: String,
    pub outputs: Value, // JSON matching model's output schema
    pub timing_ms: u64,
}

/// Progress callback for long-running operations.
pub type ProgressCallback = Box<dyn Fn(u8, &str) + Send + Sync>;

/// Generic inference backend trait for any ML model.
///
/// Implementations provide model-specific inference logic while
/// exposing a uniform interface for the protocol layer.
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Unique model identifier (e.g., "sam-base", "yolo-v8-detect").
    fn model_id(&self) -> &str;

    /// Model capability description for protocol negotiation.
    fn capability(&self) -> ModelCapability;

    /// Whether this model requires pre-computed embeddings.
    fn requires_embedding(&self) -> bool {
        false
    }

    /// Pre-compute embeddings for the given image (optional).
    ///
    /// Called when client sends `prepare_model`. For models like SAM
    /// that benefit from caching expensive encoder outputs.
    ///
    /// # Default Implementation
    /// This default returns `Ok(())` immediately for models that don't need preparation.
    /// `_image` and `_progress` are prefixed with underscore because:
    /// 1. SAM overrides this method and DOES use both parameters (see sam_adapter.rs:86)
    /// 2. Future models (YOLO, DINOv2) may override and use them differently
    /// 3. Simple models that don't need preparation use this no-op default
    /// The trait signature needs these parameters for models that DO use them.
    async fn prepare(
        &self,
        _image: &ImageContext,
        _progress: Option<ProgressCallback>,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// Check if embeddings are ready for the given image.
    ///
    /// # Default Implementation
    /// Returns `true` for models that don't need preparation.
    /// `_image_id` is prefixed with underscore because:
    /// 1. SAM overrides this and DOES use image_id (see sam_adapter.rs:118)
    /// 2. Models without embeddings use this default which doesn't need the ID
    /// The trait signature needs this parameter for models that track per-image state.
    async fn is_prepared(&self, _image_id: &str) -> bool {
        true // Models without embeddings are always "prepared"
    }

    /// Run inference with generic JSON inputs.
    ///
    /// # Arguments
    /// * `image` - Image context (may include cached embeddings)
    /// * `inputs` - JSON object matching model's input schema
    /// * `options` - JSON object with model options (or empty)
    /// * `progress` - Optional progress callback
    async fn infer(
        &self,
        image: &ImageContext,
        inputs: Value,
        options: Value,
        progress: Option<ProgressCallback>,
    ) -> anyhow::Result<InferenceResult>;

    /// Validate inputs against the model's schema.
    ///
    /// Returns `Ok(())` if valid, or an error describing the problem.
    ///
    /// # Default Implementation
    /// Accepts any input without validation.
    /// `_inputs` is prefixed with underscore because:
    /// 1. SAM overrides this and DOES use inputs (see sam_adapter.rs:192)
    /// 2. Models without strict schemas use this permissive default
    /// The trait signature needs this parameter for models that validate inputs.
    fn validate_inputs(&self, _inputs: &Value) -> anyhow::Result<()> {
        Ok(()) // Default: accept anything
    }
}
