//! SAM3 compatibility engine wrapper.
//!
//! This wrapper gives `full_sam3` its own backend type and module boundary,
//! while delegating inference to the proven SAM2 ONNX engine implementation.

use std::path::Path;

use async_trait::async_trait;

use crate::full_sam::sam::{
    EncoderOutput, ExecutionProvider, OnnxSamEngine, SamBackend, SamMaskResult, SamPoint,
    SamVariant,
};

/// Temporary SAM3 backend engine that proxies to the SAM2 ONNX runtime.
pub struct Sam3CompatOnnxEngine {
    inner: OnnxSamEngine,
}

impl Sam3CompatOnnxEngine {
    pub fn new(
        model_dir: &Path,
        variant: SamVariant,
        provider: ExecutionProvider,
    ) -> anyhow::Result<Self> {
        let inner = OnnxSamEngine::new(model_dir, variant, provider)?;
        Ok(Self { inner })
    }
}

#[async_trait]
impl SamBackend for Sam3CompatOnnxEngine {
    fn name(&self) -> &'static str {
        "sam3-compat-onnx"
    }

    fn provider(&self) -> ExecutionProvider {
        self.inner.provider()
    }

    async fn encode_image(
        &self,
        image_rgb: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<EncoderOutput> {
        self.inner.encode_image(image_rgb, width, height).await
    }

    async fn decode_mask(
        &self,
        encoder_output: &EncoderOutput,
        original_width: u32,
        original_height: u32,
        points: &[SamPoint],
        box_prompt: Option<[f32; 4]>,
    ) -> anyhow::Result<SamMaskResult> {
        self.inner
            .decode_mask(
                encoder_output,
                original_width,
                original_height,
                points,
                box_prompt,
            )
            .await
    }

    fn is_available(&self) -> bool {
        self.inner.is_available()
    }
}
