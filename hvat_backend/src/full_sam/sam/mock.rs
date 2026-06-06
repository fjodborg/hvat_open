//! Deterministic mock SAM backend for integration and E2E testing.
//!
//! This backend avoids ONNX model dependencies while exercising the same
//! prepare/infer code paths as production SAM wiring.

use std::sync::Arc;

use async_trait::async_trait;

use crate::config::ServerConfig;

use super::{EncoderOutput, ExecutionProvider, SamBackend, SamMaskResult, SamPoint};

/// Lightweight SAM backend that returns deterministic box-like polygons.
pub struct MockSamBackend;

#[async_trait]
impl SamBackend for MockSamBackend {
    fn name(&self) -> &'static str {
        "mock-sam"
    }

    fn provider(&self) -> ExecutionProvider {
        ExecutionProvider::Cpu
    }

    async fn encode_image(
        &self,
        _image_rgb: &[u8],
        _width: u32,
        _height: u32,
    ) -> anyhow::Result<EncoderOutput> {
        Ok(EncoderOutput::BackendSpecific {
            backend_name: "mock-sam".to_string(),
            format: "mock-v1".to_string(),
            data: vec![1],
        })
    }

    async fn decode_mask(
        &self,
        _encoder_output: &EncoderOutput,
        original_width: u32,
        original_height: u32,
        points: &[SamPoint],
        box_prompt: Option<[f32; 4]>,
    ) -> anyhow::Result<SamMaskResult> {
        let (x1, y1, x2, y2) = if let Some([x1, y1, x2, y2]) = box_prompt {
            (x1, y1, x2, y2)
        } else if let Some(point) = points.iter().find(|p| p.label == 1) {
            let radius = 24.0;
            (
                (point.x - radius).max(0.0),
                (point.y - radius).max(0.0),
                (point.x + radius).min(original_width as f32),
                (point.y + radius).min(original_height as f32),
            )
        } else {
            (0.0, 0.0, 32.0, 32.0)
        };

        Ok(SamMaskResult {
            masks: vec![vec![0u8; (original_width * original_height) as usize]],
            polygons: vec![vec![[x1, y1], [x2, y1], [x2, y2], [x1, y2]]],
            iou_scores: vec![0.93],
            width: original_width,
            height: original_height,
        })
    }

    fn is_available(&self) -> bool {
        true
    }
}

pub fn mock_model_name(_: &ServerConfig) -> String {
    "Mock SAM".to_string()
}

pub fn mock_expected_files(_: &ServerConfig) -> Vec<String> {
    vec!["mock".to_string()]
}

pub fn mock_factory(_: &ServerConfig) -> anyhow::Result<Arc<dyn SamBackend>> {
    Ok(Arc::new(MockSamBackend))
}
