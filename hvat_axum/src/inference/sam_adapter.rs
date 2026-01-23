//! Adapter that wraps a SamBackend as an InferenceBackend.

use super::{ImageContext, InferenceBackend, InferenceResult, ProgressCallback};
use crate::sam::{CachedEmbedding, EmbeddingCache, EncoderOutput, SamBackend, SamPoint};
use async_trait::async_trait;
use hvat_common::{InputSchema, ModelCapability, ModelType, OutputSchema};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;

/// Adapter that wraps a SamBackend as an InferenceBackend.
///
/// This allows existing SAM implementations to work with the
/// generic model-agnostic protocol without modification.
pub struct SamInferenceAdapter<B: SamBackend + ?Sized> {
    backend: Arc<B>,
    cache: Arc<EmbeddingCache>,
    model_id: String,
    model_name: String,
}

impl<B: SamBackend + ?Sized> SamInferenceAdapter<B> {
    pub fn new(
        backend: Arc<B>,
        cache: Arc<EmbeddingCache>,
        model_id: impl Into<String>,
        model_name: impl Into<String>,
    ) -> Self {
        Self {
            backend,
            cache,
            model_id: model_id.into(),
            model_name: model_name.into(),
        }
    }
}

#[async_trait]
impl<B: SamBackend + ?Sized + 'static> InferenceBackend for SamInferenceAdapter<B> {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn capability(&self) -> ModelCapability {
        ModelCapability {
            id: self.model_id.clone(),
            name: self.model_name.clone(),
            model_type: ModelType::Segmentation,
            description: "Interactive segmentation with point/box prompts".to_string(),
            inputs: vec![
                InputSchema {
                    name: "points".to_string(),
                    input_type: "point_list".to_string(),
                    required: false,
                    description: "Click points with foreground/background labels".to_string(),
                },
                InputSchema {
                    name: "box".to_string(),
                    input_type: "bbox".to_string(),
                    required: false,
                    description: "Bounding box prompt".to_string(),
                },
            ],
            outputs: vec![
                OutputSchema {
                    name: "masks".to_string(),
                    output_type: "polygon_list".to_string(),
                    description: "Segmentation masks as polygons".to_string(),
                },
                OutputSchema {
                    name: "scores".to_string(),
                    output_type: "float_list".to_string(),
                    description: "Confidence scores for each mask".to_string(),
                },
            ],
            options: HashMap::new(),
            requires_embedding: true,
            embedding_time_ms: 1000,
        }
    }

    fn requires_embedding(&self) -> bool {
        true
    }

    async fn prepare(
        &self,
        image: &ImageContext,
        progress: Option<ProgressCallback>,
    ) -> anyhow::Result<()> {
        if let Some(ref cb) = progress {
            cb(0, "Computing image embedding...");
        }

        // Run encoder
        let output = self
            .backend
            .encode_image(&image.rgb_data, image.width, image.height)
            .await?;

        // Convert EncoderOutput to CachedEmbedding and store
        let cached = CachedEmbedding {
            image_embed: output.image_embed,
            high_res_feats_0: output.high_res_feats_0,
            high_res_feats_1: output.high_res_feats_1,
            width: image.width,
            height: image.height,
        };
        self.cache.insert(image.image_id.clone(), cached).await;

        if let Some(ref cb) = progress {
            cb(100, "Embedding ready");
        }

        Ok(())
    }

    async fn is_prepared(&self, image_id: &str) -> bool {
        self.cache.contains(image_id).await
    }

    async fn infer(
        &self,
        image: &ImageContext,
        inputs: Value,
        _options: Value,
        progress: Option<ProgressCallback>,
    ) -> anyhow::Result<InferenceResult> {
        // Get cached embedding
        let cached = self
            .cache
            .get_and_touch(&image.image_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("Embedding not ready - call prepare_model first"))?;

        if let Some(ref cb) = progress {
            cb(50, "Running decoder...");
        }

        // Convert CachedEmbedding back to EncoderOutput for decode_mask
        let encoder_output = EncoderOutput {
            image_embed: cached.image_embed,
            high_res_feats_0: cached.high_res_feats_0,
            high_res_feats_1: cached.high_res_feats_1,
        };

        // Parse inputs
        let points = parse_sam_points(&inputs)?;
        let box_prompt = parse_sam_box(&inputs)?;

        let start = std::time::Instant::now();

        let result = self
            .backend
            .decode_mask(
                &encoder_output,
                image.width,
                image.height,
                &points,
                box_prompt,
            )
            .await?;

        let timing_ms = start.elapsed().as_millis() as u64;

        if let Some(ref cb) = progress {
            cb(100, "Complete");
        }

        Ok(InferenceResult {
            model_id: self.model_id.clone(),
            outputs: json!({
                "masks": result.polygons,
                "scores": result.iou_scores,
            }),
            timing_ms,
        })
    }

    fn validate_inputs(&self, inputs: &Value) -> anyhow::Result<()> {
        // Must have at least one prompt
        let has_points = inputs
            .get("points")
            .and_then(|v| v.as_array())
            .map(|a| !a.is_empty())
            .unwrap_or(false);
        let has_box = inputs.get("box").is_some() && !inputs["box"].is_null();

        if !has_points && !has_box {
            anyhow::bail!("At least one prompt (points or box) is required");
        }

        Ok(())
    }
}

fn parse_sam_points(inputs: &Value) -> anyhow::Result<Vec<SamPoint>> {
    let Some(points) = inputs.get("points") else {
        return Ok(vec![]);
    };

    let arr = points
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("points must be an array"))?;

    arr.iter()
        .map(|p| {
            Ok(SamPoint {
                x: p["x"]
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("point missing x"))? as f32,
                y: p["y"]
                    .as_f64()
                    .ok_or_else(|| anyhow::anyhow!("point missing y"))? as f32,
                label: p["label"]
                    .as_i64()
                    .ok_or_else(|| anyhow::anyhow!("point missing label"))?
                    as i32,
            })
        })
        .collect()
}

fn parse_sam_box(inputs: &Value) -> anyhow::Result<Option<[f32; 4]>> {
    let Some(box_val) = inputs.get("box") else {
        return Ok(None);
    };

    if box_val.is_null() {
        return Ok(None);
    }

    let arr = box_val
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("box must be an array [x1, y1, x2, y2]"))?;

    if arr.len() != 4 {
        anyhow::bail!("box must have exactly 4 elements");
    }

    Ok(Some([
        arr[0]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("box[0] must be a number"))? as f32,
        arr[1]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("box[1] must be a number"))? as f32,
        arr[2]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("box[2] must be a number"))? as f32,
        arr[3]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("box[3] must be a number"))? as f32,
    ]))
}
