//! SAM 2 (Segment Anything Model 2) integration for AI-assisted annotation.
//!
//! This module provides ONNX Runtime-based inference for SAM 2 models,
//! enabling interactive image segmentation with point and box prompts.
//!
//! ## Architecture
//!
//! SAM 2 uses a split encoder-decoder architecture:
//! - **Encoder**: Heavy model (~134-889 MB) that computes image embeddings once per image
//! - **Decoder**: Light model (~21 MB) that generates masks from embeddings + prompts
//!
//! This enables efficient interactive segmentation: compute embedding once,
//! then quickly generate masks for different prompt combinations.
//!
//! ## Model Sources
//!
//! Pre-exported ONNX models are available from Hugging Face:
//! <https://huggingface.co/vietanhdev/segment-anything-2-onnx-models>

pub mod backend;
pub mod cache;
pub mod engine;
pub mod models;

pub use backend::{EncoderOutput, ExecutionProvider, SamBackend, SamMaskResult, SamPoint};
pub use cache::{CachedEmbedding, EmbeddingCache};
pub use engine::OnnxSamEngine;
pub use models::SamVariant;
