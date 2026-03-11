//! SAM3 backend model integration surface.
//!
//! This module is the dedicated home for SAM3-family model wiring in the
//! `full_sam3` backend. Current implementation uses a SAM2-compatible ONNX
//! engine wrapper while native SAM3 runtime support is implemented.

pub mod engine;
pub mod models;

pub use crate::full_sam::sam::{
    EncoderOutput, ExecutionProvider, SamBackend, SamMaskResult, SamPoint,
};
pub use engine::Sam3CompatOnnxEngine;
pub use models::Sam3CompatVariant;
