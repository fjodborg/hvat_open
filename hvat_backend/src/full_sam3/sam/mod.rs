//! SAM3 backend model integration surface.
//!
//! This module is the dedicated home for SAM3-family model wiring in the
//! `full_sam3` backend. Current implementation keeps a compat-onnx runtime
//! (SAM2 ONNX internals under SAM3 wiring) while staging native SAM3 runtime
//! artifact/config plumbing.

pub mod engine;
pub mod models;

pub use crate::full_sam::sam::{
    EncoderOutput, ExecutionProvider, SamBackend, SamMaskResult, SamPoint,
};
pub use engine::Sam3CompatOnnxEngine;
pub use models::{Sam3ModelSpecError, Sam3RuntimeModelSpec};
