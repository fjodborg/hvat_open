//! Generic inference backend system for model-agnostic protocol.
//!
//! This module provides the infrastructure for pluggable ML model backends.
//! Backends implement the `InferenceBackend` trait and are registered with
//! the `ModelRegistry` for discovery by clients.

mod backend;
mod registry;
mod sam_adapter;

pub use backend::{ImageContext, InferenceBackend, InferenceResult, ProgressCallback};
pub use registry::ModelRegistry;
pub use sam_adapter::SamInferenceAdapter;
