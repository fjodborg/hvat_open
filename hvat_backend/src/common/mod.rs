//! Reusable helpers for HVAT backend implementations.
//!
//! Provides everything needed to build a protocol-compliant HVAT backend:
//! - Image loading (PNG, JPEG, NPY, etc.)
//! - RGBA band packing for GPU streaming
//! - WebSocket protocol encoding
//! - Multiplexed WebSocket handler with connection management
//! - Image streaming logic
//! - CLI configuration and error types
//! - Image catalog traversal
//! - Project state persistence
//! - Upload validation
//! - ZIP archive building

pub mod archive;
pub mod catalog;
pub mod config;
pub mod error;
pub mod loaders;
pub mod packer;
pub mod project_state;
pub mod protocol;
pub mod streaming;
pub mod upload;
#[cfg(feature = "legacy-ws")]
pub mod websocket;

mod frame;

pub use frame::{FrameBuilder, NoHeader, WithHeader};
