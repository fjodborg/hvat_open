//! Shared utilities for HVAT crates.
//!
//! This crate provides common functionality used across hvat_axum, hvat_leptos,
//! and hvat_gpu to avoid code duplication and ensure consistent behavior.

pub mod binary;
pub mod error;
pub mod interpolation;
pub mod packing;
pub mod protocol;
pub mod pyramid;

pub use binary::BinaryReader;
pub use error::{ErrorContext, ProtocolError};
pub use interpolation::bilinear_sample;
pub use packing::{BANDS_PER_LAYER, BandSlice, MIN_TEXTURE_LAYERS, pack_bands_to_rgba_layers};
pub use protocol::{
    ClientMessage, ErrorCategory, ErrorCode, InputSchema, ModelCapability, ModelType, OptionSchema,
    OutputSchema, PROTOCOL_VERSION, ServerCapabilities, ServerInfo, ServerLimits,
    ServerMessageType, Severity,
};
pub use pyramid::{MAX_PYRAMID_LEVEL, PyramidLevel};

/// Calculate pixel count with checked arithmetic.
///
/// Use this instead of raw `width * height` to make integer overflow explicit.
/// Panics if the multiplication would overflow, which is preferable to
/// silent corruption or undefined behavior.
///
/// # Panics
/// Panics if `width * height` would overflow `usize`.
#[inline]
pub fn pixel_count(width: usize, height: usize) -> usize {
    width.checked_mul(height).unwrap()
}

/// Calculate pixel count from u32 dimensions with checked arithmetic.
///
/// Converts to usize and performs checked multiplication.
///
/// # Panics
/// Panics if the multiplication would overflow `usize`.
#[inline]
pub fn pixel_count_u32(width: u32, height: u32) -> usize {
    (width as usize).checked_mul(height as usize).unwrap()
}
