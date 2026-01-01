//! Shared utilities for HVAT crates.
//!
//! This crate provides common functionality used across hvat_axum, hvat_leptos,
//! and hvat_gpu to avoid code duplication and ensure consistent behavior.

pub mod packing;

pub use packing::{BANDS_PER_LAYER, BandSlice, MIN_TEXTURE_LAYERS, pack_bands_to_rgba_layers};
