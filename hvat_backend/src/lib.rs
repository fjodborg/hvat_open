//! HVAT backend crate.
//!
//! This crate provides a unified backend layout:
//! - `common`: shared helper functionality
//! - `simple`: minimal backend mode (no SAM/pyramid caching)
//! - `full_sam`: full backend mode with pyramids and SAM inference
//! - `full_sam3`: SAM3 backend entrypoint scaffold

pub mod common;
pub mod full_sam;
pub mod full_sam3;
pub mod helper;
pub mod simple;

// Keep root-level exports for existing call sites.
pub use full_sam::AppState;
pub use full_sam::Error;
pub use full_sam::Result;
pub use full_sam::ServerConfig;
pub use full_sam::pregenerate_pyramids;

// Keep module paths stable while exposing the new full_sam namespace.
pub use full_sam::config;
pub use full_sam::error;
pub use full_sam::inference;
pub use full_sam::loaders;
pub use full_sam::packer;
pub use full_sam::protocol;
pub use full_sam::pyramid;
pub use full_sam::routes;
pub use full_sam::sam;
pub use full_sam::startup;
pub use full_sam::state;
pub use full_sam::utils;
