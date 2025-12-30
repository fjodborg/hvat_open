//! Multi-resolution image pyramid generation and caching.
//!
//! Pyramids allow progressive loading of large images by storing
//! multiple resolution levels. Each level is pre-packed into RGBA
//! format for direct GPU upload.

mod builder;
mod storage;

pub use builder::{PyramidBuilder, PyramidLevel, calculate_level_dimensions, calculate_num_levels};
pub use storage::{FilesystemStorage, PyramidStorage, compute_image_hash};

use serde::{Deserialize, Serialize};

/// Metadata about a stored pyramid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyramidMetadata {
    /// Original image width
    pub full_width: u32,
    /// Original image height
    pub full_height: u32,
    /// Number of spectral bands
    pub num_bands: usize,
    /// Available pyramid levels (0 = thumbnail, highest = full res)
    pub levels: Vec<LevelInfo>,
    /// Hash of the source file for cache invalidation
    pub source_hash: String,
    /// Timestamp when pyramid was created
    pub created_at: u64,
}

/// Information about a single pyramid level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelInfo {
    /// Level index (0 = smallest/thumbnail)
    pub level: u32,
    /// Width at this level
    pub width: u32,
    /// Height at this level
    pub height: u32,
    /// Number of RGBA texture layers
    pub num_layers: u32,
    /// Size of the level data in bytes
    pub size_bytes: u64,
}

/// Status of pyramid generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PyramidStatus {
    /// Pyramid doesn't exist yet
    Pending,
    /// Pyramid is being generated
    Building,
    /// Pyramid is ready for use
    Ready,
    /// Pyramid generation failed
    Failed,
}
