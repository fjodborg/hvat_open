//! Pyramid generation from image data.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::{LevelInfo, PyramidMetadata, PyramidStorage};
use crate::error::Result;
use crate::loaders::BandData;
use crate::packer::{downsample_bands, pack_bands_to_rgba_layers};

/// Minimum dimension for the thumbnail level.
const MIN_THUMBNAIL_SIZE: u32 = 256;

/// A single pyramid level with pre-packed RGBA data.
#[derive(Debug)]
pub struct PyramidLevel {
    /// Level index (0 = thumbnail, higher = more detail)
    pub level: u32,
    /// Width at this level
    pub width: u32,
    /// Height at this level
    pub height: u32,
    /// Number of RGBA texture layers
    pub num_layers: u32,
    /// Pre-packed RGBA layers (layer_index, rgba_data)
    pub layers: Vec<(u32, Vec<u8>)>,
}

impl PyramidLevel {
    /// Calculate the total size in bytes of this level's data.
    pub fn size_bytes(&self) -> u64 {
        self.layers.iter().map(|(_, data)| data.len() as u64).sum()
    }
}

/// Builder for creating multi-resolution image pyramids.
#[derive(Clone)]
pub struct PyramidBuilder {
    storage: Arc<dyn PyramidStorage>,
}

impl PyramidBuilder {
    /// Create a new pyramid builder with the given storage backend.
    pub fn new(storage: Arc<dyn PyramidStorage>) -> Self {
        Self { storage }
    }

    /// Build a pyramid from band data and save to storage.
    ///
    /// Returns the pyramid metadata.
    pub async fn build_and_save(
        &self,
        image_hash: &str,
        bands: &BandData,
    ) -> Result<PyramidMetadata> {
        // Mark as building
        self.storage.mark_building(image_hash).await?;

        // Generate all pyramid levels
        let levels = self.generate_levels(bands);

        // Save each level
        for level in &levels {
            self.storage
                .save_level(image_hash, level.level, &level.layers)
                .await?;
        }

        // Create and save metadata
        let metadata = PyramidMetadata {
            full_width: bands.width,
            full_height: bands.height,
            num_bands: bands.num_bands(),
            levels: levels
                .iter()
                .map(|l| LevelInfo {
                    level: l.level,
                    width: l.width,
                    height: l.height,
                    num_layers: l.num_layers,
                    size_bytes: l.size_bytes(),
                })
                .collect(),
            source_hash: image_hash.to_string(),
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        self.storage.save_metadata(image_hash, &metadata).await?;

        // Mark as ready
        self.storage.mark_ready(image_hash).await?;

        Ok(metadata)
    }

    /// Generate all pyramid levels from band data.
    pub fn generate_levels(&self, bands: &BandData) -> Vec<PyramidLevel> {
        let mut current_bands = bands.clone();

        // Generate from full resolution down to thumbnail
        // We'll reverse at the end so level 0 = thumbnail
        let mut temp_levels = Vec::new();

        loop {
            // Pack current resolution to RGBA
            let rgba_layers = pack_bands_to_rgba_layers(
                &current_bands.bands,
                current_bands.width,
                current_bands.height,
            );

            let num_layers = rgba_layers.len() as u32;

            temp_levels.push(PyramidLevel {
                level: 0, // Will be renumbered
                width: current_bands.width,
                height: current_bands.height,
                num_layers,
                layers: rgba_layers,
            });

            // Check if we've reached minimum size
            if current_bands.width <= MIN_THUMBNAIL_SIZE
                && current_bands.height <= MIN_THUMBNAIL_SIZE
            {
                break;
            }

            // Downsample for next level
            let new_width = (current_bands.width + 1) / 2;
            let new_height = (current_bands.height + 1) / 2;

            current_bands = downsample_bands(&current_bands, new_width, new_height);
        }

        // Level 0 = full resolution (largest), higher levels = smaller
        // Just renumber without reversing
        let mut levels = Vec::with_capacity(temp_levels.len());
        for (idx, mut level) in temp_levels.into_iter().enumerate() {
            level.level = idx as u32;
            levels.push(level);
        }

        levels
    }
}

/// Calculate the number of pyramid levels for an image.
pub fn calculate_num_levels(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height);
    if max_dim <= MIN_THUMBNAIL_SIZE {
        return 1;
    }

    // Each level halves the dimensions
    let levels = (max_dim as f32 / MIN_THUMBNAIL_SIZE as f32).log2().ceil() as u32;
    levels + 1 // +1 for the full resolution level
}

/// Calculate dimensions for a specific pyramid level.
///
/// Level 0 = full resolution
/// Level 1 = half resolution
/// Level 2 = quarter resolution, etc.
pub fn calculate_level_dimensions(full_width: u32, full_height: u32, level: u32) -> (u32, u32) {
    if level == 0 {
        return (full_width, full_height);
    }

    // Each level halves the resolution
    let scale = 1u32 << level; // 2^level

    ((full_width / scale).max(1), (full_height / scale).max(1))
}

// Implement Clone for BandData since we need it
impl Clone for BandData {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            bands: self.bands.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_num_levels() {
        // Small image - just 1 level
        assert_eq!(calculate_num_levels(256, 256), 1);
        assert_eq!(calculate_num_levels(128, 128), 1);

        // 512x512 -> 256x256 = 2 levels
        assert_eq!(calculate_num_levels(512, 512), 2);

        // 1024x1024 -> 512 -> 256 = 3 levels
        assert_eq!(calculate_num_levels(1024, 1024), 3);

        // 4096x4096 -> 2048 -> 1024 -> 512 -> 256 = 5 levels
        assert_eq!(calculate_num_levels(4096, 4096), 5);
    }

    #[test]
    fn test_calculate_level_dimensions() {
        // For a 1024x1024 image:
        // Level 0: 1024x1024 (full)
        // Level 1: 512x512
        // Level 2: 256x256 (thumbnail)

        let (w, h) = calculate_level_dimensions(1024, 1024, 0);
        assert_eq!((w, h), (1024, 1024));

        let (w, h) = calculate_level_dimensions(1024, 1024, 1);
        assert_eq!((w, h), (512, 512));

        let (w, h) = calculate_level_dimensions(1024, 1024, 2);
        assert_eq!((w, h), (256, 256));
    }
}
