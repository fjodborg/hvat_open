//! RGBA band packing for GPU-ready data.
//!
//! Re-exports the shared packing logic from hvat_common and provides
//! the BandPacker trait for the server's pyramid builder.

use crate::loaders::BandData;

// Re-export from hvat_common for use in this crate
pub use hvat_common::packing::{BANDS_PER_LAYER, MIN_TEXTURE_LAYERS, pack_bands_to_rgba_layers};

/// Trait for packing band data into GPU-ready formats.
pub trait BandPacker: Send + Sync {
    /// Pack f32 bands into RGBA u8 layers (4 bands per layer).
    ///
    /// Returns a vector of (layer_index, rgba_data) pairs.
    fn pack_to_rgba(&self, bands: &BandData) -> Vec<(u32, Vec<u8>)>;

    /// Downsample bands to target resolution.
    fn downsample(&self, bands: &BandData, target_width: u32, target_height: u32) -> BandData;
}

/// Default band packer implementation.
pub struct DefaultBandPacker;

impl DefaultBandPacker {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DefaultBandPacker {
    fn default() -> Self {
        Self::new()
    }
}

impl BandPacker for DefaultBandPacker {
    fn pack_to_rgba(&self, bands: &BandData) -> Vec<(u32, Vec<u8>)> {
        // Use the shared packing function from hvat_common
        pack_bands_to_rgba_layers(&bands.bands, bands.width, bands.height)
    }

    fn downsample(&self, bands: &BandData, target_width: u32, target_height: u32) -> BandData {
        downsample_bands(bands, target_width, target_height)
    }
}

/// Downsample band data to target resolution using bilinear interpolation.
pub fn downsample_bands(bands: &BandData, target_width: u32, target_height: u32) -> BandData {
    let src_width = bands.width as usize;
    let src_height = bands.height as usize;
    let dst_width = target_width as usize;
    let dst_height = target_height as usize;

    if src_width == dst_width && src_height == dst_height {
        // No downsampling needed, clone the data
        return BandData {
            width: bands.width,
            height: bands.height,
            bands: bands.bands.clone(),
        };
    }

    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    let mut downsampled_bands = Vec::with_capacity(bands.bands.len());

    for band in &bands.bands {
        let mut downsampled = Vec::with_capacity(dst_width * dst_height);

        for dst_y in 0..dst_height {
            for dst_x in 0..dst_width {
                // Map destination pixel to source coordinates
                let src_x = (dst_x as f32 + 0.5) * x_scale - 0.5;
                let src_y = (dst_y as f32 + 0.5) * y_scale - 0.5;

                // Bilinear interpolation
                let x0 = (src_x.floor() as usize).min(src_width - 1);
                let y0 = (src_y.floor() as usize).min(src_height - 1);
                let x1 = (x0 + 1).min(src_width - 1);
                let y1 = (y0 + 1).min(src_height - 1);

                let x_frac = src_x - src_x.floor();
                let y_frac = src_y - src_y.floor();

                let v00 = band[y0 * src_width + x0];
                let v10 = band[y0 * src_width + x1];
                let v01 = band[y1 * src_width + x0];
                let v11 = band[y1 * src_width + x1];

                let v0 = v00 * (1.0 - x_frac) + v10 * x_frac;
                let v1 = v01 * (1.0 - x_frac) + v11 * x_frac;
                let value = v0 * (1.0 - y_frac) + v1 * y_frac;

                downsampled.push(value);
            }
        }

        downsampled_bands.push(downsampled);
    }

    BandData {
        width: target_width,
        height: target_height,
        bands: downsampled_bands,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_downsample() {
        let bands = BandData {
            width: 4,
            height: 4,
            bands: vec![vec![
                1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
            ]],
        };

        let downsampled = downsample_bands(&bands, 2, 2);

        assert_eq!(downsampled.width, 2);
        assert_eq!(downsampled.height, 2);
        assert_eq!(downsampled.bands.len(), 1);

        // Each 2x2 block should average to its dominant value
        let band = &downsampled.bands[0];
        assert!(band[0] > 0.5); // Top-left was all 1.0
        assert!(band[1] < 0.5); // Top-right was all 0.0
        assert!(band[2] < 0.5); // Bottom-left was all 0.0
        assert!(band[3] > 0.5); // Bottom-right was all 1.0
    }
}
