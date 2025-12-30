//! RGBA band packing for GPU-ready data.
//!
//! Packs hyperspectral bands into RGBA texture layers where each layer
//! contains 4 bands (one per channel). This matches the GPU texture format
//! used by the WebGPU rendering pipeline.
//!
//! The packing logic is identical to the client-side implementation in
//! `hvat_leptos/src/state/project_state.rs`.

use crate::loaders::BandData;

/// Number of bands packed into each RGBA texture layer.
pub const BANDS_PER_LAYER: usize = 4;

/// Minimum number of texture layers required by WebGL2.
pub const MIN_TEXTURE_LAYERS: u32 = 2;

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
        pack_bands_to_rgba_layers(&bands.bands, bands.width, bands.height)
    }

    fn downsample(&self, bands: &BandData, target_width: u32, target_height: u32) -> BandData {
        downsample_bands(bands, target_width, target_height)
    }
}

/// Pack bands into RGBA layers.
///
/// Each layer contains 4 bands packed into RGBA channels.
/// Values are clamped to 0.0-1.0 and scaled to 0-255.
pub fn pack_bands_to_rgba_layers(
    bands: &[Vec<f32>],
    width: u32,
    height: u32,
) -> Vec<(u32, Vec<u8>)> {
    let num_bands = bands.len();
    let num_pixels = (width * height) as usize;

    // Calculate number of layers needed (4 bands per layer)
    let num_layers =
        ((num_bands + BANDS_PER_LAYER - 1) / BANDS_PER_LAYER).max(MIN_TEXTURE_LAYERS as usize);

    let mut layers = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        let base_band = layer_idx * BANDS_PER_LAYER;
        let mut rgba_data = vec![0u8; num_pixels * 4];

        for channel_idx in 0..BANDS_PER_LAYER {
            let band_idx = base_band + channel_idx;

            // If we have data for this band, use it; otherwise use zeros
            if band_idx < num_bands {
                let band = &bands[band_idx];
                for pixel_idx in 0..num_pixels {
                    let value = band.get(pixel_idx).copied().unwrap_or(0.0);
                    let byte_value = (value.clamp(0.0, 1.0) * 255.0) as u8;
                    rgba_data[pixel_idx * 4 + channel_idx] = byte_value;
                }
            }
            // Zeros already filled by vec![0u8; ...]
        }

        layers.push((layer_idx as u32, rgba_data));
    }

    layers
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
    fn test_pack_single_band() {
        let bands = vec![vec![0.0, 0.5, 1.0, 0.25]];
        let layers = pack_bands_to_rgba_layers(&bands, 2, 2);

        // Should produce at least 2 layers (WebGL2 requirement)
        assert_eq!(layers.len(), 2);

        // First layer should have our band in R channel
        let (idx, data) = &layers[0];
        assert_eq!(*idx, 0);
        assert_eq!(data[0], 0); // pixel 0, R
        assert_eq!(data[4], 127); // pixel 1, R (0.5 * 255 ≈ 127)
        assert_eq!(data[8], 255); // pixel 2, R
        assert_eq!(data[12], 63); // pixel 3, R (0.25 * 255 ≈ 63)
    }

    #[test]
    fn test_pack_four_bands() {
        let bands = vec![
            vec![1.0, 1.0],   // R
            vec![0.5, 0.5],   // G
            vec![0.0, 0.0],   // B
            vec![0.25, 0.25], // A
        ];
        let layers = pack_bands_to_rgba_layers(&bands, 2, 1);

        assert_eq!(layers.len(), 2);

        let (_, data) = &layers[0];
        // Pixel 0: RGBA
        assert_eq!(data[0], 255); // R
        assert_eq!(data[1], 127); // G
        assert_eq!(data[2], 0); // B
        assert_eq!(data[3], 63); // A
    }

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
