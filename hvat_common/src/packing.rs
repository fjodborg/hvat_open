//! RGBA band packing for GPU-ready data.
//!
//! Packs hyperspectral bands into RGBA texture layers where each layer
//! contains 4 bands (one per channel). This matches the GPU texture format
//! used by WebGPU/WebGL rendering pipelines.
//!
//! # Alpha Channel Handling
//!
//! When an image has fewer than 4 bands per layer, the alpha channel is
//! automatically set to 255 (fully opaque) to ensure proper rendering.
//! This is critical for standard RGB images (3 bands) which would otherwise
//! appear transparent.

/// Number of bands packed into each RGBA texture layer.
pub const BANDS_PER_LAYER: usize = 4;

/// Minimum number of texture layers required by WebGL2.
///
/// WebGL2 has issues with single-layer texture arrays, so we always
/// create at least 2 layers even if the image has fewer than 4 bands.
pub const MIN_TEXTURE_LAYERS: u32 = 2;

/// A slice of band data that can be accessed by pixel index.
///
/// This trait allows the packing function to work with different
/// band storage formats (Vec<f32>, &[f32], etc.)
pub trait BandSlice {
    /// Get the value at the given pixel index, or None if out of bounds.
    fn get_value(&self, pixel_idx: usize) -> Option<f32>;

    /// Get the number of pixels in this band.
    fn len(&self) -> usize;

    /// Check if the band is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl BandSlice for Vec<f32> {
    fn get_value(&self, pixel_idx: usize) -> Option<f32> {
        self.get(pixel_idx).copied()
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl BandSlice for &[f32] {
    fn get_value(&self, pixel_idx: usize) -> Option<f32> {
        self.get(pixel_idx).copied()
    }

    fn len(&self) -> usize {
        (*self).len()
    }
}

/// Pack bands into RGBA layers.
///
/// Each layer contains 4 bands packed into RGBA channels.
/// Values are clamped to 0.0-1.0 and scaled to 0-255.
///
/// # Alpha Channel
///
/// When a layer has fewer than 4 bands, the alpha channel (index 3)
/// is set to 255 (fully opaque). This ensures that RGB images (3 bands)
/// display correctly instead of appearing transparent.
///
/// # Arguments
///
/// * `bands` - Slice of band data, each band is a slice of f32 values normalized to 0.0-1.0
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
///
/// # Returns
///
/// Vector of (layer_index, rgba_data) pairs where rgba_data is a Vec<u8>
/// with 4 bytes per pixel in RGBA order.
pub fn pack_bands_to_rgba_layers<B: BandSlice>(
    bands: &[B],
    width: u32,
    height: u32,
) -> Vec<(u32, Vec<u8>)> {
    let num_bands = bands.len();
    let num_pixels = (width * height) as usize;

    // Calculate number of layers needed (4 bands per layer)
    let num_layers = num_bands
        .div_ceil(BANDS_PER_LAYER)
        .max(MIN_TEXTURE_LAYERS as usize);

    let mut layers = Vec::with_capacity(num_layers);

    for layer_idx in 0..num_layers {
        let base_band = layer_idx * BANDS_PER_LAYER;
        let mut rgba_data = vec![0u8; num_pixels * 4];

        // Track if we have an alpha band for this layer
        let has_alpha_band = base_band + 3 < num_bands;

        for channel_idx in 0..BANDS_PER_LAYER {
            let band_idx = base_band + channel_idx;

            if band_idx < num_bands {
                // We have data for this band
                let band = &bands[band_idx];
                for pixel_idx in 0..num_pixels {
                    let value = band.get_value(pixel_idx).unwrap_or(0.0);
                    let byte_value = (value.clamp(0.0, 1.0) * 255.0) as u8;
                    rgba_data[pixel_idx * 4 + channel_idx] = byte_value;
                }
            }
        }

        // If this layer doesn't have an alpha band, set alpha to 255 (opaque)
        if !has_alpha_band {
            for pixel_idx in 0..num_pixels {
                rgba_data[pixel_idx * 4 + 3] = 255;
            }
        }

        layers.push((layer_idx as u32, rgba_data));
    }

    layers
}

#[cfg(test)]
#[path = "packing.test.rs"]
mod tests;
