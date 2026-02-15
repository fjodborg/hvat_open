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
mod tests {
    use super::*;

    #[test]
    fn test_pack_single_band() {
        let bands: Vec<Vec<f32>> = vec![vec![0.0, 0.5, 1.0, 0.25]];
        let layers = pack_bands_to_rgba_layers(&bands, 2, 2);

        // Should produce at least 2 layers (WebGL2 requirement)
        assert_eq!(layers.len(), 2);

        // First layer should have our band in R channel
        let (idx, data) = &layers[0];
        assert_eq!(*idx, 0);
        assert_eq!(data[0], 0); // pixel 0, R
        assert_eq!(data[3], 255); // pixel 0, A (opaque - no 4th band)
        assert_eq!(data[4], 127); // pixel 1, R (0.5 * 255 ≈ 127)
        assert_eq!(data[7], 255); // pixel 1, A (opaque)
        assert_eq!(data[8], 255); // pixel 2, R
        assert_eq!(data[12], 63); // pixel 3, R (0.25 * 255 ≈ 63)
        assert_eq!(data[15], 255); // pixel 3, A (opaque)
    }

    #[test]
    fn test_pack_three_bands_rgb() {
        // Standard RGB image - 3 bands, no alpha
        let bands: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0], // R: fully red, then black
            vec![0.0, 1.0], // G: black, then fully green
            vec![0.0, 0.0], // B: all black
        ];
        let layers = pack_bands_to_rgba_layers(&bands, 2, 1);

        assert_eq!(layers.len(), 2);

        let (_, data) = &layers[0];
        // Pixel 0: red (RGB=255,0,0)
        assert_eq!(data[0], 255); // R
        assert_eq!(data[1], 0); // G
        assert_eq!(data[2], 0); // B
        assert_eq!(data[3], 255); // A (opaque - no 4th band)
        // Pixel 1: green (RGB=0,255,0)
        assert_eq!(data[4], 0); // R
        assert_eq!(data[5], 255); // G
        assert_eq!(data[6], 0); // B
        assert_eq!(data[7], 255); // A (opaque - no 4th band)

        // Layer 1 should also have opaque alpha
        let (_, layer1_data) = &layers[1];
        assert_eq!(layer1_data[3], 255);
        assert_eq!(layer1_data[7], 255);
    }

    #[test]
    fn test_pack_four_bands() {
        let bands: Vec<Vec<f32>> = vec![
            vec![1.0, 1.0],   // R
            vec![0.5, 0.5],   // G
            vec![0.0, 0.0],   // B
            vec![0.25, 0.25], // A (actual alpha band)
        ];
        let layers = pack_bands_to_rgba_layers(&bands, 2, 1);

        assert_eq!(layers.len(), 2);

        let (_, data) = &layers[0];
        // Pixel 0: RGBA
        assert_eq!(data[0], 255); // R
        assert_eq!(data[1], 127); // G (0.5 * 255 ≈ 127)
        assert_eq!(data[2], 0); // B
        assert_eq!(data[3], 63); // A (from band, not forced to 255)
    }
}
