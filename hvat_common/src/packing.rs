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
    let num_layers =
        ((num_bands + BANDS_PER_LAYER - 1) / BANDS_PER_LAYER).max(MIN_TEXTURE_LAYERS as usize);

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

/// Pack raw RGBA pixel data into layers.
///
/// This is used when we already have RGBA u8 data (e.g., from an image crate)
/// and just need to pack it into the layer format with proper alpha handling.
///
/// # Arguments
///
/// * `rgba_pixels` - Raw RGBA pixel data as chunks of 4 bytes [R, G, B, A]
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `include_alpha` - If false, alpha channel is ignored from input and set to 255
///
/// # Returns
///
/// Vector of (layer_index, rgba_data) pairs. For standard images, this will be
/// 2 layers (minimum for WebGL2) with the first layer containing RGB+opaque alpha.
pub fn pack_rgba_pixels_to_layers(
    rgba_pixels: &[u8],
    width: u32,
    height: u32,
    include_alpha: bool,
) -> Vec<(u32, Vec<u8>)> {
    let num_pixels = (width * height) as usize;
    let expected_size = num_pixels * 4;

    assert_eq!(
        rgba_pixels.len(),
        expected_size,
        "RGBA pixel data size mismatch: expected {}, got {}",
        expected_size,
        rgba_pixels.len()
    );

    // For standard images, we have 3 effective bands (R, G, B)
    // We need minimum 2 layers for WebGL2
    let mut layers = Vec::with_capacity(MIN_TEXTURE_LAYERS as usize);

    // Layer 0: RGB data with alpha = 255 (or from source if include_alpha)
    let mut layer0_data = vec![0u8; num_pixels * 4];
    for pixel_idx in 0..num_pixels {
        let src_offset = pixel_idx * 4;
        let dst_offset = pixel_idx * 4;

        layer0_data[dst_offset] = rgba_pixels[src_offset]; // R
        layer0_data[dst_offset + 1] = rgba_pixels[src_offset + 1]; // G
        layer0_data[dst_offset + 2] = rgba_pixels[src_offset + 2]; // B
        layer0_data[dst_offset + 3] = if include_alpha {
            rgba_pixels[src_offset + 3]
        } else {
            255 // Force opaque
        };
    }
    layers.push((0, layer0_data));

    // Layer 1: Padding layer (all zeros except alpha = 255)
    let mut layer1_data = vec![0u8; num_pixels * 4];
    for pixel_idx in 0..num_pixels {
        layer1_data[pixel_idx * 4 + 3] = 255; // Alpha = opaque
    }
    layers.push((1, layer1_data));

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

    #[test]
    fn test_pack_rgba_pixels() {
        // Simple 2x1 image: red pixel, green pixel
        let rgba_pixels = vec![
            255, 0, 0, 128, // Red pixel with 50% alpha
            0, 255, 0, 64, // Green pixel with 25% alpha
        ];

        // Without alpha - should force to opaque
        let layers = pack_rgba_pixels_to_layers(&rgba_pixels, 2, 1, false);
        assert_eq!(layers.len(), 2);
        let (_, data) = &layers[0];
        assert_eq!(data[0], 255); // R
        assert_eq!(data[3], 255); // A forced to 255
        assert_eq!(data[4], 0); // R
        assert_eq!(data[7], 255); // A forced to 255

        // With alpha - should preserve
        let layers = pack_rgba_pixels_to_layers(&rgba_pixels, 2, 1, true);
        let (_, data) = &layers[0];
        assert_eq!(data[3], 128); // A preserved
        assert_eq!(data[7], 64); // A preserved
    }
}
