//! RGBA band packing for GPU-ready data.

use rayon::prelude::*;

use crate::common::loaders::BandData;

pub use hvat_common::packing::{BANDS_PER_LAYER, MIN_TEXTURE_LAYERS, pack_bands_to_rgba_layers};
use hvat_common::{bilinear_sample, pixel_count};

/// Downsample band data to target resolution using bilinear interpolation.
///
/// Uses Rayon to process bands in parallel for better performance on
/// hyperspectral images with many bands.
pub fn downsample_bands(bands: &BandData, target_width: u32, target_height: u32) -> BandData {
    let src_width = bands.width as usize;
    let src_height = bands.height as usize;
    let dst_width = target_width as usize;
    let dst_height = target_height as usize;

    if src_width == dst_width && src_height == dst_height {
        return bands.clone();
    }

    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    let downsampled_bands: Vec<Vec<f32>> = bands
        .bands
        .par_iter()
        .map(|band| {
            let mut downsampled = Vec::with_capacity(pixel_count(dst_height, dst_width));

            for dst_y in 0..dst_height {
                for dst_x in 0..dst_width {
                    let src_x = (dst_x as f32 + 0.5) * x_scale - 0.5;
                    let src_y = (dst_y as f32 + 0.5) * y_scale - 0.5;

                    downsampled.push(bilinear_sample(band, src_width, src_height, src_x, src_y));
                }
            }

            downsampled
        })
        .collect();

    BandData {
        width: target_width,
        height: target_height,
        bands: downsampled_bands,
    }
}

/// Calculate dimensions for a specific pyramid level.
///
/// Level 0 = full resolution, each higher level halves dimensions.
#[must_use]
pub fn calculate_level_size(full_width: u32, full_height: u32, level: u32) -> (u32, u32) {
    if level == 0 {
        return (full_width, full_height);
    }
    let scale = 1u32 << level;
    ((full_width / scale).max(1), (full_height / scale).max(1))
}

#[cfg(test)]
#[path = "packer.test.rs"]
mod tests;
