//! Bilinear interpolation utilities for image resampling.

/// Sample a value from a 2D array using bilinear interpolation.
///
/// Coordinates are clamped to valid ranges. The data is assumed to be
/// stored in row-major order (y * width + x).
///
/// # Arguments
/// * `data` - Flat array of pixel values in row-major order
/// * `width` - Width of the 2D array
/// * `height` - Height of the 2D array
/// * `x` - X coordinate to sample (can be fractional)
/// * `y` - Y coordinate to sample (can be fractional)
#[inline]
pub fn bilinear_sample(data: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    let x0 = (x.floor() as isize).clamp(0, width as isize - 1) as usize;
    let y0 = (y.floor() as isize).clamp(0, height as isize - 1) as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let fx = (x - x.floor()).clamp(0.0, 1.0);
    let fy = (y - y.floor()).clamp(0.0, 1.0);

    let v00 = data[y0 * width + x0];
    let v10 = data[y0 * width + x1];
    let v01 = data[y1 * width + x0];
    let v11 = data[y1 * width + x1];

    let v0 = v00 * (1.0 - fx) + v10 * fx;
    let v1 = v01 * (1.0 - fx) + v11 * fx;

    v0 * (1.0 - fy) + v1 * fy
}

#[cfg(test)]
#[path = "interpolation.test.rs"]
mod tests;
