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
mod tests {
    use super::*;

    #[test]
    fn test_bilinear_sample_corners() {
        // 2x2 grid with distinct values
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let width = 2;
        let height = 2;

        // Exact corners
        assert!((bilinear_sample(&data, width, height, 0.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((bilinear_sample(&data, width, height, 1.0, 0.0) - 2.0).abs() < 1e-6);
        assert!((bilinear_sample(&data, width, height, 0.0, 1.0) - 3.0).abs() < 1e-6);
        assert!((bilinear_sample(&data, width, height, 1.0, 1.0) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_bilinear_sample_center() {
        // 2x2 grid with distinct values
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let width = 2;
        let height = 2;

        // Center should be average of all four
        let center = bilinear_sample(&data, width, height, 0.5, 0.5);
        assert!((center - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_bilinear_sample_clamping() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let width = 2;
        let height = 2;

        // Out of bounds should clamp
        let oob = bilinear_sample(&data, width, height, -1.0, -1.0);
        assert!((oob - 1.0).abs() < 1e-6); // Should clamp to corner

        let oob2 = bilinear_sample(&data, width, height, 10.0, 10.0);
        assert!((oob2 - 4.0).abs() < 1e-6); // Should clamp to opposite corner
    }
}
