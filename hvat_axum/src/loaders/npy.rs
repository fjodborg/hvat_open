//! NumPy .npy file loader for hyperspectral data.
//!
//! Expects arrays with shape (B, H, W) where:
//! - B = number of bands
//! - H = height
//! - W = width
//!
//! Supports f32 and f64 data types. Values are normalized to 0.0-1.0
//! based on the min/max values in the array.

use std::io::BufReader;
use std::path::Path;

use async_trait::async_trait;
use hvat_common::pixel_count;
use ndarray::Array3;
use ndarray_npy::ReadNpyExt;

use super::{BandData, ImageLoader, ImageMetadata};
use crate::error::{Error, Result};

/// Try to read an NPY file as f32, falling back to f64.
/// Returns the array shape (bands, height, width).
fn read_npy_shape(path: &Path) -> Result<(usize, usize, usize)> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    Array3::<f32>::read_npy(reader)
        .map(|arr| arr.dim())
        .or_else(|_| {
            let file = std::fs::File::open(path)?;
            let reader = BufReader::new(file);
            Array3::<f64>::read_npy(reader)
                .map(|arr| arr.dim())
                .map_err(|e| Error::InvalidImageData(format!("Failed to read NPY file: {}", e)))
        })
}

/// Try to read NPY bands, first as f32, falling back to f64.
fn read_npy_bands(path: &Path) -> Result<Vec<Vec<f32>>> {
    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    Array3::<f32>::read_npy(reader)
        .map(extract_bands_f32)
        .or_else(|_| {
            let file = std::fs::File::open(path)?;
            let reader = BufReader::new(file);
            Array3::<f64>::read_npy(reader)
                .map(extract_bands_f64)
                .map_err(|e| Error::InvalidImageData(format!("Failed to read NPY file: {}", e)))
        })
}

/// Loader for NumPy .npy files containing hyperspectral data.
pub struct NpyLoader;

impl NpyLoader {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NpyLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ImageLoader for NpyLoader {
    fn name(&self) -> &'static str {
        "NpyLoader"
    }

    fn supports(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| ext.to_lowercase() == "npy")
    }

    async fn load_metadata(&self, path: &Path) -> Result<ImageMetadata> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            let (num_bands, height, width) = read_npy_shape(&path)?;

            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            Ok(ImageMetadata {
                width: width as u32,
                height: height as u32,
                num_bands,
                filename,
                format: "NPY".to_string(),
            })
        })
        .await
        .map_err(|e| Error::Internal(e.to_string()))?
    }

    async fn load_bands(&self, path: &Path) -> Result<BandData> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            let bands = read_npy_bands(&path)?;

            if bands.is_empty() {
                return Err(Error::InvalidImageData("NPY file has no bands".to_string()));
            }

            let (_, height, width) = read_npy_shape(&path)?;

            // Verify dimensions match
            let num_pixels = bands[0].len();
            if num_pixels != pixel_count(width, height) {
                return Err(Error::InvalidImageData(format!(
                    "Band size {} doesn't match dimensions {}x{}",
                    num_pixels, width, height
                )));
            }

            Ok(BandData {
                width: width as u32,
                height: height as u32,
                bands,
            })
        })
        .await
        .map_err(|e| Error::Internal(e.to_string()))?
    }
}

/// Extract bands from f32 array and normalize to 0.0-1.0.
fn extract_bands_f32(arr: Array3<f32>) -> Vec<Vec<f32>> {
    let (num_bands, height, width) = arr.dim();

    // Find global min/max for normalization
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for &val in arr.iter() {
        if val.is_finite() {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    let range = max_val - min_val;
    let scale = if range > 0.0 { 1.0 / range } else { 1.0 };

    let mut bands = Vec::with_capacity(num_bands);
    for b in 0..num_bands {
        let mut band = Vec::with_capacity(pixel_count(height, width));
        for h in 0..height {
            for w in 0..width {
                let val = arr[[b, h, w]];
                let normalized = if val.is_finite() {
                    ((val - min_val) * scale).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                band.push(normalized);
            }
        }
        bands.push(band);
    }

    bands
}

/// Extract bands from f64 array and normalize to 0.0-1.0.
fn extract_bands_f64(arr: Array3<f64>) -> Vec<Vec<f32>> {
    let (num_bands, height, width) = arr.dim();

    // Find global min/max for normalization
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for &val in arr.iter() {
        if val.is_finite() {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    let range = max_val - min_val;
    let scale = if range > 0.0 { 1.0 / range } else { 1.0 };

    let mut bands = Vec::with_capacity(num_bands);
    for b in 0..num_bands {
        let mut band = Vec::with_capacity(pixel_count(height, width));
        for h in 0..height {
            for w in 0..width {
                let val = arr[[b, h, w]];
                let normalized = if val.is_finite() {
                    (((val - min_val) * scale) as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                band.push(normalized);
            }
        }
        bands.push(band);
    }

    bands
}
