//! NumPy .npy file loader for hyperspectral data.
//!
//! Expects arrays with shape (B, H, W) where:
//! - B = number of bands
//! - H = height
//! - W = width
//!
//! Supports f32 and f64 data types. Values are normalized to 0.0-1.0
//! based on the min/max values in the array.

use std::path::Path;

use async_trait::async_trait;
use ndarray::Array3;
use ndarray_npy::ReadNpyExt;

use super::{BandData, ImageLoader, ImageMetadata};
use crate::error::{Error, Result};

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
            // Try to read as f32 first, then f64
            let file = std::fs::File::open(&path)?;
            let reader = std::io::BufReader::new(file);

            // Try f32
            let shape = match Array3::<f32>::read_npy(reader) {
                Ok(arr) => arr.dim(),
                Err(_) => {
                    // Try f64
                    let file = std::fs::File::open(&path)?;
                    let reader = std::io::BufReader::new(file);
                    match Array3::<f64>::read_npy(reader) {
                        Ok(arr) => arr.dim(),
                        Err(e) => {
                            return Err(Error::InvalidImageData(format!(
                                "Failed to read NPY file: {}",
                                e
                            )));
                        }
                    }
                }
            };

            let (num_bands, height, width) = shape;

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
            let file = std::fs::File::open(&path)?;
            let reader = std::io::BufReader::new(file);

            // Try f32 first
            let bands: Vec<Vec<f32>> = match Array3::<f32>::read_npy(reader) {
                Ok(arr) => extract_bands_f32(arr),
                Err(_) => {
                    // Try f64
                    let file = std::fs::File::open(&path)?;
                    let reader = std::io::BufReader::new(file);
                    match Array3::<f64>::read_npy(reader) {
                        Ok(arr) => extract_bands_f64(arr),
                        Err(e) => {
                            return Err(Error::InvalidImageData(format!(
                                "Failed to read NPY file: {}",
                                e
                            )));
                        }
                    }
                }
            };

            if bands.is_empty() {
                return Err(Error::InvalidImageData("NPY file has no bands".to_string()));
            }

            // Infer dimensions from first band
            let num_pixels = bands[0].len();
            // We need to know height and width - for NPY with shape (B, H, W),
            // we read the file again to get the shape
            let file = std::fs::File::open(&path)?;
            let reader = std::io::BufReader::new(file);
            let shape = if let Ok(arr) = Array3::<f32>::read_npy(reader) {
                arr.dim()
            } else {
                let file = std::fs::File::open(&path)?;
                let reader = std::io::BufReader::new(file);
                Array3::<f64>::read_npy(reader)
                    .map_err(|e| Error::InvalidImageData(e.to_string()))?
                    .dim()
            };

            let (_num_bands, height, width) = shape;

            // Verify dimensions match
            if num_pixels != width * height {
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
        let mut band = Vec::with_capacity(height * width);
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
        let mut band = Vec::with_capacity(height * width);
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
