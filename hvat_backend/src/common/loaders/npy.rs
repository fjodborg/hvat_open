//! NumPy .npy file loader for hyperspectral data.
//!
//! Expects arrays with shape (B, H, W) where B = bands, H = height, W = width.
//! Values are normalized to 0.0-1.0 based on the min/max values in the array.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use async_trait::async_trait;
use hvat_common::pixel_count;
use ndarray::Array3;
use ndarray_npy::ReadNpyExt;

use super::{BandData, ImageLoader, ImageMetadata};
use crate::common::error::{Error, Result};

// =============================================================================
// NPY Header Parsing
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NpyDtype {
    F32,
    F64,
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
}

#[derive(Debug)]
struct NpyHeader {
    dtype: NpyDtype,
    shape: Vec<usize>,
}

fn dim_to_u32(value: usize, axis: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::InvalidImageData(format!(
            "NPY {} dimension {} exceeds u32 range",
            axis, value
        ))
    })
}

fn parse_npy_header(path: &Path) -> Result<NpyHeader> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 6];
    reader.read_exact(&mut magic)?;
    if &magic != b"\x93NUMPY" {
        return Err(Error::InvalidImageData("Not a valid NPY file".to_string()));
    }

    let mut version = [0u8; 2];
    reader.read_exact(&mut version)?;

    let header_len = if version[0] == 1 {
        let mut len_bytes = [0u8; 2];
        reader.read_exact(&mut len_bytes)?;
        u16::from_le_bytes(len_bytes) as usize
    } else {
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        u32::from_le_bytes(len_bytes) as usize
    };

    let mut header_bytes = vec![0u8; header_len];
    reader.read_exact(&mut header_bytes)?;
    let header_str = String::from_utf8_lossy(&header_bytes);

    let dtype = parse_dtype_from_header(&header_str)?;
    let shape = parse_shape_from_header(&header_str)?;

    Ok(NpyHeader { dtype, shape })
}

fn parse_dtype_from_header(header: &str) -> Result<NpyDtype> {
    let descr_start = header
        .find("'descr'")
        .or_else(|| header.find("\"descr\""))
        .ok_or_else(|| Error::InvalidImageData("No 'descr' in NPY header".to_string()))?;

    let after_descr = &header[descr_start..];

    let dtype_str = after_descr
        .split(['\'', '"'])
        .find(|s| s.len() >= 2 && (s.starts_with('<') || s.starts_with('>') || s.starts_with('|')))
        .ok_or_else(|| Error::InvalidImageData("Cannot parse dtype from NPY header".to_string()))?;

    let type_part = &dtype_str[1..];

    match type_part {
        "f4" => Ok(NpyDtype::F32),
        "f8" => Ok(NpyDtype::F64),
        "u1" => Ok(NpyDtype::U8),
        "u2" => Ok(NpyDtype::U16),
        "u4" => Ok(NpyDtype::U32),
        "i1" => Ok(NpyDtype::I8),
        "i2" => Ok(NpyDtype::I16),
        "i4" => Ok(NpyDtype::I32),
        _ => Err(Error::InvalidImageData(format!(
            "Unsupported NPY dtype: {}",
            dtype_str
        ))),
    }
}

fn parse_shape_from_header(header: &str) -> Result<Vec<usize>> {
    let shape_start = header
        .find("'shape'")
        .or_else(|| header.find("\"shape\""))
        .ok_or_else(|| Error::InvalidImageData("No 'shape' in NPY header".to_string()))?;

    let after_shape = &header[shape_start..];

    let paren_start = after_shape
        .find('(')
        .ok_or_else(|| Error::InvalidImageData("No shape tuple in NPY header".to_string()))?;
    let paren_end = after_shape
        .find(')')
        .ok_or_else(|| Error::InvalidImageData("Unclosed shape tuple in NPY header".to_string()))?;

    let tuple_content = &after_shape[paren_start + 1..paren_end];

    let shape: Vec<usize> = tuple_content
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if shape.is_empty() {
        return Err(Error::InvalidImageData(
            "Empty shape in NPY header".to_string(),
        ));
    }

    Ok(shape)
}

// =============================================================================
// Band Extraction
// =============================================================================

fn read_npy_bands_typed(path: &Path, dtype: NpyDtype) -> Result<Vec<Vec<f32>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    match dtype {
        NpyDtype::F32 => read_and_normalize::<f32>(reader),
        NpyDtype::F64 => read_and_normalize::<f64>(reader),
        NpyDtype::U8 => read_and_normalize::<u8>(reader),
        NpyDtype::U16 => read_and_normalize::<u16>(reader),
        NpyDtype::U32 => read_and_normalize::<u32>(reader),
        NpyDtype::I8 => read_and_normalize::<i8>(reader),
        NpyDtype::I16 => read_and_normalize::<i16>(reader),
        NpyDtype::I32 => read_and_normalize::<i32>(reader),
    }
}

fn read_and_normalize<T>(reader: BufReader<File>) -> Result<Vec<Vec<f32>>>
where
    T: ndarray_npy::ReadableElement + Copy + Into<f64>,
{
    let arr: Array3<T> = Array3::read_npy(reader)
        .map_err(|e| Error::InvalidImageData(format!("Failed to read NPY: {}", e)))?;

    Ok(extract_and_normalize(arr))
}

fn extract_and_normalize<T: Copy + Into<f64>>(arr: Array3<T>) -> Vec<Vec<f32>> {
    let (num_bands, height, width) = arr.dim();

    let (min_val, max_val) = arr.iter().fold((f64::MAX, f64::MIN), |(min, max), &val| {
        let v: f64 = val.into();
        (min.min(v), max.max(v))
    });

    let range = max_val - min_val;
    let scale = if range > 0.0 { 1.0 / range } else { 1.0 };

    (0..num_bands)
        .map(|b| {
            let mut band = Vec::with_capacity(pixel_count(height, width));
            for h in 0..height {
                for w in 0..width {
                    let val: f64 = arr[[b, h, w]].into();
                    let normalized = ((val - min_val) * scale).clamp(0.0, 1.0) as f32;
                    band.push(normalized);
                }
            }
            band
        })
        .collect()
}

/// Loader for NumPy .npy files containing hyperspectral data.
pub struct NpyLoader;

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
            let header = parse_npy_header(&path)?;

            if header.shape.len() != 3 {
                return Err(Error::InvalidImageData(format!(
                    "Expected 3D array (B, H, W), got {}D",
                    header.shape.len()
                )));
            }

            let [num_bands, height, width] = [header.shape[0], header.shape[1], header.shape[2]];
            let width = dim_to_u32(width, "width")?;
            let height = dim_to_u32(height, "height")?;

            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            Ok(ImageMetadata {
                width,
                height,
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
            let header = parse_npy_header(&path)?;

            if header.shape.len() != 3 {
                return Err(Error::InvalidImageData(format!(
                    "Expected 3D array (B, H, W), got {}D",
                    header.shape.len()
                )));
            }

            let [_num_bands, height, width] = [header.shape[0], header.shape[1], header.shape[2]];
            let width = dim_to_u32(width, "width")?;
            let height = dim_to_u32(height, "height")?;

            let bands = read_npy_bands_typed(&path, header.dtype)?;

            if bands.is_empty() {
                return Err(Error::InvalidImageData("NPY file has no bands".to_string()));
            }

            Ok(BandData {
                width,
                height,
                bands,
            })
        })
        .await
        .map_err(|e| Error::Internal(e.to_string()))?
    }
}
