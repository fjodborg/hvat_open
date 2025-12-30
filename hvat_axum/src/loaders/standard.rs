//! Standard image loader for PNG, JPEG, TIFF, etc.
//!
//! Uses the `image` crate to load common image formats.
//! Extracts R, G, B channels as separate bands.

use std::path::Path;

use async_trait::async_trait;
use image::GenericImageView;

use super::{BandData, ImageLoader, ImageMetadata};
use crate::error::{Error, Result};

/// Loader for standard image formats (PNG, JPEG, TIFF, etc.).
pub struct StandardImageLoader;

impl StandardImageLoader {
    pub fn new() -> Self {
        Self
    }

    /// Check if a file extension is supported.
    fn is_supported_extension(ext: &str) -> bool {
        matches!(
            ext.to_lowercase().as_str(),
            "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "tiff" | "tif"
        )
    }
}

impl Default for StandardImageLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ImageLoader for StandardImageLoader {
    fn name(&self) -> &'static str {
        "StandardImageLoader"
    }

    fn supports(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(Self::is_supported_extension)
    }

    async fn load_metadata(&self, path: &Path) -> Result<ImageMetadata> {
        // Use blocking task for file I/O
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            let img = image::open(&path).map_err(Error::Image)?;
            let (width, height) = img.dimensions();

            // Standard images have 3 or 4 channels, we extract RGB as bands
            let num_bands = match img.color() {
                image::ColorType::L8 | image::ColorType::L16 => 1,
                image::ColorType::La8 | image::ColorType::La16 => 1,
                _ => 3, // RGB or RGBA -> 3 bands
            };

            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            let format = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("unknown")
                .to_uppercase();

            Ok(ImageMetadata {
                width,
                height,
                num_bands,
                filename,
                format,
            })
        })
        .await
        .map_err(|e| Error::Internal(e.to_string()))?
    }

    async fn load_bands(&self, path: &Path) -> Result<BandData> {
        let path = path.to_path_buf();
        tokio::task::spawn_blocking(move || {
            let img = image::open(&path).map_err(Error::Image)?;
            let (width, height) = img.dimensions();
            let rgba = img.to_rgba8();
            let pixels = rgba.as_raw();

            let num_pixels = (width * height) as usize;

            // Extract R, G, B channels as separate f32 bands (normalized 0.0-1.0)
            let mut red = Vec::with_capacity(num_pixels);
            let mut green = Vec::with_capacity(num_pixels);
            let mut blue = Vec::with_capacity(num_pixels);

            for chunk in pixels.chunks(4) {
                red.push(chunk[0] as f32 / 255.0);
                green.push(chunk[1] as f32 / 255.0);
                blue.push(chunk[2] as f32 / 255.0);
            }

            Ok(BandData {
                width,
                height,
                bands: vec![red, green, blue],
            })
        })
        .await
        .map_err(|e| Error::Internal(e.to_string()))?
    }
}
