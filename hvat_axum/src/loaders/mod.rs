//! Image loader traits and implementations.
//!
//! The [`ImageLoader`] trait provides a unified interface for loading
//! different hyperspectral file formats into band data.

mod npy;
mod standard;

pub use npy::NpyLoader;
pub use standard::StandardImageLoader;

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;

/// Metadata about a loaded image.
#[derive(Debug, Clone)]
pub struct ImageMetadata {
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Number of spectral bands
    pub num_bands: usize,
    /// Original filename
    pub filename: String,
    /// File format (e.g., "PNG", "NPY", "ENVI")
    pub format: String,
}

/// Band data loaded from an image file.
#[derive(Debug)]
pub struct BandData {
    /// Image dimensions
    pub width: u32,
    pub height: u32,
    /// Band data as f32 arrays, normalized to 0.0-1.0
    /// Each band is width * height values in row-major order
    pub bands: Vec<Vec<f32>>,
}

impl BandData {
    /// Get the number of bands.
    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    /// Calculate the number of RGBA texture layers needed (4 bands per layer).
    pub fn num_layers(&self) -> u32 {
        let layers = self.bands.len().div_ceil(4);
        // WebGL2 requires at least 2 layers for array textures
        layers.max(2) as u32
    }
}

/// Trait for loading different image formats.
///
/// Implementations should be thread-safe and stateless.
#[async_trait]
pub trait ImageLoader: Send + Sync {
    /// Get a unique name for this loader (for logging/debugging).
    fn name(&self) -> &'static str;

    /// Check if this loader supports the given file path.
    ///
    /// Typically checks file extension.
    fn supports(&self, path: &Path) -> bool;

    /// Load image metadata without reading full band data.
    ///
    /// This should be fast - used for listing images.
    async fn load_metadata(&self, path: &Path) -> Result<ImageMetadata>;

    /// Load full band data from the image.
    ///
    /// Returns band data as f32 arrays normalized to 0.0-1.0.
    async fn load_bands(&self, path: &Path) -> Result<BandData>;
}

/// Registry of available image loaders.
#[derive(Clone)]
pub struct ImageLoaderRegistry {
    loaders: Vec<Arc<dyn ImageLoader>>,
}

impl ImageLoaderRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            loaders: Vec::new(),
        }
    }

    /// Register a new loader.
    pub fn register(&mut self, loader: Arc<dyn ImageLoader>) {
        tracing::info!("Registered image loader: {}", loader.name());
        self.loaders.push(loader);
    }

    /// Find a loader that supports the given path.
    pub fn find_loader(&self, path: &Path) -> Option<Arc<dyn ImageLoader>> {
        self.loaders
            .iter()
            .find(|loader| loader.supports(path))
            .cloned()
    }

    /// Get all registered loaders.
    pub fn loaders(&self) -> &[Arc<dyn ImageLoader>] {
        &self.loaders
    }

    /// Check if any loader supports the given path.
    pub fn supports(&self, path: &Path) -> bool {
        self.loaders.iter().any(|loader| loader.supports(path))
    }
}

impl Default for ImageLoaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}
