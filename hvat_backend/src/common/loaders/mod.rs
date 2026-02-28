//! Image loader traits and implementations.
//!
//! The [`ImageLoader`] trait provides a unified interface for loading
//! different image file formats into band data for streaming.

mod npy;
mod standard;

pub use npy::NpyLoader;
pub use standard::StandardImageLoader;

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use crate::common::error::Result;

/// Metadata about a loaded image.
#[derive(Debug, Clone)]
pub struct ImageMetadata {
    pub width: u32,
    pub height: u32,
    pub num_bands: usize,
    pub filename: String,
    pub format: String,
}

/// Band data loaded from an image file.
#[derive(Debug, Clone)]
pub struct BandData {
    pub width: u32,
    pub height: u32,
    /// Band data as f32 arrays, normalized to 0.0-1.0.
    /// Each band is width * height values in row-major order.
    pub bands: Vec<Vec<f32>>,
}

impl BandData {
    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    /// Number of RGBA texture layers needed (4 bands per layer).
    pub fn num_layers(&self) -> u32 {
        let layers = self.bands.len().div_ceil(4);
        // WebGL2 requires at least 2 layers for array textures
        layers.max(2) as u32
    }
}

/// Trait for loading different image formats.
#[async_trait]
pub trait ImageLoader: Send + Sync {
    fn name(&self) -> &'static str;
    fn supports(&self, path: &Path) -> bool;
    async fn load_metadata(&self, path: &Path) -> Result<ImageMetadata>;
    async fn load_bands(&self, path: &Path) -> Result<BandData>;
}

/// Registry of available image loaders.
#[derive(Clone)]
pub struct ImageLoaderRegistry {
    loaders: Vec<Arc<dyn ImageLoader>>,
}

impl ImageLoaderRegistry {
    pub fn new() -> Self {
        Self {
            loaders: Vec::new(),
        }
    }

    /// Create a registry pre-loaded with the standard and NPY loaders.
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();
        registry.register(Arc::new(StandardImageLoader));
        registry.register(Arc::new(NpyLoader));
        registry
    }

    pub fn register(&mut self, loader: Arc<dyn ImageLoader>) {
        tracing::info!("Registered image loader: {}", loader.name());
        self.loaders.push(loader);
    }

    pub fn find_loader(&self, path: &Path) -> Option<Arc<dyn ImageLoader>> {
        self.loaders
            .iter()
            .find(|loader| loader.supports(path))
            .cloned()
    }

    pub fn supports(&self, path: &Path) -> bool {
        self.loaders.iter().any(|loader| loader.supports(path))
    }
}

impl Default for ImageLoaderRegistry {
    fn default() -> Self {
        Self::new()
    }
}
