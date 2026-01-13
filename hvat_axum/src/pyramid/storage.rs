//! Pyramid storage backends.
//!
//! The [`PyramidStorage`] trait abstracts where pyramid data is stored,
//! allowing for filesystem, memory, or cloud storage implementations.

use std::path::{Path, PathBuf};

use async_trait::async_trait;

use super::{PyramidMetadata, PyramidStatus};
use crate::error::{Error, Result};

/// Trait for pyramid cache storage backends.
#[async_trait]
pub trait PyramidStorage: Send + Sync {
    /// Get the status of a pyramid for an image.
    async fn get_status(&self, image_hash: &str) -> PyramidStatus;

    /// Check if a pyramid exists and is valid.
    async fn has_pyramid(&self, image_hash: &str) -> bool;

    /// Load pyramid metadata.
    async fn load_metadata(&self, image_hash: &str) -> Result<PyramidMetadata>;

    /// Load a specific pyramid level's RGBA data.
    ///
    /// Returns a vector of (layer_index, rgba_data) pairs.
    async fn load_level(&self, image_hash: &str, level: u32) -> Result<Vec<(u32, Vec<u8>)>>;

    /// Save pyramid metadata.
    async fn save_metadata(&self, image_hash: &str, metadata: &PyramidMetadata) -> Result<()>;

    /// Save a pyramid level's RGBA data.
    async fn save_level(
        &self,
        image_hash: &str,
        level: u32,
        layers: &[(u32, Vec<u8>)],
    ) -> Result<()>;

    /// Mark a pyramid as building (in progress).
    async fn mark_building(&self, image_hash: &str) -> Result<()>;

    /// Update build progress (0.0 to 1.0) and optional ETA.
    async fn update_progress(
        &self,
        image_hash: &str,
        progress: f32,
        eta_seconds: Option<u32>,
    ) -> Result<()>;

    /// Get build progress for a pyramid that's currently building.
    async fn get_progress(&self, image_hash: &str) -> Option<(f32, Option<u32>)>;

    /// Mark a pyramid as failed.
    async fn mark_failed(&self, image_hash: &str, error: &str) -> Result<()>;

    /// Delete a pyramid.
    async fn delete(&self, image_hash: &str) -> Result<()>;

    /// Mark a pyramid as ready.
    async fn mark_ready(&self, image_hash: &str) -> Result<()>;
}

/// Filesystem-based pyramid storage.
///
/// Stores pyramids in a directory structure:
/// ```text
/// cache_dir/
/// ├── {image_hash}/
/// │   ├── meta.json       # PyramidMetadata
/// │   ├── status          # "building", "ready", or "failed: error message"
/// │   ├── level_0.bin     # Thumbnail level
/// │   ├── level_1.bin     # Next level up
/// │   └── ...
/// ```
pub struct FilesystemStorage {
    cache_dir: PathBuf,
}

impl FilesystemStorage {
    /// Create a new filesystem storage.
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Get the directory path for a pyramid.
    fn pyramid_dir(&self, image_hash: &str) -> PathBuf {
        self.cache_dir.join(image_hash)
    }

    /// Get the metadata file path.
    fn metadata_path(&self, image_hash: &str) -> PathBuf {
        self.pyramid_dir(image_hash).join("meta.json")
    }

    /// Get the status file path.
    fn status_path(&self, image_hash: &str) -> PathBuf {
        self.pyramid_dir(image_hash).join("status")
    }

    /// Get a level file path.
    fn level_path(&self, image_hash: &str, level: u32) -> PathBuf {
        self.pyramid_dir(image_hash)
            .join(format!("level_{}.bin", level))
    }

    /// Ensure the pyramid directory exists.
    fn ensure_dir(&self, image_hash: &str) -> Result<()> {
        let dir = self.pyramid_dir(image_hash);
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?;
        }
        Ok(())
    }
}

#[async_trait]
impl PyramidStorage for FilesystemStorage {
    async fn get_status(&self, image_hash: &str) -> PyramidStatus {
        let status_path = self.status_path(image_hash);

        if !status_path.exists() {
            return PyramidStatus::Pending;
        }

        match std::fs::read_to_string(&status_path) {
            Ok(content) => {
                let content = content.trim();
                if content == "ready" {
                    PyramidStatus::Ready
                } else if content == "building" {
                    PyramidStatus::Building
                } else if content.starts_with("failed:") {
                    PyramidStatus::Failed
                } else {
                    PyramidStatus::Pending
                }
            }
            Err(_) => PyramidStatus::Pending,
        }
    }

    async fn has_pyramid(&self, image_hash: &str) -> bool {
        self.get_status(image_hash).await == PyramidStatus::Ready
    }

    async fn load_metadata(&self, image_hash: &str) -> Result<PyramidMetadata> {
        let path = self.metadata_path(image_hash);
        let content = tokio::fs::read_to_string(&path).await?;
        let metadata: PyramidMetadata =
            serde_json::from_str(&content).map_err(|e| Error::Internal(e.to_string()))?;
        Ok(metadata)
    }

    async fn load_level(&self, image_hash: &str, level: u32) -> Result<Vec<(u32, Vec<u8>)>> {
        let path = self.level_path(image_hash, level);

        // Read the level file
        let data = tokio::fs::read(&path).await?;

        // Parse the level format:
        // [num_layers: u32]
        // For each layer:
        //   [layer_index: u32][data_len: u64][rgba_data: bytes]

        if data.len() < 4 {
            return Err(Error::InvalidImageData("Level file too small".to_string()));
        }

        let num_layers = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let mut layers = Vec::with_capacity(num_layers);
        let mut offset = 4;

        for _ in 0..num_layers {
            if offset + 12 > data.len() {
                return Err(Error::InvalidImageData("Truncated level file".to_string()));
            }

            let layer_index = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]);
            offset += 4;

            let data_len = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]) as usize;
            offset += 8;

            if offset + data_len > data.len() {
                return Err(Error::InvalidImageData("Truncated layer data".to_string()));
            }

            let rgba_data = data[offset..offset + data_len].to_vec();
            offset += data_len;

            layers.push((layer_index, rgba_data));
        }

        Ok(layers)
    }

    async fn save_metadata(&self, image_hash: &str, metadata: &PyramidMetadata) -> Result<()> {
        self.ensure_dir(image_hash)?;
        let path = self.metadata_path(image_hash);
        let content =
            serde_json::to_string_pretty(metadata).map_err(|e| Error::Internal(e.to_string()))?;
        tokio::fs::write(&path, content).await?;
        Ok(())
    }

    async fn save_level(
        &self,
        image_hash: &str,
        level: u32,
        layers: &[(u32, Vec<u8>)],
    ) -> Result<()> {
        self.ensure_dir(image_hash)?;
        let path = self.level_path(image_hash, level);

        // Build the level file format
        let mut data = Vec::new();

        // Number of layers
        data.extend_from_slice(&(layers.len() as u32).to_le_bytes());

        // Each layer
        for (layer_index, rgba_data) in layers {
            data.extend_from_slice(&layer_index.to_le_bytes());
            data.extend_from_slice(&(rgba_data.len() as u64).to_le_bytes());
            data.extend_from_slice(rgba_data);
        }

        tokio::fs::write(&path, data).await?;
        Ok(())
    }

    async fn mark_building(&self, image_hash: &str) -> Result<()> {
        self.ensure_dir(image_hash)?;
        let path = self.status_path(image_hash);
        tokio::fs::write(&path, "building").await?;

        // Initialize progress file
        let progress_path = self.pyramid_dir(image_hash).join("progress.json");
        let progress_data = serde_json::json!({
            "progress": 0.0,
            "eta_seconds": null
        });
        tokio::fs::write(&progress_path, progress_data.to_string()).await?;

        Ok(())
    }

    async fn update_progress(
        &self,
        image_hash: &str,
        progress: f32,
        eta_seconds: Option<u32>,
    ) -> Result<()> {
        self.ensure_dir(image_hash)?;
        let progress_path = self.pyramid_dir(image_hash).join("progress.json");

        let progress_data = serde_json::json!({
            "progress": progress,
            "eta_seconds": eta_seconds
        });

        tokio::fs::write(&progress_path, progress_data.to_string()).await?;
        Ok(())
    }

    async fn get_progress(&self, image_hash: &str) -> Option<(f32, Option<u32>)> {
        let progress_path = self.pyramid_dir(image_hash).join("progress.json");

        let data = tokio::fs::read_to_string(&progress_path).await.ok()?;
        let json: serde_json::Value = serde_json::from_str(&data).ok()?;

        let progress = json.get("progress")?.as_f64()? as f32;
        let eta_seconds = json.get("eta_seconds")?.as_u64().map(|v| v as u32);

        Some((progress, eta_seconds))
    }

    async fn mark_failed(&self, image_hash: &str, error: &str) -> Result<()> {
        self.ensure_dir(image_hash)?;
        let path = self.status_path(image_hash);
        tokio::fs::write(&path, format!("failed: {}", error)).await?;
        Ok(())
    }

    async fn delete(&self, image_hash: &str) -> Result<()> {
        let dir = self.pyramid_dir(image_hash);
        if dir.exists() {
            tokio::fs::remove_dir_all(&dir).await?;
        }
        Ok(())
    }

    async fn mark_ready(&self, image_hash: &str) -> Result<()> {
        self.ensure_dir(image_hash)?;
        let path = self.status_path(image_hash);
        tokio::fs::write(&path, "ready").await?;
        Ok(())
    }
}

/// Compute a hash for an image file (for cache key).
pub fn compute_image_hash(path: &Path) -> Result<String> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let metadata = std::fs::metadata(path)?;

    let mut hasher = DefaultHasher::new();

    // Hash the path
    path.to_string_lossy().hash(&mut hasher);

    // Hash the file size
    metadata.len().hash(&mut hasher);

    // Hash the modification time if available
    if let Ok(modified) = metadata.modified() {
        if let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH) {
            duration.as_secs().hash(&mut hasher);
        }
    }

    Ok(format!("{:016x}", hasher.finish()))
}
