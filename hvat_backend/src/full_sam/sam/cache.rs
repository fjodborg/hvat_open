//! Embedding cache for SAM image embeddings.
//!
//! Caches computed image embeddings to avoid re-running the expensive
//! encoder model when the user makes multiple segmentation requests
//! on the same image.

use lru::LruCache;
use std::num::NonZeroUsize;
use tokio::sync::RwLock;

use crate::sam::EncoderOutput;

/// Cached image embedding with metadata.
///
/// Encoded state is runtime-defined (`EncoderOutput`) so the cache does not
/// assume a SAM2 tensor layout.
#[derive(Clone)]
pub struct CachedEmbedding {
    /// Runtime-specific encoded representation from `SamBackend::encode_image`.
    pub encoded_state: EncoderOutput,
    /// Original image width (for coordinate scaling).
    pub width: u32,
    /// Original image height (for coordinate scaling).
    pub height: u32,
}

/// LRU cache for image embeddings.
///
/// Thread-safe cache that stores embeddings keyed by image hash.
/// Uses LRU eviction to limit memory usage.
pub struct EmbeddingCache {
    cache: RwLock<LruCache<String, CachedEmbedding>>,
    max_entries: usize,
}

impl EmbeddingCache {
    /// Create a new embedding cache.
    ///
    /// # Arguments
    /// * `max_entries` - Maximum number of embeddings to cache (must be > 0)
    pub fn new(max_entries: usize) -> Self {
        let size = NonZeroUsize::new(max_entries).unwrap_or(NonZeroUsize::MIN);
        Self {
            cache: RwLock::new(LruCache::new(size)),
            max_entries,
        }
    }

    /// Get an embedding from the cache.
    ///
    /// Returns `None` if not found. Does not update LRU order (peek).
    pub async fn get(&self, hash: &str) -> Option<CachedEmbedding> {
        self.cache.read().await.peek(hash).cloned()
    }

    /// Get an embedding and update LRU order.
    ///
    /// Use this when actually using the embedding for inference.
    pub async fn get_and_touch(&self, hash: &str) -> Option<CachedEmbedding> {
        self.cache.write().await.get(hash).cloned()
    }

    /// Insert an embedding into the cache.
    ///
    /// If the cache is full, evicts the least recently used entry.
    pub async fn insert(&self, hash: String, embedding: CachedEmbedding) {
        self.cache.write().await.put(hash, embedding);
    }

    /// Check if an embedding is cached.
    pub async fn contains(&self, hash: &str) -> bool {
        self.cache.read().await.contains(hash)
    }

    /// Remove an embedding from the cache.
    pub async fn remove(&self, hash: &str) -> Option<CachedEmbedding> {
        self.cache.write().await.pop(hash)
    }

    /// Clear all cached embeddings.
    pub async fn clear(&self) {
        self.cache.write().await.clear();
    }

    /// Get the current number of cached embeddings.
    pub async fn len(&self) -> usize {
        self.cache.read().await.len()
    }

    /// Check if the cache is empty.
    pub async fn is_empty(&self) -> bool {
        self.cache.read().await.is_empty()
    }

    /// Get the maximum capacity.
    pub fn capacity(&self) -> usize {
        self.max_entries
    }
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new(50) // Default: cache up to 50 image embeddings
    }
}

#[cfg(test)]
#[path = "cache.test.rs"]
mod tests;
