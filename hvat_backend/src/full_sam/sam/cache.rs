//! Embedding cache for SAM image embeddings.
//!
//! Caches computed image embeddings to avoid re-running the expensive
//! encoder model when the user makes multiple segmentation requests
//! on the same image.

use lru::LruCache;
use std::num::NonZeroUsize;
use tokio::sync::RwLock;

/// Cached image embedding with metadata.
///
/// SAM 2 encoder produces three outputs that are all needed for decoding:
/// - `image_embed`: Main image embedding (256, 64, 64)
/// - `high_res_feats_0`: High resolution features (32, 256, 256)
/// - `high_res_feats_1`: High resolution features (64, 128, 128)
#[derive(Clone)]
pub struct CachedEmbedding {
    /// The main image embedding vector (256 * 64 * 64 = 1048576 floats).
    pub image_embed: Vec<f32>,
    /// High resolution features 0 (32 * 256 * 256 = 2097152 floats).
    pub high_res_feats_0: Vec<f32>,
    /// High resolution features 1 (64 * 128 * 128 = 1048576 floats).
    pub high_res_feats_1: Vec<f32>,
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
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cache_insert_get() {
        let cache = EmbeddingCache::new(10);

        let embedding = CachedEmbedding {
            image_embed: vec![1.0, 2.0, 3.0],
            high_res_feats_0: vec![4.0, 5.0],
            high_res_feats_1: vec![6.0, 7.0],
            width: 100,
            height: 100,
        };

        cache.insert("hash1".to_string(), embedding.clone()).await;

        let retrieved = cache.get("hash1").await.unwrap();
        assert_eq!(retrieved.image_embed, vec![1.0, 2.0, 3.0]);
        assert_eq!(retrieved.width, 100);
    }

    #[tokio::test]
    async fn test_cache_lru_eviction() {
        let cache = EmbeddingCache::new(2);

        for i in 0..3 {
            let embedding = CachedEmbedding {
                image_embed: vec![i as f32],
                high_res_feats_0: vec![],
                high_res_feats_1: vec![],
                width: 100,
                height: 100,
            };
            cache.insert(format!("hash{}", i), embedding).await;
        }

        // First entry should be evicted
        assert!(cache.get("hash0").await.is_none());
        assert!(cache.get("hash1").await.is_some());
        assert!(cache.get("hash2").await.is_some());
    }

    #[tokio::test]
    async fn test_dimension_mismatch_detection() {
        let cache = EmbeddingCache::new(10);

        // Insert embedding with specific dimensions
        let embedding = CachedEmbedding {
            image_embed: vec![1.0, 2.0, 3.0],
            high_res_feats_0: vec![],
            high_res_feats_1: vec![],
            width: 800,
            height: 600,
        };
        cache.insert("hash1".to_string(), embedding).await;

        // Retrieve and check dimensions match expected
        let cached = cache.get("hash1").await.unwrap();
        let expected_width = 800u32;
        let expected_height = 600u32;
        assert_eq!(cached.width, expected_width);
        assert_eq!(cached.height, expected_height);

        // Simulate dimension mismatch scenario
        let different_width = 1024u32;
        let different_height = 768u32;
        let has_mismatch = cached.width != different_width || cached.height != different_height;
        assert!(has_mismatch, "Should detect dimension mismatch");

        // After removing stale entry, cache should be empty for that hash
        cache.remove("hash1").await;
        assert!(cache.get("hash1").await.is_none());
    }

    #[tokio::test]
    async fn test_cache_update_with_new_dimensions() {
        let cache = EmbeddingCache::new(10);

        // Insert initial embedding
        let embedding1 = CachedEmbedding {
            image_embed: vec![1.0],
            high_res_feats_0: vec![],
            high_res_feats_1: vec![],
            width: 800,
            height: 600,
        };
        cache.insert("hash1".to_string(), embedding1).await;

        // Update with new embedding (different dimensions)
        let embedding2 = CachedEmbedding {
            image_embed: vec![2.0],
            high_res_feats_0: vec![],
            high_res_feats_1: vec![],
            width: 1024,
            height: 768,
        };
        cache.insert("hash1".to_string(), embedding2).await;

        // Should have new dimensions
        let cached = cache.get("hash1").await.unwrap();
        assert_eq!(cached.width, 1024);
        assert_eq!(cached.height, 768);
        assert_eq!(cached.image_embed, vec![2.0]);
    }
}
