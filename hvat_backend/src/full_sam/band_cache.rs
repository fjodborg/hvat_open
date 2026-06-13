//! Packed RGBA layer cache for loaded image band data.
//!
//! Avoids re-reading and re-packing image files on every band request.
//! The packed representation is independent of which bands the client
//! wants to display, so it is cached once per image hash.

use std::num::NonZeroUsize;
use std::sync::Arc;

use lru::LruCache;
use tokio::sync::RwLock;

/// Cached packed RGBA layers for a single image.
pub struct CachedBandLayers {
    /// All bands packed as RGBA layers. Each entry is `(layer_idx, rgba_bytes)`.
    pub layers: Vec<(u32, Vec<u8>)>,
    pub width: u32,
    pub height: u32,
    pub num_bands: usize,
}

/// LRU cache for packed image band layers.
///
/// Thread-safe cache keyed by image hash. Stores all bands for an image
/// pre-packed as GPU-ready RGBA u8 data so `GET /{id}/bands` can skip the
/// full decode+pack on every request after the first.
pub struct PackedLayerCache {
    cache: RwLock<LruCache<String, Arc<CachedBandLayers>>>,
}

impl PackedLayerCache {
    pub fn new(max_entries: usize) -> Self {
        let size = NonZeroUsize::new(max_entries).unwrap_or(NonZeroUsize::MIN);
        Self {
            cache: RwLock::new(LruCache::new(size)),
        }
    }

    /// Get cached layers, updating LRU order.
    pub async fn get(&self, hash: &str) -> Option<Arc<CachedBandLayers>> {
        self.cache.write().await.get(hash).cloned()
    }

    /// Insert packed layers into the cache.
    pub async fn insert(&self, hash: String, entry: Arc<CachedBandLayers>) {
        self.cache.write().await.put(hash, entry);
    }
}

impl Default for PackedLayerCache {
    fn default() -> Self {
        Self::new(4)
    }
}
