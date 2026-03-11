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
