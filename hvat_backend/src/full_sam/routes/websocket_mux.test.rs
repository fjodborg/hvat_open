use super::*;
use crate::config::ServerConfig;
use crate::state::AppState;
use std::time::Duration;

#[test]
fn prepared_dimensions_eviction_is_lru_by_access() {
    let mut streams = ConnectionStreams::new(4, 2);

    for i in 0..MAX_PREPARED_DIMENSIONS {
        streams.cache_prepared_dimensions(format!("k{}", i), i as u32, i as u32);
    }

    // Touch k0 so it is no longer the least-recently-used entry.
    assert_eq!(streams.prepared_dimensions("k0"), Some((0, 0)));

    // Insert one more key and ensure k1 (the true LRU) is evicted.
    streams.cache_prepared_dimensions("k_new".to_string(), 999, 999);
    assert!(streams.prepared_dimensions("k1").is_none());
    assert_eq!(streams.prepared_dimensions("k0"), Some((0, 0)));
    assert_eq!(streams.prepared_dimensions("k_new"), Some((999, 999)));
    assert_eq!(streams.prepared_dimensions.len(), MAX_PREPARED_DIMENSIONS);
}

#[test]
fn set_image_keeps_prepared_dimensions_for_other_images() {
    let mut streams = ConnectionStreams::new(4, 2);
    streams.cache_prepared_dimensions("img_a#0:1:2".to_string(), 100, 200);
    streams.cache_prepared_dimensions("img_b#0:1:2".to_string(), 300, 400);

    streams.set_image("img_a".to_string());
    streams.set_image("img_b".to_string());

    assert_eq!(streams.prepared_dimensions("img_a#0:1:2"), Some((100, 200)));
    assert_eq!(streams.prepared_dimensions("img_b#0:1:2"), Some((300, 400)));
}

#[test]
fn capabilities_without_models_disable_inference_and_sam() {
    let state = AppState::new(ServerConfig::default()).expect("state init should succeed");
    let caps = build_server_capabilities(&state);

    assert!(caps.features.streaming);
    assert!(caps.features.project_state);
    assert!(caps.features.downloads);
    assert!(caps.supports_chunked_downloads());
    assert!(!caps.features.inference);
    assert!(!caps.features.sam);
    assert!(caps.models.is_empty());
    assert!(!caps.supports_inference());
    assert!(!caps.supports_sam());
}

#[tokio::test(flavor = "current_thread")]
async fn shutdown_aborts_ping_before_waiting_for_sender_task() {
    let connection_alive = Arc::new(AtomicBool::new(true));
    let streams = Arc::new(RwLock::new(ConnectionStreams::new(4, 2)));
    let (tx, mut rx) = mpsc::channel::<Message>(8);

    let send_task = tokio::spawn(async move { while rx.recv().await.is_some() {} });

    let ping_task = {
        let ping_tx = tx.clone();
        Some(tokio::spawn(async move {
            // Hold a sender clone until aborted to mimic a sleeping ping loop.
            let _held_sender = ping_tx;
            tokio::time::sleep(Duration::from_secs(60)).await;
        }))
    };

    let shutdown = shutdown_mux_connection(connection_alive, streams, tx, send_task, ping_task);
    let result = tokio::time::timeout(Duration::from_millis(200), shutdown).await;

    assert!(
        result.is_ok(),
        "shutdown must not block waiting for ping interval before closing sender task"
    );
}
