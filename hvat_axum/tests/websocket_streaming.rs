//! Integration tests for WebSocket image streaming.

use futures_util::{SinkExt, StreamExt};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use hvat_axum::{AppState, ServerConfig, routes};

/// Start a test server and return its address.
async fn start_test_server() -> SocketAddr {
    let config = ServerConfig {
        port: 0, // Let OS assign port
        data_dir: std::path::PathBuf::from("../hvat_visual_tests/test_input"),
        cache_dir: std::path::PathBuf::from("./.cache/pyramids"),
        max_cache_memory: 100 * 1024 * 1024,
        max_user_memory: 50 * 1024 * 1024,
        max_user_streams: 4,
        stream_chunk_rows: 128,
        project_name: "test_input".to_string(),
    };

    let state = Arc::new(AppState::new(config));

    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    let app = axum::Router::new()
        .nest("/api", routes::api_router())
        .layer(cors)
        .with_state(state);

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    addr
}

#[tokio::test]
async fn test_websocket_requires_start_stream_message() {
    let addr = start_test_server().await;

    // Connect to WebSocket without sending StartStream message
    let ws_url = format!(
        "ws://{}/api/images/subfolder_Screenshot_20251228_135208_png/stream",
        addr
    );

    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to WebSocket");

    // Without sending StartStream, server should not send any binary data
    // Wait a bit and check if we get anything
    let timeout =
        tokio::time::timeout(tokio::time::Duration::from_millis(500), ws_stream.next()).await;

    // Should timeout because server is waiting for StartStream message
    assert!(
        timeout.is_err(),
        "Server should not send data without StartStream message"
    );

    // Close connection
    let _ = ws_stream.close(None).await;
}

#[tokio::test]
async fn test_websocket_streams_after_start_message() {
    let addr = start_test_server().await;

    // Connect to WebSocket
    let ws_url = format!(
        "ws://{}/api/images/subfolder_Screenshot_20251228_135208_png/stream",
        addr
    );

    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to WebSocket");

    // Send StartStream message
    let start_msg = serde_json::json!({
        "action": "start_stream",
        "image_id": "subfolder_Screenshot_20251228_135208_png",
        "level": 0
    });

    ws_stream
        .send(Message::Text(start_msg.to_string().into()))
        .await
        .expect("Failed to send StartStream message");

    // Now we should receive binary data (metadata first)
    let mut received_metadata = false;
    let mut received_chunks = 0;
    let mut received_level_complete = false;

    // Collect messages with timeout
    let timeout_duration = tokio::time::Duration::from_secs(10);
    let start_time = std::time::Instant::now();

    while start_time.elapsed() < timeout_duration {
        let msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(1000), ws_stream.next()).await;

        match msg {
            Ok(Some(Ok(Message::Binary(data)))) => {
                if data.is_empty() {
                    continue;
                }

                let msg_type = data[0];
                match msg_type {
                    0x01 => {
                        // Metadata
                        received_metadata = true;
                        assert!(data.len() >= 17, "Metadata should be at least 17 bytes");

                        let width = u32::from_le_bytes([data[1], data[2], data[3], data[4]]);
                        let height = u32::from_le_bytes([data[5], data[6], data[7], data[8]]);
                        let num_bands = u32::from_le_bytes([data[9], data[10], data[11], data[12]]);
                        let num_layers =
                            u32::from_le_bytes([data[13], data[14], data[15], data[16]]);

                        println!(
                            "Received metadata: {}x{}, {} bands, {} layers",
                            width, height, num_bands, num_layers
                        );

                        assert!(width > 0, "Width should be > 0");
                        assert!(height > 0, "Height should be > 0");
                    }
                    0x02 => {
                        // Layer chunk
                        received_chunks += 1;
                    }
                    0x03 => {
                        // Layer complete
                        println!("Layer complete");
                    }
                    0x04 => {
                        // Level complete
                        received_level_complete = true;
                        println!("Level complete - streaming finished");
                        break;
                    }
                    0x05 => {
                        // Error
                        let len = u16::from_le_bytes([data[1], data[2]]) as usize;
                        let error_msg = String::from_utf8_lossy(&data[3..3 + len]);
                        panic!("Server error: {}", error_msg);
                    }
                    _ => {
                        println!("Unknown message type: {}", msg_type);
                    }
                }
            }
            Ok(Some(Ok(Message::Text(text)))) => {
                println!("Received text message: {}", text);
            }
            Ok(Some(Ok(Message::Close(_)))) => {
                println!("Connection closed");
                break;
            }
            Ok(Some(Err(e))) => {
                panic!("WebSocket error: {}", e);
            }
            Ok(None) => {
                println!("Stream ended");
                break;
            }
            Err(_) => {
                // Timeout - check if we've received what we need
                if received_level_complete {
                    break;
                }
            }
            _ => {
                // Ping, Pong, Frame - ignore
            }
        }
    }

    assert!(received_metadata, "Should have received metadata");
    assert!(
        received_chunks > 0,
        "Should have received at least one chunk"
    );
    assert!(
        received_level_complete,
        "Should have received level complete"
    );

    println!("Test passed: received {} chunks", received_chunks);

    // Close connection
    let _ = ws_stream.close(None).await;
}

#[tokio::test]
async fn test_rest_api_projects() {
    let addr = start_test_server().await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{}/api/projects", addr))
        .send()
        .await
        .expect("Failed to fetch projects");

    assert!(resp.status().is_success());

    let projects: serde_json::Value = resp.json().await.expect("Failed to parse JSON");
    assert!(projects.is_array(), "Projects should be an array");

    println!(
        "Projects: {}",
        serde_json::to_string_pretty(&projects).unwrap()
    );
}

#[tokio::test]
async fn test_rest_api_images() {
    let addr = start_test_server().await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{}/api/projects/subfolder/images", addr))
        .send()
        .await
        .expect("Failed to fetch images");

    assert!(resp.status().is_success());

    let images: serde_json::Value = resp.json().await.expect("Failed to parse JSON");
    assert!(images.is_array(), "Images should be an array");

    println!("Images: {}", serde_json::to_string_pretty(&images).unwrap());
}
