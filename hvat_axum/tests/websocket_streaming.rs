//! Integration tests for WebSocket image streaming.

use futures_util::{SinkExt, StreamExt};
use image::{ImageBuffer, Rgb};
use std::io::Cursor;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use hvat_axum::sam::{ExecutionProvider, SamVariant};
use hvat_axum::{AppState, ServerConfig, routes};

/// Test context that holds temporary directories and server address.
struct TestContext {
    _temp_dir: TempDir,
    _cache_dir: TempDir,
    addr: SocketAddr,
    /// Image ID for testing (URL-safe version of filename)
    test_image_id: String,
}

/// Create a simple PNG image (10x10 red) for testing using the image crate.
fn create_test_png() -> Vec<u8> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(10, 10, |_x, _y| {
        Rgb([255u8, 0u8, 0u8]) // Red pixel
    });

    let mut bytes: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("Failed to encode PNG");
    bytes
}

/// Create test fixtures and start server, returning context.
async fn setup_test_server() -> TestContext {
    // Create temporary directories
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let cache_dir = TempDir::new().expect("Failed to create cache dir");

    // Create test image
    let test_image_name = "test_image.png";
    let test_image_path = temp_dir.path().join(test_image_name);
    std::fs::write(&test_image_path, create_test_png()).expect("Failed to write test image");

    // Create subfolder with another image
    let subfolder = temp_dir.path().join("subfolder");
    std::fs::create_dir(&subfolder).expect("Failed to create subfolder");
    std::fs::write(subfolder.join("nested_image.png"), create_test_png())
        .expect("Failed to write nested image");

    let config = ServerConfig {
        port: 0, // Let OS assign port
        data_dir: temp_dir.path().to_path_buf(),
        cache_dir: cache_dir.path().to_path_buf(),
        max_cache_memory: 100 * 1024 * 1024,
        max_user_memory: 50 * 1024 * 1024,
        max_user_streams: 4,
        stream_chunk_rows: 128,
        project_name: "test_project".to_string(),
        sam_enabled: false,
        sam_model_dir: PathBuf::from("./.cache/models"),
        sam_variant: SamVariant::Tiny,
        sam_provider: ExecutionProvider::Cpu,
        sam_cache_size: 10,
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

    TestContext {
        _temp_dir: temp_dir,
        _cache_dir: cache_dir,
        addr,
        test_image_id: "test_image_png".to_string(),
    }
}

#[tokio::test]
async fn test_websocket_requires_start_stream_message() {
    let ctx = setup_test_server().await;

    // Connect to WebSocket without sending StartStream message
    let ws_url = format!("ws://{}/api/images/{}/stream", ctx.addr, ctx.test_image_id);

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
    let ctx = setup_test_server().await;

    // Connect to WebSocket
    let ws_url = format!("ws://{}/api/images/{}/stream", ctx.addr, ctx.test_image_id);

    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to WebSocket");

    // Send StartStream message
    let start_msg = serde_json::json!({
        "action": "start_stream",
        "image_id": ctx.test_image_id,
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
async fn test_rest_api_info() {
    let ctx = setup_test_server().await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{}/api/info", ctx.addr))
        .send()
        .await
        .expect("Failed to fetch info");

    assert!(resp.status().is_success());

    let info: serde_json::Value = resp.json().await.expect("Failed to parse JSON");
    assert!(info.is_object(), "Info should be an object");
    assert!(info.get("name").is_some(), "Info should have a name");
    assert!(
        info.get("image_count").is_some(),
        "Info should have image_count"
    );

    // Should have 2 images (test_image.png and subfolder/nested_image.png)
    let image_count = info["image_count"].as_u64().unwrap();
    assert_eq!(image_count, 2, "Should have 2 test images");

    println!(
        "Server Info: {}",
        serde_json::to_string_pretty(&info).unwrap()
    );
}

#[tokio::test]
async fn test_rest_api_images() {
    let ctx = setup_test_server().await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{}/api/images", ctx.addr))
        .send()
        .await
        .expect("Failed to fetch images");

    assert!(resp.status().is_success());

    let images: serde_json::Value = resp.json().await.expect("Failed to parse JSON");
    assert!(images.is_array(), "Images should be an array");

    // Should have 2 images created in setup
    let images_arr = images.as_array().unwrap();
    assert_eq!(images_arr.len(), 2, "Should have 2 test images");

    // Verify image structure
    for img in images_arr {
        assert!(img.get("id").is_some(), "Image should have id");
        assert!(img.get("name").is_some(), "Image should have name");
        assert!(img.get("path").is_some(), "Image should have path");
        assert!(img.get("format").is_some(), "Image should have format");
    }

    println!("Images: {}", serde_json::to_string_pretty(&images).unwrap());
}

#[tokio::test]
#[should_panic(expected = "SAM initialization failed")]
async fn test_sam_enabled_but_models_missing_should_error() {
    // When sam_enabled=true but models don't exist, the server should panic
    // with a clear error message instead of silently disabling SAM.

    let config = ServerConfig {
        port: 0,
        data_dir: std::path::PathBuf::from("../hvat_visual_tests/test_input"),
        cache_dir: std::path::PathBuf::from("./.cache/pyramids"),
        max_cache_memory: 100 * 1024 * 1024,
        max_user_memory: 50 * 1024 * 1024,
        max_user_streams: 4,
        stream_chunk_rows: 128,
        project_name: "test_input".to_string(),
        sam_enabled: true, // <-- USER ENABLED SAM
        sam_model_dir: std::path::PathBuf::from("/nonexistent/path/to/models"), // Models don't exist
        sam_variant: SamVariant::Tiny,
        sam_provider: ExecutionProvider::Cpu,
        sam_cache_size: 10,
    };

    // This should panic with a clear error about missing models
    let _state = Arc::new(AppState::new(config));
}

/// Start a test server with SAM enabled (requires models in .cache/models).
async fn start_test_server_with_sam() -> Option<SocketAddr> {
    let model_dir = std::path::PathBuf::from("../.cache/models");

    // Skip if models don't exist
    if !model_dir.join("sam2_hiera_tiny.encoder.onnx").exists() {
        println!("Skipping SAM test: models not found in {:?}", model_dir);
        return None;
    }

    let config = ServerConfig {
        port: 0,
        data_dir: std::path::PathBuf::from("../hvat_visual_tests/test_input"),
        cache_dir: std::path::PathBuf::from("./.cache/pyramids"),
        max_cache_memory: 100 * 1024 * 1024,
        max_user_memory: 50 * 1024 * 1024,
        max_user_streams: 4,
        stream_chunk_rows: 128,
        project_name: "test_input".to_string(),
        sam_enabled: true,
        sam_model_dir: model_dir,
        sam_variant: SamVariant::Tiny,
        sam_provider: ExecutionProvider::Cpu,
        sam_cache_size: 10,
    };

    let state = Arc::new(AppState::new(config));

    // Verify SAM engine is available
    assert!(
        state.sam_engine.is_some(),
        "SAM engine should be initialized"
    );
    println!("SAM engine initialized successfully");

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

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    Some(addr)
}

#[tokio::test]
async fn test_sam_segment_e2e_with_real_models() {
    // Real e2e test: start server with SAM, send segment request, get mask back
    let Some(addr) = start_test_server_with_sam().await else {
        println!("Test skipped: SAM models not available");
        return;
    };

    // Connect to WebSocket
    let ws_url = format!(
        "ws://{}/api/images/subfolder_Screenshot_20251228_135208_png/stream",
        addr
    );

    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to WebSocket");

    // Send SAM segment request with a point in the middle of the image
    let sam_segment_msg = serde_json::json!({
        "action": "sam_segment",
        "image_id": "subfolder_Screenshot_20251228_135208_png",
        "points": [
            {"x": 200.0, "y": 150.0, "label": 1}
        ],
        "box": null
    });

    println!("Sending SAM segment request...");
    ws_stream
        .send(Message::Text(sam_segment_msg.to_string().into()))
        .await
        .expect("Failed to send SAM segment message");

    // Wait for SAM mask response with generous timeout (first request computes embedding)
    let mut got_sam_mask = false;
    let mut got_error = false;
    let mut error_message = String::new();
    let mut polygon_count = 0;

    // SAM encoding can take 10-30 seconds on CPU for first request
    let timeout_duration = tokio::time::Duration::from_secs(60);
    let start_time = std::time::Instant::now();

    println!("Waiting for SAM response (may take up to 60s for first embedding)...");

    while start_time.elapsed() < timeout_duration {
        let msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(5000), ws_stream.next()).await;

        match msg {
            Ok(Some(Ok(Message::Text(text)))) => {
                println!("Received text: {}", text);
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if json["type"] == "sam_mask" {
                        got_sam_mask = true;
                        if let Some(polygons) = json["polygons"].as_array() {
                            polygon_count = polygons.len();
                        }
                        println!("Got SAM mask with {} polygons!", polygon_count);
                        break;
                    } else if json["type"] == "error" {
                        got_error = true;
                        error_message = json["message"].as_str().unwrap_or("").to_string();
                        println!("Got error: {}", error_message);
                        break;
                    }
                }
            }
            Ok(Some(Ok(Message::Binary(data)))) => {
                if !data.is_empty() && data[0] == 0x05 {
                    let len = u16::from_le_bytes([data[1], data[2]]) as usize;
                    error_message =
                        String::from_utf8_lossy(&data[3..3 + len.min(data.len() - 3)]).to_string();
                    got_error = true;
                    println!("Got binary error: {}", error_message);
                    break;
                }
            }
            Ok(Some(Ok(Message::Close(_)))) | Ok(None) => {
                println!("Connection closed unexpectedly");
                break;
            }
            Ok(Some(Err(e))) => {
                println!("WebSocket error: {}", e);
                break;
            }
            Err(_) => {
                // Timeout on this iteration, keep waiting
                println!(
                    "Still waiting... ({:.1}s elapsed)",
                    start_time.elapsed().as_secs_f32()
                );
            }
            _ => {}
        }
    }

    let _ = ws_stream.close(None).await;

    // Assertions
    assert!(
        !got_error,
        "SAM request failed with error: {}",
        error_message
    );
    assert!(
        got_sam_mask,
        "Did not receive SAM mask within timeout! This indicates the request is stuck."
    );
    assert!(
        polygon_count > 0,
        "SAM mask should contain at least one polygon"
    );

    println!(
        "SAM e2e test passed! Got {} polygons in {:.1}s",
        polygon_count,
        start_time.elapsed().as_secs_f32()
    );
}

#[tokio::test]
async fn test_sam_segment_without_sam_enabled_returns_error() {
    // Test that SAM segment request returns an error when SAM is not enabled
    // This tests the protocol handling without requiring the actual models
    let ctx = setup_test_server().await;

    // Connect to WebSocket
    let ws_url = format!("ws://{}/api/images/{}/stream", ctx.addr, ctx.test_image_id);

    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to WebSocket");

    // Send SAM segment request (should fail since SAM is not enabled)
    let sam_segment_msg = serde_json::json!({
        "action": "sam_segment",
        "image_id": ctx.test_image_id,
        "points": [
            {"x": 100.0, "y": 100.0, "label": 1}
        ],
        "box": null
    });

    ws_stream
        .send(Message::Text(sam_segment_msg.to_string().into()))
        .await
        .expect("Failed to send SAM segment message");

    // We should receive an error response (JSON text message)
    let mut received_error = false;
    let timeout_duration = tokio::time::Duration::from_secs(5);
    let start_time = std::time::Instant::now();

    while start_time.elapsed() < timeout_duration {
        let msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(1000), ws_stream.next()).await;

        match msg {
            Ok(Some(Ok(Message::Text(text)))) => {
                println!("Received text message: {}", text);
                // Check if it's an error about SAM not being enabled
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                    if json["type"] == "error" {
                        println!("Got expected error: {}", json["message"]);
                        received_error = true;
                        break;
                    }
                }
            }
            Ok(Some(Ok(Message::Binary(data)))) => {
                // Binary message - could be an error
                if !data.is_empty() && data[0] == 0x05 {
                    // Error message type
                    let len = u16::from_le_bytes([data[1], data[2]]) as usize;
                    let error_msg = String::from_utf8_lossy(&data[3..3 + len.min(data.len() - 3)]);
                    println!("Got binary error: {}", error_msg);
                    received_error = true;
                    break;
                }
            }
            Ok(Some(Ok(Message::Close(_)))) | Ok(None) => {
                println!("Connection closed");
                break;
            }
            Ok(Some(Err(e))) => {
                println!("WebSocket error: {}", e);
                break;
            }
            Err(_) => {
                // Timeout
                break;
            }
            _ => {}
        }
    }

    assert!(
        received_error,
        "Should have received an error since SAM is not enabled"
    );

    let _ = ws_stream.close(None).await;
}
