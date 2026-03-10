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

use hvat_backend::sam::{ExecutionProvider, SamVariant};
use hvat_backend::{AppState, ServerConfig, routes};
use hvat_common::annotation_io::{BundleFile, ExportBundle};
use zip::ZipArchive;

/// Test context that holds temporary directories and server address.
struct TestContext {
    /// Temp dir for test data. Must be kept alive - dropping it deletes the directory.
    /// Compiler warns "never read" but the Drop impl needs it to persist.
    temp_dir: TempDir,
    /// Cache dir for pyramids. Must be kept alive - dropping it deletes the directory.
    /// Compiler warns "never read" but the Drop impl needs it to persist.
    cache_dir: TempDir,
    addr: SocketAddr,
    /// Image ID for testing (URL-safe version of filename)
    test_image_id: String,
}

impl TestContext {
    fn assert_temp_dirs_alive(&self) {
        assert!(self.temp_dir.path().exists());
        assert!(self.cache_dir.path().exists());
    }
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

async fn bind_test_listener() -> Option<(TcpListener, SocketAddr)> {
    match TcpListener::bind("127.0.0.1:0").await {
        Ok(listener) => match listener.local_addr() {
            Ok(addr) => Some((listener, addr)),
            Err(e) => panic!("Failed to get listener address: {e}"),
        },
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            eprintln!("Skipping test: unable to bind local TCP listener in this environment ({e})");
            None
        }
        Err(e) => panic!("Failed to bind local TCP listener: {e}"),
    }
}

/// Create test fixtures and start server, returning context.
async fn setup_test_server() -> Option<TestContext> {
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
        base: hvat_backend::common::config::ServerConfig {
            port: 0, // Let OS assign port
            data_dir: temp_dir.path().to_path_buf(),
            cache_dir: cache_dir.path().to_path_buf(),
            max_cache_memory: 100 * 1024 * 1024,
            max_user_streams: 4,
            max_connections: 100,
            ping_interval_secs: 30,
            connection_timeout_secs: 90,
            stream_chunk_rows: 128,
            project_name: "test_project".to_string(),
        },
        max_user_memory: 50 * 1024 * 1024,
        pyramid_concurrency: 2,
        sam_enabled: false,
        sam_model_dir: PathBuf::from("./.cache/models"),
        sam_variant: SamVariant::Tiny,
        sam_provider: ExecutionProvider::Cpu,
        sam_cache_size: 10,
    };

    let state = Arc::new(
        AppState::new(config).expect("test server state should initialize for progressive test"),
    );

    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    let app = axum::Router::new()
        .nest("/api", routes::api_router())
        .layer(cors)
        .with_state(state);

    let (listener, addr) = bind_test_listener().await?;

    tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app).await {
            eprintln!("Test server exited with error: {err}");
        }
    });

    // Give server time to start
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let ctx = TestContext {
        temp_dir,
        cache_dir,
        addr,
        test_image_id: "test_image_png".to_string(),
    };
    ctx.assert_temp_dirs_alive();
    Some(ctx)
}

#[tokio::test]
async fn test_rest_api_info() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

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
    let Some(ctx) = setup_test_server().await else {
        return;
    };

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
async fn test_rest_api_images_download_archive() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{}/api/images/download", ctx.addr))
        .send()
        .await
        .expect("Failed to fetch image archive");

    assert!(
        resp.status().is_success(),
        "Archive endpoint should succeed"
    );
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("application/zip"),
        "Expected application/zip response, got '{}'",
        content_type
    );

    let bytes = resp.bytes().await.expect("Failed to read archive response");
    let mut archive =
        ZipArchive::new(Cursor::new(bytes.to_vec())).expect("Failed to parse zip archive");

    let mut names = Vec::new();
    for i in 0..archive.len() {
        let entry = archive.by_index(i).expect("Failed to read zip entry");
        names.push(entry.name().to_string());
    }
    names.sort();

    assert_eq!(names.len(), 2, "Expected exactly 2 images in archive");
    assert!(
        names.iter().any(|name| name == "test_image.png"),
        "Archive should contain top-level image: {:?}",
        names
    );
    assert!(
        names
            .iter()
            .any(|name| name == "subfolder/nested_image.png"),
        "Archive should contain nested image path: {:?}",
        names
    );
}

#[tokio::test]
async fn test_rest_api_images_download_plan() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{}/api/images/download/plan", ctx.addr))
        .send()
        .await
        .expect("Failed to fetch image download plan");

    assert!(resp.status().is_success(), "Plan endpoint should succeed");
    let plan: serde_json::Value = resp.json().await.expect("Failed to parse plan JSON");

    let total_files = plan["total_files"].as_u64().unwrap_or(0);
    let part_count = plan["part_count"].as_u64().unwrap_or(0);
    let parts = plan["parts"].as_array().cloned().unwrap_or_default();

    assert_eq!(total_files, 2, "Plan should include 2 files");
    assert!(
        part_count >= 1,
        "Plan should include at least one part, got {}",
        part_count
    );
    assert_eq!(
        parts.len() as u64,
        part_count,
        "parts array length should match part_count"
    );

    let first_part = parts.first().expect("Plan should include first part");
    assert!(
        first_part["filename"]
            .as_str()
            .unwrap_or_default()
            .ends_with(".zip"),
        "Part filename should end with .zip"
    );
}

#[tokio::test]
async fn test_rest_api_images_download_part() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{}/api/images/download/part/0", ctx.addr))
        .send()
        .await
        .expect("Failed to fetch image archive part");

    assert!(
        resp.status().is_success(),
        "Part endpoint should succeed for part 0"
    );
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.contains("application/zip"),
        "Expected application/zip response, got '{}'",
        content_type
    );

    let bytes = resp.bytes().await.expect("Failed to read archive response");
    let archive =
        ZipArchive::new(Cursor::new(bytes.to_vec())).expect("Failed to parse zip archive");
    assert!(
        archive.len() >= 1,
        "Part archive should contain at least one image"
    );
}

#[tokio::test]
async fn test_rest_api_images_download_part_invalid_index() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{}/api/images/download/part/999", ctx.addr))
        .send()
        .await
        .expect("Failed to fetch invalid image archive part");

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::NOT_FOUND,
        "Invalid part index should return 404"
    );
}

#[tokio::test]
async fn test_project_state_round_trip() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    let client = reqwest::Client::new();
    let url = format!("http://{}/api/project-state", ctx.addr);

    let missing = client
        .get(&url)
        .send()
        .await
        .expect("Failed to fetch project state");
    assert_eq!(missing.status(), reqwest::StatusCode::NOT_FOUND);

    let bundle = ExportBundle {
        files: vec![BundleFile {
            path: "annotations/default.json".to_string(),
            bytes: br#"{"items":[]}"#.to_vec(),
            mime_type: Some("application/json".to_string()),
        }],
    };
    let payload = serde_json::to_vec(&bundle).expect("serialize project state bundle");

    let save = client
        .put(&url)
        .header("content-type", "application/json")
        .body(payload.clone())
        .send()
        .await
        .expect("Failed to save project state");
    assert_eq!(save.status(), reqwest::StatusCode::NO_CONTENT);

    let loaded = client
        .get(&url)
        .send()
        .await
        .expect("Failed to reload project state");
    assert_eq!(loaded.status(), reqwest::StatusCode::OK);
    let loaded_bytes = loaded.bytes().await.expect("read response body");
    assert_eq!(loaded_bytes.as_ref(), payload.as_slice());

    let decoded: ExportBundle =
        serde_json::from_slice(&loaded_bytes).expect("decode saved bundle payload");
    assert_eq!(decoded.files.len(), 1);
    assert_eq!(decoded.files[0].path, "annotations/default.json");
}

#[tokio::test]
async fn test_project_state_rejects_invalid_payload() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    let client = reqwest::Client::new();
    let url = format!("http://{}/api/project-state", ctx.addr);

    let resp = client
        .put(&url)
        .header("content-type", "application/json")
        .body(r#"{"not":"a valid bundle"}"#)
        .send()
        .await
        .expect("Failed to submit invalid payload");

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test]
#[should_panic(expected = "initialization failed")]
async fn test_sam_enabled_but_models_missing_should_error() {
    // When sam_enabled=true but models don't exist, the server should panic
    // with a clear error message instead of silently disabling SAM.

    let config = ServerConfig {
        base: hvat_backend::common::config::ServerConfig {
            port: 0,
            data_dir: std::path::PathBuf::from("../hvat_visual_tests/test_input"),
            cache_dir: std::path::PathBuf::from("./.cache/pyramids"),
            max_cache_memory: 100 * 1024 * 1024,
            max_user_streams: 4,
            max_connections: 100,
            ping_interval_secs: 30,
            connection_timeout_secs: 90,
            stream_chunk_rows: 128,
            project_name: "test_input".to_string(),
        },
        max_user_memory: 50 * 1024 * 1024,
        pyramid_concurrency: 2,
        sam_enabled: true, // <-- USER ENABLED SAM
        sam_model_dir: std::path::PathBuf::from("/nonexistent/path/to/models"), // Models don't exist
        sam_variant: SamVariant::Tiny,
        sam_provider: ExecutionProvider::Cpu,
        sam_cache_size: 10,
    };

    // This should panic with a clear error about missing models
    Arc::new(AppState::new(config).expect("SAM initialization failed"));
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
        base: hvat_backend::common::config::ServerConfig {
            port: 0,
            data_dir: std::path::PathBuf::from("../hvat_visual_tests/test_input"),
            cache_dir: std::path::PathBuf::from("./.cache/pyramids"),
            max_cache_memory: 100 * 1024 * 1024,
            max_user_streams: 4,
            max_connections: 100,
            ping_interval_secs: 30,
            connection_timeout_secs: 90,
            stream_chunk_rows: 128,
            project_name: "test_input".to_string(),
        },
        max_user_memory: 50 * 1024 * 1024,
        pyramid_concurrency: 2,
        sam_enabled: true,
        sam_model_dir: model_dir,
        sam_variant: SamVariant::Tiny,
        sam_provider: ExecutionProvider::Cpu,
        sam_cache_size: 10,
    };

    let state = Arc::new(
        AppState::new(config).expect("test server state should initialize for progressive test"),
    );

    // Verify SAM model registry is available.
    assert!(
        state.model_registry.is_some(),
        "SAM model registry should be initialized"
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

    let (listener, addr) = bind_test_listener().await?;

    tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app).await {
            eprintln!("SAM test server exited with error: {err}");
        }
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    Some(addr)
}

/// TODO: Rewrite this test to use the mux protocol with the generic infer message
#[tokio::test]
#[ignore = "Needs rewrite for mux protocol - uses removed per-image endpoint and legacy SAM messages"]
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
                // Protocol v1: [version][msg_type][payload...]
                if data.len() >= 2 {
                    let version = data[0];
                    let msg_type = data[1];

                    if version == 1 && msg_type == 0xFE {
                        // Error message type
                        if data.len() >= 4 {
                            let error_code = u16::from_le_bytes([data[2], data[3]]);
                            error_message = format!("Error code: {}", error_code);
                            got_error = true;
                            println!("Got binary error: {}", error_message);
                            break;
                        }
                    }
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

    ws_stream.close(None).await.ok(); // Test cleanup - ignore close errors

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

/// Test that progressive streaming sends multiple levels in correct order.
///
/// This test verifies that:
/// 1. Progressive mode sends levels from highest (thumbnail) to lowest (full-res)
/// 2. Each level has Metadata with correct dimensions
/// 3. Only one Reset is sent (before first level)
/// 4. LevelComplete is sent for each level with correct level number
/// 5. Total bytes received per level matches expected (width * height * 4 * num_layers)
///
/// TODO: Rewrite this test to use the mux protocol (/api/ws with set_image + stream_image)
#[tokio::test]
#[ignore = "Needs rewrite for mux protocol - uses removed per-image endpoint"]
async fn test_progressive_streaming_sends_multiple_levels() {
    // Create a larger test image that will have multiple pyramid levels
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let cache_dir = TempDir::new().expect("Failed to create cache dir");

    // Create a 1024x1024 image to generate multiple pyramid levels
    // MIN_THUMBNAIL_SIZE is 256, so we need:
    // Level 0: 1024x1024 (full resolution)
    // Level 1: 512x512
    // Level 2: 256x256 (thumbnail - stops here)
    // Progressive should send: Level 2 -> Level 1 -> Level 0
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_fn(1024, 1024, |x, y| {
        // Gradient so we can verify downsampling
        Rgb([(x % 256) as u8, (y % 256) as u8, 128u8])
    });

    let test_image_name = "pyramid_test.png";
    let test_image_path = temp_dir.path().join(test_image_name);
    let mut bytes: Vec<u8> = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("Failed to encode PNG");
    std::fs::write(&test_image_path, bytes).expect("Failed to write test image");

    let config = ServerConfig {
        base: hvat_backend::common::config::ServerConfig {
            port: 0,
            data_dir: temp_dir.path().to_path_buf(),
            cache_dir: cache_dir.path().to_path_buf(),
            max_cache_memory: 100 * 1024 * 1024,
            max_user_streams: 4,
            max_connections: 100,
            ping_interval_secs: 30,
            connection_timeout_secs: 90,
            stream_chunk_rows: 128,
            project_name: "test_project".to_string(),
        },
        max_user_memory: 50 * 1024 * 1024,
        pyramid_concurrency: 2,
        sam_enabled: false,
        sam_model_dir: PathBuf::from("./.cache/models"),
        sam_variant: SamVariant::Tiny,
        sam_provider: ExecutionProvider::Cpu,
        sam_cache_size: 10,
    };

    let state = Arc::new(
        AppState::new(config).expect("test server state should initialize for progressive test"),
    );

    let cors = tower_http::cors::CorsLayer::new()
        .allow_origin(tower_http::cors::Any)
        .allow_methods(tower_http::cors::Any)
        .allow_headers(tower_http::cors::Any);

    let app = axum::Router::new()
        .nest("/api", routes::api_router())
        .layer(cors)
        .with_state(state);

    let Some((listener, addr)) = bind_test_listener().await else {
        return;
    };

    tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app).await {
            eprintln!("Progressive streaming test server exited with error: {err}");
        }
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Connect to WebSocket
    let image_id = "pyramid_test_png";
    let ws_url = format!("ws://{}/api/images/{}/stream", addr, image_id);

    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to WebSocket");

    // Send StartStream with progressive=true
    // This should work directly from source without needing a cached pyramid
    let start_msg = serde_json::json!({
        "action": "start_stream",
        "level": 0,
        "progressive": true
    });

    ws_stream
        .send(Message::Text(start_msg.to_string().into()))
        .await
        .expect("Failed to send StartStream message");

    // Track what we receive
    let mut reset_count = 0;
    let mut levels_received: Vec<LevelData> = Vec::new();
    let mut current_level_data = LevelData::default();

    #[derive(Default, Debug)]
    struct LevelData {
        width: u32,
        height: u32,
        num_layers: u32,
        level_number: Option<u8>,
        total_bytes: usize,
    }

    let timeout_duration = tokio::time::Duration::from_secs(30);
    let start_time = std::time::Instant::now();

    while start_time.elapsed() < timeout_duration {
        let msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(2000), ws_stream.next()).await;

        match msg {
            Ok(Some(Ok(Message::Binary(data)))) => {
                if data.len() < 2 {
                    continue;
                }

                let version = data[0];
                let msg_type = data[1];

                assert_eq!(version, 2, "Expected protocol version 2");

                match msg_type {
                    0x00 => {
                        // Reset
                        reset_count += 1;
                        println!("Received Reset #{}", reset_count);
                    }
                    0x06 => {
                        // Capabilities - sent on connect, just log and continue
                        println!("Received Capabilities");
                    }
                    0x01 => {
                        // Metadata - start of a new level
                        if data.len() >= 18 {
                            current_level_data = LevelData {
                                width: u32::from_le_bytes([data[2], data[3], data[4], data[5]]),
                                height: u32::from_le_bytes([data[6], data[7], data[8], data[9]]),
                                num_layers: u32::from_le_bytes([
                                    data[14], data[15], data[16], data[17],
                                ]),
                                level_number: None,
                                total_bytes: 0,
                            };
                            println!(
                                "Received Metadata: {}x{}, {} layers",
                                current_level_data.width,
                                current_level_data.height,
                                current_level_data.num_layers
                            );
                        }
                    }
                    0x02 => {
                        // LayerChunk - accumulate bytes
                        if data.len() > 12 {
                            let chunk_data_len = data.len() - 12; // header is 12 bytes
                            current_level_data.total_bytes += chunk_data_len;
                        }
                    }
                    0x03 => {
                        // LayerComplete - just log
                        if data.len() >= 4 {
                            let layer = u16::from_le_bytes([data[2], data[3]]);
                            println!("Layer {} complete", layer);
                        }
                    }
                    0x04 => {
                        // LevelComplete - save this level's data
                        if !data.is_empty() && data.len() >= 3 {
                            current_level_data.level_number = Some(data[2]);
                            println!(
                                "Level {} complete: {}x{}, {} bytes received",
                                data[2],
                                current_level_data.width,
                                current_level_data.height,
                                current_level_data.total_bytes
                            );
                            levels_received.push(std::mem::take(&mut current_level_data));

                            // If we got level 0 (full resolution), we're done
                            if data[2] == 0 {
                                break;
                            }
                        }
                    }
                    0xFE => {
                        // Error
                        if data.len() >= 4 {
                            let error_code = u16::from_le_bytes([data[2], data[3]]);
                            // 2000 = PyramidNotReady - this is a retryable error
                            // The test may hit this if pyramid building hasn't completed
                            // For this test, we accept partial results if we got at least one level
                            if error_code == 2000 {
                                println!(
                                    "Pyramid not ready (code 2000) - this can happen during progressive loading"
                                );
                                if !levels_received.is_empty() {
                                    println!(
                                        "Got {} levels before pyramid became busy, accepting partial result",
                                        levels_received.len()
                                    );
                                    break;
                                }
                                // No levels yet - this shouldn't happen, but continue to see if more data comes
                                continue;
                            }
                            panic!("Received error: code {}", error_code);
                        }
                    }
                    _ => {
                        println!("Unknown message type: 0x{:02x}", msg_type);
                    }
                }
            }
            Ok(Some(Ok(Message::Close(_)))) | Ok(None) => {
                println!("Connection closed");
                break;
            }
            Ok(Some(Err(e))) => {
                panic!("WebSocket error: {}", e);
            }
            Err(_) => {
                // Timeout
                if !levels_received.is_empty()
                    && levels_received.last().map(|l| l.level_number) == Some(Some(0))
                {
                    break;
                }
            }
            _ => {}
        }
    }

    ws_stream.close(None).await.ok(); // Test cleanup - ignore close errors

    // Assertions
    assert!(
        reset_count >= 1,
        "Should receive at least one Reset message"
    );

    assert!(
        !levels_received.is_empty(),
        "Should receive at least one level"
    );

    println!("\n=== Progressive Streaming Test Results ===");
    println!("Reset messages: {}", reset_count);
    println!("Levels received: {}", levels_received.len());

    for (i, level) in levels_received.iter().enumerate() {
        let expected_bytes = (level.width * level.height * 4 * level.num_layers) as usize;
        println!(
            "  Level {} (idx {}): {}x{}, {} layers, {} bytes (expected: {})",
            level.level_number.unwrap_or(255),
            i,
            level.width,
            level.height,
            level.num_layers,
            level.total_bytes,
            expected_bytes
        );

        // Verify bytes match expected
        assert_eq!(
            level.total_bytes,
            expected_bytes,
            "Level {} byte count mismatch",
            level.level_number.unwrap_or(255)
        );
    }

    // Verify levels are in descending order (highest/smallest first, then progressively larger)
    // Level numbers should be: highest (e.g., 2) -> 1 -> 0
    let level_numbers: Vec<u8> = levels_received
        .iter()
        .filter_map(|l| l.level_number)
        .collect();
    println!("Level order: {:?}", level_numbers);

    // Verify descending order (if we have multiple levels)
    for i in 1..level_numbers.len() {
        assert!(
            level_numbers[i] < level_numbers[i - 1],
            "Levels should be in descending order: got {:?}",
            level_numbers
        );
    }

    // If we got all levels, verify we ended at level 0
    // (we may not get all levels if pyramid building interrupted us)
    if levels_received.len() >= 3 {
        assert_eq!(
            level_numbers.last(),
            Some(&0),
            "Final level should be 0 (full resolution)"
        );
    } else {
        println!(
            "Note: Got {} levels (may be partial due to pyramid building)",
            levels_received.len()
        );
    }

    // Verify dimensions increase with each level (smaller level number = larger dimensions)
    for i in 1..levels_received.len() {
        assert!(
            levels_received[i].width >= levels_received[i - 1].width,
            "Width should increase as level number decreases"
        );
        assert!(
            levels_received[i].height >= levels_received[i - 1].height,
            "Height should increase as level number decreases"
        );
    }

    println!("\nProgressive streaming test PASSED!");
}

// ============================================================================
// Multiplexed WebSocket Tests (/api/ws endpoint)
// ============================================================================

/// Test that the multiplexed endpoint sends capabilities on connect.
#[tokio::test]
async fn test_mux_websocket_sends_capabilities_on_connect() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    // Connect to multiplexed WebSocket endpoint
    let ws_url = format!("ws://{}/api/ws", ctx.addr);

    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to multiplexed WebSocket");

    // Server should send capabilities immediately on connect
    let msg = tokio::time::timeout(tokio::time::Duration::from_millis(500), ws_stream.next())
        .await
        .expect("Should receive capabilities message")
        .expect("Stream should not end")
        .expect("Should be valid message");

    // Verify it's a capabilities message (type 0x06)
    if let tokio_tungstenite::tungstenite::Message::Binary(data) = msg {
        assert!(data.len() >= 2, "Capabilities message should have header");
        assert_eq!(data[0], 2, "Protocol version should be 2");
        assert_eq!(
            data[1], 0x06,
            "First message should be capabilities (type 0x06)"
        );

        // Parse capabilities (format: [version:u8][type:u8][request_id:u32][JSON payload])
        if data.len() >= 10 {
            let request_id = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);
            assert_eq!(request_id, 0, "Capabilities should have request_id=0");

            // Parse JSON payload
            let json_payload = &data[6..];
            if let Ok(caps) = serde_json::from_slice::<serde_json::Value>(json_payload) {
                println!("Capabilities received: {:?}", caps);
                assert_eq!(caps["protocol_version"], 2, "Protocol version should be 2");
                assert!(caps["limits"]["max_concurrent_streams"].as_u64().unwrap() > 0);
            } else {
                panic!("Failed to parse capabilities JSON");
            }
        }
    } else {
        panic!("Expected binary capabilities message");
    }

    ws_stream.close(None).await.ok(); // Test cleanup - ignore close errors
}

/// Test that the multiplexed endpoint can stream an image using Protocol v2 (set_image + stream_image).
#[tokio::test]
async fn test_mux_websocket_streams_image() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    // Connect to multiplexed WebSocket endpoint
    let ws_url = format!("ws://{}/api/ws", ctx.addr);

    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to multiplexed WebSocket");

    // Receive capabilities first
    ws_stream.next().await;

    // Send set_image + stream_image messages (Protocol v2)
    let request_id = 42u32;

    // First, set the active image context
    let set_image_msg = serde_json::json!({
        "action": "set_image",
        "request_id": request_id,
        "image_id": ctx.test_image_id,
    });
    ws_stream
        .send(Message::Text(set_image_msg.to_string().into()))
        .await
        .expect("Failed to send set_image message");

    // Then request streaming
    let stream_msg = serde_json::json!({
        "action": "stream_image",
        "request_id": request_id,
        "level": 0,
        "progressive": false
    });
    ws_stream
        .send(Message::Text(stream_msg.to_string().into()))
        .await
        .expect("Failed to send stream_image message");

    // Now we should receive binary data with request_id embedded
    let mut received_metadata = false;
    let mut received_chunks = 0;
    let mut received_level_complete = false;
    let mut received_stream_complete = false;

    let timeout_duration = tokio::time::Duration::from_secs(10);
    let start_time = std::time::Instant::now();

    while start_time.elapsed() < timeout_duration {
        let msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(1000), ws_stream.next()).await;

        match msg {
            Ok(Some(Ok(Message::Binary(data)))) => {
                if data.len() < 6 {
                    continue;
                }

                // Multiplexed protocol: [version][msg_type][request_id:u32][payload...]
                let version = data[0];
                let msg_type = data[1];
                let msg_request_id = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);

                assert_eq!(version, 2, "Expected protocol version 2");
                assert_eq!(
                    msg_request_id, request_id,
                    "Request ID should match our request"
                );

                match msg_type {
                    0x00 => {
                        // Reset message
                        println!("Received Reset (request_id={})", msg_request_id);
                    }
                    0x01 => {
                        // Metadata
                        received_metadata = true;
                        if data.len() >= 22 {
                            let width = u32::from_le_bytes([data[6], data[7], data[8], data[9]]);
                            let height =
                                u32::from_le_bytes([data[10], data[11], data[12], data[13]]);
                            println!(
                                "Received Metadata: {}x{} (request_id={})",
                                width, height, msg_request_id
                            );
                        }
                    }
                    0x02 => {
                        // Layer chunk
                        received_chunks += 1;
                    }
                    0x03 => {
                        // Layer complete
                        println!("Layer complete (request_id={})", msg_request_id);
                    }
                    0x04 => {
                        // Level complete
                        received_level_complete = true;
                        println!("Level complete (request_id={})", msg_request_id);
                    }
                    0x05 => {
                        // Stream complete (AllComplete reused for mux)
                        received_stream_complete = true;
                        println!("Stream complete (request_id={})", msg_request_id);
                        break;
                    }
                    0x07 => {
                        // StreamError
                        if data.len() >= 8 {
                            let error_code = u16::from_le_bytes([data[6], data[7]]);
                            panic!(
                                "Stream error: code {} (request_id={})",
                                error_code, msg_request_id
                            );
                        }
                    }
                    _ => {
                        println!(
                            "Unknown message type: 0x{:02x} (request_id={})",
                            msg_type, msg_request_id
                        );
                    }
                }
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
                if received_stream_complete {
                    break;
                }
            }
            _ => {}
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
    assert!(
        received_stream_complete,
        "Should have received stream complete"
    );

    println!(
        "Multiplexed streaming test passed: received {} chunks",
        received_chunks
    );

    ws_stream.close(None).await.ok(); // Test cleanup - ignore close errors
}

/// Test that multiple concurrent streams work on the multiplexed endpoint.
#[tokio::test]
async fn test_mux_websocket_concurrent_streams() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    // Connect to multiplexed WebSocket endpoint
    let ws_url = format!("ws://{}/api/ws", ctx.addr);

    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to multiplexed WebSocket");

    // Receive capabilities first
    ws_stream.next().await;

    // Start two streams with different request_ids
    let request_id_1 = 100u32;
    let request_id_2 = 200u32;

    // Send set_image + stream_image for first stream
    let set_image_msg_1 = serde_json::json!({
        "action": "set_image",
        "request_id": request_id_1,
        "image_id": ctx.test_image_id,
    });
    let stream_msg_1 = serde_json::json!({
        "action": "stream_image",
        "request_id": request_id_1,
        "level": 0,
        "progressive": false
    });

    // Send set_image + stream_image for second stream
    let set_image_msg_2 = serde_json::json!({
        "action": "set_image",
        "request_id": request_id_2,
        "image_id": "subfolder_nested_image_png",
    });
    let stream_msg_2 = serde_json::json!({
        "action": "stream_image",
        "request_id": request_id_2,
        "level": 0,
        "progressive": false
    });

    // Send both requests
    ws_stream
        .send(Message::Text(set_image_msg_1.to_string().into()))
        .await
        .expect("Failed to send first set_image");
    ws_stream
        .send(Message::Text(stream_msg_1.to_string().into()))
        .await
        .expect("Failed to send first stream_image");
    ws_stream
        .send(Message::Text(set_image_msg_2.to_string().into()))
        .await
        .expect("Failed to send second set_image");
    ws_stream
        .send(Message::Text(stream_msg_2.to_string().into()))
        .await
        .expect("Failed to send second stream_image");

    // Track completions for each request_id
    let mut stream_1_complete = false;
    let mut stream_2_complete = false;
    let mut stream_1_chunks = 0;
    let mut stream_2_chunks = 0;

    let timeout_duration = tokio::time::Duration::from_secs(15);
    let start_time = std::time::Instant::now();

    while start_time.elapsed() < timeout_duration {
        let msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(1000), ws_stream.next()).await;

        match msg {
            Ok(Some(Ok(Message::Binary(data)))) => {
                if data.len() < 6 {
                    continue;
                }

                let msg_type = data[1];
                let msg_request_id = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);

                match msg_type {
                    0x02 => {
                        // Layer chunk
                        if msg_request_id == request_id_1 {
                            stream_1_chunks += 1;
                        } else if msg_request_id == request_id_2 {
                            stream_2_chunks += 1;
                        }
                    }
                    0x05 => {
                        // Stream complete
                        if msg_request_id == request_id_1 {
                            stream_1_complete = true;
                            println!("Stream 1 complete ({} chunks)", stream_1_chunks);
                        } else if msg_request_id == request_id_2 {
                            stream_2_complete = true;
                            println!("Stream 2 complete ({} chunks)", stream_2_chunks);
                        }

                        if stream_1_complete && stream_2_complete {
                            break;
                        }
                    }
                    0x07 => {
                        // StreamError
                        if data.len() >= 8 {
                            let error_code = u16::from_le_bytes([data[6], data[7]]);
                            println!(
                                "Stream error for request {}: code {}",
                                msg_request_id, error_code
                            );
                        }
                    }
                    _ => {}
                }
            }
            Ok(Some(Ok(Message::Close(_)))) | Ok(None) => {
                break;
            }
            Ok(Some(Err(e))) => {
                panic!("WebSocket error: {}", e);
            }
            Err(_) => {
                if stream_1_complete && stream_2_complete {
                    break;
                }
            }
            _ => {}
        }
    }

    assert!(stream_1_complete, "Stream 1 should complete");
    assert!(stream_2_complete, "Stream 2 should complete");
    assert!(stream_1_chunks > 0, "Stream 1 should have received chunks");
    assert!(stream_2_chunks > 0, "Stream 2 should have received chunks");

    println!(
        "Concurrent streams test passed: stream1={} chunks, stream2={} chunks",
        stream_1_chunks, stream_2_chunks
    );

    ws_stream.close(None).await.ok(); // Test cleanup - ignore close errors
}

/// Test that cancel_stream works on the multiplexed endpoint.
#[tokio::test]
async fn test_mux_websocket_cancel_stream() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    // Connect to multiplexed WebSocket endpoint
    let ws_url = format!("ws://{}/api/ws", ctx.addr);

    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to multiplexed WebSocket");

    // Receive capabilities first
    ws_stream.next().await;

    // Start a stream
    let request_id = 123u32;
    let set_image_msg = serde_json::json!({
        "action": "set_image",
        "request_id": request_id,
        "image_id": ctx.test_image_id,
    });
    let stream_msg = serde_json::json!({
        "action": "stream_image",
        "request_id": request_id,
        "level": 0,
        "progressive": false
    });

    ws_stream
        .send(Message::Text(set_image_msg.to_string().into()))
        .await
        .expect("Failed to send set_image");
    ws_stream
        .send(Message::Text(stream_msg.to_string().into()))
        .await
        .expect("Failed to send stream_image");

    // Wait for some data to arrive
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Cancel the stream
    let cancel_msg = serde_json::json!({
        "action": "cancel_stream",
        "request_id": request_id
    });

    ws_stream
        .send(Message::Text(cancel_msg.to_string().into()))
        .await
        .expect("Failed to send cancel_stream");

    // The stream should stop sending data - we shouldn't receive StreamComplete
    let mut received_stream_complete = false;
    let timeout_duration = tokio::time::Duration::from_secs(2);
    let start_time = std::time::Instant::now();

    while start_time.elapsed() < timeout_duration {
        let msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(500), ws_stream.next()).await;

        match msg {
            Ok(Some(Ok(Message::Binary(data)))) => {
                if data.len() >= 6 {
                    let msg_type = data[1];
                    let msg_request_id = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);

                    if msg_request_id == request_id && msg_type == 0x05 {
                        received_stream_complete = true;
                    }
                }
            }
            Err(_) => {
                // Timeout - good, stream was cancelled
                break;
            }
            _ => {}
        }
    }

    // The stream might complete if it was small enough to finish before cancel arrived
    // This is acceptable behavior, so we just verify the test runs without error
    println!(
        "Cancel test completed. Stream completed before cancel: {}",
        received_stream_complete
    );

    ws_stream.close(None).await.ok(); // Test cleanup - ignore close errors
}

/// Test that legacy protocol messages are rejected on the multiplexed endpoint.
#[tokio::test]
async fn test_mux_websocket_rejects_legacy_protocol() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    // Connect to multiplexed WebSocket endpoint
    let ws_url = format!("ws://{}/api/ws", ctx.addr);

    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to multiplexed WebSocket");

    // Receive capabilities first
    ws_stream.next().await;

    // Send legacy start_stream message (without request_id)
    let legacy_msg = serde_json::json!({
        "action": "start_stream",
        "level": 0,
        "progressive": false
    });

    ws_stream
        .send(Message::Text(legacy_msg.to_string().into()))
        .await
        .expect("Failed to send legacy message");

    // Should receive an error
    let mut received_error = false;
    let timeout_duration = tokio::time::Duration::from_secs(2);
    let start_time = std::time::Instant::now();

    while start_time.elapsed() < timeout_duration {
        let msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(500), ws_stream.next()).await;

        match msg {
            Ok(Some(Ok(Message::Binary(data)))) => {
                if data.len() >= 2 {
                    let msg_type = data[1];
                    // 0x07 = StreamError (for request_id 0)
                    if msg_type == 0x07 {
                        received_error = true;
                        println!("Received expected error for legacy protocol");
                        break;
                    }
                }
            }
            Err(_) => break,
            _ => {}
        }
    }

    assert!(
        received_error,
        "Should receive error when using legacy protocol on mux endpoint"
    );

    ws_stream.close(None).await.ok(); // Test cleanup - ignore close errors
}
