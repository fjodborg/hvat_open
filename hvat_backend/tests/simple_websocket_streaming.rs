#![cfg(feature = "legacy-ws")]

//! Integration tests for the HVAT simple backend.

use futures_util::{SinkExt, StreamExt};
use image::{ImageBuffer, Rgb};
use std::io::Cursor;
use std::net::SocketAddr;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio_tungstenite::{connect_async, tungstenite::Message};

use hvat_backend::simple::{AppState, ServerConfig, routes};
use hvat_common::annotation_io::{BundleFile, ExportBundle};
use zip::ZipArchive;

struct TestContext {
    #[expect(dead_code)]
    temp_dir: TempDir,
    addr: SocketAddr,
    test_image_id: String,
}

fn create_test_png() -> Vec<u8> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_fn(10, 10, |_x, _y| Rgb([255u8, 0u8, 0u8]));

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

async fn setup_test_server() -> Option<TestContext> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    let test_image_name = "test_image.png";
    let test_image_path = temp_dir.path().join(test_image_name);
    std::fs::write(&test_image_path, create_test_png()).expect("Failed to write test image");

    let subfolder = temp_dir.path().join("subfolder");
    std::fs::create_dir(&subfolder).expect("Failed to create subfolder");
    std::fs::write(subfolder.join("nested_image.png"), create_test_png())
        .expect("Failed to write nested image");

    let config = ServerConfig {
        port: 0,
        data_dir: temp_dir.path().to_path_buf(),
        cache_dir: temp_dir.path().join(".cache"),
        max_cache_memory: 100 * 1024 * 1024,
        max_user_streams: 4,
        max_connections: 100,
        ping_interval_secs: 30,
        connection_timeout_secs: 90,
        stream_chunk_rows: 128,
        project_name: "test_project".to_string(),
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

    let (listener, addr) = bind_test_listener().await?;

    tokio::spawn(async move {
        if let Err(err) = axum::serve(listener, app).await {
            eprintln!("Test server exited with error: {err}");
        }
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    Some(TestContext {
        temp_dir,
        addr,
        test_image_id: "test_image_png".to_string(),
    })
}

// ============================================================================
// REST API Tests
// ============================================================================

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
    assert!(info.get("name").is_some());
    assert_eq!(info["image_count"].as_u64().unwrap(), 2);
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
    let images_arr = images.as_array().unwrap();
    assert_eq!(images_arr.len(), 2);

    for img in images_arr {
        assert!(img.get("id").is_some());
        assert!(img.get("name").is_some());
        assert!(img.get("path").is_some());
        assert!(img.get("format").is_some());
    }
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

    assert!(resp.status().is_success());
    let content_type = resp
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    assert!(content_type.contains("application/zip"));

    let bytes = resp.bytes().await.expect("Failed to read archive response");
    let mut archive =
        ZipArchive::new(Cursor::new(bytes.to_vec())).expect("Failed to parse zip archive");

    let mut names = Vec::new();
    for i in 0..archive.len() {
        let entry = archive.by_index(i).expect("Failed to read zip entry");
        names.push(entry.name().to_string());
    }
    names.sort();

    assert_eq!(names.len(), 2);
    assert!(names.iter().any(|name| name == "test_image.png"));
    assert!(
        names
            .iter()
            .any(|name| name == "subfolder/nested_image.png")
    );
}

#[tokio::test]
async fn test_project_state_round_trip() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    let client = reqwest::Client::new();
    let url = format!("http://{}/api/project-state", ctx.addr);

    let missing = client.get(&url).send().await.expect("GET failed");
    assert_eq!(missing.status(), reqwest::StatusCode::NOT_FOUND);

    let bundle = ExportBundle {
        files: vec![BundleFile {
            path: "annotations/default.json".to_string(),
            bytes: br#"{"items":[]}"#.to_vec(),
            mime_type: Some("application/json".to_string()),
        }],
    };
    let payload = serde_json::to_vec(&bundle).expect("serialize bundle");

    let save = client
        .put(&url)
        .header("content-type", "application/json")
        .body(payload.clone())
        .send()
        .await
        .expect("PUT failed");
    assert_eq!(save.status(), reqwest::StatusCode::NO_CONTENT);

    let loaded = client.get(&url).send().await.expect("GET failed");
    assert_eq!(loaded.status(), reqwest::StatusCode::OK);
    let loaded_bytes = loaded.bytes().await.expect("read body");
    assert_eq!(loaded_bytes.as_ref(), payload.as_slice());

    let decoded: ExportBundle = serde_json::from_slice(&loaded_bytes).expect("decode bundle");
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
        .expect("PUT failed");

    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
}

// ============================================================================
// WebSocket Tests
// ============================================================================

#[tokio::test]
async fn test_mux_websocket_sends_capabilities_on_connect() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    let ws_url = format!("ws://{}/api/ws", ctx.addr);
    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to WebSocket");

    let msg = tokio::time::timeout(tokio::time::Duration::from_millis(500), ws_stream.next())
        .await
        .expect("Should receive capabilities")
        .expect("Stream should not end")
        .expect("Should be valid message");

    if let tokio_tungstenite::tungstenite::Message::Binary(data) = msg {
        assert!(data.len() >= 6);
        assert_eq!(data[0], 2, "Protocol version should be 2");
        assert_eq!(data[1], 0x06, "Should be capabilities (0x06)");

        let request_id = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);
        assert_eq!(request_id, 0);

        let json_payload = &data[6..];
        let caps: serde_json::Value = serde_json::from_slice(json_payload).unwrap();
        assert_eq!(caps["protocol_version"], 2);
        assert!(caps["limits"]["max_concurrent_streams"].as_u64().unwrap() > 0);
        assert_eq!(caps["download_mode"], "single_zip");
    } else {
        panic!("Expected binary capabilities message");
    }

    ws_stream.close(None).await.ok();
}

#[tokio::test]
async fn test_mux_websocket_streams_image() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    let ws_url = format!("ws://{}/api/ws", ctx.addr);
    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to WebSocket");

    // Receive capabilities
    ws_stream.next().await;

    let request_id = 42u32;

    let set_image_msg = serde_json::json!({
        "action": "set_image",
        "request_id": request_id,
        "image_id": ctx.test_image_id,
    });
    ws_stream
        .send(Message::Text(set_image_msg.to_string().into()))
        .await
        .unwrap();

    let stream_msg = serde_json::json!({
        "action": "stream_image",
        "request_id": request_id,
        "level": 0,
        "progressive": false
    });
    ws_stream
        .send(Message::Text(stream_msg.to_string().into()))
        .await
        .unwrap();

    let mut received_metadata = false;
    let mut received_chunks = 0;
    let mut received_level_complete = false;
    let mut received_stream_complete = false;

    let timeout = tokio::time::Duration::from_secs(10);
    let start = std::time::Instant::now();

    while start.elapsed() < timeout {
        let msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(1000), ws_stream.next()).await;

        match msg {
            Ok(Some(Ok(Message::Binary(data)))) => {
                if data.len() < 6 {
                    continue;
                }

                let version = data[0];
                let msg_type = data[1];
                let msg_request_id = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);

                assert_eq!(version, 2);
                assert_eq!(msg_request_id, request_id);

                match msg_type {
                    0x00 => {} // Reset
                    0x01 => received_metadata = true,
                    0x02 => received_chunks += 1,
                    0x03 => {} // LayerComplete
                    0x04 => received_level_complete = true,
                    0x05 => {
                        received_stream_complete = true;
                        break;
                    }
                    0x07 => {
                        if data.len() >= 8 {
                            let error_code = u16::from_le_bytes([data[6], data[7]]);
                            panic!("Stream error: code {}", error_code);
                        }
                    }
                    _ => {}
                }
            }
            Ok(Some(Ok(Message::Close(_)))) | Ok(None) => break,
            Ok(Some(Err(e))) => panic!("WebSocket error: {}", e),
            Err(_) => {
                if received_stream_complete {
                    break;
                }
            }
            _ => {}
        }
    }

    assert!(received_metadata, "Should have received metadata");
    assert!(received_chunks > 0, "Should have received chunks");
    assert!(
        received_level_complete,
        "Should have received level complete"
    );
    assert!(
        received_stream_complete,
        "Should have received stream complete"
    );

    ws_stream.close(None).await.ok();
}

#[tokio::test]
async fn test_mux_websocket_concurrent_streams() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    let ws_url = format!("ws://{}/api/ws", ctx.addr);
    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to WebSocket");

    // Receive capabilities
    ws_stream.next().await;

    let request_id_1 = 100u32;
    let request_id_2 = 200u32;

    // Stream 1: test_image.png
    ws_stream
        .send(Message::Text(
            serde_json::json!({
                "action": "set_image",
                "request_id": request_id_1,
                "image_id": ctx.test_image_id,
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
    ws_stream
        .send(Message::Text(
            serde_json::json!({
                "action": "stream_image",
                "request_id": request_id_1,
                "level": 0,
                "progressive": false
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    // Stream 2: nested image
    ws_stream
        .send(Message::Text(
            serde_json::json!({
                "action": "set_image",
                "request_id": request_id_2,
                "image_id": "subfolder_nested_image_png",
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
    ws_stream
        .send(Message::Text(
            serde_json::json!({
                "action": "stream_image",
                "request_id": request_id_2,
                "level": 0,
                "progressive": false
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let mut stream_1_complete = false;
    let mut stream_2_complete = false;
    let mut stream_1_chunks = 0;
    let mut stream_2_chunks = 0;

    let timeout = tokio::time::Duration::from_secs(15);
    let start = std::time::Instant::now();

    while start.elapsed() < timeout {
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
                        if msg_request_id == request_id_1 {
                            stream_1_chunks += 1;
                        } else if msg_request_id == request_id_2 {
                            stream_2_chunks += 1;
                        }
                    }
                    0x05 => {
                        if msg_request_id == request_id_1 {
                            stream_1_complete = true;
                        } else if msg_request_id == request_id_2 {
                            stream_2_complete = true;
                        }
                        if stream_1_complete && stream_2_complete {
                            break;
                        }
                    }
                    _ => {}
                }
            }
            Ok(Some(Ok(Message::Close(_)))) | Ok(None) => break,
            Ok(Some(Err(e))) => panic!("WebSocket error: {}", e),
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
    assert!(stream_1_chunks > 0, "Stream 1 should have chunks");
    assert!(stream_2_chunks > 0, "Stream 2 should have chunks");

    ws_stream.close(None).await.ok();
}

#[tokio::test]
async fn test_mux_websocket_cancel_stream() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    let ws_url = format!("ws://{}/api/ws", ctx.addr);
    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to WebSocket");

    // Receive capabilities
    ws_stream.next().await;

    let request_id = 123u32;

    ws_stream
        .send(Message::Text(
            serde_json::json!({
                "action": "set_image",
                "request_id": request_id,
                "image_id": ctx.test_image_id,
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();
    ws_stream
        .send(Message::Text(
            serde_json::json!({
                "action": "stream_image",
                "request_id": request_id,
                "level": 0,
                "progressive": false
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    ws_stream
        .send(Message::Text(
            serde_json::json!({
                "action": "cancel_stream",
                "request_id": request_id
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    // Drain remaining messages — small images may complete before cancel arrives
    let timeout = tokio::time::Duration::from_secs(2);
    let start = std::time::Instant::now();
    while start.elapsed() < timeout {
        let msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(500), ws_stream.next()).await;
        match msg {
            Ok(Some(Ok(Message::Binary(_)))) => {}
            Err(_) => break,
            _ => break,
        }
    }

    ws_stream.close(None).await.ok();
}

#[tokio::test]
async fn test_mux_websocket_rejects_legacy_protocol() {
    let Some(ctx) = setup_test_server().await else {
        return;
    };

    let ws_url = format!("ws://{}/api/ws", ctx.addr);
    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("Failed to connect to WebSocket");

    // Receive capabilities
    ws_stream.next().await;

    // Send legacy message without request_id
    ws_stream
        .send(Message::Text(
            serde_json::json!({
                "action": "start_stream",
                "level": 0,
                "progressive": false
            })
            .to_string()
            .into(),
        ))
        .await
        .unwrap();

    let mut received_error = false;
    let timeout = tokio::time::Duration::from_secs(2);
    let start = std::time::Instant::now();

    while start.elapsed() < timeout {
        let msg =
            tokio::time::timeout(tokio::time::Duration::from_millis(500), ws_stream.next()).await;

        match msg {
            Ok(Some(Ok(Message::Binary(data)))) => {
                if data.len() >= 2 && data[1] == 0x07 {
                    received_error = true;
                    break;
                }
            }
            Err(_) => break,
            _ => {}
        }
    }

    assert!(received_error, "Should receive error for legacy protocol");

    ws_stream.close(None).await.ok();
}
