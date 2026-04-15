//! SAM3 wiring integration test for prepare -> infer over mux websocket.

use std::io::Cursor;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use image::{ImageBuffer, Rgb};
use tempfile::TempDir;
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};

use hvat_backend::ServerConfig;
use hvat_backend::common::catalog::encode_image_id_from_relative;
use hvat_backend::full_sam::sam::{
    EncoderOutput, ExecutionProvider, SamBackend, SamMaskResult, SamPoint,
};
use hvat_backend::full_sam::state::{AppState, SamBackendWiring};
use hvat_backend::full_sam3::routes as sam3_routes;
use hvat_backend::full_sam3::state::SAM3_COMPAT_MODEL_ID;
use hvat_common::{
    ErrorCode, PROTOCOL_VERSION, ProtocolError, ServerCapabilities, ServerMessageType,
};

type WsClient = WebSocketStream<MaybeTlsStream<TcpStream>>;

type SamBackendFactory = fn(&ServerConfig) -> anyhow::Result<Arc<dyn SamBackend>>;

#[derive(Clone, Copy, PartialEq, Eq)]
enum MockBehavior {
    Success,
    EncodeFails,
    DecodeFails,
}

struct MockSamBackend {
    behavior: MockBehavior,
}

impl MockSamBackend {
    const fn success() -> Self {
        Self {
            behavior: MockBehavior::Success,
        }
    }

    const fn encode_fails() -> Self {
        Self {
            behavior: MockBehavior::EncodeFails,
        }
    }

    const fn decode_fails() -> Self {
        Self {
            behavior: MockBehavior::DecodeFails,
        }
    }
}

#[async_trait]
impl SamBackend for MockSamBackend {
    fn name(&self) -> &'static str {
        "sam3-mock-backend"
    }

    fn provider(&self) -> ExecutionProvider {
        ExecutionProvider::Cpu
    }

    async fn encode_image(
        &self,
        image_rgb: &[u8],
        width: u32,
        height: u32,
    ) -> anyhow::Result<EncoderOutput> {
        if self.behavior == MockBehavior::EncodeFails {
            anyhow::bail!("mock encode failure");
        }

        let expected_len = width as usize * height as usize * 3;
        anyhow::ensure!(
            image_rgb.len() == expected_len,
            "expected RGB len {}, got {}",
            expected_len,
            image_rgb.len()
        );

        Ok(EncoderOutput::sam2_onnx(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.1, 0.2],
            vec![0.3, 0.4],
        ))
    }

    async fn decode_mask(
        &self,
        _encoder_output: &EncoderOutput,
        original_width: u32,
        original_height: u32,
        points: &[SamPoint],
        box_prompt: Option<[f32; 4]>,
    ) -> anyhow::Result<SamMaskResult> {
        if self.behavior == MockBehavior::DecodeFails {
            anyhow::bail!("mock decode failure");
        }

        anyhow::ensure!(!points.is_empty(), "expected at least one point prompt");
        anyhow::ensure!(box_prompt.is_some(), "expected box prompt");

        let point = points[0];
        let x0 = point.x.clamp(0.0, original_width.saturating_sub(1) as f32);
        let y0 = point.y.clamp(0.0, original_height.saturating_sub(1) as f32);
        let x1 = (x0 + 4.0).min(original_width as f32);
        let y1 = (y0 + 4.0).min(original_height as f32);
        let polygon = vec![[x0, y0], [x1, y0], [x1, y1], [x0, y1]];

        Ok(SamMaskResult {
            masks: Vec::new(),
            polygons: vec![polygon],
            iou_scores: vec![0.91],
            width: original_width,
            height: original_height,
        })
    }

    fn is_available(&self) -> bool {
        true
    }
}

fn mock_model_name(_: &ServerConfig) -> String {
    "SAM3 Mock".to_string()
}

fn mock_expected_files(_: &ServerConfig) -> Vec<String> {
    vec![
        "sam3_mock_encoder.bin".to_string(),
        "sam3_mock_decoder.bin".to_string(),
    ]
}

fn success_factory(_: &ServerConfig) -> anyhow::Result<Arc<dyn SamBackend>> {
    Ok(Arc::new(MockSamBackend::success()))
}

fn encode_fail_factory(_: &ServerConfig) -> anyhow::Result<Arc<dyn SamBackend>> {
    Ok(Arc::new(MockSamBackend::encode_fails()))
}

fn decode_fail_factory(_: &ServerConfig) -> anyhow::Result<Arc<dyn SamBackend>> {
    Ok(Arc::new(MockSamBackend::decode_fails()))
}

fn make_config(data_dir: &Path, cache_dir: &Path) -> ServerConfig {
    let mut config = ServerConfig::default();
    config.base.port = 0;
    config.base.data_dir = data_dir.to_path_buf();
    config.base.cache_dir = cache_dir.to_path_buf();
    config.sam_enabled = true;
    config
}

fn create_test_png() -> Vec<u8> {
    let img: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_fn(12, 10, |_x, _y| Rgb([64u8, 128u8, 255u8]));

    let mut bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
        .expect("failed to encode PNG");
    bytes
}

async fn bind_test_listener() -> Option<(TcpListener, SocketAddr)> {
    match TcpListener::bind("127.0.0.1:0").await {
        Ok(listener) => match listener.local_addr() {
            Ok(addr) => Some((listener, addr)),
            Err(e) => panic!("failed to get local listener addr: {e}"),
        },
        Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
            eprintln!("Skipping test: unable to bind local TCP listener in this environment ({e})");
            None
        }
        Err(e) => panic!("failed to bind local TCP listener: {e}"),
    }
}

fn parse_frame(data: &[u8]) -> (u8, ServerMessageType, u32, &[u8]) {
    assert!(data.len() >= 6, "frame must include 6-byte header");
    let version = data[0];
    let msg_type = ServerMessageType::from_byte(data[1])
        .unwrap_or_else(|| panic!("unknown message type: 0x{:02x}", data[1]));
    let request_id = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);
    (version, msg_type, request_id, &data[6..])
}

async fn next_binary_frame(ws_stream: &mut WsClient) -> Vec<u8> {
    loop {
        let msg = tokio::time::timeout(Duration::from_secs(5), ws_stream.next())
            .await
            .expect("timed out waiting for websocket frame")
            .expect("websocket stream ended unexpectedly")
            .expect("received invalid websocket frame");

        match msg {
            Message::Binary(data) => return data.to_vec(),
            Message::Close(frame) => panic!("websocket closed unexpectedly: {frame:?}"),
            _ => {}
        }
    }
}

async fn wait_for_message(
    ws_stream: &mut WsClient,
    expected_request_id: u32,
    expected_type: ServerMessageType,
) -> Vec<u8> {
    loop {
        let data = next_binary_frame(ws_stream).await;
        let (version, msg_type, request_id, payload) = parse_frame(&data);
        assert_eq!(version, PROTOCOL_VERSION);

        if msg_type == ServerMessageType::StreamError {
            let error = ProtocolError::decode(payload)
                .unwrap_or_else(|| panic!("failed to decode stream error payload: {payload:?}"));
            panic!(
                "unexpected stream error for request {}: {:?} {}",
                request_id, error.code, error.message
            );
        }

        if request_id == expected_request_id && msg_type == expected_type {
            return payload.to_vec();
        }
    }
}

async fn wait_for_stream_error(
    ws_stream: &mut WsClient,
    expected_request_id: u32,
) -> ProtocolError {
    loop {
        let data = next_binary_frame(ws_stream).await;
        let (version, msg_type, request_id, payload) = parse_frame(&data);
        assert_eq!(version, PROTOCOL_VERSION);

        if request_id == expected_request_id && msg_type == ServerMessageType::StreamError {
            return ProtocolError::decode(payload)
                .unwrap_or_else(|| panic!("failed to decode stream error payload: {payload:?}"));
        }
    }
}

fn decode_len_prefixed_utf8(payload: &[u8]) -> String {
    assert!(payload.len() >= 2, "payload must contain length prefix");
    let len = u16::from_le_bytes([payload[0], payload[1]]) as usize;
    assert!(
        payload.len() >= 2 + len,
        "payload length {} is too short for {} bytes",
        payload.len(),
        len
    );
    String::from_utf8(payload[2..2 + len].to_vec()).expect("payload string must be valid UTF-8")
}

struct TestSession {
    _data_dir: TempDir,
    _cache_dir: TempDir,
    image_id: String,
    ws_stream: WsClient,
}

async fn setup_test_session(factory: SamBackendFactory) -> Option<TestSession> {
    let data_dir = TempDir::new().expect("temp data dir");
    let cache_dir = TempDir::new().expect("temp cache dir");
    let relative_image_path = "sam3_prepare_infer.png";
    let image_id = encode_image_id_from_relative(relative_image_path);
    std::fs::write(data_dir.path().join(relative_image_path), create_test_png())
        .expect("failed to write test image");

    let state = Arc::new(
        AppState::new_with_sam_wiring(
            make_config(data_dir.path(), cache_dir.path()),
            SamBackendWiring {
                backend_name: "sam3-mock",
                model_id: SAM3_COMPAT_MODEL_ID,
                model_name: mock_model_name,
                expected_files: mock_expected_files,
                create_backend: factory,
            },
        )
        .expect("sam3 test state should init"),
    );

    let app = axum::Router::new()
        .nest("/api", sam3_routes::api_router())
        .with_state(state);

    let Some((listener, addr)) = bind_test_listener().await else {
        return None;
    };
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    tokio::time::sleep(Duration::from_millis(100)).await;

    let ws_url = format!("ws://{}/api/ws", addr);
    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("websocket connect should succeed");

    let caps_payload = wait_for_message(&mut ws_stream, 0, ServerMessageType::Capabilities).await;
    let caps: ServerCapabilities =
        serde_json::from_slice(&caps_payload).expect("capabilities payload should decode");
    assert!(caps.features.inference);
    assert!(caps.features.sam);
    let model = caps
        .models
        .iter()
        .find(|m| m.id == SAM3_COMPAT_MODEL_ID)
        .expect("SAM3 compat model should be advertised");
    assert!(model.supports_point_to_mask());
    assert!(model.supports_bbox_to_mask());
    assert!(model.supports_point_bbox_to_mask());

    Some(TestSession {
        _data_dir: data_dir,
        _cache_dir: cache_dir,
        image_id,
        ws_stream,
    })
}

async fn set_active_image(ws_stream: &mut WsClient, image_id: &str, request_id: u32) {
    let set_image_msg = serde_json::json!({
        "action": "set_image",
        "request_id": request_id,
        "image_id": image_id,
    });
    ws_stream
        .send(Message::Text(set_image_msg.to_string().into()))
        .await
        .expect("set_image send should succeed");

    let image_set_payload =
        wait_for_message(ws_stream, request_id, ServerMessageType::ImageSet).await;
    assert_eq!(decode_len_prefixed_utf8(&image_set_payload), image_id);
}

#[tokio::test]
async fn sam3_mux_prepare_then_infer_succeeds_with_sam3_model_id() {
    let Some(mut session) = setup_test_session(success_factory).await else {
        return;
    };

    set_active_image(&mut session.ws_stream, &session.image_id, 101).await;

    let prepare_request_id = 102;
    let prepare_msg = serde_json::json!({
        "action": "prepare_model",
        "request_id": prepare_request_id,
        "model_id": SAM3_COMPAT_MODEL_ID,
    });
    session
        .ws_stream
        .send(Message::Text(prepare_msg.to_string().into()))
        .await
        .expect("prepare_model send should succeed");

    let model_ready_payload = wait_for_message(
        &mut session.ws_stream,
        prepare_request_id,
        ServerMessageType::ModelReady,
    )
    .await;
    assert_eq!(
        decode_len_prefixed_utf8(&model_ready_payload),
        SAM3_COMPAT_MODEL_ID
    );

    let infer_request_id = 103;
    let infer_msg = serde_json::json!({
        "action": "infer",
        "request_id": infer_request_id,
        "model_id": SAM3_COMPAT_MODEL_ID,
        "inputs": {
            "points": [{ "x": 3.0, "y": 4.0, "label": 1 }],
            "box": [1.0, 1.0, 8.0, 9.0]
        },
        "options": {}
    });
    session
        .ws_stream
        .send(Message::Text(infer_msg.to_string().into()))
        .await
        .expect("infer send should succeed");

    let infer_payload = wait_for_message(
        &mut session.ws_stream,
        infer_request_id,
        ServerMessageType::InferResult,
    )
    .await;
    let infer_json: serde_json::Value =
        serde_json::from_slice(&infer_payload).expect("infer payload should decode");

    assert_eq!(infer_json["model_id"], SAM3_COMPAT_MODEL_ID);
    assert_eq!(
        infer_json["outputs"]["masks"],
        serde_json::json!([[3.0, 4.0, 7.0, 4.0, 7.0, 8.0, 3.0, 8.0]])
    );
    let score = infer_json["outputs"]["scores"][0]
        .as_f64()
        .expect("score should be numeric");
    assert!((score - 0.91).abs() < 0.001);
    assert!(
        infer_json["timing_ms"].as_u64().is_some(),
        "timing_ms should be present"
    );
}

#[tokio::test]
async fn sam3_mux_prepare_maps_encode_failure_to_model_encode_failed() {
    let Some(mut session) = setup_test_session(encode_fail_factory).await else {
        return;
    };

    set_active_image(&mut session.ws_stream, &session.image_id, 201).await;

    let prepare_request_id = 202;
    let prepare_msg = serde_json::json!({
        "action": "prepare_model",
        "request_id": prepare_request_id,
        "model_id": SAM3_COMPAT_MODEL_ID,
    });
    session
        .ws_stream
        .send(Message::Text(prepare_msg.to_string().into()))
        .await
        .expect("prepare_model send should succeed");

    let error = wait_for_stream_error(&mut session.ws_stream, prepare_request_id).await;
    assert_eq!(error.code, ErrorCode::ModelEncodeFailed);
    assert!(error.message.contains("Failed to prepare model"));
}

#[tokio::test]
async fn sam3_mux_infer_maps_decode_failure_to_model_decode_failed() {
    let Some(mut session) = setup_test_session(decode_fail_factory).await else {
        return;
    };

    set_active_image(&mut session.ws_stream, &session.image_id, 301).await;

    let prepare_request_id = 302;
    let prepare_msg = serde_json::json!({
        "action": "prepare_model",
        "request_id": prepare_request_id,
        "model_id": SAM3_COMPAT_MODEL_ID,
    });
    session
        .ws_stream
        .send(Message::Text(prepare_msg.to_string().into()))
        .await
        .expect("prepare_model send should succeed");
    let _ = wait_for_message(
        &mut session.ws_stream,
        prepare_request_id,
        ServerMessageType::ModelReady,
    )
    .await;

    let infer_request_id = 303;
    let infer_msg = serde_json::json!({
        "action": "infer",
        "request_id": infer_request_id,
        "model_id": SAM3_COMPAT_MODEL_ID,
        "inputs": {
            "points": [{ "x": 3.0, "y": 4.0, "label": 1 }],
            "box": [1.0, 1.0, 8.0, 9.0]
        },
        "options": {}
    });
    session
        .ws_stream
        .send(Message::Text(infer_msg.to_string().into()))
        .await
        .expect("infer send should succeed");

    let error = wait_for_stream_error(&mut session.ws_stream, infer_request_id).await;
    assert_eq!(error.code, ErrorCode::ModelDecodeFailed);
    assert!(error.message.contains("Inference failed"));
}

#[tokio::test]
async fn sam3_mux_infer_invalid_inputs_returns_invalid_input_error() {
    let Some(mut session) = setup_test_session(success_factory).await else {
        return;
    };

    set_active_image(&mut session.ws_stream, &session.image_id, 401).await;

    let infer_request_id = 402;
    let infer_msg = serde_json::json!({
        "action": "infer",
        "request_id": infer_request_id,
        "model_id": SAM3_COMPAT_MODEL_ID,
        "inputs": {},
        "options": {}
    });
    session
        .ws_stream
        .send(Message::Text(infer_msg.to_string().into()))
        .await
        .expect("infer send should succeed");

    let error = wait_for_stream_error(&mut session.ws_stream, infer_request_id).await;
    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert!(error.message.contains("Invalid inputs"));
}
