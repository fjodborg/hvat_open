//! Capability identity tests for SAM2 vs SAM3 backend route stacks.

use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use futures_util::StreamExt;
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio_tungstenite::connect_async;

use hvat_backend::full_sam::routes as sam2_routes;
use hvat_backend::full_sam::sam::{
    EncoderOutput, ExecutionProvider, SamBackend, SamMaskResult, SamPoint,
};
use hvat_backend::full_sam::state::{AppState, SamBackendWiring};
use hvat_backend::full_sam3::ServerConfig;
use hvat_backend::full_sam3::routes as sam3_routes;
use hvat_common::{PROTOCOL_VERSION, ServerCapabilities, ServerMessageType};

struct MockSamBackend;

#[async_trait]
impl SamBackend for MockSamBackend {
    fn name(&self) -> &'static str {
        "mock-sam"
    }

    fn provider(&self) -> ExecutionProvider {
        ExecutionProvider::Cpu
    }

    async fn encode_image(
        &self,
        _image_rgb: &[u8],
        _width: u32,
        _height: u32,
    ) -> anyhow::Result<EncoderOutput> {
        anyhow::bail!("unused in capability tests")
    }

    async fn decode_mask(
        &self,
        _encoder_output: &EncoderOutput,
        _original_width: u32,
        _original_height: u32,
        _points: &[SamPoint],
        _box_prompt: Option<[f32; 4]>,
    ) -> anyhow::Result<SamMaskResult> {
        anyhow::bail!("unused in capability tests")
    }

    fn is_available(&self) -> bool {
        true
    }
}

fn mock_model_name(_: &ServerConfig) -> String {
    "Mock SAM".to_string()
}

fn mock_expected_files(_: &ServerConfig) -> Vec<String> {
    vec!["encoder.onnx".to_string(), "decoder.onnx".to_string()]
}

fn mock_factory(_: &ServerConfig) -> anyhow::Result<Arc<dyn SamBackend>> {
    Ok(Arc::new(MockSamBackend))
}

fn make_config(data_dir: &std::path::Path, cache_dir: &std::path::Path) -> ServerConfig {
    let mut config = ServerConfig::default();
    config.base.port = 0;
    config.base.data_dir = data_dir.to_path_buf();
    config.base.cache_dir = cache_dir.to_path_buf();
    config.sam_enabled = true;
    config
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

async fn read_caps_from_ws(addr: SocketAddr) -> ServerCapabilities {
    let ws_url = format!("ws://{}/api/ws", addr);
    let (mut ws_stream, _) = connect_async(&ws_url)
        .await
        .expect("websocket connect should succeed");

    let msg = ws_stream
        .next()
        .await
        .expect("capabilities frame should be sent")
        .expect("frame should be valid");

    let data = match msg {
        tokio_tungstenite::tungstenite::Message::Binary(data) => data,
        other => panic!("expected binary capabilities frame, got {other:?}"),
    };

    assert!(data.len() >= 6);
    assert_eq!(data[0], PROTOCOL_VERSION);
    assert_eq!(data[1], ServerMessageType::Capabilities.to_byte());

    let request_id = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);
    assert_eq!(request_id, 0);

    serde_json::from_slice(&data[6..]).expect("capabilities payload should decode")
}

#[tokio::test]
async fn sam2_and_sam3_routes_advertise_distinct_model_ids() {
    let data_dir = TempDir::new().expect("temp data dir");
    let cache_dir = TempDir::new().expect("temp cache dir");

    let sam2_state = Arc::new(
        AppState::new_with_sam_wiring(
            make_config(data_dir.path(), cache_dir.path()),
            SamBackendWiring {
                backend_name: "sam2-mock",
                model_id: "sam-base",
                model_name: mock_model_name,
                expected_files: mock_expected_files,
                create_backend: mock_factory,
            },
        )
        .expect("sam2 state should init"),
    );
    let sam3_state = Arc::new(
        AppState::new_with_sam_wiring(
            make_config(data_dir.path(), cache_dir.path()),
            SamBackendWiring {
                backend_name: "sam3-mock",
                model_id: "sam3-compat-base",
                model_name: mock_model_name,
                expected_files: mock_expected_files,
                create_backend: mock_factory,
            },
        )
        .expect("sam3 state should init"),
    );

    let sam2_app = axum::Router::new()
        .nest("/api", sam2_routes::api_router())
        .with_state(sam2_state);
    let Some((sam2_listener, sam2_addr)) = bind_test_listener().await else {
        return;
    };
    tokio::spawn(async move {
        let _ = axum::serve(sam2_listener, sam2_app).await;
    });

    let sam3_app = axum::Router::new()
        .nest("/api", sam3_routes::api_router())
        .with_state(sam3_state);
    let Some((sam3_listener, sam3_addr)) = bind_test_listener().await else {
        return;
    };
    tokio::spawn(async move {
        let _ = axum::serve(sam3_listener, sam3_app).await;
    });

    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    let sam2_caps = read_caps_from_ws(sam2_addr).await;
    let sam3_caps = read_caps_from_ws(sam3_addr).await;

    assert_eq!(sam2_caps.models.len(), 1);
    assert_eq!(sam3_caps.models.len(), 1);

    assert_eq!(sam2_caps.models[0].id, "sam-base");
    assert_eq!(sam3_caps.models[0].id, "sam3-compat-base");
    assert_ne!(sam2_caps.models[0].id, sam3_caps.models[0].id);

    assert!(sam2_caps.models[0].is_sam_compatible());
    assert!(sam3_caps.models[0].is_sam_compatible());
}
