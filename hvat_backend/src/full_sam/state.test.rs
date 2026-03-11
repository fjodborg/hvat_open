use super::*;
use async_trait::async_trait;

use crate::sam::{EncoderOutput, ExecutionProvider, SamMaskResult, SamPoint};

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
        anyhow::bail!("unused in test")
    }

    async fn decode_mask(
        &self,
        _encoder_output: &EncoderOutput,
        _original_width: u32,
        _original_height: u32,
        _points: &[SamPoint],
        _box_prompt: Option<[f32; 4]>,
    ) -> anyhow::Result<SamMaskResult> {
        anyhow::bail!("unused in test")
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

fn failing_factory(_: &ServerConfig) -> anyhow::Result<Arc<dyn SamBackend>> {
    anyhow::bail!("boom")
}

#[test]
fn custom_wiring_registers_custom_model_id() {
    let config = ServerConfig {
        sam_enabled: true,
        ..ServerConfig::default()
    };

    let state = AppState::new_with_sam_wiring(
        config,
        SamBackendWiring {
            backend_name: "mock-backend",
            model_id: "sam-custom",
            model_name: mock_model_name,
            expected_files: mock_expected_files,
            create_backend: mock_factory,
        },
    )
    .expect("state init should succeed");

    let registry = state.model_registry.expect("model registry should exist");
    assert!(registry.get("sam-custom").is_some());
}

#[test]
fn custom_wiring_surfaces_backend_context_on_init_failure() {
    let config = ServerConfig {
        sam_enabled: true,
        ..ServerConfig::default()
    };

    let result = AppState::new_with_sam_wiring(
        config,
        SamBackendWiring {
            backend_name: "failing-backend",
            model_id: "sam-custom",
            model_name: mock_model_name,
            expected_files: mock_expected_files,
            create_backend: failing_factory,
        },
    );
    let err = result.err().expect("state init should fail");

    match err {
        StateInitError::SamInit {
            backend_name,
            expected_files,
            ..
        } => {
            assert_eq!(backend_name, "failing-backend");
            assert!(expected_files.contains("encoder.onnx"));
            assert!(expected_files.contains("decoder.onnx"));
        }
    }
}

#[test]
fn different_wiring_ids_produce_distinct_capability_ids() {
    let config = ServerConfig {
        sam_enabled: true,
        ..ServerConfig::default()
    };

    let sam2_like = AppState::new_with_sam_wiring(
        config.clone(),
        SamBackendWiring {
            backend_name: "sam2-like",
            model_id: "sam-base",
            model_name: mock_model_name,
            expected_files: mock_expected_files,
            create_backend: mock_factory,
        },
    )
    .expect("sam2-like state should init");
    let sam3_like = AppState::new_with_sam_wiring(
        config,
        SamBackendWiring {
            backend_name: "sam3-like",
            model_id: "sam3-compat-base",
            model_name: mock_model_name,
            expected_files: mock_expected_files,
            create_backend: mock_factory,
        },
    )
    .expect("sam3-like state should init");

    let sam2_caps = sam2_like
        .model_registry
        .as_ref()
        .expect("registry should exist")
        .capabilities();
    let sam3_caps = sam3_like
        .model_registry
        .as_ref()
        .expect("registry should exist")
        .capabilities();

    assert_eq!(sam2_caps.len(), 1);
    assert_eq!(sam3_caps.len(), 1);
    assert_eq!(sam2_caps[0].id, "sam-base");
    assert_eq!(sam3_caps[0].id, "sam3-compat-base");
    assert_ne!(sam2_caps[0].id, sam3_caps[0].id);
}
