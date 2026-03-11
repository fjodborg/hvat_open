use super::*;

#[test]
fn test_message_type_roundtrip() {
    assert_eq!(
        ServerMessageType::from_byte(ServerMessageType::Reset.to_byte()),
        Some(ServerMessageType::Reset)
    );
    assert_eq!(
        ServerMessageType::from_byte(ServerMessageType::Error.to_byte()),
        Some(ServerMessageType::Error)
    );
    assert_eq!(
        ServerMessageType::from_byte(ServerMessageType::StreamError.to_byte()),
        Some(ServerMessageType::StreamError)
    );
    assert_eq!(
        ServerMessageType::from_byte(ServerMessageType::ImageSet.to_byte()),
        Some(ServerMessageType::ImageSet)
    );
    assert_eq!(
        ServerMessageType::from_byte(ServerMessageType::ModelReady.to_byte()),
        Some(ServerMessageType::ModelReady)
    );
    assert_eq!(
        ServerMessageType::from_byte(ServerMessageType::InferProgress.to_byte()),
        Some(ServerMessageType::InferProgress)
    );
    assert_eq!(
        ServerMessageType::from_byte(ServerMessageType::InferResult.to_byte()),
        Some(ServerMessageType::InferResult)
    );
}

#[test]
fn test_connection_level_messages() {
    assert!(ServerMessageType::Capabilities.is_connection_level());
    assert!(ServerMessageType::Ping.is_connection_level());
    assert!(ServerMessageType::Error.is_connection_level());
    assert!(!ServerMessageType::Reset.is_connection_level());
    assert!(!ServerMessageType::StreamError.is_connection_level());
}

#[test]
fn test_error_code_category() {
    assert_eq!(ErrorCode::ImageNotFound.category(), ErrorCategory::Image);
    assert_eq!(
        ErrorCode::PyramidNotReady.category(),
        ErrorCategory::Pyramid
    );
    assert_eq!(ErrorCode::ModelNotFound.category(), ErrorCategory::Model);
    assert_eq!(ErrorCode::NoActiveImage.category(), ErrorCategory::Image);
    assert_eq!(
        ErrorCode::EmbeddingRequired.category(),
        ErrorCategory::Model
    );
}

#[test]
fn test_error_code_retryable() {
    assert!(ErrorCode::PyramidNotReady.default_retryable());
    assert!(ErrorCode::ModelBusy.default_retryable());
    assert!(!ErrorCode::ImageNotFound.default_retryable());
    assert!(!ErrorCode::ModelNotFound.default_retryable());
}

#[test]
fn test_severity_ordering() {
    assert!(Severity::Info < Severity::Warning);
    assert!(Severity::Warning < Severity::Error);
    assert!(Severity::Error < Severity::Fatal);
}

#[test]
fn test_capabilities_default() {
    let caps = ServerCapabilities::default();
    assert_eq!(caps.protocol_version, PROTOCOL_VERSION);
    assert!(caps.models.is_empty());
    assert_eq!(caps.server.name, "hvat-axum");
    assert_eq!(caps.features, ServerFeatures::default());
}

#[test]
fn test_capabilities_find_model() {
    let mut caps = ServerCapabilities::default();
    caps.models.push(ModelCapability {
        id: "sam-base".to_string(),
        name: "Segment Anything".to_string(),
        model_type: ModelType::Segmentation,
        description: "SAM model".to_string(),
        inputs: vec![],
        outputs: vec![],
        options: HashMap::new(),
        requires_embedding: true,
        embedding_time_ms: 1000,
    });

    assert!(caps.has_model("sam-base"));
    assert!(!caps.has_model("yolo-v8"));
    assert!(caps.find_model_by_type(ModelType::Segmentation).is_some());
    assert!(caps.find_model_by_type(ModelType::Detection).is_none());
}

#[test]
fn test_capabilities_json_roundtrip() {
    let caps = ServerCapabilities {
        protocol_version: 2,
        server: ServerInfo {
            name: "test-server".to_string(),
            version: "1.0.0".to_string(),
        },
        limits: ServerLimits {
            max_image_size: 1024 * 1024,
            max_pyramid_levels: 4,
            max_concurrent_streams: 2,
            max_concurrent_inferences: 1,
        },
        features: ServerFeatures {
            streaming: true,
            project_state: true,
            downloads: true,
            thumbnails: true,
            inference: true,
            sam: false,
            progressive_streaming: true,
        },
        download_mode: DownloadMode::Chunked,
        models: vec![ModelCapability {
            id: "test-model".to_string(),
            name: "Test Model".to_string(),
            model_type: ModelType::Detection,
            description: "A test model".to_string(),
            inputs: vec![InputSchema {
                name: "image".to_string(),
                input_type: "image".to_string(),
                required: true,
                description: "Input image".to_string(),
            }],
            outputs: vec![OutputSchema {
                name: "boxes".to_string(),
                output_type: "bbox_list".to_string(),
                description: "Detected boxes".to_string(),
            }],
            options: HashMap::new(),
            requires_embedding: false,
            embedding_time_ms: 0,
        }],
    };

    let json = serde_json::to_string(&caps).unwrap();
    let decoded: ServerCapabilities = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, caps);
}

#[test]
fn test_capabilities_deserialize_without_features_defaults() {
    let json = serde_json::json!({
        "protocol_version": PROTOCOL_VERSION,
        "server": {
            "name": "legacy",
            "version": "0.0.1"
        },
        "limits": {
            "max_image_size": 1024,
            "max_pyramid_levels": 4,
            "max_concurrent_streams": 2,
            "max_concurrent_inferences": 1
        },
        "models": []
    });

    let decoded: ServerCapabilities = serde_json::from_value(json).unwrap();
    assert_eq!(decoded.features, ServerFeatures::default());
    assert_eq!(decoded.download_mode, DownloadMode::Unknown);
    assert!(!decoded.supports_inference());
    assert!(!decoded.supports_sam());
}

#[test]
fn test_capabilities_deserialize_without_download_mode_defaults_unknown() {
    let json = serde_json::json!({
        "protocol_version": PROTOCOL_VERSION,
        "server": {
            "name": "legacy",
            "version": "0.0.1"
        },
        "limits": {
            "max_image_size": 1024,
            "max_pyramid_levels": 4,
            "max_concurrent_streams": 2,
            "max_concurrent_inferences": 1
        },
        "features": {
            "streaming": true,
            "project_state": true,
            "downloads": true,
            "thumbnails": true,
            "inference": false,
            "sam": false,
            "progressive_streaming": false
        },
        "models": []
    });

    let decoded: ServerCapabilities = serde_json::from_value(json).unwrap();
    assert_eq!(decoded.download_mode, DownloadMode::Unknown);
    assert!(!decoded.supports_chunked_downloads());
}

#[test]
fn test_supports_thumbnails_prefers_features_and_legacy_downloads() {
    let mut caps = ServerCapabilities::default();
    assert!(!caps.supports_thumbnails());

    caps.features.downloads = true;
    assert!(caps.supports_thumbnails());

    caps.features.downloads = false;
    caps.download_mode = DownloadMode::Chunked;
    assert!(caps.supports_thumbnails());

    caps.download_mode = DownloadMode::SingleZip;
    caps.features.thumbnails = true;
    assert!(caps.supports_thumbnails());
}

#[test]
fn test_find_sam_prompt_model_from_capabilities() {
    let mut caps = ServerCapabilities::default();
    caps.models.push(ModelCapability {
        id: "sam-like".to_string(),
        name: "SAM Like".to_string(),
        model_type: ModelType::Segmentation,
        description: String::new(),
        inputs: vec![
            InputSchema {
                name: "points".to_string(),
                input_type: "point_list".to_string(),
                required: true,
                description: String::new(),
            },
            InputSchema {
                name: "box".to_string(),
                input_type: "bbox".to_string(),
                required: false,
                description: String::new(),
            },
        ],
        outputs: vec![OutputSchema {
            name: "polygons".to_string(),
            output_type: "polygon_list".to_string(),
            description: String::new(),
        }],
        options: HashMap::new(),
        requires_embedding: true,
        embedding_time_ms: 100,
    });

    let model = caps.find_sam_prompt_model().expect("SAM model not found");
    assert_eq!(model.id, "sam-like");
    assert!(caps.supports_inference());
    assert!(caps.supports_sam());
}

#[test]
fn test_client_message_serialization() {
    let msg = ClientMessage::SetImage {
        request_id: 1,
        image_id: "test".to_string(),
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"action\":\"set_image\""));
    assert!(json.contains("\"request_id\":1"));
    assert!(json.contains("\"image_id\":\"test\""));

    let msg = ClientMessage::PrepareModel {
        request_id: 2,
        model_id: "sam-base".to_string(),
        image_id: None,
        config: None,
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"action\":\"prepare_model\""));

    let msg = ClientMessage::Infer {
        request_id: 3,
        model_id: "sam-base".to_string(),
        image_id: None,
        inputs: serde_json::json!({"points": []}),
        options: serde_json::json!({}),
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(json.contains("\"action\":\"infer\""));
}

#[test]
fn test_capabilities_compatibility() {
    let caps = ServerCapabilities::default();
    assert!(caps.is_compatible(PROTOCOL_VERSION));
    assert!(!caps.is_compatible(PROTOCOL_VERSION + 1));
}
