use super::*;

#[test]
fn test_error_encoding() {
    let error = ProtocolError::retryable(ErrorCode::PyramidNotReady, "Pyramid is building", 5000);
    let encoded = encode_error(&error);
    assert_eq!(encoded[0], PROTOCOL_VERSION);
    assert_eq!(encoded[1], ServerMessageType::Error.to_byte());
}

#[test]
fn test_capabilities_encoding_v2() {
    use hvat_common::{DownloadMode, ServerFeatures, ServerInfo, ServerLimits};

    let caps = ServerCapabilities {
        protocol_version: PROTOCOL_VERSION,
        server: ServerInfo {
            name: "test-backend".to_string(),
            version: "0.1.0".to_string(),
        },
        limits: ServerLimits {
            max_image_size: 4 * 1024 * 1024 * 1024,
            max_pyramid_levels: 8,
            max_concurrent_streams: 4,
            max_concurrent_inferences: 0,
        },
        features: ServerFeatures {
            streaming: true,
            project_state: true,
            downloads: true,
            thumbnails: true,
            inference: false,
            sam: false,
            progressive_streaming: false,
        },
        download_mode: DownloadMode::SingleZip,
        models: vec![],
    };

    let encoded = encode_capabilities_v2(&caps);
    assert_eq!(encoded[0], PROTOCOL_VERSION);
    assert_eq!(encoded[1], ServerMessageType::Capabilities.to_byte());
    assert_eq!(
        u32::from_le_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]),
        0
    );

    let json_payload = &encoded[6..];
    let decoded: ServerCapabilities = serde_json::from_slice(json_payload).unwrap();
    assert_eq!(decoded.protocol_version, PROTOCOL_VERSION);
    assert_eq!(decoded.server.name, "test-backend");
}

#[test]
fn test_ping_encoding() {
    let timestamp = 123_456_789_u64;
    let encoded = encode_ping(timestamp);
    assert_eq!(encoded[0], PROTOCOL_VERSION);
    assert_eq!(encoded[1], ServerMessageType::Ping.to_byte());
    assert_eq!(
        u32::from_le_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]),
        0
    );
    assert_eq!(
        u64::from_le_bytes([
            encoded[6],
            encoded[7],
            encoded[8],
            encoded[9],
            encoded[10],
            encoded[11],
            encoded[12],
            encoded[13]
        ]),
        timestamp
    );
}

#[test]
fn test_reset_mux_encoding() {
    let reset = encode_reset_mux(42);
    assert_eq!(reset.len(), 6);
    assert_eq!(reset[0], PROTOCOL_VERSION);
    assert_eq!(reset[1], ServerMessageType::Reset.to_byte());
    assert_eq!(
        u32::from_le_bytes([reset[2], reset[3], reset[4], reset[5]]),
        42
    );
}

#[test]
fn test_stream_complete_encoding() {
    let complete = encode_stream_complete(99);
    assert_eq!(complete.len(), 6);
    assert_eq!(complete[0], PROTOCOL_VERSION);
    assert_eq!(complete[1], ServerMessageType::StreamComplete.to_byte());
}

#[test]
fn test_stream_error_encoding() {
    let error = ProtocolError::error(ErrorCode::ImageNotFound, "Image not found");
    let encoded = encode_stream_error(42, &error);
    assert_eq!(encoded[0], PROTOCOL_VERSION);
    assert_eq!(encoded[1], ServerMessageType::StreamError.to_byte());
    assert_eq!(
        u32::from_le_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]),
        42
    );

    let payload = &encoded[6..];
    let decoded_error = ProtocolError::decode(payload).unwrap();
    assert_eq!(decoded_error.code, ErrorCode::ImageNotFound);
}

#[test]
fn test_model_ready_encoding() {
    let encoded = encode_model_ready(77, "sam-base");
    assert_eq!(encoded[0], PROTOCOL_VERSION);
    assert_eq!(encoded[1], ServerMessageType::ModelReady.to_byte());
    assert_eq!(
        u32::from_le_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]),
        77
    );

    let payload_len = u16::from_le_bytes([encoded[6], encoded[7]]) as usize;
    assert_eq!(payload_len, "sam-base".len());
    assert_eq!(&encoded[8..], b"sam-base");
}

#[test]
fn test_infer_progress_encoding() {
    let encoded = encode_infer_progress(5, 42, "embedding");
    assert_eq!(encoded[0], PROTOCOL_VERSION);
    assert_eq!(encoded[1], ServerMessageType::InferProgress.to_byte());
    assert_eq!(
        u32::from_le_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]),
        5
    );
    assert_eq!(encoded[6], 42);

    let len = u16::from_le_bytes([encoded[7], encoded[8]]) as usize;
    assert_eq!(len, "embedding".len());
    assert_eq!(&encoded[9..], b"embedding");
}

#[test]
fn test_infer_result_encoding() {
    let payload = r#"{"ok":true}"#;
    let encoded = encode_infer_result(9, payload);
    assert_eq!(encoded[0], PROTOCOL_VERSION);
    assert_eq!(encoded[1], ServerMessageType::InferResult.to_byte());
    assert_eq!(
        u32::from_le_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]),
        9
    );
    assert_eq!(&encoded[6..], payload.as_bytes());
}
