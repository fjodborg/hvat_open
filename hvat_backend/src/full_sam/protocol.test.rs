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
    use hvat_common::{
        DownloadMode, ModelCapability, ModelType, ServerFeatures, ServerInfo, ServerLimits,
    };
    use std::collections::HashMap;

    let caps = ServerCapabilities {
        protocol_version: PROTOCOL_VERSION,
        server: ServerInfo {
            name: "hvat-axum".to_string(),
            version: "0.1.0".to_string(),
        },
        limits: ServerLimits {
            max_image_size: 4 * 1024 * 1024 * 1024,
            max_pyramid_levels: 8,
            max_concurrent_streams: 4,
            max_concurrent_inferences: 2,
        },
        features: ServerFeatures {
            streaming: true,
            project_state: true,
            downloads: true,
            thumbnails: true,
            inference: true,
            sam: true,
            progressive_streaming: true,
        },
        download_mode: DownloadMode::Chunked,
        models: vec![ModelCapability {
            id: "sam-tiny".to_string(),
            name: "SAM Tiny".to_string(),
            model_type: ModelType::Segmentation,
            description: "Segment Anything Model (Tiny)".to_string(),
            inputs: vec![],
            outputs: vec![],
            options: HashMap::new(),
            requires_embedding: true,
            embedding_time_ms: 500,
        }],
    };

    let encoded = encode_capabilities_v2(&caps);

    // Verify header
    assert_eq!(encoded[0], PROTOCOL_VERSION);
    assert_eq!(encoded[1], ServerMessageType::Capabilities.to_byte());

    // Verify request_id is 0 (connection-level)
    assert_eq!(
        u32::from_le_bytes([encoded[2], encoded[3], encoded[4], encoded[5]]),
        0
    );

    // Verify it can be decoded as JSON by client
    let json_payload = &encoded[6..];
    let decoded: ServerCapabilities = serde_json::from_slice(json_payload).unwrap();
    assert_eq!(decoded.protocol_version, PROTOCOL_VERSION);
    assert_eq!(decoded.server.name, "hvat-axum");
    assert_eq!(decoded.limits.max_concurrent_streams, 4);
    assert_eq!(decoded.models.len(), 1);
    assert_eq!(decoded.models[0].id, "sam-tiny");
}

#[test]
fn test_ping_encoding_includes_connection_request_id() {
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

/// Test version mismatch detection.
#[test]
fn test_version_mismatch_detection() {
    // Create capabilities with a different version
    let caps = ServerCapabilities {
        protocol_version: 99,
        ..ServerCapabilities::default()
    }; // Future version

    // Client checks compatibility
    assert!(!caps.is_compatible(PROTOCOL_VERSION));
    assert!(caps.is_compatible(99));

    // Verify error can be created for mismatch
    let error = ProtocolError::fatal(
        ErrorCode::ProtocolMismatch,
        format!(
            "Protocol version mismatch: server uses v{}, client uses v{}",
            caps.protocol_version, PROTOCOL_VERSION
        ),
    );

    assert_eq!(error.code, ErrorCode::ProtocolMismatch);
    assert!(!error.retryable);
}

// ========================================================================
// Multiplexed Protocol Tests
// ========================================================================

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
fn test_metadata_mux_encoding() {
    let meta = StreamMetadata {
        width: 1024,
        height: 768,
        num_bands: 10,
        num_layers: 3,
        full_width: 2048,
        full_height: 1536,
    };
    let bytes = meta.to_bytes_mux(123);

    assert_eq!(bytes.len(), 30); // 2 header + 4 request_id + 24 payload
    assert_eq!(bytes[0], PROTOCOL_VERSION);
    assert_eq!(bytes[1], ServerMessageType::Metadata.to_byte());

    // Verify request_id
    assert_eq!(
        u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]),
        123
    );

    // Verify width (offset by 4 for request_id)
    assert_eq!(
        u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]),
        1024
    );
}

#[test]
fn test_layer_chunk_mux_encoding() {
    let data = vec![255u8, 128, 64, 32];
    let chunk = encode_layer_chunk_mux(42, 0, 0, 10, &data);

    assert_eq!(chunk[0], PROTOCOL_VERSION);
    assert_eq!(chunk[1], ServerMessageType::LayerChunk.to_byte());

    // Verify request_id
    assert_eq!(
        u32::from_le_bytes([chunk[2], chunk[3], chunk[4], chunk[5]]),
        42
    );

    // Verify layer index (offset by 4)
    assert_eq!(u16::from_le_bytes([chunk[6], chunk[7]]), 0);

    // Verify data (offset by 4)
    assert_eq!(&chunk[16..], &data[..]);
}

#[test]
fn test_stream_complete_encoding() {
    let complete = encode_stream_complete(99);
    assert_eq!(complete.len(), 6);
    assert_eq!(complete[0], PROTOCOL_VERSION);
    assert_eq!(complete[1], ServerMessageType::StreamComplete.to_byte());
    assert_eq!(
        u32::from_le_bytes([complete[2], complete[3], complete[4], complete[5]]),
        99
    );
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

    // Verify the error payload follows the request_id
    let payload = &encoded[6..];
    let decoded_error = ProtocolError::decode(payload).unwrap();
    assert_eq!(decoded_error.code, ErrorCode::ImageNotFound);
}
