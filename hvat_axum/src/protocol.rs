//! WebSocket protocol definitions for client-server communication.
//!
//! The protocol uses:
//! - JSON text frames for client→server commands and server→client responses
//! - Binary frames for image layer data (server→client only)
//!
//! # Protocol
//!
//! **Multiplexed (single WebSocket endpoint: `/api/ws`):**
//! - Binary header: `[version:u8][type:u8][request_id:u32][payload...]`
//! - All streams share one WebSocket, identified by request_id
//! - Connection-level messages (Capabilities, Ping) use request_id = 0

use serde::{Deserialize, Serialize};

// Re-export shared protocol types from hvat_common
pub use hvat_common::protocol::{
    ClientMessage, ErrorCode, PROTOCOL_VERSION, ServerMessageType, Severity,
};
pub use hvat_common::{ErrorContext, ProtocolError, ServerCapabilities};

/// Server-to-client JSON response types.
///
/// These are sent as JSON text frames for non-streaming data like
/// annotation responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServerResponse {
    /// Annotations loaded from storage
    AnnotationsLoaded {
        image_id: String,
        annotations: serde_json::Value,
        categories: serde_json::Value,
    },

    /// Annotations saved successfully
    AnnotationsSaved { image_id: String, success: bool },

    /// SAM segmentation result
    SamMask {
        request_id: String,
        /// Each polygon is a flat array [x1, y1, x2, y2, ...]
        polygons: Vec<Vec<f32>>,
        /// IoU scores for each mask (0.0 to 1.0)
        iou_scores: Vec<f32>,
    },

    /// SAM embedding is ready
    SamEmbeddingReady { image_id: String },

    /// Error response (JSON version, binary version is preferred)
    Error { message: String },
}

/// Stream metadata sent at the start of streaming.
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    /// Width of this level's texture data
    pub width: u32,
    /// Height of this level's texture data
    pub height: u32,
    pub num_bands: u32,
    pub num_layers: u32,
    /// Full resolution width (for progressive loading - canvas should use this)
    pub full_width: u32,
    /// Full resolution height (for progressive loading - canvas should use this)
    pub full_height: u32,
}

impl StreamMetadata {
    /// Encode metadata with request_id (multiplexed protocol).
    ///
    /// Format: `[version:u8][type:u8][request_id:u32][width:u32][height:u32][num_bands:u32][num_layers:u32][full_width:u32][full_height:u32]`
    pub fn to_bytes_mux(&self, request_id: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity(30);
        buf.push(PROTOCOL_VERSION);
        buf.push(ServerMessageType::Metadata.to_byte());
        buf.extend_from_slice(&request_id.to_le_bytes());
        buf.extend_from_slice(&self.width.to_le_bytes());
        buf.extend_from_slice(&self.height.to_le_bytes());
        buf.extend_from_slice(&self.num_bands.to_le_bytes());
        buf.extend_from_slice(&self.num_layers.to_le_bytes());
        buf.extend_from_slice(&self.full_width.to_le_bytes());
        buf.extend_from_slice(&self.full_height.to_le_bytes());
        buf
    }
}

// ============================================================================
// Multiplexed Encoding Functions (single WebSocket)
// ============================================================================

/// Encode a Reset message with request_id (multiplexed protocol).
pub fn encode_reset_mux(request_id: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(6);
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::Reset.to_byte());
    buf.extend_from_slice(&request_id.to_le_bytes());
    buf
}

/// Encode a layer chunk message with request_id (multiplexed protocol).
pub fn encode_layer_chunk_mux(
    request_id: u32,
    layer: u16,
    row_start: u32,
    row_end: u32,
    rgba_data: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(16 + rgba_data.len());
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::LayerChunk.to_byte());
    buf.extend_from_slice(&request_id.to_le_bytes());
    buf.extend_from_slice(&layer.to_le_bytes());
    buf.extend_from_slice(&row_start.to_le_bytes());
    buf.extend_from_slice(&row_end.to_le_bytes());
    buf.extend_from_slice(rgba_data);
    buf
}

/// Encode a layer complete message with request_id (multiplexed protocol).
pub fn encode_layer_complete_mux(request_id: u32, layer: u16) -> Vec<u8> {
    let mut buf = Vec::with_capacity(8);
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::LayerComplete.to_byte());
    buf.extend_from_slice(&request_id.to_le_bytes());
    buf.extend_from_slice(&layer.to_le_bytes());
    buf
}

/// Encode a level complete message with request_id (multiplexed protocol).
pub fn encode_level_complete_mux(request_id: u32, level: u8) -> Vec<u8> {
    let mut buf = Vec::with_capacity(7);
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::LevelComplete.to_byte());
    buf.extend_from_slice(&request_id.to_le_bytes());
    buf.push(level);
    buf
}

/// Encode a stream complete message with request_id (multiplexed protocol).
///
/// This signals that all data for the given request_id has been sent.
pub fn encode_stream_complete(request_id: u32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(6);
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::StreamComplete.to_byte());
    buf.extend_from_slice(&request_id.to_le_bytes());
    buf
}

/// Encode a stream-specific error (multiplexed protocol).
///
/// This sends an error for a specific stream without affecting other streams.
pub fn encode_stream_error(request_id: u32, error: &ProtocolError) -> Vec<u8> {
    let error_payload = error.encode_payload();
    let mut buf = Vec::with_capacity(6 + error_payload.len());
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::StreamError.to_byte());
    buf.extend_from_slice(&request_id.to_le_bytes());
    buf.extend_from_slice(&error_payload);
    buf
}

/// Encode an error message using the new protocol.
///
/// This is a convenience wrapper around `ProtocolError::encode()`.
pub fn encode_error(error: &ProtocolError) -> Vec<u8> {
    error.encode(PROTOCOL_VERSION)
}

/// Encode a simple error message from a string.
///
/// Creates a non-retryable error with the given code and message.
pub fn encode_simple_error(code: ErrorCode, message: &str) -> Vec<u8> {
    ProtocolError::error(code, message).encode(PROTOCOL_VERSION)
}

/// Encode a Ping message for keepalive.
///
/// Format: `[version:u8][type:u8][request_id=0:u32][timestamp:u64]`
///
/// The timestamp is typically the server's monotonic time in milliseconds,
/// which the client echoes back in a Pong message.
pub fn encode_ping(timestamp: u64) -> Vec<u8> {
    let mut buf = Vec::with_capacity(14);
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::Ping.to_byte());
    buf.extend_from_slice(&0u32.to_le_bytes()); // request_id = 0 for connection-level
    buf.extend_from_slice(&timestamp.to_le_bytes());
    buf
}

// ============================================================================
// Inference Protocol Messages (Protocol v2)
// ============================================================================

/// Encode an ImageSet message (multiplexed protocol).
///
/// Confirms that the active image context has been set successfully.
///
/// Format: `[version:u8][type:u8][request_id:u32][image_id_len:u16][image_id:utf8]`
pub fn encode_image_set(request_id: u32, image_id: &str) -> Vec<u8> {
    let image_id_bytes = image_id.as_bytes();
    let mut buf = Vec::with_capacity(8 + image_id_bytes.len());
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::ImageSet.to_byte());
    buf.extend_from_slice(&request_id.to_le_bytes());
    buf.extend_from_slice(&(image_id_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(image_id_bytes);
    buf
}

/// Encode a ModelReady message (multiplexed protocol).
///
/// Indicates that model embedding has been computed and is ready for inference.
///
/// Format: `[version:u8][type:u8][request_id:u32][model_id_len:u16][model_id:utf8]`
pub fn encode_model_ready(request_id: u32, model_id: &str) -> Vec<u8> {
    let model_id_bytes = model_id.as_bytes();
    let mut buf = Vec::with_capacity(8 + model_id_bytes.len());
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::ModelReady.to_byte());
    buf.extend_from_slice(&request_id.to_le_bytes());
    buf.extend_from_slice(&(model_id_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(model_id_bytes);
    buf
}

/// Encode an InferProgress message (multiplexed protocol).
///
/// Reports progress during model inference or embedding computation.
///
/// Format: `[version:u8][type:u8][request_id:u32][progress:u8][status_len:u16][status:utf8]`
pub fn encode_infer_progress(request_id: u32, progress: u8, status: &str) -> Vec<u8> {
    let status_bytes = status.as_bytes();
    let mut buf = Vec::with_capacity(9 + status_bytes.len());
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::InferProgress.to_byte());
    buf.extend_from_slice(&request_id.to_le_bytes());
    buf.push(progress);
    buf.extend_from_slice(&(status_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(status_bytes);
    buf
}

/// Encode an InferResult message (multiplexed protocol).
///
/// Returns inference results as JSON matching the model's output schema.
///
/// Format: `[version:u8][type:u8][request_id:u32][json_payload:utf8]`
pub fn encode_infer_result(request_id: u32, result_json: &str) -> Vec<u8> {
    let json_bytes = result_json.as_bytes();
    let mut buf = Vec::with_capacity(6 + json_bytes.len());
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::InferResult.to_byte());
    buf.extend_from_slice(&request_id.to_le_bytes());
    buf.extend_from_slice(json_bytes);
    buf
}

/// Encode server capabilities as JSON (Protocol v2).
///
/// This replaces the binary encoding used in v1. The capabilities are sent
/// as a binary message with a JSON payload.
///
/// Format: `[version:u8][type:u8][request_id=0:u32][json_payload:utf8]`
pub fn encode_capabilities_v2(capabilities: &hvat_common::ServerCapabilities) -> Vec<u8> {
    let json = serde_json::to_string(capabilities).unwrap();
    let json_bytes = json.as_bytes();
    let mut buf = Vec::with_capacity(6 + json_bytes.len());
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::Capabilities.to_byte());
    buf.extend_from_slice(&0u32.to_le_bytes()); // request_id = 0 for connection-level
    buf.extend_from_slice(json_bytes);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_encoding() {
        let error =
            ProtocolError::retryable(ErrorCode::PyramidNotReady, "Pyramid is building", 5000);

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
                inference: true,
                sam: true,
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
}
