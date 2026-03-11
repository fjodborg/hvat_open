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
        crate::common::protocol::StreamMetadata {
            width: self.width,
            height: self.height,
            num_bands: self.num_bands,
            num_layers: self.num_layers,
            full_width: self.full_width,
            full_height: self.full_height,
        }
        .to_bytes_mux(request_id)
    }
}

// ============================================================================
// Multiplexed Encoding Functions (single WebSocket)
// ============================================================================

/// Encode a Reset message with request_id (multiplexed protocol).
pub fn encode_reset_mux(request_id: u32) -> Vec<u8> {
    crate::common::protocol::encode_reset_mux(request_id)
}

/// Encode a layer chunk message with request_id (multiplexed protocol).
pub fn encode_layer_chunk_mux(
    request_id: u32,
    layer: u16,
    row_start: u32,
    row_end: u32,
    rgba_data: &[u8],
) -> Vec<u8> {
    crate::common::protocol::encode_layer_chunk_mux(
        request_id, layer, row_start, row_end, rgba_data,
    )
}

/// Encode a layer complete message with request_id (multiplexed protocol).
pub fn encode_layer_complete_mux(request_id: u32, layer: u16) -> Vec<u8> {
    crate::common::protocol::encode_layer_complete_mux(request_id, layer)
}

/// Encode a level complete message with request_id (multiplexed protocol).
pub fn encode_level_complete_mux(request_id: u32, level: u8) -> Vec<u8> {
    crate::common::protocol::encode_level_complete_mux(request_id, level)
}

/// Encode a stream complete message with request_id (multiplexed protocol).
///
/// This signals that all data for the given request_id has been sent.
pub fn encode_stream_complete(request_id: u32) -> Vec<u8> {
    crate::common::protocol::encode_stream_complete(request_id)
}

/// Encode a stream-specific error (multiplexed protocol).
///
/// This sends an error for a specific stream without affecting other streams.
pub fn encode_stream_error(request_id: u32, error: &ProtocolError) -> Vec<u8> {
    crate::common::protocol::encode_stream_error(request_id, error)
}

/// Encode an error message using the new protocol.
///
/// This is a convenience wrapper around `ProtocolError::encode()`.
pub fn encode_error(error: &ProtocolError) -> Vec<u8> {
    crate::common::protocol::encode_error(error)
}

/// Encode a simple error message from a string.
///
/// Creates a non-retryable error with the given code and message.
pub fn encode_simple_error(code: ErrorCode, message: &str) -> Vec<u8> {
    crate::common::protocol::encode_simple_error(code, message)
}

/// Encode a Ping message for keepalive.
///
/// Format: `[version:u8][type:u8][request_id=0:u32][timestamp:u64]`
///
/// The timestamp is typically the server's monotonic time in milliseconds,
/// which the client echoes back in a Pong message.
pub fn encode_ping(timestamp: u64) -> Vec<u8> {
    crate::common::protocol::encode_ping(timestamp)
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
    crate::common::protocol::encode_image_set(request_id, image_id)
}

/// Encode a ModelReady message (multiplexed protocol).
///
/// Indicates that model embedding has been computed and is ready for inference.
///
/// Format: `[version:u8][type:u8][request_id:u32][model_id_len:u16][model_id:utf8]`
pub fn encode_model_ready(request_id: u32, model_id: &str) -> Vec<u8> {
    crate::common::protocol::encode_model_ready(request_id, model_id)
}

/// Encode an InferProgress message (multiplexed protocol).
///
/// Reports progress during model inference or embedding computation.
///
/// Format: `[version:u8][type:u8][request_id:u32][progress:u8][status_len:u16][status:utf8]`
pub fn encode_infer_progress(request_id: u32, progress: u8, status: &str) -> Vec<u8> {
    crate::common::protocol::encode_infer_progress(request_id, progress, status)
}

/// Encode an InferResult message (multiplexed protocol).
///
/// Returns inference results as JSON matching the model's output schema.
///
/// Format: `[version:u8][type:u8][request_id:u32][json_payload:utf8]`
pub fn encode_infer_result(request_id: u32, result_json: &str) -> Vec<u8> {
    crate::common::protocol::encode_infer_result(request_id, result_json)
}

/// Encode server capabilities as JSON (Protocol v2).
///
/// This replaces the binary encoding used in v1. The capabilities are sent
/// as a binary message with a JSON payload.
///
/// Format: `[version:u8][type:u8][request_id=0:u32][json_payload:utf8]`
pub fn encode_capabilities_v2(capabilities: &hvat_common::ServerCapabilities) -> Vec<u8> {
    crate::common::protocol::encode_capabilities_v2(capabilities)
}

#[cfg(test)]
#[path = "protocol.test.rs"]
mod tests;
