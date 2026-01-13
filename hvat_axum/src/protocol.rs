//! WebSocket protocol definitions for client-server communication.
//!
//! The protocol uses:
//! - JSON text frames for client→server commands and server→client responses
//! - Binary frames for image layer data (server→client only)

use serde::{Deserialize, Serialize};

// Re-export shared protocol types from hvat_common
pub use hvat_common::protocol::{
    ClientMessage, ErrorCode, PROTOCOL_VERSION, SamPoint, ServerMessageType, Severity,
};
pub use hvat_common::{ErrorContext, ProtocolError};

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
    /// Encode metadata as binary (protocol v1).
    ///
    /// Format: `[version:u8][type:u8][width:u32][height:u32][num_bands:u32][num_layers:u32][full_width:u32][full_height:u32]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(26);
        buf.push(PROTOCOL_VERSION);
        buf.push(ServerMessageType::Metadata.to_byte());
        buf.extend_from_slice(&self.width.to_le_bytes());
        buf.extend_from_slice(&self.height.to_le_bytes());
        buf.extend_from_slice(&self.num_bands.to_le_bytes());
        buf.extend_from_slice(&self.num_layers.to_le_bytes());
        buf.extend_from_slice(&self.full_width.to_le_bytes());
        buf.extend_from_slice(&self.full_height.to_le_bytes());
        buf
    }
}

/// Encode a Reset message.
///
/// This signals the client to clear the current display before a new image loads.
pub fn encode_reset() -> Vec<u8> {
    vec![PROTOCOL_VERSION, ServerMessageType::Reset.to_byte()]
}

/// Encode a layer chunk message.
pub fn encode_layer_chunk(layer: u16, row_start: u32, row_end: u32, rgba_data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(12 + rgba_data.len());
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::LayerChunk.to_byte());
    buf.extend_from_slice(&layer.to_le_bytes());
    buf.extend_from_slice(&row_start.to_le_bytes());
    buf.extend_from_slice(&row_end.to_le_bytes());
    buf.extend_from_slice(rgba_data);
    buf
}

/// Encode a layer complete message.
pub fn encode_layer_complete(layer: u16) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4);
    buf.push(PROTOCOL_VERSION);
    buf.push(ServerMessageType::LayerComplete.to_byte());
    buf.extend_from_slice(&layer.to_le_bytes());
    buf
}

/// Encode a level complete message.
pub fn encode_level_complete(level: u8) -> Vec<u8> {
    vec![
        PROTOCOL_VERSION,
        ServerMessageType::LevelComplete.to_byte(),
        level,
    ]
}

/// Encode an all complete message (all levels sent).
pub fn encode_all_complete() -> Vec<u8> {
    vec![PROTOCOL_VERSION, ServerMessageType::AllComplete.to_byte()]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reset_encoding() {
        let reset = encode_reset();
        assert_eq!(reset.len(), 2);
        assert_eq!(reset[0], PROTOCOL_VERSION);
        assert_eq!(reset[1], ServerMessageType::Reset.to_byte());
    }

    #[test]
    fn test_metadata_encoding() {
        let meta = StreamMetadata {
            width: 1024,
            height: 768,
            num_bands: 10,
            num_layers: 3,
            full_width: 2048,
            full_height: 1536,
        };
        let bytes = meta.to_bytes();

        assert_eq!(bytes.len(), 26);
        assert_eq!(bytes[0], PROTOCOL_VERSION);
        assert_eq!(bytes[1], ServerMessageType::Metadata.to_byte());

        // Verify width
        assert_eq!(
            u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]),
            1024
        );
        // Verify height
        assert_eq!(
            u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]),
            768
        );
    }

    #[test]
    fn test_layer_chunk_encoding() {
        let data = vec![255u8, 128, 64, 32];
        let chunk = encode_layer_chunk(0, 0, 10, &data);

        assert_eq!(chunk[0], PROTOCOL_VERSION);
        assert_eq!(chunk[1], ServerMessageType::LayerChunk.to_byte());

        // Verify layer index
        assert_eq!(u16::from_le_bytes([chunk[2], chunk[3]]), 0);

        // Verify data
        assert_eq!(&chunk[12..], &data[..]);
    }

    #[test]
    fn test_error_encoding() {
        let error =
            ProtocolError::retryable(ErrorCode::PyramidNotReady, "Pyramid is building", 5000);

        let encoded = encode_error(&error);
        assert_eq!(encoded[0], PROTOCOL_VERSION);
        assert_eq!(encoded[1], ServerMessageType::Error.to_byte());
    }
}
