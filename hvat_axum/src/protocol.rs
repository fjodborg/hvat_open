//! WebSocket protocol definitions for client-server communication.
//!
//! The protocol uses:
//! - JSON text frames for client→server commands and server→client responses
//! - Binary frames for image layer data (server→client only)

use serde::{Deserialize, Serialize};

/// Client-to-server message types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Start streaming an image at a specific pyramid level
    StartStream { image_id: String, level: u32 },

    /// Cancel the current stream
    CancelStream,

    /// Change to a different pyramid level mid-stream
    ChangeLevel { level: u32 },

    /// Save annotations for an image
    SaveAnnotations {
        image_id: String,
        annotations: serde_json::Value,
        categories: serde_json::Value,
    },

    /// Load annotations for an image
    LoadAnnotations { image_id: String },

    /// Request SAM segmentation
    SamSegment {
        image_id: String,
        points: Vec<SamPoint>,
        #[serde(rename = "box")]
        box_prompt: Option<[f32; 4]>,
    },

    /// Pre-compute SAM embedding for an image
    SamEmbed { image_id: String },
}

/// Point prompt for SAM segmentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamPoint {
    pub x: f32,
    pub y: f32,
    /// 1 = foreground, 0 = background
    pub label: i32,
}

/// Server-to-client JSON response types.
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
    },

    /// SAM embedding is ready
    SamEmbeddingReady { image_id: String },

    /// Error response
    Error { message: String },
}

/// Binary message types for image streaming.
/// These are sent as the first byte of binary WebSocket frames.
#[repr(u8)]
pub enum BinaryMessageType {
    /// Image metadata: [width:u32][height:u32][num_bands:u32][num_layers:u32]
    Metadata = 0x01,

    /// Layer chunk: [layer:u16][row_start:u32][row_end:u32][rgba_bytes...]
    LayerChunk = 0x02,

    /// Layer complete: [layer:u16]
    LayerComplete = 0x03,

    /// Level complete: [level:u8]
    LevelComplete = 0x04,

    /// Error: [error_len:u16][error_utf8...]
    Error = 0x05,
}

impl BinaryMessageType {
    pub fn to_byte(self) -> u8 {
        self as u8
    }
}

/// Stream metadata sent at the start of streaming.
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    pub width: u32,
    pub height: u32,
    pub num_bands: u32,
    pub num_layers: u32,
}

impl StreamMetadata {
    /// Encode metadata as binary (16 bytes + type byte).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(17);
        buf.push(BinaryMessageType::Metadata.to_byte());
        buf.extend_from_slice(&self.width.to_le_bytes());
        buf.extend_from_slice(&self.height.to_le_bytes());
        buf.extend_from_slice(&self.num_bands.to_le_bytes());
        buf.extend_from_slice(&self.num_layers.to_le_bytes());
        buf
    }
}

/// Encode a layer chunk message.
pub fn encode_layer_chunk(layer: u16, row_start: u32, row_end: u32, rgba_data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(11 + rgba_data.len());
    buf.push(BinaryMessageType::LayerChunk.to_byte());
    buf.extend_from_slice(&layer.to_le_bytes());
    buf.extend_from_slice(&row_start.to_le_bytes());
    buf.extend_from_slice(&row_end.to_le_bytes());
    buf.extend_from_slice(rgba_data);
    buf
}

/// Encode a layer complete message.
pub fn encode_layer_complete(layer: u16) -> Vec<u8> {
    let mut buf = Vec::with_capacity(3);
    buf.push(BinaryMessageType::LayerComplete.to_byte());
    buf.extend_from_slice(&layer.to_le_bytes());
    buf
}

/// Encode a level complete message.
pub fn encode_level_complete(level: u8) -> Vec<u8> {
    vec![BinaryMessageType::LevelComplete.to_byte(), level]
}

/// Encode an error message.
pub fn encode_error(message: &str) -> Vec<u8> {
    let bytes = message.as_bytes();
    let len = bytes.len().min(u16::MAX as usize) as u16;
    let mut buf = Vec::with_capacity(3 + len as usize);
    buf.push(BinaryMessageType::Error.to_byte());
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&bytes[..len as usize]);
    buf
}
