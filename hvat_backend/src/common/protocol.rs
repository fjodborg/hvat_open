//! WebSocket protocol encoding for the HVAT binary streaming protocol.
//!
//! Binary header: `[version:u8][type:u8][request_id:u32][payload...]`

use crate::common::frame::FrameBuilder;
pub use hvat_common::protocol::{
    ClientMessage, ErrorCode, PROTOCOL_VERSION, ServerMessageType, Severity,
};
pub use hvat_common::{ErrorContext, ProtocolError, ServerCapabilities};

const MAX_U16_BYTES: usize = u16::MAX as usize;

fn truncate_utf8_to_u16(value: &str) -> &str {
    if value.len() <= MAX_U16_BYTES {
        return value;
    }

    let mut end = MAX_U16_BYTES;
    while end > 0 && !value.is_char_boundary(end) {
        end -= 1;
    }
    &value[..end]
}

/// Stream metadata sent at the start of streaming.
#[derive(Debug, Clone)]
pub struct StreamMetadata {
    pub width: u32,
    pub height: u32,
    pub num_bands: u32,
    pub num_layers: u32,
    /// Full resolution width (for progressive loading)
    pub full_width: u32,
    /// Full resolution height (for progressive loading)
    pub full_height: u32,
}

impl StreamMetadata {
    pub fn to_bytes_mux(&self, request_id: u32) -> Vec<u8> {
        FrameBuilder::new()
            .with_header(
                PROTOCOL_VERSION,
                ServerMessageType::Metadata.to_byte(),
                request_id,
            )
            .reserve_payload(24)
            .push_u32_le(self.width)
            .push_u32_le(self.height)
            .push_u32_le(self.num_bands)
            .push_u32_le(self.num_layers)
            .push_u32_le(self.full_width)
            .push_u32_le(self.full_height)
            .finish()
    }
}

pub fn encode_reset_mux(request_id: u32) -> Vec<u8> {
    FrameBuilder::new()
        .with_header(
            PROTOCOL_VERSION,
            ServerMessageType::Reset.to_byte(),
            request_id,
        )
        .finish()
}

pub fn encode_layer_chunk_mux(
    request_id: u32,
    layer: u16,
    row_start: u32,
    row_end: u32,
    rgba_data: &[u8],
) -> Vec<u8> {
    FrameBuilder::new()
        .with_header(
            PROTOCOL_VERSION,
            ServerMessageType::LayerChunk.to_byte(),
            request_id,
        )
        .reserve_payload(10 + rgba_data.len())
        .push_u16_le(layer)
        .push_u32_le(row_start)
        .push_u32_le(row_end)
        .extend_bytes(rgba_data)
        .finish()
}

pub fn encode_layer_complete_mux(request_id: u32, layer: u16) -> Vec<u8> {
    FrameBuilder::new()
        .with_header(
            PROTOCOL_VERSION,
            ServerMessageType::LayerComplete.to_byte(),
            request_id,
        )
        .reserve_payload(2)
        .push_u16_le(layer)
        .finish()
}

pub fn encode_level_complete_mux(request_id: u32, level: u8) -> Vec<u8> {
    FrameBuilder::new()
        .with_header(
            PROTOCOL_VERSION,
            ServerMessageType::LevelComplete.to_byte(),
            request_id,
        )
        .reserve_payload(1)
        .push_u8(level)
        .finish()
}

pub fn encode_stream_complete(request_id: u32) -> Vec<u8> {
    FrameBuilder::new()
        .with_header(
            PROTOCOL_VERSION,
            ServerMessageType::StreamComplete.to_byte(),
            request_id,
        )
        .finish()
}

pub fn encode_stream_error(request_id: u32, error: &ProtocolError) -> Vec<u8> {
    let error_payload = error.encode_payload();
    FrameBuilder::new()
        .with_header(
            PROTOCOL_VERSION,
            ServerMessageType::StreamError.to_byte(),
            request_id,
        )
        .reserve_payload(error_payload.len())
        .extend_bytes(&error_payload)
        .finish()
}

pub fn encode_error(error: &ProtocolError) -> Vec<u8> {
    error.encode(PROTOCOL_VERSION)
}

pub fn encode_simple_error(code: ErrorCode, message: &str) -> Vec<u8> {
    ProtocolError::error(code, message).encode(PROTOCOL_VERSION)
}

pub fn encode_ping(timestamp: u64) -> Vec<u8> {
    FrameBuilder::new()
        .with_header(PROTOCOL_VERSION, ServerMessageType::Ping.to_byte(), 0)
        .reserve_payload(8)
        .push_u64_le(timestamp)
        .finish()
}

pub fn encode_image_set(request_id: u32, image_id: &str) -> Vec<u8> {
    let image_id = truncate_utf8_to_u16(image_id);
    let image_id_bytes = image_id.as_bytes();
    FrameBuilder::new()
        .with_header(
            PROTOCOL_VERSION,
            ServerMessageType::ImageSet.to_byte(),
            request_id,
        )
        .reserve_payload(2 + image_id_bytes.len())
        .push_u16_le(image_id_bytes.len() as u16)
        .extend_bytes(image_id_bytes)
        .finish()
}

pub fn encode_model_ready(request_id: u32, model_id: &str) -> Vec<u8> {
    let model_id = truncate_utf8_to_u16(model_id);
    let model_id_bytes = model_id.as_bytes();
    FrameBuilder::new()
        .with_header(
            PROTOCOL_VERSION,
            ServerMessageType::ModelReady.to_byte(),
            request_id,
        )
        .reserve_payload(2 + model_id_bytes.len())
        .push_u16_le(model_id_bytes.len() as u16)
        .extend_bytes(model_id_bytes)
        .finish()
}

pub fn encode_infer_progress(request_id: u32, progress: u8, status: &str) -> Vec<u8> {
    let status = truncate_utf8_to_u16(status);
    let status_bytes = status.as_bytes();
    FrameBuilder::new()
        .with_header(
            PROTOCOL_VERSION,
            ServerMessageType::InferProgress.to_byte(),
            request_id,
        )
        .reserve_payload(3 + status_bytes.len())
        .push_u8(progress)
        .push_u16_le(status_bytes.len() as u16)
        .extend_bytes(status_bytes)
        .finish()
}

pub fn encode_infer_result(request_id: u32, result_json: &str) -> Vec<u8> {
    let json_bytes = result_json.as_bytes();
    FrameBuilder::new()
        .with_header(
            PROTOCOL_VERSION,
            ServerMessageType::InferResult.to_byte(),
            request_id,
        )
        .reserve_payload(json_bytes.len())
        .extend_bytes(json_bytes)
        .finish()
}

pub fn encode_capabilities_v2(capabilities: &hvat_common::ServerCapabilities) -> Vec<u8> {
    let json_bytes = serde_json::to_vec(capabilities).unwrap_or_else(|_| b"{}".to_vec());
    FrameBuilder::new()
        .with_header(
            PROTOCOL_VERSION,
            ServerMessageType::Capabilities.to_byte(),
            0,
        )
        .reserve_payload(json_bytes.len())
        .extend_bytes(&json_bytes)
        .finish()
}

#[cfg(test)]
#[path = "protocol.test.rs"]
mod tests;
