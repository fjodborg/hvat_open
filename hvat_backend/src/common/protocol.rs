//! WebSocket protocol encoding for the HVAT binary streaming protocol.
//!
//! Binary header: `[version:u8][type:u8][request_id:u32][payload...]`

use crate::common::frame::FrameBuilder;
pub use hvat_common::protocol::{
    ClientMessage, ErrorCode, PROTOCOL_VERSION, ServerMessageType, Severity,
};
pub use hvat_common::{ErrorContext, ProtocolError, ServerCapabilities};

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
    let json = serde_json::to_string(capabilities).unwrap();
    let json_bytes = json.as_bytes();
    FrameBuilder::new()
        .with_header(
            PROTOCOL_VERSION,
            ServerMessageType::Capabilities.to_byte(),
            0,
        )
        .reserve_payload(json_bytes.len())
        .extend_bytes(json_bytes)
        .finish()
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
}
