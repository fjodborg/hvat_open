//! WebSocket protocol types shared between client and server.
//!
//! This module defines the binary protocol for streaming hyperspectral images
//! and running model inference over WebSocket connections. The protocol is
//! versioned to allow future extensions while maintaining backward compatibility.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SAM point for legacy protocol (will be removed).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SamPoint {
    pub x: f32,
    pub y: f32,
    pub label: i32,
}

/// Current protocol version.
///
/// Version 2: Model-agnostic inference protocol with self-describing capabilities.
/// Version 1: Legacy protocol with hardcoded SAM messages.
pub const PROTOCOL_VERSION: u8 = 2;

/// Binary message types (server → client).
///
/// The first byte of every binary WebSocket message identifies its type.
///
/// # Protocol Format
///
/// **Multiplexed WebSocket:**
/// - Header: `[version:u8][type:u8][request_id:u32][payload...]`
/// - All streams share one WebSocket at `/api/ws`, identified by request_id
/// - Connection-level messages (Capabilities, Ping) use request_id = 0
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerMessageType {
    // ========================================================================
    // Streaming messages (0x00-0x0F)
    // ========================================================================
    /// Clear display, new image starting (no payload)
    Reset = 0x00,

    /// Image metadata for current pyramid level
    Metadata = 0x01,

    /// Chunk of layer data
    LayerChunk = 0x02,

    /// Layer upload complete
    LayerComplete = 0x03,

    /// Pyramid level complete
    LevelComplete = 0x04,

    /// Stream complete (all requested levels sent)
    StreamComplete = 0x05,

    /// Server capabilities (sent on connect)
    Capabilities = 0x06,

    /// Stream error - error for specific request_id
    StreamError = 0x07,

    // ========================================================================
    // Inference messages (0x10-0x1F)
    // ========================================================================
    /// Image context set successfully
    ImageSet = 0x10,

    /// Model embedding ready for inference
    ModelReady = 0x11,

    /// Inference progress update (optional)
    InferProgress = 0x12,

    /// Inference result (JSON payload)
    InferResult = 0x13,

    // ========================================================================
    // Connection-level messages (0xFE-0xFF)
    // ========================================================================
    /// Error with code and context (connection-level)
    Error = 0xFE,

    /// Keepalive ping
    Ping = 0xFF,
}

impl ServerMessageType {
    /// Parse from byte value.
    pub fn from_byte(value: u8) -> Option<Self> {
        match value {
            0x00 => Some(Self::Reset),
            0x01 => Some(Self::Metadata),
            0x02 => Some(Self::LayerChunk),
            0x03 => Some(Self::LayerComplete),
            0x04 => Some(Self::LevelComplete),
            0x05 => Some(Self::StreamComplete),
            0x06 => Some(Self::Capabilities),
            0x07 => Some(Self::StreamError),
            0x10 => Some(Self::ImageSet),
            0x11 => Some(Self::ModelReady),
            0x12 => Some(Self::InferProgress),
            0x13 => Some(Self::InferResult),
            0xFE => Some(Self::Error),
            0xFF => Some(Self::Ping),
            _ => None,
        }
    }

    /// Convert to byte value.
    pub fn to_byte(self) -> u8 {
        self as u8
    }

    /// Returns true if this message type is connection-level (not stream-specific).
    ///
    /// Connection-level messages use request_id = 0 in the multiplexed protocol.
    pub fn is_connection_level(&self) -> bool {
        matches!(self, Self::Capabilities | Self::Ping | Self::Error)
    }
}

/// Client message types (client → server, JSON).
///
/// These are sent as JSON text frames over the WebSocket.
///
/// # Multiplexed Protocol
///
/// All messages include `request_id` for correlation with server responses.
/// The client generates unique request IDs per connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ClientMessage {
    // ========================================================================
    // Image context messages
    // ========================================================================
    /// Set active image context for subsequent operations.
    ///
    /// This must be called before `stream_image` or `infer`.
    SetImage { request_id: u32, image_id: String },

    // ========================================================================
    // Streaming messages
    // ========================================================================
    /// Request image streaming.
    ///
    /// If `progressive` is true, sends all levels from highest (smallest) to
    /// `level` (largest). If false, sends only the requested level.
    StreamImage {
        request_id: u32,
        level: u32,
        #[serde(default)]
        progressive: bool,
    },

    /// Cancel a specific stream.
    CancelStream { request_id: u32 },

    // ========================================================================
    // Model inference messages
    // ========================================================================
    /// Pre-compute model embeddings (for models that require it).
    ///
    /// Called when user selects a model with `requires_embedding: true`.
    /// Server responds with `ModelReady` when done.
    ///
    /// Optional `config` field allows model-specific configuration.
    /// For SAM models, config can include band selection for hyperspectral images.
    PrepareModel {
        request_id: u32,
        model_id: String,
        #[serde(default)]
        config: Option<serde_json::Value>,
    },

    /// Run inference with the active image.
    ///
    /// The `inputs` object must match the model's input schema.
    /// The `options` object can override model defaults.
    Infer {
        request_id: u32,
        model_id: String,
        inputs: serde_json::Value,
        #[serde(default)]
        options: serde_json::Value,
    },

    /// Cancel an ongoing inference operation.
    CancelInfer { request_id: u32 },

    // ========================================================================
    // Connection messages
    // ========================================================================
    /// Respond to server ping.
    Pong { timestamp: u64 },
}

/// Error codes organized by category.
///
/// Error codes follow a hierarchical scheme:
/// - 1xxx: Image-related errors
/// - 2xxx: Pyramid-related errors
/// - 3xxx: Inference/model-related errors
/// - 4xxx: Connection-related errors
/// - 5xxx: Server-related errors
/// - 6xxx: Client-related errors
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    // Image errors (1xxx)
    ImageNotFound = 1000,
    InvalidLevel = 1001,
    ImageCorrupted = 1002,
    UnsupportedFormat = 1003,
    ImageTooLarge = 1004,
    NoActiveImage = 1010,

    // Pyramid errors (2xxx)
    PyramidNotReady = 2000,
    PyramidBuildFailed = 2001,
    PyramidCorrupted = 2002,
    PyramidStorageFull = 2003,
    PyramidTimeout = 2004,

    // Inference/model errors (3xxx)
    ModelNotFound = 3010,
    InvalidInput = 3011,
    MissingInput = 3012,
    EmbeddingRequired = 3013,
    EmbeddingExpired = 3014,
    ModelBusy = 3015,
    ModelEncodeFailed = 3016,
    ModelDecodeFailed = 3017,
    ModelOutOfMemory = 3018,

    // Connection errors (4xxx)
    ConnectionTimeout = 4000,
    ConnectionRefused = 4001,
    ConnectionDropped = 4002,
    ProtocolMismatch = 4003,
    AuthRequired = 4004,
    AuthFailed = 4005,
    RateLimited = 4006,
    SessionExpired = 4007,
    MaxConnectionsReached = 4008,

    // Server errors (5xxx)
    InternalError = 5000,
    ServiceUnavailable = 5001,
    StorageError = 5002,
    OutOfMemory = 5003,
    ConfigurationError = 5004,

    // Client errors (6xxx)
    InvalidRequest = 6000,
    InvalidAction = 6001,
    MissingField = 6002,
    InvalidValue = 6003,
}

impl ErrorCode {
    /// Get the error category.
    pub fn category(&self) -> ErrorCategory {
        match *self as u16 {
            1000..=1999 => ErrorCategory::Image,
            2000..=2999 => ErrorCategory::Pyramid,
            3000..=3999 => ErrorCategory::Model,
            4000..=4999 => ErrorCategory::Connection,
            5000..=5999 => ErrorCategory::Server,
            6000..=6999 => ErrorCategory::Client,
            _ => ErrorCategory::Unknown,
        }
    }

    /// Is this error retryable by default?
    pub fn default_retryable(&self) -> bool {
        matches!(
            self,
            Self::PyramidNotReady
                | Self::ModelBusy
                | Self::ConnectionTimeout
                | Self::ConnectionDropped
                | Self::RateLimited
                | Self::SessionExpired
                | Self::ServiceUnavailable
        )
    }

    /// Default retry delay in milliseconds.
    pub fn default_retry_ms(&self) -> u32 {
        match self {
            Self::PyramidNotReady => 5000,
            Self::ModelBusy => 2000,
            Self::RateLimited => 10000,
            Self::ServiceUnavailable => 30000,
            _ => 1000,
        }
    }

    /// Parse from u16 value.
    pub fn from_u16(value: u16) -> Option<Self> {
        match value {
            1000 => Some(Self::ImageNotFound),
            1001 => Some(Self::InvalidLevel),
            1002 => Some(Self::ImageCorrupted),
            1003 => Some(Self::UnsupportedFormat),
            1004 => Some(Self::ImageTooLarge),
            1010 => Some(Self::NoActiveImage),
            2000 => Some(Self::PyramidNotReady),
            2001 => Some(Self::PyramidBuildFailed),
            2002 => Some(Self::PyramidCorrupted),
            2003 => Some(Self::PyramidStorageFull),
            2004 => Some(Self::PyramidTimeout),
            3010 => Some(Self::ModelNotFound),
            3011 => Some(Self::InvalidInput),
            3012 => Some(Self::MissingInput),
            3013 => Some(Self::EmbeddingRequired),
            3014 => Some(Self::EmbeddingExpired),
            3015 => Some(Self::ModelBusy),
            3016 => Some(Self::ModelEncodeFailed),
            3017 => Some(Self::ModelDecodeFailed),
            3018 => Some(Self::ModelOutOfMemory),
            4000 => Some(Self::ConnectionTimeout),
            4001 => Some(Self::ConnectionRefused),
            4002 => Some(Self::ConnectionDropped),
            4003 => Some(Self::ProtocolMismatch),
            4004 => Some(Self::AuthRequired),
            4005 => Some(Self::AuthFailed),
            4006 => Some(Self::RateLimited),
            4007 => Some(Self::SessionExpired),
            4008 => Some(Self::MaxConnectionsReached),
            5000 => Some(Self::InternalError),
            5001 => Some(Self::ServiceUnavailable),
            5002 => Some(Self::StorageError),
            5003 => Some(Self::OutOfMemory),
            5004 => Some(Self::ConfigurationError),
            6000 => Some(Self::InvalidRequest),
            6001 => Some(Self::InvalidAction),
            6002 => Some(Self::MissingField),
            6003 => Some(Self::InvalidValue),
            _ => None,
        }
    }
}

/// Error category for grouping related errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    Image,
    Pyramid,
    Model,
    Connection,
    Server,
    Client,
    Unknown,
}

/// Error severity level.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational message, operation continues
    Info = 0,
    /// Warning, operation continues with degraded functionality
    Warning = 1,
    /// Error, operation failed but connection remains open
    Error = 2,
    /// Fatal error, connection will be closed
    Fatal = 3,
}

impl Severity {
    /// Parse from byte value.
    pub fn from_byte(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Info),
            1 => Some(Self::Warning),
            2 => Some(Self::Error),
            3 => Some(Self::Fatal),
            _ => None,
        }
    }

    /// Convert to byte value.
    pub fn to_byte(self) -> u8 {
        self as u8
    }
}

// ============================================================================
// Capabilities and Model Definitions
// ============================================================================

/// Server information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// Server resource limits.
///
/// These limits can be configured on the server side based on available resources.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerLimits {
    /// Maximum image file size in bytes (not pixel dimensions).
    ///
    /// This limits the raw file size to prevent memory exhaustion, but does not
    /// restrict image dimensions. Very large hyperspectral images are supported
    /// through pyramid levels and chunked streaming.
    ///
    /// Default: 4GB (4,294,967,296 bytes)
    pub max_image_size: u64,

    /// Maximum number of pyramid levels for progressive loading.
    ///
    /// Default: 8
    pub max_pyramid_levels: u8,

    /// Maximum concurrent image streams per connection.
    ///
    /// Default: 4
    pub max_concurrent_streams: u32,

    /// Maximum concurrent inference operations per connection.
    ///
    /// Default: 2
    pub max_concurrent_inferences: u32,
}

impl Default for ServerLimits {
    fn default() -> Self {
        Self {
            max_image_size: 4 * 1024 * 1024 * 1024, // 4GB
            max_pyramid_levels: 8,
            max_concurrent_streams: 4,
            max_concurrent_inferences: 2,
        }
    }
}

/// Server capabilities sent on WebSocket connect.
///
/// This allows clients to discover available models and validate compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerCapabilities {
    /// Protocol version supported by server
    pub protocol_version: u8,

    /// Server identification
    pub server: ServerInfo,

    /// Resource limits
    pub limits: ServerLimits,

    /// Available models
    pub models: Vec<ModelCapability>,
}

impl Default for ServerCapabilities {
    fn default() -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            server: ServerInfo {
                name: "hvat-axum".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            limits: ServerLimits::default(),
            models: vec![],
        }
    }
}

impl ServerCapabilities {
    /// Find a model by type (convenience method).
    pub fn find_model_by_type(&self, model_type: ModelType) -> Option<&ModelCapability> {
        self.models.iter().find(|m| m.model_type == model_type)
    }

    /// Check if a specific model is available.
    pub fn has_model(&self, model_id: &str) -> bool {
        self.models.iter().any(|m| m.id == model_id)
    }

    /// Check if client version is compatible with server.
    pub fn is_compatible(&self, client_version: u8) -> bool {
        // For now, require exact match. In the future, we might allow
        // minor version differences.
        self.protocol_version == client_version
    }
}

/// Model type category for UI hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelType {
    Segmentation,
    Detection,
    Classification,
    Feature,
    Custom,
}

/// Input schema for a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InputSchema {
    pub name: String,
    #[serde(rename = "type")]
    pub input_type: String,
    #[serde(default)]
    pub required: bool,
    #[serde(default)]
    pub description: String,
}

/// Output schema for a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutputSchema {
    pub name: String,
    #[serde(rename = "type")]
    pub output_type: String,
    #[serde(default)]
    pub description: String,
}

/// Option schema for configurable model parameters.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct OptionSchema {
    #[serde(rename = "type")]
    pub option_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<serde_json::Value>,
    #[serde(default)]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
}

/// Model capability definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelCapability {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub model_type: ModelType,
    #[serde(default)]
    pub description: String,
    pub inputs: Vec<InputSchema>,
    pub outputs: Vec<OutputSchema>,
    #[serde(default)]
    pub options: HashMap<String, OptionSchema>,
    #[serde(default)]
    pub requires_embedding: bool,
    #[serde(default)]
    pub embedding_time_ms: u32,
}

#[cfg(test)]
mod tests {
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
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"action\":\"prepare_model\""));

        let msg = ClientMessage::Infer {
            request_id: 3,
            model_id: "sam-base".to_string(),
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
}
