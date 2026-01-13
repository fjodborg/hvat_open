//! WebSocket protocol types shared between client and server.
//!
//! This module defines the binary protocol for streaming hyperspectral images
//! over WebSocket connections. The protocol is versioned to allow future
//! extensions while maintaining backward compatibility.

use serde::{Deserialize, Serialize};

/// Current protocol version.
///
/// Increment this on breaking changes. Clients and servers negotiate
/// compatibility during connection establishment.
pub const PROTOCOL_VERSION: u8 = 1;

/// Binary message types (server → client).
///
/// The first byte of every binary WebSocket message identifies its type.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerMessageType {
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

    /// All requested levels sent
    AllComplete = 0x05,

    /// Server capabilities (sent on connect)
    Capabilities = 0x06,

    /// SAM embedding ready
    SamReady = 0x10,

    /// SAM segmentation result
    SamMask = 0x11,

    /// SAM embedding progress update
    SamProgress = 0x12,

    /// Error with code and context
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
            0x05 => Some(Self::AllComplete),
            0x06 => Some(Self::Capabilities),
            0x10 => Some(Self::SamReady),
            0x11 => Some(Self::SamMask),
            0x12 => Some(Self::SamProgress),
            0xFE => Some(Self::Error),
            0xFF => Some(Self::Ping),
            _ => None,
        }
    }

    /// Convert to byte value.
    pub fn to_byte(self) -> u8 {
        self as u8
    }
}

/// Client message types (client → server, JSON).
///
/// These are sent as JSON text frames over the WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum ClientMessage {
    /// Start streaming an image at a specific pyramid level.
    ///
    /// If `progressive` is true, sends all levels from highest (smallest) to
    /// `level` (largest). If false, sends only the requested level.
    StartStream {
        level: u32,
        #[serde(default)]
        progressive: bool,
    },

    /// Cancel the current stream.
    Cancel,

    /// Request SAM embedding pre-computation.
    SamEmbed,

    /// Request SAM segmentation with point/box prompts.
    SamSegment {
        points: Vec<SamPoint>,
        #[serde(rename = "box", skip_serializing_if = "Option::is_none")]
        box_prompt: Option<[f32; 4]>,
    },

    /// Respond to server ping.
    Pong { timestamp: u64 },
}

/// SAM point prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamPoint {
    pub x: f32,
    pub y: f32,
    /// 1 = foreground, 0 = background
    pub label: i32,
}

/// Error codes organized by category.
///
/// Error codes follow a hierarchical scheme:
/// - 1xxx: Image-related errors
/// - 2xxx: Pyramid-related errors
/// - 3xxx: SAM-related errors
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

    // Pyramid errors (2xxx)
    PyramidNotReady = 2000,
    PyramidBuildFailed = 2001,
    PyramidCorrupted = 2002,
    PyramidStorageFull = 2003,
    PyramidTimeout = 2004,

    // SAM errors (3xxx)
    SamNotEnabled = 3000,
    SamModelNotLoaded = 3001,
    SamEncodeFailed = 3002,
    SamDecodeFailed = 3003,
    SamOutOfMemory = 3004,
    SamInvalidPrompt = 3005,
    SamEmbeddingExpired = 3006,
    SamBusy = 3007,

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
            3000..=3999 => ErrorCategory::Sam,
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
                | Self::SamBusy
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
            Self::SamBusy => 2000,
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
            2000 => Some(Self::PyramidNotReady),
            2001 => Some(Self::PyramidBuildFailed),
            2002 => Some(Self::PyramidCorrupted),
            2003 => Some(Self::PyramidStorageFull),
            2004 => Some(Self::PyramidTimeout),
            3000 => Some(Self::SamNotEnabled),
            3001 => Some(Self::SamModelNotLoaded),
            3002 => Some(Self::SamEncodeFailed),
            3003 => Some(Self::SamDecodeFailed),
            3004 => Some(Self::SamOutOfMemory),
            3005 => Some(Self::SamInvalidPrompt),
            3006 => Some(Self::SamEmbeddingExpired),
            3007 => Some(Self::SamBusy),
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
    Sam,
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
    }

    #[test]
    fn test_error_code_category() {
        assert_eq!(ErrorCode::ImageNotFound.category(), ErrorCategory::Image);
        assert_eq!(
            ErrorCode::PyramidNotReady.category(),
            ErrorCategory::Pyramid
        );
        assert_eq!(ErrorCode::SamNotEnabled.category(), ErrorCategory::Sam);
    }

    #[test]
    fn test_error_code_retryable() {
        assert!(ErrorCode::PyramidNotReady.default_retryable());
        assert!(!ErrorCode::ImageNotFound.default_retryable());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Fatal);
    }
}
