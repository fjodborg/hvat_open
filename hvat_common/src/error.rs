//! Error types for WebSocket protocol.
//!
//! Provides structured error information that can be serialized to the
//! binary protocol format and transmitted over WebSocket connections.

use serde::{Deserialize, Serialize};

use crate::protocol::{ErrorCode, ServerMessageType, Severity};

/// Full error information for protocol transmission.
///
/// This structure contains all information needed to properly display
/// and handle errors on the client side, including retry logic.
#[derive(Debug, Clone)]
pub struct ProtocolError {
    /// Error code from the hierarchy
    pub code: ErrorCode,

    /// Severity level
    pub severity: Severity,

    /// Whether this error can be retried
    pub retryable: bool,

    /// Suggested retry delay in milliseconds
    pub retry_after_ms: u32,

    /// Additional structured context
    pub context: ErrorContext,

    /// Human-readable error message
    pub message: String,
}

impl ProtocolError {
    /// Create a non-retryable error.
    pub fn fatal(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            severity: Severity::Fatal,
            retryable: false,
            retry_after_ms: 0,
            context: ErrorContext::None,
            message: message.into(),
        }
    }

    /// Create an error (connection stays open).
    pub fn error(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            severity: Severity::Error,
            retryable: false,
            retry_after_ms: 0,
            context: ErrorContext::None,
            message: message.into(),
        }
    }

    /// Create a retryable error.
    pub fn retryable(code: ErrorCode, message: impl Into<String>, retry_after_ms: u32) -> Self {
        Self {
            code,
            severity: Severity::Error,
            retryable: true,
            retry_after_ms,
            context: ErrorContext::None,
            message: message.into(),
        }
    }

    /// Create a warning.
    pub fn warning(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            severity: Severity::Warning,
            retryable: false,
            retry_after_ms: 0,
            context: ErrorContext::None,
            message: message.into(),
        }
    }

    /// Add context to the error.
    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = context;
        self
    }

    /// Encode to binary protocol format.
    ///
    /// Format:
    /// ```text
    /// [version:u8][type:u8][code:u16][severity:u8][retryable:u8][retry_after:u32]
    /// [context_len:u16][msg_len:u16][context:utf8][message:utf8]
    /// ```
    pub fn encode(&self, version: u8) -> Vec<u8> {
        let context_json = match &self.context {
            ErrorContext::None => String::new(),
            ctx => serde_json::to_string(ctx).unwrap_or_default(),
        };

        let context_bytes = context_json.as_bytes();
        let message_bytes = self.message.as_bytes();

        let mut buf = Vec::with_capacity(
            2 + 2 + 1 + 1 + 4 + 2 + 2 + context_bytes.len() + message_bytes.len(),
        );

        buf.push(version);
        buf.push(ServerMessageType::Error.to_byte());
        buf.extend_from_slice(&(self.code as u16).to_le_bytes());
        buf.push(self.severity.to_byte());
        buf.push(self.retryable as u8);
        buf.extend_from_slice(&self.retry_after_ms.to_le_bytes());
        buf.extend_from_slice(&(context_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(&(message_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(context_bytes);
        buf.extend_from_slice(message_bytes);

        buf
    }

    /// Decode from binary protocol format.
    ///
    /// Expects data to start AFTER the version and type bytes.
    pub fn decode(data: &[u8]) -> Option<Self> {
        use crate::BinaryReader;

        let mut reader = BinaryReader::new(data);

        let code = reader.read_u16()?;
        let code = ErrorCode::from_u16(code)?;

        let severity = Severity::from_byte(reader.read_u8()?)?;
        let retryable = reader.read_u8()? != 0;
        let retry_after_ms = reader.read_u32()?;
        let context_len = reader.read_u16()? as usize;
        let message_len = reader.read_u16()? as usize;

        let context = if context_len > 0 {
            let context_json = reader.read_str(context_len)?;
            serde_json::from_str(context_json).unwrap_or(ErrorContext::None)
        } else {
            ErrorContext::None
        };

        let message = reader
            .read_str_lossy(message_len)
            .unwrap_or_else(String::new);

        Some(Self {
            code,
            severity,
            retryable,
            retry_after_ms,
            context,
            message,
        })
    }
}

/// Structured error context for additional information.
///
/// This is serialized as JSON and included in the error message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ErrorContext {
    /// No additional context
    None,

    /// Image-related context
    Image {
        image_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        level: Option<u8>,
    },

    /// Pyramid build progress context
    Pyramid {
        image_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        progress: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        eta_seconds: Option<u32>,
    },

    /// SAM operation context
    Sam {
        image_id: String,
        operation: String, // "embed" or "segment"
        #[serde(skip_serializing_if = "Option::is_none")]
        required_vram_mb: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        available_vram_mb: Option<u32>,
    },

    /// Protocol version mismatch context
    Protocol {
        expected_version: u8,
        received_version: u8,
    },

    /// Connection error context
    Connection { server_url: String, attempt: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::PROTOCOL_VERSION;

    #[test]
    fn test_error_encode_decode_roundtrip() {
        let error =
            ProtocolError::retryable(ErrorCode::PyramidNotReady, "Pyramid is building", 5000)
                .with_context(ErrorContext::Pyramid {
                    image_id: "test.png".to_string(),
                    progress: Some(0.5),
                    eta_seconds: Some(10),
                });

        let encoded = error.encode(PROTOCOL_VERSION);
        let decoded = ProtocolError::decode(&encoded[2..]).unwrap();

        assert_eq!(decoded.code as u16, error.code as u16);
        assert_eq!(decoded.severity as u8, error.severity as u8);
        assert_eq!(decoded.retryable, error.retryable);
        assert_eq!(decoded.retry_after_ms, error.retry_after_ms);
        assert_eq!(decoded.message, error.message);
    }

    #[test]
    fn test_error_without_context() {
        let error = ProtocolError::error(ErrorCode::ImageNotFound, "Image not found");

        let encoded = error.encode(PROTOCOL_VERSION);
        let decoded = ProtocolError::decode(&encoded[2..]).unwrap();

        assert!(matches!(decoded.context, ErrorContext::None));
    }

    #[test]
    fn test_fatal_error() {
        let error = ProtocolError::fatal(ErrorCode::ProtocolMismatch, "Incompatible version");

        assert_eq!(error.severity, Severity::Fatal);
        assert!(!error.retryable);
    }
}
