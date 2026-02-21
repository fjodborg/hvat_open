//! Error model for annotation import/export.

use thiserror::Error;

use super::types::FormatId;

/// Hard-fail conditions for annotation I/O.
#[derive(Debug, Error)]
pub enum AnnotationIoError {
    #[error("Unsupported format: {0:?}")]
    UnsupportedFormat(FormatId),

    #[error("Invalid bundle: {0}")]
    InvalidBundle(String),

    #[error("Parse error for {format:?}: {message}")]
    ParseError { format: FormatId, message: String },

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Strict policy violation: {0}")]
    StrictPolicyViolation(String),

    #[error("Missing image context: {0}")]
    MissingImageContext(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl AnnotationIoError {
    /// Build a parse error with a typed format tag.
    pub fn parse(format: FormatId, message: impl Into<String>) -> Self {
        Self::ParseError {
            format,
            message: message.into(),
        }
    }
}
