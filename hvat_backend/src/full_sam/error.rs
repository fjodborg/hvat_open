//! Error types for the HVAT server.

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Result type alias using our Error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Server error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Image not found: {0}")]
    ImageNotFound(String),

    #[error("Project not found: {0}")]
    ProjectNotFound(String),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Invalid image data: {0}")]
    InvalidImageData(String),

    #[error("Pyramid not ready: {0}")]
    PyramidNotReady(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<crate::common::error::Error> for Error {
    fn from(value: crate::common::error::Error) -> Self {
        match value {
            crate::common::error::Error::ImageNotFound(msg) => Self::ImageNotFound(msg),
            crate::common::error::Error::ProjectNotFound(msg) => Self::ProjectNotFound(msg),
            crate::common::error::Error::UnsupportedFormat(msg) => Self::UnsupportedFormat(msg),
            crate::common::error::Error::InvalidImageData(msg) => Self::InvalidImageData(msg),
            crate::common::error::Error::PyramidNotReady(msg) => Self::PyramidNotReady(msg),
            crate::common::error::Error::Io(err) => Self::Io(err),
            crate::common::error::Error::Image(err) => Self::Image(err),
            crate::common::error::Error::Internal(msg) => Self::Internal(msg),
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            Error::ImageNotFound(_) | Error::ProjectNotFound(_) => {
                (StatusCode::NOT_FOUND, self.to_string())
            }
            Error::UnsupportedFormat(_) | Error::InvalidImageData(_) => {
                (StatusCode::BAD_REQUEST, self.to_string())
            }
            Error::PyramidNotReady(_) => (StatusCode::ACCEPTED, self.to_string()),
            Error::Io(_) | Error::Image(_) | Error::Internal(_) => {
                (StatusCode::INTERNAL_SERVER_ERROR, self.to_string())
            }
        };

        let body = Json(json!({
            "error": message,
            "status": status.as_u16(),
        }));

        (status, body).into_response()
    }
}
