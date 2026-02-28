use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Result type alias using shared backend error type.
pub type Result<T> = std::result::Result<T, Error>;

/// Shared backend error type.
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

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            Self::ImageNotFound(_) | Self::ProjectNotFound(_) => {
                (StatusCode::NOT_FOUND, self.to_string())
            }
            Self::UnsupportedFormat(_) | Self::InvalidImageData(_) => {
                (StatusCode::BAD_REQUEST, self.to_string())
            }
            Self::PyramidNotReady(_) => (StatusCode::ACCEPTED, self.to_string()),
            Self::Io(_) | Self::Image(_) | Self::Internal(_) => {
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
