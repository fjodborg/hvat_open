//! Server info, image listing, and project-state endpoints.
//!
//! The server exposes a single "project" which is the configured data directory.
//! Clients connecting to this server see it as one project.

use std::{io::ErrorKind, sync::Arc};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use crate::common::catalog::{ImageInfo, count_supported_images, list_supported_images};
use crate::common::project_state::{
    MAX_PROJECT_STATE_BYTES, read_project_state, validate_project_state_payload,
    write_project_state,
};
use crate::common::upload::save_uploaded_images;
use serde::Serialize;

use crate::Result;
use crate::state::AppState;

/// Server/project information.
#[derive(Debug, Serialize)]
pub struct ServerInfo {
    pub name: String,
    pub image_count: usize,
    pub version: String,
}

#[derive(Debug, Serialize)]
struct UploadImagesResponse {
    uploaded: usize,
    files: Vec<String>,
}

/// Create the router for server info and images.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/info", get(get_info))
        .route("/images", get(list_images))
        .route(
            "/images/upload",
            post(upload_images).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/project-state",
            get(get_project_state).put(put_project_state),
        )
}

async fn get_info(State(state): State<Arc<AppState>>) -> Result<Json<ServerInfo>> {
    let image_count =
        count_supported_images(&state.config.data_dir, |path| state.loaders.supports(path));

    Ok(Json(ServerInfo {
        name: state.config.project_name.clone(),
        image_count,
        version: env!("CARGO_PKG_VERSION").to_string(),
    }))
}

async fn list_images(State(state): State<Arc<AppState>>) -> Result<Json<Vec<ImageInfo>>> {
    let images = list_supported_images(&state.config.data_dir, |path| state.loaders.supports(path));
    Ok(Json(images))
}

async fn upload_images(
    State(state): State<Arc<AppState>>,
    multipart: Multipart,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let uploaded = save_uploaded_images(&state.config.data_dir, multipart, |path| {
        state.loaders.supports(path)
    })
    .await?;

    Ok((
        StatusCode::CREATED,
        Json(UploadImagesResponse {
            uploaded: uploaded.uploaded_count(),
            files: uploaded.files,
        }),
    ))
}

async fn get_project_state(
    State(state): State<Arc<AppState>>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let bytes = match read_project_state(&state.config.data_dir).await {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == ErrorKind::NotFound => {
            return Err((StatusCode::NOT_FOUND, "No saved project state".to_string()));
        }
        Err(err) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to read saved project state: {err}"),
            ));
        }
    };

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    Ok((StatusCode::OK, headers, bytes))
}

async fn put_project_state(
    State(state): State<Arc<AppState>>,
    body: Bytes,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    if body.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Request body is empty".to_string()));
    }
    if body.len() > MAX_PROJECT_STATE_BYTES {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "Project state payload exceeds limit of {} bytes",
                MAX_PROJECT_STATE_BYTES
            ),
        ));
    }

    if !validate_project_state_payload(&body) {
        return Err((
            StatusCode::BAD_REQUEST,
            "Invalid native project bundle payload".to_string(),
        ));
    }

    write_project_state(&state.config.data_dir, &body)
        .await
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to save project state: {err}"),
            )
        })?;

    Ok(StatusCode::NO_CONTENT)
}
