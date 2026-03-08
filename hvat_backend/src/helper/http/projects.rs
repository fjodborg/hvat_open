//! Shared server info, image listing, upload, and project-state endpoints.

use std::io::ErrorKind;
use std::path::Path;
use std::sync::Arc;

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, Multipart, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use serde::Serialize;

use crate::helper::catalog::{ImageInfo, count_supported_images, list_supported_images};
use crate::helper::error::Result;
use crate::helper::loaders::ImageLoaderRegistry;
use crate::helper::project_state::{
    MAX_PROJECT_STATE_BYTES, read_project_state, validate_project_state_payload,
    write_project_state,
};
use crate::helper::upload::save_uploaded_images;

/// Minimal state contract needed by shared project routes.
pub trait ProjectRoutesState: Send + Sync + 'static {
    fn project_name(&self) -> &str;
    fn data_dir(&self) -> &Path;
    fn loaders(&self) -> &ImageLoaderRegistry;
}

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

/// Build shared `/api` routes for project metadata/state operations.
pub fn router<S>() -> Router<Arc<S>>
where
    S: ProjectRoutesState,
{
    Router::new()
        .route("/info", get(get_info::<S>))
        .route("/images", get(list_images::<S>))
        .route(
            "/images/upload",
            post(upload_images::<S>).layer(DefaultBodyLimit::disable()),
        )
        .route(
            "/project-state",
            get(get_project_state::<S>).put(put_project_state::<S>),
        )
}

async fn get_info<S>(State(state): State<Arc<S>>) -> Result<Json<ServerInfo>>
where
    S: ProjectRoutesState,
{
    let image_count =
        count_supported_images(state.data_dir(), |path| state.loaders().supports(path));

    Ok(Json(ServerInfo {
        name: state.project_name().to_string(),
        image_count,
        version: env!("CARGO_PKG_VERSION").to_string(),
    }))
}

async fn list_images<S>(State(state): State<Arc<S>>) -> Result<Json<Vec<ImageInfo>>>
where
    S: ProjectRoutesState,
{
    let images = list_supported_images(state.data_dir(), |path| state.loaders().supports(path));
    Ok(Json(images))
}

async fn upload_images<S>(
    State(state): State<Arc<S>>,
    multipart: Multipart,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)>
where
    S: ProjectRoutesState,
{
    let uploaded = save_uploaded_images(state.data_dir(), multipart, |path| {
        state.loaders().supports(path)
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

async fn get_project_state<S>(
    State(state): State<Arc<S>>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)>
where
    S: ProjectRoutesState,
{
    let bytes = match read_project_state(state.data_dir()).await {
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

async fn put_project_state<S>(
    State(state): State<Arc<S>>,
    body: Bytes,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)>
where
    S: ProjectRoutesState,
{
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

    write_project_state(state.data_dir(), &body)
        .await
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to save project state: {err}"),
            )
        })?;

    Ok(StatusCode::NO_CONTENT)
}
