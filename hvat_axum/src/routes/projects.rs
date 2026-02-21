//! Server info and image listing endpoints.
//!
//! The server exposes a single "project" which is the configured data directory.
//! Clients connecting to this server see it as one project.

use std::{io::ErrorKind, path::PathBuf, sync::Arc};

use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::IntoResponse,
    routing::get,
};
use hvat_common::annotation_io::{ExportBundle, ImportBundle};
use serde::Serialize;

use crate::error::Result;
use crate::state::AppState;
use crate::utils::make_url_safe;

const PROJECT_STATE_DIR: &str = ".hvat";
const PROJECT_STATE_FILE: &str = "project_annotations.hvatbundle.json";
const MAX_PROJECT_STATE_BYTES: usize = 64 * 1024 * 1024;

/// Server/project information.
#[derive(Debug, Serialize)]
pub struct ServerInfo {
    /// Project name (from --name or derived from data_dir)
    pub name: String,
    /// Total number of images
    pub image_count: usize,
    /// Server version
    pub version: String,
}

/// Image entry.
#[derive(Debug, Serialize)]
pub struct ImageInfo {
    /// Image ID (URL-safe path)
    pub id: String,
    /// Display name (filename)
    pub name: String,
    /// Relative path within data_dir (for tree structure)
    pub path: String,
    /// Format (e.g., "PNG", "NPY")
    pub format: String,
}

/// Create the router for server info and images.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/info", get(get_info))
        .route("/images", get(list_images))
        .route(
            "/project-state",
            get(get_project_state).put(put_project_state),
        )
}

/// Get server/project information.
async fn get_info(State(state): State<Arc<AppState>>) -> Result<Json<ServerInfo>> {
    let data_dir = &state.config.data_dir;

    // Count all images recursively
    let image_count = count_images(data_dir, &state);

    Ok(Json(ServerInfo {
        name: state.config.project_name.clone(),
        image_count,
        version: env!("CARGO_PKG_VERSION").to_string(),
    }))
}

/// List all images in the data directory (recursive).
async fn list_images(State(state): State<Arc<AppState>>) -> Result<Json<Vec<ImageInfo>>> {
    let data_dir = &state.config.data_dir;

    let mut images = Vec::new();

    if data_dir.exists() {
        collect_images(data_dir, data_dir, &state, &mut images);
    }

    // Sort by path for consistent ordering
    images.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(Json(images))
}

/// Fetch saved project annotations in native HVAT bundle format.
async fn get_project_state(
    State(state): State<Arc<AppState>>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let path = project_state_path(&state);
    let bytes = match tokio::fs::read(&path).await {
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

/// Persist project annotations in native HVAT bundle format.
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

    validate_project_state_payload(&body)?;

    let path = project_state_path(&state);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await.map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to create state directory: {err}"),
            )
        })?;
    }

    let temp_path = path.with_extension("tmp");
    tokio::fs::write(&temp_path, &body).await.map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to write temporary project state: {err}"),
        )
    })?;
    tokio::fs::rename(&temp_path, &path).await.map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to finalize project state save: {err}"),
        )
    })?;

    Ok(StatusCode::NO_CONTENT)
}

/// Count images in a directory (recursive).
fn count_images(dir: &std::path::Path, state: &AppState) -> usize {
    let mut count = 0;

    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                count += count_images(&path, state);
            } else if state.loaders.supports(&path) {
                count += 1;
            }
        }
    }

    count
}

/// Collect images from a directory (recursive).
fn collect_images(
    dir: &std::path::Path,
    base_dir: &std::path::Path,
    state: &AppState,
    images: &mut Vec<ImageInfo>,
) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_images(&path, base_dir, state, images);
            } else if state.loaders.supports(&path) {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                // Path relative to data_dir (for tree structure)
                let relative_path = path
                    .strip_prefix(base_dir)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();

                let format = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("unknown")
                    .to_uppercase();

                // ID is URL-safe version of relative path
                let id = make_url_safe(&relative_path);

                images.push(ImageInfo {
                    id,
                    name,
                    path: relative_path,
                    format,
                });
            }
        }
    }
}

fn project_state_path(state: &AppState) -> PathBuf {
    state
        .config
        .data_dir
        .join(PROJECT_STATE_DIR)
        .join(PROJECT_STATE_FILE)
}

fn validate_project_state_payload(payload: &[u8]) -> std::result::Result<(), (StatusCode, String)> {
    let import_valid = serde_json::from_slice::<ImportBundle>(payload)
        .map(|bundle| !bundle.files.is_empty())
        .unwrap_or(false);
    if import_valid {
        return Ok(());
    }

    let export_valid = serde_json::from_slice::<ExportBundle>(payload)
        .map(|bundle| !bundle.files.is_empty())
        .unwrap_or(false);
    if export_valid {
        return Ok(());
    }

    Err((
        StatusCode::BAD_REQUEST,
        "Invalid native project bundle payload".to_string(),
    ))
}
