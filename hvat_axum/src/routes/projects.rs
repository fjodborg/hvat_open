//! Server info, image listing, and project-state endpoints.
//!
//! The server exposes a single "project" which is the configured data directory.
//! Clients connecting to this server see it as one project.

use std::{io::ErrorKind, path::PathBuf, sync::Arc};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{Multipart, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::IntoResponse,
    routing::{get, post},
};
use hvat_common::annotation_io::{ExportBundle, ImportBundle};
use serde::Serialize;
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

use crate::error::Result;
use crate::state::AppState;
use crate::utils::make_url_safe;

const PROJECT_STATE_DIR: &str = ".hvat";
const PROJECT_STATE_FILE: &str = "project_annotations.hvatbundle.json";
const MAX_PROJECT_STATE_BYTES: usize = 64 * 1024 * 1024;

const UPLOAD_FIELD_TARGET_ROOT: &str = "target_root";
const UPLOAD_FIELD_FILES: &str = "files";
const MAX_UPLOAD_FILES: usize = 2_000;
const MAX_UPLOAD_FILE_BYTES: usize = 512 * 1024 * 1024;
const MAX_UPLOAD_TOTAL_BYTES: usize = 2 * 1024 * 1024 * 1024;

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
        .route("/images/upload", post(upload_images))
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

/// Upload one or more images into the server data directory.
///
/// Multipart fields:
/// - `target_root` (optional text): destination folder relative to data root
/// - `files` (one or more file fields): filename can include nested relative path
async fn upload_images(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let mut target_root = String::new();
    let mut total_bytes = 0usize;
    let mut file_count = 0usize;
    let mut uploaded_files = Vec::new();
    let mut seen_file_fields = false;

    while let Some(mut field) = multipart.next_field().await.map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid multipart payload: {err}"),
        )
    })? {
        match field.name() {
            Some(UPLOAD_FIELD_TARGET_ROOT) => {
                if seen_file_fields {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        "'target_root' must appear before file fields".to_string(),
                    ));
                }
                let raw_target = field.text().await.map_err(|err| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Failed to read target_root: {err}"),
                    )
                })?;

                // Validate and normalize once; store as normalized slash path.
                target_root = normalize_upload_path(&raw_target, true)?;
            }
            Some(UPLOAD_FIELD_FILES) => {
                seen_file_fields = true;
                file_count += 1;
                if file_count > MAX_UPLOAD_FILES {
                    return Err((
                        StatusCode::PAYLOAD_TOO_LARGE,
                        format!("Upload exceeds maximum of {} files", MAX_UPLOAD_FILES),
                    ));
                }

                let client_file_name = field.file_name().ok_or_else(|| {
                    (
                        StatusCode::BAD_REQUEST,
                        "Each uploaded file must include a filename".to_string(),
                    )
                })?;
                let relative_path = build_upload_relative_path(&target_root, client_file_name)?;
                let destination = state.config.data_dir.join(&relative_path);

                if !state.loaders.supports(&destination) {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!("Unsupported file type: {}", relative_path),
                    ));
                }

                let parent = destination.parent().ok_or_else(|| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Invalid destination path: {}", relative_path),
                    )
                })?;

                tokio::fs::create_dir_all(parent).await.map_err(|err| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to create upload directory: {err}"),
                    )
                })?;

                let temp_name = format!(
                    ".{}.uploading-{}",
                    destination
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("upload"),
                    Uuid::new_v4(),
                );
                let temp_path = parent.join(temp_name);

                let mut output = tokio::fs::File::create(&temp_path).await.map_err(|err| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to create temp file '{}': {err}", relative_path),
                    )
                })?;

                let mut file_bytes = 0usize;
                while let Some(chunk) = field.chunk().await.map_err(|err| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Failed to read upload chunk for '{}': {err}", relative_path),
                    )
                })? {
                    file_bytes = file_bytes.saturating_add(chunk.len());
                    total_bytes = total_bytes.saturating_add(chunk.len());

                    if file_bytes > MAX_UPLOAD_FILE_BYTES {
                        let _ = tokio::fs::remove_file(&temp_path).await;
                        return Err((
                            StatusCode::PAYLOAD_TOO_LARGE,
                            format!(
                                "File '{}' exceeds per-file limit ({} bytes)",
                                relative_path, MAX_UPLOAD_FILE_BYTES
                            ),
                        ));
                    }
                    if total_bytes > MAX_UPLOAD_TOTAL_BYTES {
                        let _ = tokio::fs::remove_file(&temp_path).await;
                        return Err((
                            StatusCode::PAYLOAD_TOO_LARGE,
                            format!(
                                "Upload exceeds total limit ({} bytes)",
                                MAX_UPLOAD_TOTAL_BYTES
                            ),
                        ));
                    }

                    output.write_all(&chunk).await.map_err(|err| {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed writing upload '{}': {err}", relative_path),
                        )
                    })?;
                }

                if file_bytes == 0 {
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!("Uploaded file '{}' is empty", relative_path),
                    ));
                }

                output.flush().await.map_err(|err| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to flush upload '{}': {err}", relative_path),
                    )
                })?;
                drop(output);

                tokio::fs::rename(&temp_path, &destination).await.map_err(|err| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to finalize upload '{}': {err}", relative_path),
                    )
                })?;

                uploaded_files.push(relative_path);
            }
            _ => {
                // Ignore unknown multipart fields.
            }
        }
    }

    if uploaded_files.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "No files were uploaded".to_string(),
        ));
    }

    Ok((
        StatusCode::CREATED,
        Json(UploadImagesResponse {
            uploaded: uploaded_files.len(),
            files: uploaded_files,
        }),
    ))
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

fn build_upload_relative_path(
    target_root: &str,
    client_file_name: &str,
) -> std::result::Result<String, (StatusCode, String)> {
    let mut segments = split_upload_segments(target_root, true)?;
    segments.extend(split_upload_segments(client_file_name, false)?);

    if segments.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Upload path resolved to empty".to_string(),
        ));
    }

    Ok(segments.join("/"))
}

fn normalize_upload_path(
    raw: &str,
    allow_empty: bool,
) -> std::result::Result<String, (StatusCode, String)> {
    let segments = split_upload_segments(raw, allow_empty)?;
    Ok(segments.join("/"))
}

fn split_upload_segments(
    raw: &str,
    allow_empty: bool,
) -> std::result::Result<Vec<String>, (StatusCode, String)> {
    let normalized = raw.trim().replace('\\', "/");

    if normalized.is_empty() {
        if allow_empty {
            return Ok(Vec::new());
        }
        return Err((
            StatusCode::BAD_REQUEST,
            "Upload path is empty".to_string(),
        ));
    }

    if normalized.starts_with('/') {
        return Err((
            StatusCode::BAD_REQUEST,
            "Absolute upload paths are not allowed".to_string(),
        ));
    }

    let mut segments = Vec::new();
    for segment in normalized.split('/') {
        let part = segment.trim();
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            return Err((
                StatusCode::BAD_REQUEST,
                "Upload path contains parent traversal ('..')".to_string(),
            ));
        }
        if part.contains('\0') {
            return Err((
                StatusCode::BAD_REQUEST,
                "Upload path contains NUL byte".to_string(),
            ));
        }
        if part.contains(':') {
            return Err((
                StatusCode::BAD_REQUEST,
                "Upload path contains invalid ':' character".to_string(),
            ));
        }
        segments.push(part.to_string());
    }

    if !allow_empty && segments.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Upload path is empty".to_string(),
        ));
    }

    Ok(segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_upload_segments_rejects_parent_traversal() {
        let err = split_upload_segments("../secret.png", false).expect_err("must fail");
        assert_eq!(err.0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn split_upload_segments_normalizes_slashes() {
        let segments = split_upload_segments(r"folder\\nested/file.png", false)
            .expect("should normalize");
        assert_eq!(segments, vec!["folder", "nested", "file.png"]);
    }

    #[test]
    fn build_upload_relative_path_with_target_root() {
        let path = build_upload_relative_path("incoming", "sub/a.png").expect("should build");
        assert_eq!(path, "incoming/sub/a.png");
    }

    #[test]
    fn normalize_upload_path_allows_empty_root() {
        let path = normalize_upload_path("", true).expect("should allow empty");
        assert!(path.is_empty());
    }
}
