//! Image metadata and streaming endpoints.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State, WebSocketUpgrade},
    response::Response,
    routing::get,
};
use serde::Serialize;

use crate::error::{Error, Result};
use crate::pyramid::{PyramidStatus, compute_image_hash};
use crate::state::AppState;

use super::websocket::handle_websocket;

/// Image metadata response.
#[derive(Debug, Serialize)]
pub struct ImageMetadataResponse {
    /// Image ID
    pub id: String,
    /// Filename
    pub name: String,
    /// File format
    pub format: String,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Number of spectral bands
    pub bands: usize,
    /// Pyramid information
    pub pyramid: PyramidInfo,
}

/// Pyramid status and level information.
#[derive(Debug, Serialize)]
pub struct PyramidInfo {
    /// Pyramid status: "ready", "building", "pending"
    pub status: String,
    /// Available pyramid levels
    pub levels: Vec<PyramidLevel>,
}

/// Information about a single pyramid level.
#[derive(Debug, Serialize)]
pub struct PyramidLevel {
    /// Level index (0 = full resolution)
    pub level: u32,
    /// Width at this level
    pub width: u32,
    /// Height at this level
    pub height: u32,
}

/// Create the images router.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/{id}/meta", get(get_metadata))
        .route("/{id}/stream", get(stream_handler))
}

/// Get image metadata.
async fn get_metadata(
    State(state): State<Arc<AppState>>,
    Path(image_id): Path<String>,
) -> Result<Json<ImageMetadataResponse>> {
    // Find the image file
    let image_path = find_image(&state, &image_id)?;

    // Find a loader for this file
    let loader = state
        .loaders
        .find_loader(&image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_id.clone()))?;

    // Load metadata
    let meta = loader.load_metadata(&image_path).await?;

    // Check pyramid cache status
    let image_hash = compute_image_hash(&image_path)?;
    let pyramid_status = state.pyramid_storage.get_status(&image_hash).await;

    // Get pyramid levels - from cache if available, otherwise calculate
    let (status_str, levels) = match pyramid_status {
        PyramidStatus::Ready => {
            // Load actual level info from cached metadata
            match state.pyramid_storage.load_metadata(&image_hash).await {
                Ok(pyramid_meta) => {
                    let levels = pyramid_meta
                        .levels
                        .iter()
                        .map(|l| PyramidLevel {
                            level: l.level,
                            width: l.width,
                            height: l.height,
                        })
                        .collect();
                    ("ready".to_string(), levels)
                }
                Err(_) => {
                    // Fallback to calculated levels
                    (
                        "pending".to_string(),
                        calculate_pyramid_levels(meta.width, meta.height),
                    )
                }
            }
        }
        PyramidStatus::Building => (
            "building".to_string(),
            calculate_pyramid_levels(meta.width, meta.height),
        ),
        PyramidStatus::Pending => (
            "pending".to_string(),
            calculate_pyramid_levels(meta.width, meta.height),
        ),
        PyramidStatus::Failed => (
            "failed".to_string(),
            calculate_pyramid_levels(meta.width, meta.height),
        ),
    };

    Ok(Json(ImageMetadataResponse {
        id: image_id,
        name: meta.filename,
        format: meta.format,
        width: meta.width,
        height: meta.height,
        bands: meta.num_bands,
        pyramid: PyramidInfo {
            status: status_str,
            levels,
        },
    }))
}

/// WebSocket upgrade handler for streaming.
async fn stream_handler(
    State(state): State<Arc<AppState>>,
    Path(image_id): Path<String>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| handle_websocket(socket, state, image_id))
}

/// Find an image file by ID.
fn find_image(state: &AppState, image_id: &str) -> Result<std::path::PathBuf> {
    // The image_id is URL-safe, we need to search for the actual file
    let data_dir = &state.config.data_dir;

    // Search recursively in data_dir
    if let Some(path) = search_for_image(data_dir, image_id, state) {
        return Ok(path);
    }

    Err(Error::ImageNotFound(image_id.to_string()))
}

/// Recursively search for an image matching the ID.
fn search_for_image(
    dir: &std::path::Path,
    image_id: &str,
    state: &AppState,
) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = search_for_image(&path, image_id, state) {
                return Some(found);
            }
        } else if state.loaders.supports(&path) {
            // Check if this file matches the image_id
            let relative_path = path
                .strip_prefix(&state.config.data_dir)
                .unwrap_or(&path)
                .to_string_lossy();

            let file_id = make_url_safe(&relative_path);
            if file_id == image_id {
                return Some(path);
            }
        }
    }

    None
}

/// Make a string URL-safe.
fn make_url_safe(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Calculate pyramid levels for an image.
fn calculate_pyramid_levels(width: u32, height: u32) -> Vec<PyramidLevel> {
    let mut levels = Vec::new();
    let mut w = width;
    let mut h = height;
    let mut level = 0u32;

    // Level 0 is full resolution
    levels.push(PyramidLevel {
        level,
        width: w,
        height: h,
    });

    // Generate levels until we reach minimum size (256x256 or smaller)
    while w > 256 || h > 256 {
        w = (w + 1) / 2; // Round up division
        h = (h + 1) / 2;
        level += 1;

        levels.push(PyramidLevel {
            level,
            width: w,
            height: h,
        });
    }

    // Reverse so highest level (smallest) is first
    levels.reverse();

    // Re-number levels (0 = thumbnail, highest = full res)
    for (i, l) in levels.iter_mut().enumerate() {
        l.level = i as u32;
    }

    levels
}
