//! Image metadata and streaming endpoints.

use std::sync::Arc;

use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State, WebSocketUpgrade},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::get,
};
use hvat_common::pixel_count_u32;
use serde::Serialize;

use crate::error::{Error, Result};
use crate::pyramid::{PyramidStatus, compute_image_hash};
use crate::state::AppState;
use crate::utils::find_image;

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
        .route("/{id}/thumbnail", get(get_thumbnail))
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

/// Get a thumbnail PNG for an image.
///
/// Returns the pre-generated thumbnail (level 0 of pyramid) as a PNG image.
/// If the thumbnail is not ready, returns 404.
async fn get_thumbnail(
    State(state): State<Arc<AppState>>,
    Path(image_id): Path<String>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    // Find the image file
    let image_path = find_image(&state, &image_id)
        .map_err(|e| (StatusCode::NOT_FOUND, format!("Image not found: {}", e)))?;

    // Compute image hash for cache lookup
    let image_hash = compute_image_hash(&image_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Hash error: {}", e),
        )
    })?;

    // Check if pyramid is ready
    let status = state.pyramid_storage.get_status(&image_hash).await;
    if status != PyramidStatus::Ready {
        return Err((StatusCode::NOT_FOUND, "Thumbnail not ready".to_string()));
    }

    // Load pyramid metadata to get thumbnail dimensions
    let metadata = state
        .pyramid_storage
        .load_metadata(&image_hash)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Metadata error: {}", e),
            )
        })?;

    // Use the smallest level (highest index) as the thumbnail
    // Level 0 = full resolution, highest level = smallest/thumbnail
    let level_info = metadata.levels.last().ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "No pyramid levels".to_string(),
        )
    })?;

    // Load the smallest level
    let layers = state
        .pyramid_storage
        .load_level(&image_hash, level_info.level)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Level load error: {}", e),
            )
        })?;

    if layers.is_empty() {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            "No layer data".to_string(),
        ));
    }

    // Use the first layer as the thumbnail (it's RGBA packed)
    // Take the first layer (usually contains first 4 bands packed as RGBA)
    let (_, rgba_data) = &layers[0];

    // Create PNG image
    let width = level_info.width;
    let height = level_info.height;
    let expected_size = pixel_count_u32(width, height).checked_mul(4).unwrap();

    if rgba_data.len() != expected_size {
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "Data size mismatch: got {} expected {}",
                rgba_data.len(),
                expected_size
            ),
        ));
    }

    // Create image from RGBA data and encode as PNG
    let img = image::RgbaImage::from_raw(width, height, rgba_data.clone()).ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to create image from RGBA data".to_string(),
        )
    })?;

    let mut png_data = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut png_data);
    img.write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("PNG encode error: {}", e),
            )
        })?;

    // Return PNG response
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        Body::from(png_data),
    ))
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
