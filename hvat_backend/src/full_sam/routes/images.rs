//! Image metadata and streaming endpoints.

use std::sync::Arc;

use crate::common::archive::{
    ArchiveError, DownloadPartInfo, DownloadPlanResponse, DownloadQuery,
    build_chunked_download_plan, build_images_zip, build_project_archive, resolve_part_size_bytes,
};
use axum::{
    Json, Router,
    body::Body,
    extract::{Path, Query, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::IntoResponse,
    routing::get,
};
use hvat_common::pixel_count_u32;
use serde::Serialize;

use crate::error::{Error, Result};
use crate::pyramid::{PyramidStatus, compute_image_hash};
use crate::state::AppState;
use crate::utils::find_image;

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
        .route("/download", get(download_all_images))
        .route("/download/plan", get(download_images_plan))
        .route("/download/part/{part_index}", get(download_images_part))
        .route("/{id}/meta", get(get_metadata))
        // Legacy /{id}/stream route removed - use /api/ws multiplexed endpoint instead
        .route("/{id}/thumbnail", get(get_thumbnail))
        .route("/{id}/raw", get(get_raw_image))
}

/// Get image metadata.
async fn get_metadata(
    State(state): State<Arc<AppState>>,
    Path(image_id): Path<String>,
) -> Result<Json<ImageMetadataResponse>> {
    let image_path = find_image(&state, &image_id)?;

    let loader = state
        .loaders
        .find_loader(&image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_id.clone()))?;

    let meta = loader.load_metadata(&image_path).await?;

    let image_hash = compute_image_hash(&image_path)?;
    let pyramid_status = state.pyramid_storage.get_status(&image_hash).await;

    let (status_str, levels) = match pyramid_status {
        PyramidStatus::Ready => match state.pyramid_storage.load_metadata(&image_hash).await {
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
            Err(_) => (
                "pending".to_string(),
                calculate_pyramid_levels(meta.width, meta.height),
            ),
        },
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

/// Get a thumbnail PNG for an image.
///
/// Returns the pre-generated thumbnail (smallest pyramid level) as PNG.
async fn get_thumbnail(
    State(state): State<Arc<AppState>>,
    Path(image_id): Path<String>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let image_path = find_image(&state, &image_id)
        .map_err(|e| (StatusCode::NOT_FOUND, format!("Image not found: {}", e)))?;

    let image_hash = compute_image_hash(&image_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Hash error: {}", e),
        )
    })?;

    const THUMBNAIL_MAX_DIM: u32 = 256;

    let status = state.pyramid_storage.get_status(&image_hash).await;
    let png_data = if status == PyramidStatus::Ready {
        match load_pyramid_thumbnail_png(&state, &image_hash).await {
            Ok(png) => png,
            Err(err) => {
                log::warn!(
                    "Failed to read pyramid thumbnail for '{}': {}. Falling back to on-demand thumbnail generation.",
                    image_id,
                    err
                );
                crate::common::streaming::generate_thumbnail(
                    &state.config.data_dir,
                    &state.loaders,
                    &image_id,
                    THUMBNAIL_MAX_DIM,
                )
                .await
                .map_err(|e| (StatusCode::NOT_FOUND, format!("Thumbnail failed: {e}")))?
            }
        }
    } else {
        crate::common::streaming::generate_thumbnail(
            &state.config.data_dir,
            &state.loaders,
            &image_id,
            THUMBNAIL_MAX_DIM,
        )
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, format!("Thumbnail failed: {e}")))?
    };

    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        Body::from(png_data),
    ))
}

async fn load_pyramid_thumbnail_png(
    state: &AppState,
    image_hash: &str,
) -> std::result::Result<Vec<u8>, String> {
    let metadata = state
        .pyramid_storage
        .load_metadata(image_hash)
        .await
        .map_err(|e| format!("Metadata error: {}", e))?;

    let level_info = metadata
        .levels
        .last()
        .ok_or_else(|| "No pyramid levels".to_string())?;

    let layers = state
        .pyramid_storage
        .load_level(image_hash, level_info.level)
        .await
        .map_err(|e| format!("Level load error: {}", e))?;

    if layers.is_empty() {
        return Err("No layer data".to_string());
    }

    let (_, rgba_data) = &layers[0];
    let width = level_info.width;
    let height = level_info.height;
    let expected_size = pixel_count_u32(width, height)
        .checked_mul(4)
        .ok_or_else(|| format!("Thumbnail dimensions too large: {}x{}", width, height))?;

    if rgba_data.len() != expected_size {
        return Err(format!(
            "Data size mismatch: got {} expected {}",
            rgba_data.len(),
            expected_size
        ));
    }

    let img = image::RgbaImage::from_raw(width, height, rgba_data.clone())
        .ok_or_else(|| "Failed to create image from RGBA data".to_string())?;

    let mut png_data = Vec::new();
    let mut cursor = std::io::Cursor::new(&mut png_data);
    img.write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|e| format!("PNG encode error: {}", e))?;

    Ok(png_data)
}

/// Download the original image bytes.
async fn get_raw_image(
    State(state): State<Arc<AppState>>,
    Path(image_id): Path<String>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let image_path = find_image(&state, &image_id)
        .map_err(|e| (StatusCode::NOT_FOUND, format!("Image not found: {}", e)))?;

    let bytes = tokio::fs::read(&image_path).await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Read error: {}", e),
        )
    })?;

    let filename = image_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("image.bin")
        .to_string();
    let content_disposition = format!("attachment; filename=\"{}\"", filename.replace('"', "_"));
    let disposition = HeaderValue::from_str(&content_disposition).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Invalid header value: {}", e),
        )
    })?;

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/octet-stream"),
    );
    headers.insert(header::CONTENT_DISPOSITION, disposition);

    Ok((StatusCode::OK, headers, Body::from(bytes)))
}

/// Download all project images as a ZIP archive.
async fn download_all_images(
    State(state): State<Arc<AppState>>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let data_dir = state.config.data_dir.clone();
    let project_name = state.config.project_name.clone();
    let loaders = state.loaders.clone();

    let archive_result = tokio::task::spawn_blocking(move || {
        build_project_archive(&data_dir, &project_name, |path| loaders.supports(path))
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to build image archive: {e}"),
        )
    })?;

    let archive = archive_result.map_err(map_archive_error)?;

    let content_disposition = format!("attachment; filename=\"{}\"", archive.filename);
    let disposition = HeaderValue::from_str(&content_disposition).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Invalid header value: {}", e),
        )
    })?;

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/zip"),
    );
    headers.insert(header::CONTENT_DISPOSITION, disposition);

    Ok((StatusCode::OK, headers, Body::from(archive.bytes)))
}

/// Build a chunked download plan for project images.
async fn download_images_plan(
    State(state): State<Arc<AppState>>,
    Query(query): Query<DownloadQuery>,
) -> std::result::Result<Json<DownloadPlanResponse>, (StatusCode, String)> {
    let part_size_bytes = resolve_part_size_bytes(query.part_size_mb);

    let planned = build_chunked_download_plan(
        &state.config.data_dir,
        &state.config.project_name,
        part_size_bytes,
        |path| state.loaders.supports(path),
    )
    .map_err(map_archive_error)?;

    let parts = planned
        .parts
        .iter()
        .map(|part| DownloadPartInfo {
            index: part.index,
            filename: part.filename.clone(),
            file_count: part.entries.len(),
            total_bytes: part.total_bytes,
        })
        .collect::<Vec<_>>();

    Ok(Json(DownloadPlanResponse {
        total_files: planned.total_files,
        total_bytes: planned.total_bytes,
        part_size_bytes: planned.part_size_bytes,
        part_count: parts.len(),
        parts,
    }))
}

/// Download one ZIP part from the chunked plan.
async fn download_images_part(
    State(state): State<Arc<AppState>>,
    Path(part_index): Path<usize>,
    Query(query): Query<DownloadQuery>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let part_size_bytes = resolve_part_size_bytes(query.part_size_mb);

    let planned = build_chunked_download_plan(
        &state.config.data_dir,
        &state.config.project_name,
        part_size_bytes,
        |path| state.loaders.supports(path),
    )
    .map_err(map_archive_error)?;

    let Some(part) = planned.parts.get(part_index).cloned() else {
        return Err((
            StatusCode::NOT_FOUND,
            format!(
                "Invalid download part index {} (available parts: {})",
                part_index,
                planned.parts.len()
            ),
        ));
    };

    let zip_bytes = tokio::task::spawn_blocking(move || build_images_zip(&part.entries))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to build image archive part: {e}"),
            )
        })?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let content_disposition = format!("attachment; filename=\"{}\"", part.filename);
    let disposition = HeaderValue::from_str(&content_disposition).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Invalid header value: {}", e),
        )
    })?;

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/zip"),
    );
    headers.insert(header::CONTENT_DISPOSITION, disposition);

    Ok((StatusCode::OK, headers, Body::from(zip_bytes)))
}

fn map_archive_error(err: ArchiveError) -> (StatusCode, String) {
    match err {
        ArchiveError::NoImages => (
            StatusCode::NOT_FOUND,
            "No project images available for download".to_string(),
        ),
        ArchiveError::BuildFailed(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
    }
}

/// Calculate pyramid levels for an image.
fn calculate_pyramid_levels(width: u32, height: u32) -> Vec<PyramidLevel> {
    let mut levels = Vec::new();
    let mut w = width;
    let mut h = height;
    let mut level = 0u32;

    levels.push(PyramidLevel {
        level,
        width: w,
        height: h,
    });

    while w > 256 || h > 256 {
        w = w.div_ceil(2);
        h = h.div_ceil(2);
        level += 1;

        levels.push(PyramidLevel {
            level,
            width: w,
            height: h,
        });
    }

    levels
}
