//! Image metadata and streaming endpoints.

use std::{
    io::{Cursor, Write},
    path::{Path as FsPath, PathBuf},
    sync::Arc,
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
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::pyramid::{PyramidStatus, compute_image_hash};
use crate::state::AppState;
use crate::utils::find_image;

const DEFAULT_PART_SIZE_MB: u64 = 256;
const MIN_PART_SIZE_MB: u64 = 16;
const MAX_PART_SIZE_MB: u64 = 1024;

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

#[derive(Debug, Deserialize, Default)]
struct DownloadQuery {
    #[serde(default)]
    part_size_mb: Option<u64>,
}

#[derive(Debug, Serialize)]
struct DownloadPlanResponse {
    total_files: usize,
    total_bytes: u64,
    part_size_bytes: u64,
    part_count: usize,
    parts: Vec<DownloadPartInfo>,
}

#[derive(Debug, Serialize)]
struct DownloadPartInfo {
    index: usize,
    filename: String,
    file_count: usize,
    total_bytes: u64,
}

#[derive(Debug, Clone)]
struct ImageEntry {
    relative_path: String,
    file_path: PathBuf,
    size_bytes: u64,
}

#[derive(Debug, Clone)]
struct PlannedDownloadPart {
    index: usize,
    filename: String,
    entries: Vec<ImageEntry>,
    total_bytes: u64,
}

#[derive(Debug)]
struct PlannedDownload {
    part_size_bytes: u64,
    total_files: usize,
    total_bytes: u64,
    parts: Vec<PlannedDownloadPart>,
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
    let expected_size = pixel_count_u32(width, height)
        .checked_mul(4)
        .ok_or_else(|| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Thumbnail dimensions too large: {}x{}", width, height),
            )
        })?;

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
    let mut image_entries = collect_image_entries(&data_dir, &state);
    if image_entries.is_empty() {
        return Err((
            StatusCode::NOT_FOUND,
            "No project images available for download".to_string(),
        ));
    }
    image_entries.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    let zip_bytes = tokio::task::spawn_blocking(move || build_images_zip(image_entries))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to build image archive: {e}"),
            )
        })?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let archive_name = format!(
        "{}_images.zip",
        sanitize_filename_component(&state.config.project_name)
    );
    let content_disposition = format!("attachment; filename=\"{}\"", archive_name);
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

/// Build a chunked download plan for project images.
async fn download_images_plan(
    State(state): State<Arc<AppState>>,
    Query(query): Query<DownloadQuery>,
) -> std::result::Result<Json<DownloadPlanResponse>, (StatusCode, String)> {
    let part_size_bytes = resolve_part_size_bytes(query.part_size_mb);
    let planned = build_chunked_download_plan(&state, part_size_bytes)?;
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
    let planned = build_chunked_download_plan(&state, part_size_bytes)?;
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

    let zip_bytes = tokio::task::spawn_blocking(move || build_images_zip(part.entries))
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
        w = w.div_ceil(2); // Round up division
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

fn collect_image_entries(base_dir: &FsPath, state: &AppState) -> Vec<ImageEntry> {
    let mut entries = Vec::new();
    collect_image_entries_recursive(base_dir, base_dir, state, &mut entries);
    entries
}

fn collect_image_entries_recursive(
    dir: &FsPath,
    base_dir: &FsPath,
    state: &AppState,
    out: &mut Vec<ImageEntry>,
) {
    if let Ok(read_dir) = std::fs::read_dir(dir) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_image_entries_recursive(&path, base_dir, state, out);
                continue;
            }
            if !state.loaders.supports(&path) {
                continue;
            }

            let relative = path
                .strip_prefix(base_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let size_bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            out.push(ImageEntry {
                relative_path: relative,
                file_path: path,
                size_bytes,
            });
        }
    }
}

fn build_images_zip(entries: Vec<ImageEntry>) -> std::result::Result<Vec<u8>, String> {
    let cursor = Cursor::new(Vec::<u8>::new());
    let mut zip = zip::ZipWriter::new(cursor);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .unix_permissions(0o644);

    for entry in entries {
        let bytes = std::fs::read(&entry.file_path)
            .map_err(|e| format!("Failed to read {:?}: {}", entry.file_path, e))?;
        zip.start_file(entry.relative_path, options)
            .map_err(|e| format!("Failed to add file to zip: {}", e))?;
        zip.write_all(&bytes)
            .map_err(|e| format!("Failed to write zip entry: {}", e))?;
    }

    zip.finish()
        .map(|cursor| cursor.into_inner())
        .map_err(|e| format!("Failed to finalize zip: {}", e))
}

fn resolve_part_size_bytes(part_size_mb: Option<u64>) -> u64 {
    let mb = part_size_mb
        .unwrap_or(DEFAULT_PART_SIZE_MB)
        .clamp(MIN_PART_SIZE_MB, MAX_PART_SIZE_MB);
    mb.saturating_mul(1024 * 1024)
}

fn build_chunked_download_plan(
    state: &AppState,
    part_size_bytes: u64,
) -> std::result::Result<PlannedDownload, (StatusCode, String)> {
    let data_dir = state.config.data_dir.clone();
    let mut entries = collect_image_entries(&data_dir, state);
    if entries.is_empty() {
        return Err((
            StatusCode::NOT_FOUND,
            "No project images available for download".to_string(),
        ));
    }
    entries.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    let total_files = entries.len();
    let total_bytes = entries.iter().map(|entry| entry.size_bytes).sum::<u64>();
    let partitions = partition_image_entries(entries, part_size_bytes);

    let base_name = sanitize_filename_component(&state.config.project_name);
    let parts = partitions
        .into_iter()
        .enumerate()
        .map(|(index, part_entries)| {
            let total_bytes = part_entries
                .iter()
                .map(|entry| entry.size_bytes)
                .sum::<u64>();
            PlannedDownloadPart {
                index,
                filename: format!("{}_images_part{:03}.zip", base_name, index + 1),
                entries: part_entries,
                total_bytes,
            }
        })
        .collect::<Vec<_>>();

    Ok(PlannedDownload {
        part_size_bytes,
        total_files,
        total_bytes,
        parts,
    })
}

fn partition_image_entries(entries: Vec<ImageEntry>, part_size_bytes: u64) -> Vec<Vec<ImageEntry>> {
    let mut parts: Vec<Vec<ImageEntry>> = Vec::new();
    let mut current_part = Vec::new();
    let mut current_size = 0u64;

    for entry in entries {
        let entry_size = entry.size_bytes.max(1);
        let would_overflow =
            !current_part.is_empty() && current_size.saturating_add(entry_size) > part_size_bytes;
        if would_overflow {
            parts.push(current_part);
            current_part = Vec::new();
            current_size = 0;
        }

        current_size = current_size.saturating_add(entry_size);
        current_part.push(entry);
    }

    if !current_part.is_empty() {
        parts.push(current_part);
    }

    parts
}

fn sanitize_filename_component(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return "project".to_string();
    }

    let sanitized: String = trimmed
        .chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => ch,
        })
        .collect();

    if sanitized.trim().is_empty() {
        "project".to_string()
    } else {
        sanitized
    }
}
