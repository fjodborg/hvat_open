//! Image metadata and download endpoints.

use std::sync::Arc;

use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    response::IntoResponse,
    routing::get,
};
use crate::common::archive::{ArchiveError, build_project_archive};
use crate::common::catalog::find_supported_image_by_id;
use serde::Serialize;

use crate::simple::state::AppState;
use crate::common::error::{Error, Result};

#[derive(Debug, Serialize)]
pub struct ImageMetadataResponse {
    pub id: String,
    pub name: String,
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub bands: usize,
}

pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/download", get(download_all_images))
        .route("/{id}/meta", get(get_metadata))
        .route("/{id}/thumbnail", get(get_thumbnail))
        .route("/{id}/raw", get(get_raw_image))
}

const THUMBNAIL_MAX_DIM: u32 = 256;

async fn get_thumbnail(
    State(state): State<Arc<AppState>>,
    Path(image_id): Path<String>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let png_bytes = crate::common::streaming::generate_thumbnail(
        &state.config.data_dir,
        &state.loaders,
        &image_id,
        THUMBNAIL_MAX_DIM,
    )
    .await
    .map_err(|e| (StatusCode::NOT_FOUND, format!("Thumbnail failed: {e}")))?;

    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("image/png"));
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=3600"),
    );

    Ok((StatusCode::OK, headers, Body::from(png_bytes)))
}

async fn get_metadata(
    State(state): State<Arc<AppState>>,
    Path(image_id): Path<String>,
) -> Result<Json<ImageMetadataResponse>> {
    let image_path = find_supported_image_by_id(&state.config.data_dir, &image_id, |path| {
        state.loaders.supports(path)
    })
    .ok_or_else(|| Error::ImageNotFound(image_id.clone()))?;

    let loader = state
        .loaders
        .find_loader(&image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_id.clone()))?;

    let meta = loader.load_metadata(&image_path).await?;

    Ok(Json(ImageMetadataResponse {
        id: image_id,
        name: meta.filename,
        format: meta.format,
        width: meta.width,
        height: meta.height,
        bands: meta.num_bands,
    }))
}

async fn get_raw_image(
    State(state): State<Arc<AppState>>,
    Path(image_id): Path<String>,
) -> std::result::Result<impl IntoResponse, (StatusCode, String)> {
    let image_path = find_supported_image_by_id(&state.config.data_dir, &image_id, |path| {
        state.loaders.supports(path)
    })
    .ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            format!("Image not found: {image_id}"),
        )
    })?;

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

    let archive = match archive_result {
        Ok(archive) => archive,
        Err(ArchiveError::NoImages) => {
            return Err((
                StatusCode::NOT_FOUND,
                "No project images available for download".to_string(),
            ));
        }
        Err(ArchiveError::BuildFailed(msg)) => {
            return Err((StatusCode::INTERNAL_SERVER_ERROR, msg));
        }
    };

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
