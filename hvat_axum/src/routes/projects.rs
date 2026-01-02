//! Server info and image listing endpoints.
//!
//! The server exposes a single "project" which is the configured data directory.
//! Clients connecting to this server see it as one project.

use std::sync::Arc;

use axum::{Json, Router, extract::State, routing::get};
use serde::Serialize;

use crate::error::Result;
use crate::state::AppState;
use crate::utils::make_url_safe;

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
