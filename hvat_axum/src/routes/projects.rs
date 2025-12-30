//! Project listing endpoints.
//!
//! Projects are directories containing image files.

use std::sync::Arc;

use axum::{
    Json, Router,
    extract::{Path, State},
    routing::get,
};
use serde::Serialize;

use crate::error::{Error, Result};
use crate::state::AppState;

/// Project information.
#[derive(Debug, Serialize)]
pub struct ProjectInfo {
    /// Project ID (directory name, URL-safe)
    pub id: String,
    /// Display name
    pub name: String,
    /// Full path on server
    pub path: String,
    /// Number of images in project
    pub image_count: usize,
}

/// Image entry in a project.
#[derive(Debug, Serialize)]
pub struct ImageInfo {
    /// Image ID (filename without extension, URL-safe)
    pub id: String,
    /// Display name (filename)
    pub name: String,
    /// Relative path within project
    pub path: String,
    /// Format (e.g., "PNG", "NPY")
    pub format: String,
}

/// Create the projects router.
pub fn router() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", get(list_projects))
        .route("/{id}/images", get(list_images))
}

/// List all available projects (directories in data_dir).
async fn list_projects(State(state): State<Arc<AppState>>) -> Result<Json<Vec<ProjectInfo>>> {
    let data_dir = &state.config.data_dir;

    if !data_dir.exists() {
        // Create data directory if it doesn't exist
        std::fs::create_dir_all(data_dir)?;
        return Ok(Json(vec![]));
    }

    let mut projects = Vec::new();

    let entries = std::fs::read_dir(data_dir)?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            // Count images in this directory
            let image_count = count_images(&path, &state);

            let id = make_url_safe(&name);

            projects.push(ProjectInfo {
                id,
                name: name.clone(),
                path: path.to_string_lossy().to_string(),
                image_count,
            });
        }
    }

    // Sort by name
    projects.sort_by(|a, b| a.name.cmp(&b.name));

    Ok(Json(projects))
}

/// List images in a project.
async fn list_images(
    State(state): State<Arc<AppState>>,
    Path(project_id): Path<String>,
) -> Result<Json<Vec<ImageInfo>>> {
    let data_dir = &state.config.data_dir;

    // Find project directory
    let project_dir = find_project_dir(data_dir, &project_id)?;

    let mut images = Vec::new();
    // Use data_dir as base for ID generation so IDs match what websocket.rs expects
    collect_images(&project_dir, data_dir, &state, &mut images);

    // Sort by path
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

/// Find project directory by ID.
fn find_project_dir(data_dir: &std::path::Path, project_id: &str) -> Result<std::path::PathBuf> {
    if let Ok(entries) = std::fs::read_dir(data_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

                if make_url_safe(name) == project_id {
                    return Ok(path);
                }
            }
        }
    }

    Err(Error::ProjectNotFound(project_id.to_string()))
}

/// Make a string URL-safe by replacing non-alphanumeric characters.
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
