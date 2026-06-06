//! API routes for the HVAT server.

mod capabilities;
mod images;
mod metrics;
mod projects;
mod sam;
mod sam_service;

use std::sync::Arc;

use axum::Router;
use axum::routing::get;

use crate::state::AppState;

/// Create the main API router.
///
/// API endpoints:
/// - `GET /api/info` - Server/project information
/// - `GET /api/images` - List all images (for tree display)
/// - `POST /api/images/upload` - Upload images/folders into server data directory
/// - `GET /api/project-state` - Load saved native project annotations
/// - `PUT /api/project-state` - Save native project annotations
/// - `GET /api/capabilities` - REST capabilities payload
/// - `GET /api/metrics` - REST route telemetry snapshot
/// - `GET /api/images/:id/meta` - Image metadata
/// - `GET /api/images/download` - Download all project images as ZIP
/// - `GET /api/images/download/plan` - Chunked image-download plan
/// - `GET /api/images/download/part/:index` - Download one ZIP chunk
pub fn api_router() -> Router<Arc<AppState>> {
    Router::new()
        // Server info and image listing from projects module
        .merge(projects::router())
        // REST capabilities contract
        .route("/capabilities", get(capabilities::get_capabilities))
        // REST endpoint telemetry snapshot for dashboards
        .route("/metrics", get(metrics::get_metrics))
        // Image metadata and streaming from images module
        .nest("/images", images::router())
        // REST SAM routes
        .nest("/sam", sam::router())
}
