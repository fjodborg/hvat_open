//! API routes for the HVAT server.

mod images;
mod projects;
mod websocket;

use std::sync::Arc;

use axum::Router;

use crate::state::AppState;

/// Create the main API router.
///
/// API endpoints:
/// - `GET /api/info` - Server/project information
/// - `GET /api/images` - List all images (for tree display)
/// - `GET /api/images/:id/meta` - Image metadata
/// - `WS /api/images/:id/stream` - WebSocket streaming
pub fn api_router() -> Router<Arc<AppState>> {
    Router::new()
        // Server info and image listing from projects module
        .merge(projects::router())
        // Image metadata and streaming from images module
        .nest("/images", images::router())
}
