//! API routes for the HVAT server.

mod images;
mod projects;
mod websocket;

use std::sync::Arc;

use axum::Router;

use crate::state::AppState;

/// Create the main API router.
pub fn api_router() -> Router<Arc<AppState>> {
    Router::new()
        .nest("/projects", projects::router())
        .nest("/images", images::router())
}
