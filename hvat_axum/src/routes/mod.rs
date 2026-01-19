//! API routes for the HVAT server.

mod images;
mod projects;
pub mod websocket;
mod websocket_mux;

use std::sync::Arc;

use axum::Router;
use axum::routing::get;

use crate::state::AppState;

/// Create the main API router.
///
/// API endpoints:
/// - `GET /api/info` - Server/project information
/// - `GET /api/images` - List all images (for tree display)
/// - `GET /api/images/:id/meta` - Image metadata
/// - `WS /api/images/:id/stream` - WebSocket streaming (legacy, per-image)
/// - `WS /api/ws` - Multiplexed WebSocket streaming (new, single connection)
pub fn api_router() -> Router<Arc<AppState>> {
    Router::new()
        // Server info and image listing from projects module
        .merge(projects::router())
        // Image metadata and streaming from images module
        .nest("/images", images::router())
        // Multiplexed WebSocket endpoint
        .route("/ws", get(ws_handler))
}

/// WebSocket upgrade handler for multiplexed streaming.
async fn ws_handler(
    ws: axum::extract::WebSocketUpgrade,
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> axum::response::Response {
    ws.on_upgrade(move |socket| websocket_mux::handle_websocket_mux(socket, state))
}
