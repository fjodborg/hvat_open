//! API routes for the HVAT simple backend.

mod images;
mod projects;

use std::sync::Arc;

use axum::Router;
use axum::routing::get;

use crate::simple::state::AppState;

/// Create the main API router.
pub fn api_router() -> Router<Arc<AppState>> {
    Router::new()
        .merge(projects::router())
        .nest("/images", images::router())
        .route("/ws", get(ws_handler))
}

/// WebSocket upgrade handler — delegates to the helper's generic handler.
async fn ws_handler(
    ws: axum::extract::WebSocketUpgrade,
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> axum::response::Response {
    let handler = state.clone();
    let ws_state = state.ws.clone();
    ws.on_upgrade(move |socket| {
        crate::common::websocket::handle_websocket(socket, handler, ws_state)
    })
}
