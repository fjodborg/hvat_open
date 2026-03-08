//! Server info, image listing, and project-state endpoints for full backend.

use std::sync::Arc;

use axum::Router;

use crate::state::AppState;

pub fn router() -> Router<Arc<AppState>> {
    crate::helper::http::projects::router::<AppState>()
}
