//! Server info, image listing, and project-state endpoints for simple backend.

use std::sync::Arc;

use axum::Router;

use crate::simple::state::AppState;

pub fn router() -> Router<Arc<AppState>> {
    crate::helper::http::projects::router::<AppState>()
}
