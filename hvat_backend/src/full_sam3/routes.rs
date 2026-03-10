//! SAM3 backend route scaffold.

use std::sync::Arc;

use axum::Router;

use crate::full_sam3::AppState;

pub fn api_router() -> Router<Arc<AppState>> {
    crate::full_sam::routes::api_router()
}
