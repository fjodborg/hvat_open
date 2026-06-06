//! REST telemetry endpoint for operational dashboards.

use std::sync::Arc;

use axum::{Json, extract::State};

use crate::full_sam::state::RestMetricsSnapshot;
use crate::state::AppState;

/// `GET /api/metrics`
pub async fn get_metrics(State(state): State<Arc<AppState>>) -> Json<RestMetricsSnapshot> {
    Json(state.rest_metrics.snapshot())
}
