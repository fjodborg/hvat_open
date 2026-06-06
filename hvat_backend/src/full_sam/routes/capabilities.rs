//! REST capabilities endpoint.

use std::sync::Arc;

use axum::{Json, extract::State};
use hvat_common::{
    DownloadMode, PROTOCOL_VERSION, ServerCapabilities, ServerFeatures, ServerInfo, ServerLimits,
};

use crate::state::AppState;

/// Upper bound for concurrent inferences exposed via REST capabilities.
pub const MAX_CONCURRENT_INFERENCES: u32 = 2;

/// Build server capabilities from current application state.
#[must_use]
pub fn build_server_capabilities(state: &AppState) -> ServerCapabilities {
    let models = if let Some(ref registry) = state.model_registry {
        registry.capabilities()
    } else {
        vec![]
    };
    let inference_enabled = !models.is_empty();

    let mut capabilities = ServerCapabilities {
        protocol_version: PROTOCOL_VERSION,
        server: ServerInfo {
            name: "hvat-axum".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
        limits: ServerLimits {
            max_image_size: state.config.max_cache_memory,
            max_pyramid_levels: hvat_common::MAX_PYRAMID_LEVEL,
            max_concurrent_streams: state.config.max_user_streams as u32,
            max_concurrent_inferences: MAX_CONCURRENT_INFERENCES,
        },
        features: ServerFeatures {
            streaming: true,
            project_state: true,
            downloads: true,
            thumbnails: true,
            inference: inference_enabled,
            sam: false,
            progressive_streaming: true,
        },
        download_mode: DownloadMode::Chunked,
        models,
    };
    capabilities.features.sam = capabilities.find_sam_prompt_model().is_some();
    capabilities
}

/// `GET /api/capabilities`
pub async fn get_capabilities(State(state): State<Arc<AppState>>) -> Json<ServerCapabilities> {
    Json(build_server_capabilities(&state))
}
