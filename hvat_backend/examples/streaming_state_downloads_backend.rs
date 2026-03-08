//! Streaming backend example with project-state + downloads enabled.
//!
//! This is the "streaming_state_downloads" profile:
//! - `/api/info`
//! - `/api/images`
//! - `/api/project-state` (GET/PUT)
//! - `/api/images/download`
//! - `/api/ws`
//! - No inference/SAM
//!
//! Run:
//! `cargo run -p hvat_backend --example streaming_state_downloads_backend -- ./data`

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use hvat_backend::helper::config::ServerConfig;
use hvat_backend::simple::{AppState, routes};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hvat_backend::examples=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let mut config = ServerConfig::default();
    if let Some(arg_data_dir) = std::env::args().nth(1) {
        config.data_dir = arg_data_dir.into();
    }
    config.project_name = "streaming-state-downloads-example".to_string();

    tracing::warn!("Example limitations: inference=false, sam=false, progressive_streaming=false");

    let state = Arc::new(AppState::new(config.clone()));

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .nest("/api", routes::api_router())
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    let listener = TcpListener::bind(addr).await?;
    tracing::info!(
        "Streaming+state+downloads example listening on http://{}",
        addr
    );
    axum::serve(listener, app).await?;

    Ok(())
}
