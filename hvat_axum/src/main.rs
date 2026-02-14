//! HVAT Axum Server Entry Point
//!
//! Run with: `cargo run -p hvat_axum -- --data-dir /path/to/images`
//!
//! Use `--help` to see all options.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use clap::Parser;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use hvat_axum::config::CliArgs;
use hvat_axum::{AppState, ServerConfig, pregenerate_pyramids, routes};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hvat_axum=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Parse CLI arguments
    let args = CliArgs::parse();

    // Load configuration from CLI
    let config = ServerConfig::from_cli(args);
    tracing::info!("Starting HVAT server with config: {:?}", config);
    tracing::info!(
        "Project: {} ({})",
        config.project_name,
        config.data_dir.display()
    );

    // Create application state
    let state = Arc::new(AppState::new(config.clone()));

    // Pre-generate pyramids/thumbnails in background so startup is non-blocking.
    let pregen_state = state.clone();
    tokio::spawn(async move {
        pregenerate_pyramids(pregen_state).await;
    });

    // Build CORS layer for development
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build the router
    let app = Router::new()
        .nest("/api", routes::api_router())
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start the server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Listening on http://{}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}
