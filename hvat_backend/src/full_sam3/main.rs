//! HVAT Axum SAM3 Backend Entry Point
//!
//! Run with:
//! `cargo run -p hvat_backend --bin hvat_backend_sam3 -- --data-dir /path/to/images`

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use clap::Parser;
use tokio::net::TcpListener;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use hvat_backend::full_sam3::{
    CliArgs, ServerConfig, build_app_state, pregenerate_pyramids, routes,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                "hvat_backend::full_sam3=debug,hvat_backend::full_sam=debug,tower_http=debug".into()
            }),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = CliArgs::parse();
    let config = ServerConfig::from_cli(args)?;
    tracing::info!("Starting HVAT SAM3 backend with config: {:?}", config);
    tracing::info!(
        "Project: {} ({})",
        config.project_name,
        config.data_dir.display()
    );
    if let Ok(resolved_data_dir) = config.data_dir.canonicalize() {
        tracing::info!("Resolved data directory: {}", resolved_data_dir.display());
    }
    if let Ok(resolved_model_dir) = config.sam_model_dir.canonicalize() {
        tracing::info!(
            "Resolved SAM model directory: {}",
            resolved_model_dir.display()
        );
    }

    let state = Arc::new(build_app_state(config.clone())?);

    let pregen_state = state.clone();
    tokio::spawn(async move {
        pregenerate_pyramids(pregen_state).await;
    });

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
    tracing::info!("Listening on http://{}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}
