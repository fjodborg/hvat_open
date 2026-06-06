//! HVAT Axum server entrypoint with deterministic mock SAM backend.
//!
//! This is used for E2E integration testing where real SAM model artifacts are
//! unavailable or too heavy.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::Router;
use clap::Parser;
use tokio::net::TcpListener;
use tower_http::compression::CompressionLayer;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use hvat_backend::full_sam::config::CliArgs;
use hvat_backend::full_sam::sam::{mock_expected_files, mock_factory, mock_model_name};
use hvat_backend::full_sam::state::SamBackendWiring;
use hvat_backend::full_sam::{AppState, ServerConfig, pregenerate_pyramids, routes};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hvat_backend::full_sam=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = CliArgs::parse();
    let mut config = ServerConfig::from_cli(args)?;
    config.sam_enabled = true;

    tracing::info!("Starting HVAT mock SAM server with config: {:?}", config);
    tracing::info!(
        "Project: {} ({})",
        config.project_name,
        config.data_dir.display()
    );

    let state = Arc::new(AppState::new_with_sam_wiring(
        config.clone(),
        SamBackendWiring {
            backend_name: "mock-sam",
            model_id: "sam-mock",
            model_name: mock_model_name,
            expected_files: mock_expected_files,
            create_backend: mock_factory,
        },
    )?);

    let pregen_state = state.clone();
    tokio::spawn(async move {
        pregenerate_pyramids(pregen_state).await;
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
        .expose_headers(Any);

    let app = Router::new()
        .nest("/api", routes::api_router())
        .layer(CompressionLayer::new())
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Listening on http://{}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}
