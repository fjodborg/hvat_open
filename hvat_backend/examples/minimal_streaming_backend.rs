//! Minimal streaming-only backend built with `hvat_backend::helper`.
//!
//! Run:
//! `cargo run -p hvat_backend --example minimal_streaming_backend -- ./data`

use std::net::SocketAddr;
use std::sync::Arc;

use async_trait::async_trait;
use axum::extract::ws::Message;
use axum::extract::{State, WebSocketUpgrade};
use axum::response::Response;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use hvat_backend::helper::catalog::{ImageInfo, count_supported_images, list_supported_images};
use hvat_backend::helper::config::ServerConfig;
use hvat_backend::helper::error::Result;
use hvat_backend::helper::loaders::ImageLoaderRegistry;
use hvat_backend::helper::streaming;
use hvat_backend::helper::websocket::{BackendHandler, WebSocketState, handle_websocket};

#[derive(Debug, Serialize)]
struct ServerInfo {
    name: String,
    image_count: usize,
    version: String,
}

struct AppState {
    config: ServerConfig,
    loaders: ImageLoaderRegistry,
    ws: Arc<WebSocketState>,
}

impl AppState {
    fn new(config: ServerConfig) -> Self {
        Self {
            loaders: ImageLoaderRegistry::with_defaults(),
            ws: Arc::new(WebSocketState::new(config.clone())),
            config,
        }
    }
}

#[async_trait]
impl BackendHandler for AppState {
    async fn stream_image(
        &self,
        tx: mpsc::Sender<Message>,
        request_id: u32,
        image_id: &str,
        level: u32,
        progressive: bool,
    ) -> Result<()> {
        streaming::stream_image(
            &self.config.data_dir,
            &self.loaders,
            self.config.stream_chunk_rows,
            tx,
            request_id,
            image_id,
            level,
            progressive,
        )
        .await
    }

    fn capabilities(&self) -> hvat_common::ServerCapabilities {
        use hvat_common::{DownloadMode, ServerFeatures, ServerInfo, ServerLimits};

        hvat_common::ServerCapabilities {
            protocol_version: hvat_backend::helper::protocol::PROTOCOL_VERSION,
            server: ServerInfo {
                name: "minimal-streaming-example".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            limits: ServerLimits {
                max_image_size: self.config.max_cache_memory,
                max_pyramid_levels: hvat_common::MAX_PYRAMID_LEVEL,
                max_concurrent_streams: self.config.max_user_streams as u32,
                max_concurrent_inferences: 0,
            },
            features: ServerFeatures {
                streaming: true,
                project_state: false,
                downloads: false,
                thumbnails: false,
                inference: false,
                sam: false,
                progressive_streaming: false,
            },
            download_mode: DownloadMode::SingleZip,
            models: vec![],
        }
    }
}

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

    tracing::warn!(
        "Example limitations: project_state=false, downloads=false, thumbnails=false, inference=false, sam=false, progressive_streaming=false"
    );

    let state = Arc::new(AppState::new(config.clone()));

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/info", get(get_info))
        .route("/api/images", get(list_images))
        .route("/api/ws", get(ws_handler))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Minimal streaming backend listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

async fn get_info(State(state): State<Arc<AppState>>) -> Result<Json<ServerInfo>> {
    let image_count =
        count_supported_images(&state.config.data_dir, |path| state.loaders.supports(path));

    Ok(Json(ServerInfo {
        name: state.config.project_name.clone(),
        image_count,
        version: env!("CARGO_PKG_VERSION").to_string(),
    }))
}

async fn list_images(State(state): State<Arc<AppState>>) -> Result<Json<Vec<ImageInfo>>> {
    let images = list_supported_images(&state.config.data_dir, |path| state.loaders.supports(path));
    Ok(Json(images))
}

async fn ws_handler(ws: WebSocketUpgrade, State(state): State<Arc<AppState>>) -> Response {
    let handler = state.clone();
    let ws_state = state.ws.clone();
    ws.on_upgrade(move |socket| handle_websocket(socket, handler, ws_state))
}
