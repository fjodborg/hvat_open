//! Application state shared across handlers.

use std::sync::Arc;

use async_trait::async_trait;
use axum::extract::ws::Message;
use tokio::sync::mpsc;

use crate::common::config::ServerConfig;
use crate::common::error::Error;
use crate::common::loaders::ImageLoaderRegistry;
use crate::common::streaming;
use crate::common::websocket::{BackendHandler, WebSocketState};

/// Shared application state.
pub struct AppState {
    pub config: ServerConfig,
    pub loaders: ImageLoaderRegistry,
    pub ws: Arc<WebSocketState>,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        let ws = Arc::new(WebSocketState::new(config.clone()));

        Self {
            loaders: ImageLoaderRegistry::with_defaults(),
            config,
            ws,
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
    ) -> Result<(), Error> {
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
        streaming::build_default_capabilities(&self.config)
    }
}
