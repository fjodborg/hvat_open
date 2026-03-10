//! Multiplexed WebSocket handler for concurrent image streaming.
//!
//! Provides a generic WebSocket handler that backends plug into by implementing
//! [`BackendHandler`]. This handles all protocol boilerplate: connection management,
//! ping/pong keepalive, concurrent stream tracking, and message routing.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;

use crate::common::config::ServerConfig;
use crate::common::error::Error;
use crate::common::protocol::{
    ClientMessage, ErrorCode, ProtocolError, encode_capabilities_v2, encode_image_set, encode_ping,
    encode_stream_error,
};

/// Trait that backends implement to handle image streaming requests.
///
/// The WebSocket handler takes care of all protocol-level concerns
/// (connection management, multiplexing, keepalive). Backends only
/// need to implement the actual image loading and streaming logic.
#[async_trait]
pub trait BackendHandler: Send + Sync + 'static {
    /// Stream image data to the client.
    ///
    /// Send binary frames via `tx` using the protocol encoding functions.
    /// The handler already tracks this stream and will clean up on completion.
    async fn stream_image(
        &self,
        tx: mpsc::Sender<Message>,
        request_id: u32,
        image_id: &str,
        level: u32,
        progressive: bool,
    ) -> Result<(), Error>;

    /// Build server capabilities to send on connection.
    fn capabilities(&self) -> hvat_common::ServerCapabilities;
}

/// Shared state needed by the WebSocket handler.
///
/// Backends should wrap this with their own state and implement `BackendHandler`.
pub struct WebSocketState {
    pub config: ServerConfig,
    active_connections: AtomicU64,
    connection_id_counter: AtomicU64,
}

impl WebSocketState {
    pub fn new(config: ServerConfig) -> Self {
        Self {
            config,
            active_connections: AtomicU64::new(0),
            connection_id_counter: AtomicU64::new(0),
        }
    }

    pub fn try_acquire_connection(&self) -> Option<u64> {
        let max_connections = self.config.max_connections as u64;
        let current = self.active_connections.fetch_add(1, Ordering::SeqCst);

        if current >= max_connections {
            self.active_connections.fetch_sub(1, Ordering::SeqCst);
            None
        } else {
            let id = self.connection_id_counter.fetch_add(1, Ordering::SeqCst);
            Some(id)
        }
    }

    pub fn release_connection(&self) {
        self.active_connections.fetch_sub(1, Ordering::SeqCst);
    }

    pub fn connection_count(&self) -> u64 {
        self.active_connections.load(Ordering::SeqCst)
    }
}

// ============================================================================
// Internal types
// ============================================================================

struct ActiveStream {
    task: JoinHandle<()>,
    image_id: String,
}

struct ConnectionStreams {
    streams: HashMap<u32, ActiveStream>,
    max_streams: u32,
    active_image_id: Option<String>,
}

impl ConnectionStreams {
    fn new(max_streams: u32) -> Self {
        Self {
            streams: HashMap::new(),
            max_streams,
            active_image_id: None,
        }
    }

    fn set_image(&mut self, image_id: String) {
        self.active_image_id = Some(image_id);
    }

    fn active_image(&self) -> Option<&str> {
        self.active_image_id.as_deref()
    }

    fn can_start_stream(&self) -> bool {
        (self.streams.len() as u32) < self.max_streams
    }

    fn max_streams(&self) -> u32 {
        self.max_streams
    }

    fn insert(&mut self, request_id: u32, stream: ActiveStream) {
        self.streams.insert(request_id, stream);
    }

    fn cancel(&mut self, request_id: u32) -> bool {
        if let Some(stream) = self.streams.remove(&request_id) {
            stream.task.abort();
            tracing::debug!(
                "Cancelled stream {} for image '{}'",
                request_id,
                stream.image_id
            );
            true
        } else {
            false
        }
    }

    fn remove(&mut self, request_id: u32) {
        self.streams.remove(&request_id);
    }

    fn cancel_all(&mut self) {
        for (request_id, stream) in self.streams.drain() {
            stream.task.abort();
            tracing::debug!("Cancelled stream {} (connection closing)", request_id);
        }
    }
}

#[derive(Clone)]
struct MuxCtx<H: BackendHandler> {
    handler: Arc<H>,
    streams: Arc<RwLock<ConnectionStreams>>,
    tx: mpsc::Sender<Message>,
    connection_id: u64,
}

// ============================================================================
// Public API
// ============================================================================

/// Handle a multiplexed WebSocket connection.
///
/// Call this from your WebSocket upgrade handler:
/// ```ignore
/// ws.on_upgrade(move |socket| {
///     handle_websocket(socket, handler, ws_state)
/// })
/// ```
#[allow(clippy::cognitive_complexity)]
pub async fn handle_websocket<H: BackendHandler>(
    socket: WebSocket,
    handler: Arc<H>,
    ws_state: Arc<WebSocketState>,
) {
    let connection_id = match ws_state.try_acquire_connection() {
        Some(id) => id,
        None => {
            let (mut sender, _) = socket.split();
            let error = ProtocolError::retryable(
                ErrorCode::MaxConnectionsReached,
                format!(
                    "Server at maximum connections ({}). Please try again later.",
                    ws_state.config.max_connections
                ),
                5000,
            );
            sender
                .send(Message::Binary(encode_stream_error(0, &error).into()))
                .await
                .ok();
            sender.close().await.ok();
            tracing::warn!(
                "Rejected connection: max connections ({}) reached",
                ws_state.config.max_connections
            );
            return;
        }
    };

    tracing::info!(
        "WebSocket connection {} established (active: {})",
        connection_id,
        ws_state.connection_count()
    );

    let result = handle_inner(socket, handler, ws_state.clone(), connection_id).await;

    ws_state.release_connection();
    tracing::info!(
        "WebSocket connection {} closed (active: {})",
        connection_id,
        ws_state.connection_count()
    );

    if let Err(e) = result {
        tracing::error!("WebSocket connection {} error: {}", connection_id, e);
    }
}

#[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
async fn handle_inner<H: BackendHandler>(
    socket: WebSocket,
    handler: Arc<H>,
    ws_state: Arc<WebSocketState>,
    connection_id: u64,
) -> Result<(), Error> {
    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::channel::<Message>(64);

    let last_pong = Arc::new(AtomicU64::new(current_timestamp_ms()));
    let connection_alive = Arc::new(AtomicBool::new(true));
    let streams = Arc::new(RwLock::new(ConnectionStreams::new(
        ws_state.config.max_user_streams as u32,
    )));

    let capabilities = handler.capabilities();
    let capabilities_bytes = encode_capabilities_v2(&capabilities);

    let connection_alive_send = connection_alive.clone();
    let send_task = tokio::spawn(async move {
        if sender
            .send(Message::Binary(capabilities_bytes.into()))
            .await
            .is_err()
        {
            connection_alive_send.store(false, Ordering::SeqCst);
            return;
        }

        while let Some(msg) = rx.recv().await {
            if sender.send(msg).await.is_err() {
                connection_alive_send.store(false, Ordering::SeqCst);
                break;
            }
        }
    });

    let ping_interval = ws_state.config.ping_interval_secs;
    let connection_timeout = ws_state.config.connection_timeout_secs;
    let ping_task = if ping_interval > 0 {
        let tx_ping = tx.clone();
        let last_pong_ping = last_pong.clone();
        let connection_alive_ping = connection_alive.clone();

        Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(ping_interval));
            interval.tick().await;

            loop {
                interval.tick().await;

                if !connection_alive_ping.load(Ordering::SeqCst) {
                    break;
                }

                let last_pong_time = last_pong_ping.load(Ordering::SeqCst);
                let now = current_timestamp_ms();
                let elapsed_secs = (now.saturating_sub(last_pong_time)) / 1000;

                if elapsed_secs > connection_timeout {
                    tracing::warn!(
                        "Connection {} timed out: no pong for {} seconds",
                        connection_id,
                        elapsed_secs
                    );
                    connection_alive_ping.store(false, Ordering::SeqCst);
                    break;
                }

                let ping_msg = encode_ping(now);
                if tx_ping
                    .send(Message::Binary(ping_msg.into()))
                    .await
                    .is_err()
                {
                    connection_alive_ping.store(false, Ordering::SeqCst);
                    break;
                }
            }
        }))
    } else {
        None
    };

    while let Some(result) = receiver.next().await {
        if !connection_alive.load(Ordering::SeqCst) {
            break;
        }

        match result {
            Ok(msg) => match msg {
                Message::Text(text) => match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(client_msg) => {
                        if let ClientMessage::Pong { timestamp } = &client_msg {
                            last_pong.store(current_timestamp_ms(), Ordering::SeqCst);
                            tracing::trace!(
                                "Received pong from connection {} (ts: {})",
                                connection_id,
                                timestamp
                            );
                            continue;
                        }

                        let ctx = MuxCtx {
                            handler: handler.clone(),
                            streams: streams.clone(),
                            tx: tx.clone(),
                            connection_id,
                        };
                        handle_message(client_msg, ctx).await;
                    }
                    Err(e) => {
                        tracing::warn!("Invalid client message: {}", e);
                        let error = ProtocolError::error(
                            ErrorCode::InvalidRequest,
                            format!("Invalid message: {}", e),
                        );
                        tx.send(Message::Binary(encode_stream_error(0, &error).into()))
                            .await
                            .ok();
                    }
                },
                Message::Close(_) => {
                    tracing::debug!("Received close frame on connection {}", connection_id);
                    break;
                }
                _ => {}
            },
            Err(e) => {
                tracing::warn!("WebSocket error on connection {}: {}", connection_id, e);
                break;
            }
        }
    }

    connection_alive.store(false, Ordering::SeqCst);

    let mut streams_guard = streams.write().await;
    streams_guard.cancel_all();

    drop(tx);
    send_task.await.ok();
    if let Some(ping_task) = ping_task {
        ping_task.abort();
    }

    Ok(())
}

#[allow(clippy::cognitive_complexity)]
async fn handle_message<H: BackendHandler>(msg: ClientMessage, ctx: MuxCtx<H>) {
    match msg {
        ClientMessage::SetImage {
            request_id,
            image_id,
        } => {
            tracing::info!(
                "Connection {}: set_image {} (request_id={})",
                ctx.connection_id,
                image_id,
                request_id
            );
            ctx.streams.write().await.set_image(image_id.clone());
            let msg = encode_image_set(request_id, &image_id);
            ctx.tx.send(Message::Binary(msg.into())).await.ok();
        }

        ClientMessage::StreamImage {
            request_id,
            level,
            progressive,
        } => {
            let image_id = {
                let streams_guard = ctx.streams.read().await;
                match streams_guard.active_image() {
                    Some(id) => id.to_string(),
                    None => {
                        send_error(
                            &ctx.tx,
                            request_id,
                            ErrorCode::NoActiveImage,
                            "No active image set. Call set_image first.",
                        )
                        .await;
                        return;
                    }
                }
            };

            {
                let streams_guard = ctx.streams.read().await;
                if !streams_guard.can_start_stream() {
                    let error = ProtocolError::retryable(
                        ErrorCode::RateLimited,
                        format!(
                            "Maximum concurrent streams ({}) reached",
                            streams_guard.max_streams()
                        ),
                        1000,
                    );
                    ctx.tx
                        .send(Message::Binary(
                            encode_stream_error(request_id, &error).into(),
                        ))
                        .await
                        .ok();
                    return;
                }
            }

            tracing::info!(
                "Connection {}: stream_image {} (request_id={}, level={}, progressive={})",
                ctx.connection_id,
                image_id,
                request_id,
                level,
                progressive
            );

            let handler = ctx.handler.clone();
            let streams = ctx.streams.clone();
            let tx = ctx.tx.clone();
            let image_id_clone = image_id.clone();

            let task = tokio::spawn(async move {
                let result = handler
                    .stream_image(tx.clone(), request_id, &image_id_clone, level, progressive)
                    .await;

                if let Err(e) = result {
                    tracing::error!("Stream {} error: {}", request_id, e);
                    let error = ProtocolError::error(ErrorCode::InternalError, e.to_string());
                    tx.send(Message::Binary(
                        encode_stream_error(request_id, &error).into(),
                    ))
                    .await
                    .ok();
                }

                streams.write().await.remove(request_id);
                tracing::debug!("Stream {} completed and removed", request_id);
            });

            ctx.streams
                .write()
                .await
                .insert(request_id, ActiveStream { task, image_id });
        }

        ClientMessage::CancelStream { request_id } => {
            let cancelled = ctx.streams.write().await.cancel(request_id);
            if cancelled {
                tracing::info!(
                    "Connection {}: cancelled stream {}",
                    ctx.connection_id,
                    request_id
                );
            }
        }

        ClientMessage::PrepareModel { request_id, .. }
        | ClientMessage::Infer { request_id, .. }
        | ClientMessage::CancelInfer { request_id } => {
            send_error(
                &ctx.tx,
                request_id,
                ErrorCode::InvalidAction,
                "Inference is not supported by this backend",
            )
            .await;
        }

        ClientMessage::Pong { .. } => {}
    }
}

async fn send_error(
    tx: &mpsc::Sender<Message>,
    request_id: u32,
    code: ErrorCode,
    message: impl Into<String>,
) {
    let error = ProtocolError::error(code, message);
    tx.send(Message::Binary(
        encode_stream_error(request_id, &error).into(),
    ))
    .await
    .ok();
}

pub fn current_timestamp_ms() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
