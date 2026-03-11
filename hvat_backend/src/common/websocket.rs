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

struct MuxCtx<H: BackendHandler> {
    handler: Arc<H>,
    streams: Arc<RwLock<ConnectionStreams>>,
    tx: mpsc::Sender<Message>,
    connection_id: u64,
}

impl<H: BackendHandler> Clone for MuxCtx<H> {
    fn clone(&self) -> Self {
        Self {
            handler: Arc::clone(&self.handler),
            streams: Arc::clone(&self.streams),
            tx: self.tx.clone(),
            connection_id: self.connection_id,
        }
    }
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
pub async fn handle_websocket<H: BackendHandler>(
    socket: WebSocket,
    handler: Arc<H>,
    ws_state: Arc<WebSocketState>,
) {
    let Some(connection_id) = ws_state.try_acquire_connection() else {
        reject_max_connections(socket, ws_state.config.max_connections).await;
        return;
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

async fn reject_max_connections(socket: WebSocket, max_connections: usize) {
    let (mut sender, _) = socket.split();
    let error = ProtocolError::retryable(
        ErrorCode::MaxConnectionsReached,
        format!(
            "Server at maximum connections ({}). Please try again later.",
            max_connections
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
        max_connections
    );
}

async fn handle_inner<H: BackendHandler>(
    socket: WebSocket,
    handler: Arc<H>,
    ws_state: Arc<WebSocketState>,
    connection_id: u64,
) -> Result<(), Error> {
    let (sender, mut receiver) = socket.split();
    let (tx, rx) = mpsc::channel::<Message>(64);

    let last_pong = Arc::new(AtomicU64::new(current_timestamp_ms()));
    let connection_alive = Arc::new(AtomicBool::new(true));
    let streams = Arc::new(RwLock::new(ConnectionStreams::new(
        ws_state.config.max_user_streams as u32,
    )));

    let capabilities = handler.capabilities();
    let capabilities_bytes = encode_capabilities_v2(&capabilities);

    let send_task = spawn_send_task(sender, rx, capabilities_bytes, connection_alive.clone());

    let ping_task = spawn_ping_task(
        tx.clone(),
        last_pong.clone(),
        connection_alive.clone(),
        ws_state.config.ping_interval_secs,
        ws_state.config.connection_timeout_secs,
        connection_id,
    );

    let loop_ctx = MuxCtx {
        handler,
        streams: streams.clone(),
        tx: tx.clone(),
        connection_id,
    };
    receive_loop(&mut receiver, connection_alive.clone(), last_pong, loop_ctx).await;

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

fn spawn_send_task(
    mut sender: futures::stream::SplitSink<WebSocket, Message>,
    mut rx: mpsc::Receiver<Message>,
    capabilities_bytes: Vec<u8>,
    connection_alive: Arc<AtomicBool>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        if sender
            .send(Message::Binary(capabilities_bytes.into()))
            .await
            .is_err()
        {
            connection_alive.store(false, Ordering::SeqCst);
            return;
        }

        while let Some(msg) = rx.recv().await {
            if sender.send(msg).await.is_err() {
                connection_alive.store(false, Ordering::SeqCst);
                break;
            }
        }
    })
}

fn spawn_ping_task(
    tx: mpsc::Sender<Message>,
    last_pong: Arc<AtomicU64>,
    connection_alive: Arc<AtomicBool>,
    ping_interval_secs: u64,
    connection_timeout_secs: u64,
    connection_id: u64,
) -> Option<JoinHandle<()>> {
    if ping_interval_secs == 0 {
        return None;
    }

    Some(tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(ping_interval_secs));
        interval.tick().await;

        loop {
            interval.tick().await;

            if !connection_alive.load(Ordering::SeqCst) {
                break;
            }

            if pong_timeout_exceeded(&last_pong, connection_timeout_secs) {
                let elapsed_secs = elapsed_pong_seconds(&last_pong);
                tracing::warn!(
                    "Connection {} timed out: no pong for {} seconds",
                    connection_id,
                    elapsed_secs
                );
                connection_alive.store(false, Ordering::SeqCst);
                break;
            }

            let ping_msg = encode_ping(current_timestamp_ms());
            if tx.send(Message::Binary(ping_msg.into())).await.is_err() {
                connection_alive.store(false, Ordering::SeqCst);
                break;
            }
        }
    }))
}

fn elapsed_pong_seconds(last_pong: &AtomicU64) -> u64 {
    let last_pong_time = last_pong.load(Ordering::SeqCst);
    let now = current_timestamp_ms();
    (now.saturating_sub(last_pong_time)) / 1000
}

fn pong_timeout_exceeded(last_pong: &AtomicU64, timeout_secs: u64) -> bool {
    elapsed_pong_seconds(last_pong) > timeout_secs
}

async fn receive_loop<H: BackendHandler>(
    receiver: &mut futures::stream::SplitStream<WebSocket>,
    connection_alive: Arc<AtomicBool>,
    last_pong: Arc<AtomicU64>,
    ctx: MuxCtx<H>,
) {
    while let Some(result) = receiver.next().await {
        if !connection_alive.load(Ordering::SeqCst) {
            break;
        }

        let keep_running = handle_socket_event(
            result,
            connection_alive.clone(),
            last_pong.clone(),
            ctx.clone(),
        )
        .await;
        if !keep_running {
            break;
        }
    }
}

async fn handle_socket_event<H: BackendHandler>(
    result: Result<Message, axum::Error>,
    connection_alive: Arc<AtomicBool>,
    last_pong: Arc<AtomicU64>,
    ctx: MuxCtx<H>,
) -> bool {
    match result {
        Ok(msg) => handle_socket_message(msg, last_pong, ctx).await,
        Err(e) => {
            tracing::warn!("WebSocket error on connection {}: {}", ctx.connection_id, e);
            connection_alive.store(false, Ordering::SeqCst);
            false
        }
    }
}

async fn handle_socket_message<H: BackendHandler>(
    msg: Message,
    last_pong: Arc<AtomicU64>,
    ctx: MuxCtx<H>,
) -> bool {
    match msg {
        Message::Text(text) => {
            handle_text_message(&text, last_pong, ctx).await;
            true
        }
        Message::Close(_) => {
            tracing::debug!("Received close frame on connection {}", ctx.connection_id);
            false
        }
        _ => true,
    }
}

async fn handle_text_message<H: BackendHandler>(
    text: &str,
    last_pong: Arc<AtomicU64>,
    ctx: MuxCtx<H>,
) {
    let client_msg = match serde_json::from_str::<ClientMessage>(text) {
        Ok(message) => message,
        Err(e) => {
            tracing::warn!("Invalid client message: {}", e);
            let error =
                ProtocolError::error(ErrorCode::InvalidRequest, format!("Invalid message: {}", e));
            ctx.tx
                .send(Message::Binary(encode_stream_error(0, &error).into()))
                .await
                .ok();
            return;
        }
    };

    if let ClientMessage::Pong { timestamp } = &client_msg {
        last_pong.store(current_timestamp_ms(), Ordering::SeqCst);
        tracing::trace!(
            "Received pong from connection {} (ts: {})",
            ctx.connection_id,
            timestamp
        );
        return;
    }

    handle_message(client_msg, ctx).await;
}

async fn handle_message<H: BackendHandler>(msg: ClientMessage, ctx: MuxCtx<H>) {
    match msg {
        ClientMessage::SetImage {
            request_id,
            image_id,
        } => handle_set_image_message(&ctx, request_id, image_id).await,

        ClientMessage::StreamImage {
            request_id,
            level,
            progressive,
        } => handle_stream_image_message(&ctx, request_id, level, progressive).await,

        ClientMessage::CancelStream { request_id } => {
            handle_cancel_stream_message(&ctx, request_id).await;
        }

        ClientMessage::PrepareModel { request_id, .. }
        | ClientMessage::Infer { request_id, .. }
        | ClientMessage::CancelInfer { request_id } => {
            handle_unsupported_inference_message(&ctx, request_id).await;
        }

        ClientMessage::Pong { .. } => {}
    }
}

async fn handle_set_image_message<H: BackendHandler>(
    ctx: &MuxCtx<H>,
    request_id: u32,
    image_id: String,
) {
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

async fn handle_stream_image_message<H: BackendHandler>(
    ctx: &MuxCtx<H>,
    request_id: u32,
    level: u32,
    progressive: bool,
) {
    let Some(image_id) = resolve_active_image_for_stream(ctx, request_id).await else {
        return;
    };
    if !reserve_stream_slot(ctx, request_id).await {
        return;
    }

    tracing::info!(
        "Connection {}: stream_image {} (request_id={}, level={}, progressive={})",
        ctx.connection_id,
        image_id,
        request_id,
        level,
        progressive
    );

    let task = spawn_stream_task(
        ctx.handler.clone(),
        ctx.streams.clone(),
        ctx.tx.clone(),
        request_id,
        image_id.clone(),
        level,
        progressive,
    );
    ctx.streams
        .write()
        .await
        .insert(request_id, ActiveStream { task, image_id });
}

async fn resolve_active_image_for_stream<H: BackendHandler>(
    ctx: &MuxCtx<H>,
    request_id: u32,
) -> Option<String> {
    let active_image = ctx.streams.read().await.active_image().map(str::to_string);
    if active_image.is_some() {
        return active_image;
    }

    send_error(
        &ctx.tx,
        request_id,
        ErrorCode::NoActiveImage,
        "No active image set. Call set_image first.",
    )
    .await;
    None
}

async fn reserve_stream_slot<H: BackendHandler>(ctx: &MuxCtx<H>, request_id: u32) -> bool {
    let streams_guard = ctx.streams.read().await;
    if streams_guard.can_start_stream() {
        return true;
    }

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
    false
}

fn spawn_stream_task<H: BackendHandler>(
    handler: Arc<H>,
    streams: Arc<RwLock<ConnectionStreams>>,
    tx: mpsc::Sender<Message>,
    request_id: u32,
    image_id: String,
    level: u32,
    progressive: bool,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let result = handler
            .stream_image(tx.clone(), request_id, &image_id, level, progressive)
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
    })
}

async fn handle_cancel_stream_message<H: BackendHandler>(ctx: &MuxCtx<H>, request_id: u32) {
    let cancelled = ctx.streams.write().await.cancel(request_id);
    if cancelled {
        tracing::info!(
            "Connection {}: cancelled stream {}",
            ctx.connection_id,
            request_id
        );
    }
}

async fn handle_unsupported_inference_message<H: BackendHandler>(ctx: &MuxCtx<H>, request_id: u32) {
    send_error(
        &ctx.tx,
        request_id,
        ErrorCode::InvalidAction,
        "Inference is not supported by this backend",
    )
    .await;
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
