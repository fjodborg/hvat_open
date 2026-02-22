//! Multiplexed WebSocket handler for concurrent image streaming.
//!
//! This module provides a single WebSocket endpoint (`/api/ws`) that handles
//! multiple concurrent image streams over one connection. Each stream is
//! identified by a client-generated `request_id`.
//!
//! # Protocol
//!
//! **Client → Server (JSON text frames):**
//! ```json
//! { "action": "set_image", "request_id": 1, "image_id": "img.png" }
//! { "action": "stream_image", "request_id": 1, "level": 0, "progressive": true }
//! { "action": "cancel_stream", "request_id": 1 }
//! { "action": "pong", "timestamp": 123456789 }
//! ```
//!
//! **Server → Client (binary frames):**
//! ```text
//! [version:u8][type:u8][request_id:u32][payload...]
//! ```
//!
//! Connection-level messages (Capabilities, Ping) use `request_id = 0`.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use hvat_backend_helper::websocket::current_timestamp_ms;
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;

use crate::error::Error;
use crate::packer::{downsample_bands, pack_bands_to_rgba_layers};
use crate::protocol::{
    ClientMessage, ErrorCode, PROTOCOL_VERSION, ProtocolError, ServerCapabilities, StreamMetadata,
    encode_capabilities_v2, encode_image_set, encode_infer_progress, encode_infer_result,
    encode_layer_chunk_mux, encode_layer_complete_mux, encode_level_complete_mux,
    encode_model_ready, encode_ping, encode_reset_mux, encode_stream_complete, encode_stream_error,
};
use crate::pyramid::{PyramidStatus, compute_image_hash};
use crate::state::AppState;
use crate::utils::find_image;

const YIELD_EVERY_N_CHUNKS: u32 = 8;
const MAX_CONCURRENT_INFERENCES: u32 = 2;
const MAX_PREPARED_DIMENSIONS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SamBands {
    red: u32,
    green: u32,
    blue: u32,
}

impl SamBands {
    const fn default_rgb() -> Self {
        Self {
            red: 0,
            green: 1,
            blue: 2,
        }
    }

    fn from_config(config: Option<&serde_json::Value>) -> Self {
        let Some(cfg) = config else {
            return Self::default_rgb();
        };

        let Some(bands) = cfg.get("bands") else {
            return Self::default_rgb();
        };

        let red = bands.get("red").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let green = bands.get("green").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
        let blue = bands.get("blue").and_then(|v| v.as_u64()).unwrap_or(2) as u32;

        Self { red, green, blue }
    }

    const fn is_default_rgb(self) -> bool {
        self.red == 0 && self.green == 1 && self.blue == 2
    }

    fn embedding_key(self, image_id: &str) -> String {
        format!("{}#{}:{}:{}", image_id, self.red, self.green, self.blue)
    }

    fn clamp_to_num_bands(self, num_bands: usize) -> Self {
        if num_bands == 0 {
            return Self::default_rgb();
        }
        let max_idx = num_bands.saturating_sub(1) as u32;
        Self {
            red: self.red.min(max_idx),
            green: self.green.min(max_idx),
            blue: self.blue.min(max_idx),
        }
    }
}

/// State for a single active stream within a multiplexed connection.
struct ActiveStream {
    /// Task handle for the streaming coroutine
    task: JoinHandle<()>,
    /// Image ID being streamed
    image_id: String,
}

/// Cached image data for inference operations.
struct CachedImageData {
    image_id: String,
    rgb_data: Vec<u8>,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy)]
struct PreparedDimensionsEntry {
    width: u32,
    height: u32,
    last_access_tick: u64,
}

/// State for an active inference task.
struct ActiveInference {
    /// Task handle for the inference coroutine
    task: JoinHandle<()>,
    /// Model ID being used
    model_id: String,
}

/// State for an active model preparation task.
struct ActivePrepare {
    /// Task handle for the prepare coroutine
    task: JoinHandle<()>,
    /// Model ID being prepared
    model_id: String,
}

#[derive(Clone)]
struct MuxRequestCtx {
    state: Arc<AppState>,
    streams: Arc<RwLock<ConnectionStreams>>,
    tx: mpsc::Sender<Message>,
    connection_id: u64,
}

struct PrepareModelRequest {
    request_id: u32,
    model_id: String,
    image_id: Option<String>,
    config: Option<serde_json::Value>,
}

struct InferModelRequest {
    request_id: u32,
    model_id: String,
    image_id: Option<String>,
    inputs: serde_json::Value,
    options: serde_json::Value,
}

#[derive(Clone, Copy)]
struct StreamLevelRequest<'a> {
    image_id: &'a str,
    image_path: &'a Path,
    image_hash: &'a str,
    level: u32,
    full_width: u32,
    full_height: u32,
    send_reset: bool,
}

#[derive(Clone, Copy)]
struct StreamSourceRequest<'a> {
    image_path: &'a Path,
    image_id: &'a str,
    level: u32,
    full_width: u32,
    full_height: u32,
    send_reset: bool,
}

/// Manages all active streams for a single WebSocket connection.
struct ConnectionStreams {
    /// Active streams indexed by request_id
    streams: HashMap<u32, ActiveStream>,
    /// Active prepare tasks indexed by request_id
    prepares: HashMap<u32, ActivePrepare>,
    /// Active inference tasks indexed by request_id
    inferences: HashMap<u32, ActiveInference>,
    /// Maximum concurrent streams allowed
    max_streams: u32,
    /// Maximum concurrent inferences allowed
    max_inferences: u32,
    /// Active image ID (set by SetImage message)
    active_image_id: Option<String>,
    /// Cached RGB image data for inference (avoid reloading for each infer call)
    cached_image: Option<CachedImageData>,
    /// Prepared embedding dimensions keyed by embedding cache key.
    prepared_dimensions: HashMap<String, PreparedDimensionsEntry>,
    prepared_dimensions_tick: u64,
}

impl ConnectionStreams {
    fn new(max_streams: u32, max_inferences: u32) -> Self {
        Self {
            streams: HashMap::new(),
            prepares: HashMap::new(),
            inferences: HashMap::new(),
            max_streams,
            max_inferences,
            active_image_id: None,
            cached_image: None,
            prepared_dimensions: HashMap::new(),
            prepared_dimensions_tick: 0,
        }
    }

    /// Set the active image context.
    fn set_image(&mut self, image_id: String) {
        // Clear cached image if switching to a different image
        if self.active_image_id.as_ref() != Some(&image_id) {
            self.cached_image = None;
        }
        self.active_image_id = Some(image_id);
    }

    /// Get the active image ID.
    fn active_image(&self) -> Option<&str> {
        self.active_image_id.as_deref()
    }

    /// Cache RGB image data for inference.
    fn cache_image(&mut self, image_id: String, rgb_data: Vec<u8>, width: u32, height: u32) {
        self.cached_image = Some(CachedImageData {
            image_id,
            rgb_data,
            width,
            height,
        });
    }

    fn next_prepared_dimensions_tick(&mut self) -> u64 {
        self.prepared_dimensions_tick = self.prepared_dimensions_tick.wrapping_add(1);
        self.prepared_dimensions_tick
    }

    fn evict_prepared_dimensions_if_needed(&mut self) {
        while self.prepared_dimensions.len() > MAX_PREPARED_DIMENSIONS {
            let oldest_key = self
                .prepared_dimensions
                .iter()
                .min_by_key(|(_, entry)| entry.last_access_tick)
                .map(|(key, _)| key.clone());

            let Some(oldest_key) = oldest_key else {
                break;
            };
            self.prepared_dimensions.remove(&oldest_key);
        }
    }

    /// Record dimensions for a prepared embedding key.
    fn cache_prepared_dimensions(&mut self, embedding_key: String, width: u32, height: u32) {
        let tick = self.next_prepared_dimensions_tick();
        self.prepared_dimensions.insert(
            embedding_key,
            PreparedDimensionsEntry {
                width,
                height,
                last_access_tick: tick,
            },
        );
        self.evict_prepared_dimensions_if_needed();
    }

    /// Read dimensions for a prepared embedding key and mark as recently used.
    fn prepared_dimensions(&mut self, embedding_key: &str) -> Option<(u32, u32)> {
        let tick = self.next_prepared_dimensions_tick();
        self.prepared_dimensions
            .get_mut(embedding_key)
            .map(|entry| {
                entry.last_access_tick = tick;
                (entry.width, entry.height)
            })
    }

    /// Get cached image data if available and matches the image_id.
    fn get_cached_image(&self, image_id: &str) -> Option<&CachedImageData> {
        self.cached_image
            .as_ref()
            .filter(|c| c.image_id == image_id)
    }

    /// Check if we can start a new stream.
    fn can_start_stream(&self) -> bool {
        (self.streams.len() as u32) < self.max_streams
    }

    fn max_streams(&self) -> u32 {
        self.max_streams
    }

    /// Register a new stream.
    fn insert(&mut self, request_id: u32, stream: ActiveStream) {
        self.streams.insert(request_id, stream);
    }

    /// Cancel and remove a stream by request_id.
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

    /// Remove a completed stream (without aborting).
    fn remove(&mut self, request_id: u32) {
        self.streams.remove(&request_id);
    }

    /// Cancel all active streams.
    fn cancel_all(&mut self) {
        for (request_id, stream) in self.streams.drain() {
            stream.task.abort();
            tracing::debug!("Cancelled stream {} (connection closing)", request_id);
        }
    }

    /// Register a new prepare task.
    fn insert_prepare(&mut self, request_id: u32, prepare: ActivePrepare) {
        self.prepares.insert(request_id, prepare);
    }

    /// Remove a completed prepare task (without aborting).
    fn remove_prepare(&mut self, request_id: u32) {
        self.prepares.remove(&request_id);
    }

    /// Cancel all active prepare tasks.
    fn cancel_all_prepares(&mut self) {
        for (request_id, prepare) in self.prepares.drain() {
            prepare.task.abort();
            tracing::debug!(
                "Cancelled prepare {} for model '{}' (replaced/new priority task)",
                request_id,
                prepare.model_id
            );
        }
    }

    /// Check if we can start a new inference task.
    fn can_start_inference(&self) -> bool {
        (self.inferences.len() as u32) < self.max_inferences
    }

    fn max_inferences(&self) -> u32 {
        self.max_inferences
    }

    /// Register a new inference task.
    fn insert_inference(&mut self, request_id: u32, inference: ActiveInference) {
        self.inferences.insert(request_id, inference);
    }

    /// Cancel and remove an inference task by request_id.
    fn cancel_inference(&mut self, request_id: u32) -> bool {
        if let Some(inference) = self.inferences.remove(&request_id) {
            inference.task.abort();
            tracing::info!(
                "Cancelled inference {} for model '{}'",
                request_id,
                inference.model_id
            );
            true
        } else {
            false
        }
    }

    /// Remove a completed inference task (without aborting).
    fn remove_inference(&mut self, request_id: u32) {
        self.inferences.remove(&request_id);
    }

    /// Cancel all active inference tasks.
    fn cancel_all_inferences(&mut self) {
        for (request_id, inference) in self.inferences.drain() {
            inference.task.abort();
            tracing::debug!("Cancelled inference {} (connection closing)", request_id);
        }
    }
}

/// Handle a multiplexed WebSocket connection.
///
/// This is the entry point for the `/api/ws` endpoint. It manages multiple
/// concurrent image streams over a single WebSocket connection.
#[allow(clippy::cognitive_complexity)]
pub async fn handle_websocket_mux(socket: WebSocket, state: Arc<AppState>) {
    // Try to acquire a connection slot
    let connection_id = match state.try_acquire_connection() {
        Some(id) => id,
        None => {
            // At connection limit - reject with error
            let (mut sender, _) = socket.split();
            let error = ProtocolError::retryable(
                ErrorCode::MaxConnectionsReached,
                format!(
                    "Server at maximum connections ({}). Please try again later.",
                    state.config.max_connections
                ),
                5000,
            );
            // If sending the error fails, client is likely already disconnected - nothing to do
            sender
                .send(Message::Binary(encode_stream_error(0, &error).into()))
                .await
                .ok();
            sender.close().await.ok();
            tracing::warn!(
                "Rejected multiplexed connection: max connections ({}) reached",
                state.config.max_connections
            );
            return;
        }
    };

    tracing::info!(
        "Multiplexed WebSocket connection {} established (active: {})",
        connection_id,
        state.connection_count()
    );

    // Run the connection handler with cleanup on exit
    let result = handle_websocket_mux_inner(socket, state.clone(), connection_id).await;

    // Always release the connection slot
    state.release_connection();
    tracing::info!(
        "Multiplexed WebSocket connection {} closed (active: {})",
        connection_id,
        state.connection_count()
    );

    if let Err(e) = result {
        tracing::error!(
            "Multiplexed WebSocket connection {} error: {}",
            connection_id,
            e
        );
    }
}

/// Inner handler for the multiplexed WebSocket connection.
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)]
async fn handle_websocket_mux_inner(
    socket: WebSocket,
    state: Arc<AppState>,
    connection_id: u64,
) -> Result<(), Error> {
    let (mut sender, mut receiver) = socket.split();

    // Channel for sending messages to the client (shared by all streams)
    let (tx, mut rx) = mpsc::channel::<Message>(64);

    // Shared state
    let last_pong = Arc::new(AtomicU64::new(current_timestamp_ms()));
    let connection_alive = Arc::new(AtomicBool::new(true));
    let streams = Arc::new(RwLock::new(ConnectionStreams::new(
        state.config.max_user_streams as u32,
        MAX_CONCURRENT_INFERENCES,
    )));

    // Build and send server capabilities immediately
    let capabilities = build_server_capabilities(&state);
    let capabilities_bytes = encode_capabilities_v2(&capabilities);

    // Spawn task to forward messages to WebSocket
    let connection_alive_send = connection_alive.clone();
    let send_task = tokio::spawn(async move {
        // Send capabilities as first message
        if sender
            .send(Message::Binary(capabilities_bytes.into()))
            .await
            .is_err()
        {
            connection_alive_send.store(false, Ordering::SeqCst);
            return;
        }

        // Forward all messages from streams
        while let Some(msg) = rx.recv().await {
            if sender.send(msg).await.is_err() {
                connection_alive_send.store(false, Ordering::SeqCst);
                break;
            }
        }
    });

    // Spawn ping task if keepalive is enabled
    let ping_interval = state.config.ping_interval_secs;
    let connection_timeout = state.config.connection_timeout_secs;
    let ping_task = if ping_interval > 0 {
        let tx_ping = tx.clone();
        let last_pong_ping = last_pong.clone();
        let connection_alive_ping = connection_alive.clone();

        Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(ping_interval));
            interval.tick().await; // Skip immediate first tick

            loop {
                interval.tick().await;

                if !connection_alive_ping.load(Ordering::SeqCst) {
                    break;
                }

                // Check if client has timed out
                let last_pong_time = last_pong_ping.load(Ordering::SeqCst);
                let now = current_timestamp_ms();
                let elapsed_secs = (now.saturating_sub(last_pong_time)) / 1000;

                if elapsed_secs > connection_timeout {
                    tracing::warn!(
                        "Multiplexed connection {} timed out: no pong for {} seconds",
                        connection_id,
                        elapsed_secs
                    );
                    connection_alive_ping.store(false, Ordering::SeqCst);
                    break;
                }

                // Send ping (request_id = 0 for connection-level)
                let ping_msg = encode_ping(now);
                if tx_ping
                    .send(Message::Binary(ping_msg.into()))
                    .await
                    .is_err()
                {
                    connection_alive_ping.store(false, Ordering::SeqCst);
                    break;
                }

                tracing::trace!("Sent ping to multiplexed connection {}", connection_id);
            }
        }))
    } else {
        None
    };

    // Process incoming messages
    while let Some(result) = receiver.next().await {
        if !connection_alive.load(Ordering::SeqCst) {
            break;
        }

        match result {
            Ok(msg) => match msg {
                Message::Text(text) => {
                    tracing::debug!("Mux received: {}", text);

                    match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(client_msg) => {
                            // Handle Pong specially
                            if let ClientMessage::Pong { timestamp } = &client_msg {
                                last_pong.store(current_timestamp_ms(), Ordering::SeqCst);
                                tracing::trace!(
                                    "Received pong from mux connection {} (ts: {})",
                                    connection_id,
                                    timestamp
                                );
                                continue;
                            }

                            handle_mux_message(
                                client_msg,
                                state.clone(),
                                streams.clone(),
                                tx.clone(),
                                connection_id,
                            )
                            .await;
                        }
                        Err(e) => {
                            tracing::warn!("Invalid client message on mux connection: {}", e);
                            let error = ProtocolError::error(
                                ErrorCode::InvalidRequest,
                                format!("Invalid message: {}", e),
                            );
                            // Ignore send errors - client likely disconnected
                            tx.send(Message::Binary(encode_stream_error(0, &error).into()))
                                .await
                                .ok();
                        }
                    }
                }
                Message::Close(_) => {
                    tracing::debug!("Received close frame on mux connection {}", connection_id);
                    break;
                }
                _ => {} // Ignore binary messages from client
            },
            Err(e) => {
                tracing::warn!(
                    "WebSocket receive error on mux connection {}: {}",
                    connection_id,
                    e
                );
                break;
            }
        }
    }

    // Signal shutdown
    connection_alive.store(false, Ordering::SeqCst);

    // Cancel all active streams and inferences
    let mut streams_guard = streams.write().await;
    streams_guard.cancel_all();
    streams_guard.cancel_all_prepares();
    streams_guard.cancel_all_inferences();

    // Clean up
    drop(tx);
    send_task.await.ok();
    if let Some(ping_task) = ping_task {
        ping_task.abort();
    }

    Ok(())
}

/// Handle a parsed client message on the multiplexed connection.
async fn handle_mux_message(
    msg: ClientMessage,
    state: Arc<AppState>,
    streams: Arc<RwLock<ConnectionStreams>>,
    tx: mpsc::Sender<Message>,
    connection_id: u64,
) {
    let ctx = MuxRequestCtx {
        state,
        streams,
        tx,
        connection_id,
    };

    match msg {
        ClientMessage::SetImage {
            request_id,
            image_id,
        } => handle_set_image(&ctx, request_id, image_id).await,

        ClientMessage::StreamImage {
            request_id,
            level,
            progressive,
        } => handle_stream_image(&ctx, request_id, level, progressive).await,

        ClientMessage::CancelStream { request_id } => handle_cancel_stream(&ctx, request_id).await,

        ClientMessage::PrepareModel {
            request_id,
            model_id,
            image_id,
            config,
        } => spawn_prepare_model_task(&ctx, request_id, model_id, image_id, config).await,

        ClientMessage::Infer {
            request_id,
            model_id,
            image_id,
            inputs,
            options,
        } => spawn_infer_task(&ctx, request_id, model_id, image_id, inputs, options).await,

        ClientMessage::CancelInfer { request_id } => handle_cancel_infer(&ctx, request_id).await,

        ClientMessage::Pong { .. } => {} // Handled in the main receive loop
    }
}

async fn handle_set_image(ctx: &MuxRequestCtx, request_id: u32, image_id: String) {
    tracing::info!(
        "Mux connection {}: set_image {} (request_id={})",
        ctx.connection_id,
        image_id,
        request_id
    );
    ctx.streams.write().await.set_image(image_id.clone());
    let msg = encode_image_set(request_id, &image_id);
    ctx.tx.send(Message::Binary(msg.into())).await.ok();
}

async fn handle_stream_image(ctx: &MuxRequestCtx, request_id: u32, level: u32, progressive: bool) {
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
        "Mux connection {}: stream_image {} (request_id={}, level={}, progressive={})",
        ctx.connection_id,
        image_id,
        request_id,
        level,
        progressive
    );

    let task = spawn_stream_task(
        ctx.state.clone(),
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

async fn handle_cancel_stream(ctx: &MuxRequestCtx, request_id: u32) {
    let cancelled = ctx.streams.write().await.cancel(request_id);
    if cancelled {
        tracing::info!(
            "Mux connection {}: cancelled stream {}",
            ctx.connection_id,
            request_id
        );
    } else {
        tracing::debug!(
            "Mux connection {}: stream {} not found (already complete?)",
            ctx.connection_id,
            request_id
        );
    }
}

async fn spawn_prepare_model_task(
    ctx: &MuxRequestCtx,
    request_id: u32,
    model_id: String,
    image_id: Option<String>,
    config: Option<serde_json::Value>,
) {
    // Keep only the newest prepare task. Stale preloads are low priority.
    ctx.streams.write().await.cancel_all_prepares();

    let state_clone = ctx.state.clone();
    let streams_clone = ctx.streams.clone();
    let tx_clone = ctx.tx.clone();
    let connection_id = ctx.connection_id;
    let model_id_for_tracking = model_id.clone();

    let task = tokio::spawn(async move {
        handle_prepare_model(
            MuxRequestCtx {
                state: state_clone,
                streams: streams_clone.clone(),
                tx: tx_clone,
                connection_id,
            },
            PrepareModelRequest {
                request_id,
                model_id,
                image_id,
                config,
            },
        )
        .await;
        streams_clone.write().await.remove_prepare(request_id);
    });

    ctx.streams.write().await.insert_prepare(
        request_id,
        ActivePrepare {
            task,
            model_id: model_id_for_tracking,
        },
    );
}

async fn spawn_infer_task(
    ctx: &MuxRequestCtx,
    request_id: u32,
    model_id: String,
    image_id: Option<String>,
    inputs: serde_json::Value,
    options: serde_json::Value,
) {
    {
        let mut streams_guard = ctx.streams.write().await;
        if !streams_guard.can_start_inference() {
            let error = ProtocolError::retryable(
                ErrorCode::ModelBusy,
                format!(
                    "Maximum concurrent inferences ({}) reached",
                    streams_guard.max_inferences()
                ),
                100,
            );
            ctx.tx
                .send(Message::Binary(
                    encode_stream_error(request_id, &error).into(),
                ))
                .await
                .ok();
            return;
        }
        // Prioritize active inference over preloading prepares.
        streams_guard.cancel_all_prepares();
    }

    let state_clone = ctx.state.clone();
    let streams_clone = ctx.streams.clone();
    let tx_clone = ctx.tx.clone();
    let connection_id = ctx.connection_id;
    let model_id_for_tracking = model_id.clone();

    let task = tokio::spawn(async move {
        handle_infer(
            MuxRequestCtx {
                state: state_clone,
                streams: streams_clone.clone(),
                tx: tx_clone,
                connection_id,
            },
            InferModelRequest {
                request_id,
                model_id,
                image_id,
                inputs,
                options,
            },
        )
        .await;
        streams_clone.write().await.remove_inference(request_id);
    });

    ctx.streams.write().await.insert_inference(
        request_id,
        ActiveInference {
            task,
            model_id: model_id_for_tracking,
        },
    );
}

async fn handle_cancel_infer(ctx: &MuxRequestCtx, request_id: u32) {
    tracing::info!(
        "Mux connection {}: cancel_infer (request_id={})",
        ctx.connection_id,
        request_id
    );
    let cancelled = ctx.streams.write().await.cancel_inference(request_id);
    if cancelled {
        tracing::debug!("Inference task {} cancelled successfully", request_id);
    } else {
        tracing::debug!(
            "Inference task {} not found (may have already completed)",
            request_id
        );
    }
}

/// Send a protocol error to the client (convenience helper).
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

// ---------------------------------------------------------------------------
// Shared helpers for prepare_model / infer
// ---------------------------------------------------------------------------

use crate::inference::InferenceBackend;

/// Resolve model registry + backend, sending protocol error on failure.
async fn resolve_model_backend(
    state: &AppState,
    model_id: &str,
    request_id: u32,
    tx: &mpsc::Sender<Message>,
) -> Option<Arc<dyn InferenceBackend>> {
    let registry = match state.model_registry.as_ref() {
        Some(r) => r.clone(),
        None => {
            send_error(
                tx,
                request_id,
                ErrorCode::ModelNotFound,
                "No models available on this server",
            )
            .await;
            return None;
        }
    };
    match registry.get(model_id) {
        Some(b) => Some(b),
        None => {
            send_error(
                tx,
                request_id,
                ErrorCode::ModelNotFound,
                format!("Model '{}' not found", model_id),
            )
            .await;
            None
        }
    }
}

/// Resolve image_id from an explicit parameter or the active connection state.
async fn resolve_image_id(
    streams: &RwLock<ConnectionStreams>,
    explicit: Option<String>,
    request_id: u32,
    tx: &mpsc::Sender<Message>,
) -> Option<String> {
    if let Some(id) = explicit {
        return Some(id);
    }
    let guard = streams.read().await;
    match guard.active_image() {
        Some(id) => Some(id.to_string()),
        None => {
            send_error(
                tx,
                request_id,
                ErrorCode::NoActiveImage,
                "No active image set. Provide image_id or call set_image first.",
            )
            .await;
            None
        }
    }
}

/// Resolved band selection and embedding key for an image.
struct ResolvedBands {
    requested: SamBands,
    resolved: SamBands,
    embedding_key: String,
}

/// Parse and resolve SAM band selection, clamping to the image's actual band count.
async fn resolve_band_selection(
    state: &AppState,
    image_id: &str,
    config: Option<&serde_json::Value>,
    request_id: u32,
    tx: &mpsc::Sender<Message>,
) -> Option<ResolvedBands> {
    let requested = SamBands::from_config(config);
    let resolved = if requested.is_default_rgb() {
        requested
    } else {
        match resolve_sam_bands(state, image_id, requested).await {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("Failed to resolve band selection for '{}': {}", image_id, e);
                send_error(
                    tx,
                    request_id,
                    ErrorCode::ImageNotFound,
                    format!("Failed to resolve image bands: {}", e),
                )
                .await;
                return None;
            }
        }
    };
    if resolved != requested {
        tracing::debug!(
            "Resolved SAM bands [{}, {}, {}] to [{}, {}, {}] for '{}'",
            requested.red,
            requested.green,
            requested.blue,
            resolved.red,
            resolved.green,
            resolved.blue,
            image_id
        );
    }
    Some(ResolvedBands {
        requested,
        resolved,
        embedding_key: resolved.embedding_key(image_id),
    })
}

/// Loaded image data ready for inference.
struct LoadedImage {
    cache_key: String,
    rgb_data: Vec<u8>,
    width: u32,
    height: u32,
}

/// Load RGB image data with caching. If `dimensions_only` is true, only
/// width/height are needed (rgb_data will be empty) — used when the backend
/// already has a prepared embedding.
async fn load_image_for_inference(
    state: &AppState,
    streams: &Arc<RwLock<ConnectionStreams>>,
    image_id: &str,
    bands: &ResolvedBands,
    dimensions_only: bool,
    request_id: u32,
    tx: &mpsc::Sender<Message>,
) -> Option<LoadedImage> {
    if dimensions_only {
        // Fast path: try prepared-dimensions cache first.
        if let Some((w, h)) = streams
            .write()
            .await
            .prepared_dimensions(&bands.embedding_key)
        {
            return Some(LoadedImage {
                cache_key: bands.embedding_key.clone(),
                rgb_data: Vec::new(),
                width: w,
                height: h,
            });
        }
    }

    if bands.resolved.is_default_rgb() {
        // Default RGB [0,1,2]: use per-connection image cache.
        let cached = {
            let guard = streams.read().await;
            guard
                .get_cached_image(image_id)
                .map(|c| (c.rgb_data.clone(), c.width, c.height))
        };
        if let Some((rgb, w, h)) = cached {
            let rgb_data = if dimensions_only { Vec::new() } else { rgb };
            return Some(LoadedImage {
                cache_key: bands.embedding_key.clone(),
                rgb_data,
                width: w,
                height: h,
            });
        }
        match load_image_rgb(state, image_id).await {
            Ok((rgb, w, h)) => {
                streams
                    .write()
                    .await
                    .cache_image(image_id.to_string(), rgb.clone(), w, h);
                let rgb_data = if dimensions_only { Vec::new() } else { rgb };
                Some(LoadedImage {
                    cache_key: bands.embedding_key.clone(),
                    rgb_data,
                    width: w,
                    height: h,
                })
            }
            Err(e) => {
                tracing::error!("Failed to load image '{}': {}", image_id, e);
                send_error(
                    tx,
                    request_id,
                    ErrorCode::ImageNotFound,
                    format!("Failed to load image: {}", e),
                )
                .await;
                None
            }
        }
    } else {
        // Custom band selection — no caching (varies per request).
        match load_image_rgb_with_bands(
            state,
            image_id,
            bands.resolved.red,
            bands.resolved.green,
            bands.resolved.blue,
        )
        .await
        {
            Ok((rgb, w, h, _)) => {
                let rgb_data = if dimensions_only { Vec::new() } else { rgb };
                Some(LoadedImage {
                    cache_key: bands.embedding_key.clone(),
                    rgb_data,
                    width: w,
                    height: h,
                })
            }
            Err(e) => {
                tracing::error!(
                    "Failed to load image '{}' with bands [{}, {}, {}] (resolved [{}, {}, {}]): {}",
                    image_id,
                    bands.requested.red,
                    bands.requested.green,
                    bands.requested.blue,
                    bands.resolved.red,
                    bands.resolved.green,
                    bands.resolved.blue,
                    e
                );
                send_error(
                    tx,
                    request_id,
                    ErrorCode::ImageNotFound,
                    format!("Failed to load image with bands: {}", e),
                )
                .await;
                None
            }
        }
    }
}

/// For embedding-backed models, resolve clamped band key lazily and verify
/// the embedding is ready. Returns `false` if an error was sent to the client.
async fn ensure_embedding_ready(
    ctx: &MuxRequestCtx,
    backend: &dyn InferenceBackend,
    image_id: &str,
    bands: &mut ResolvedBands,
    request_id: u32,
) -> bool {
    if !backend.requires_embedding() {
        return true;
    }

    // Resolve clamped key so e.g. [19,1,2] and [2,1,2] share the same
    // embedding on a 3-band image.
    if !bands.requested.is_default_rgb() && !backend.is_prepared(&bands.embedding_key).await {
        match resolve_sam_bands(&ctx.state, image_id, bands.requested).await {
            Ok(resolved) => {
                bands.resolved = resolved;
                bands.embedding_key = resolved.embedding_key(image_id);
            }
            Err(e) => {
                tracing::error!("Failed to resolve band selection for '{}': {}", image_id, e);
                send_error(
                    &ctx.tx,
                    request_id,
                    ErrorCode::ImageNotFound,
                    format!("Failed to resolve image bands: {}", e),
                )
                .await;
                return false;
            }
        }
    }

    if !backend.is_prepared(&bands.embedding_key).await {
        let error = ProtocolError::error(
            ErrorCode::EmbeddingRequired,
            format!(
                "Model embedding not ready for image '{}' and bands [{}, {}, {}] (resolved [{}, {}, {}]). Call prepare_model first.",
                image_id,
                bands.requested.red,
                bands.requested.green,
                bands.requested.blue,
                bands.resolved.red,
                bands.resolved.green,
                bands.resolved.blue
            ),
        );
        ctx.tx
            .send(Message::Binary(
                encode_stream_error(request_id, &error).into(),
            ))
            .await
            .ok();
        return false;
    }

    true
}

/// Build a non-blocking progress callback that forwards to the client channel.
fn make_progress_callback(
    tx: &mpsc::Sender<Message>,
    request_id: u32,
) -> Option<crate::inference::ProgressCallback> {
    let progress_tx = tx.clone();
    Some(Box::new(move |progress: u8, status: &str| {
        let msg = encode_infer_progress(request_id, progress, status);
        // try_send is non-blocking — if channel is full, progress update is dropped.
        progress_tx.try_send(Message::Binary(msg.into())).ok();
    }))
}

// ---------------------------------------------------------------------------
// prepare_model / infer handlers
// ---------------------------------------------------------------------------

/// Handle prepare_model request.
async fn handle_prepare_model(ctx: MuxRequestCtx, request: PrepareModelRequest) {
    let PrepareModelRequest {
        request_id,
        model_id,
        image_id,
        config,
    } = request;

    tracing::info!(
        "Mux connection {}: prepare_model {} for {:?} (request_id={})",
        ctx.connection_id,
        model_id,
        image_id,
        request_id
    );

    let backend = match resolve_model_backend(&ctx.state, &model_id, request_id, &ctx.tx).await {
        Some(b) => b,
        None => return,
    };
    let image_id = match resolve_image_id(&ctx.streams, image_id, request_id, &ctx.tx).await {
        Some(id) => id,
        None => return,
    };
    let bands =
        match resolve_band_selection(&ctx.state, &image_id, config.as_ref(), request_id, &ctx.tx)
            .await
        {
            Some(b) => b,
            None => return,
        };
    let loaded = match load_image_for_inference(
        &ctx.state,
        &ctx.streams,
        &image_id,
        &bands,
        false,
        request_id,
        &ctx.tx,
    )
    .await
    {
        Some(l) => l,
        None => return,
    };

    let image_context = crate::inference::ImageContext {
        image_id: bands.embedding_key.clone(),
        rgb_data: loaded.rgb_data,
        width: loaded.width,
        height: loaded.height,
    };

    match backend
        .prepare(&image_context, make_progress_callback(&ctx.tx, request_id))
        .await
    {
        Ok(()) => {
            ctx.streams.write().await.cache_prepared_dimensions(
                bands.embedding_key,
                loaded.width,
                loaded.height,
            );
            let msg = encode_model_ready(request_id, &model_id);
            ctx.tx.send(Message::Binary(msg.into())).await.ok();
            tracing::info!(
                "Mux connection {}: model {} ready for '{}'",
                ctx.connection_id,
                model_id,
                image_id
            );
        }
        Err(e) => {
            tracing::error!("Model prepare failed: {}", e);
            send_error(
                &ctx.tx,
                request_id,
                ErrorCode::ModelEncodeFailed,
                format!("Failed to prepare model: {}", e),
            )
            .await;
        }
    }
}

/// Handle infer request.
async fn handle_infer(ctx: MuxRequestCtx, request: InferModelRequest) {
    let InferModelRequest {
        request_id,
        model_id,
        image_id,
        inputs,
        options,
    } = request;

    tracing::debug!(
        "Mux connection {}: infer {} (request_id={})",
        ctx.connection_id,
        model_id,
        request_id
    );

    let backend = match resolve_model_backend(&ctx.state, &model_id, request_id, &ctx.tx).await {
        Some(b) => b,
        None => return,
    };

    if let Err(e) = backend.validate_inputs(&inputs) {
        send_error(
            &ctx.tx,
            request_id,
            ErrorCode::InvalidInput,
            format!("Invalid inputs: {}", e),
        )
        .await;
        return;
    }

    let image_id = match resolve_image_id(&ctx.streams, image_id, request_id, &ctx.tx).await {
        Some(id) => id,
        None => return,
    };

    // Resolve bands — for infer, band config comes from `options`.
    let mut bands =
        match resolve_band_selection(&ctx.state, &image_id, Some(&options), request_id, &ctx.tx)
            .await
        {
            Some(b) => b,
            None => return,
        };

    if !ensure_embedding_ready(&ctx, backend.as_ref(), &image_id, &mut bands, request_id).await {
        return;
    }

    let dimensions_only = backend.requires_embedding();
    let loaded = match load_image_for_inference(
        &ctx.state,
        &ctx.streams,
        &image_id,
        &bands,
        dimensions_only,
        request_id,
        &ctx.tx,
    )
    .await
    {
        Some(l) => l,
        None => return,
    };

    // For non-embedding backends, cache_key is the plain image_id.
    let cache_key = if backend.requires_embedding() {
        loaded.cache_key
    } else {
        image_id.clone()
    };

    let image_context = crate::inference::ImageContext {
        image_id: cache_key,
        rgb_data: loaded.rgb_data,
        width: loaded.width,
        height: loaded.height,
    };

    match backend
        .infer(
            &image_context,
            inputs,
            options,
            make_progress_callback(&ctx.tx, request_id),
        )
        .await
    {
        Ok(result) => {
            let result_json = serde_json::json!({
                "model_id": result.model_id,
                "outputs": result.outputs,
                "timing_ms": result.timing_ms,
            });
            let json_str = serde_json::to_string(&result_json).unwrap();
            let msg = encode_infer_result(request_id, &json_str);
            ctx.tx.send(Message::Binary(msg.into())).await.ok();
            tracing::debug!(
                "Mux connection {}: inference complete ({}ms)",
                ctx.connection_id,
                result.timing_ms
            );
        }
        Err(e) => {
            tracing::error!("Inference failed: {}", e);
            send_error(
                &ctx.tx,
                request_id,
                ErrorCode::ModelDecodeFailed,
                format!("Inference failed: {}", e),
            )
            .await;
        }
    }
}

/// Spawn a tracked pyramid build task.
///
/// The task is registered in `AppState::pyramid_tasks` so it can be:
/// - Detected to avoid duplicate builds for the same image
/// - Cancelled if needed (future enhancement)
/// - Cleaned up when complete
async fn spawn_pyramid_task(state: Arc<AppState>, image_path: &Path, image_hash: &str) {
    let builder = state.pyramid_builder.clone();
    let storage = state.pyramid_storage.clone();
    let loaders = state.loaders.clone();
    let path = image_path.to_path_buf();
    let hash = image_hash.to_string();
    let hash_for_insert = hash.clone();
    let state_for_cleanup = state.clone();

    let handle = tokio::spawn(async move {
        tracing::info!("Starting pyramid generation for {}", hash);

        if let Some(loader) = loaders.find_loader(&path) {
            match loader.load_bands(&path).await {
                Ok(bands) => {
                    if let Err(e) = builder.build_and_save(&hash, &bands).await {
                        tracing::error!("Failed to build pyramid: {}", e);
                        storage.mark_failed(&hash, &e.to_string()).await.ok();
                    } else {
                        tracing::info!("Pyramid {} built successfully", hash);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to load bands for pyramid: {}", e);
                    storage.mark_failed(&hash, &e.to_string()).await.ok();
                }
            }
        }

        // Clean up: remove ourselves from the task registry
        state_for_cleanup.pyramid_tasks.write().await.remove(&hash);
    });

    // Register the task
    state
        .pyramid_tasks
        .write()
        .await
        .insert(hash_for_insert, handle);
}

/// Resolve SAM band selection against image metadata so keys are canonical.
async fn resolve_sam_bands(
    state: &AppState,
    image_id: &str,
    requested: SamBands,
) -> Result<SamBands, Error> {
    let image_path = find_image(state, image_id)?;
    let loader = state
        .loaders
        .find_loader(&image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_id.to_string()))?;
    let metadata = loader.load_metadata(&image_path).await?;
    Ok(requested.clamp_to_num_bands(metadata.num_bands))
}

/// Load RGB image data for inference.
async fn load_image_rgb(state: &AppState, image_id: &str) -> Result<(Vec<u8>, u32, u32), Error> {
    let image_path = find_image(state, image_id)?;

    // For hyperspectral images, we need to extract RGB bands
    let loader = state
        .loaders
        .find_loader(&image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_id.to_string()))?;

    let bands = loader.load_bands(&image_path).await?;

    // Extract first 3 bands as RGB (or single band as grayscale)
    let width = bands.width;
    let height = bands.height;
    let pixel_count = (width * height) as usize;

    let rgb_data = if bands.num_bands() >= 3 {
        // Use first 3 bands as RGB
        let r = &bands.bands[0];
        let g = &bands.bands[1];
        let b = &bands.bands[2];

        let mut rgb = Vec::with_capacity(pixel_count * 3);
        for i in 0..pixel_count {
            // Convert f32 [0.0, 1.0] to u8 [0, 255]
            rgb.push((r[i].clamp(0.0, 1.0) * 255.0) as u8);
            rgb.push((g[i].clamp(0.0, 1.0) * 255.0) as u8);
            rgb.push((b[i].clamp(0.0, 1.0) * 255.0) as u8);
        }
        rgb
    } else if bands.num_bands() == 1 {
        // Grayscale - replicate to RGB
        let gray = &bands.bands[0];
        let mut rgb = Vec::with_capacity(pixel_count * 3);
        for &pixel in gray {
            // Convert f32 [0.0, 1.0] to u8 [0, 255]
            let byte = (pixel.clamp(0.0, 1.0) * 255.0) as u8;
            rgb.push(byte);
            rgb.push(byte);
            rgb.push(byte);
        }
        rgb
    } else {
        return Err(Error::Internal(format!(
            "Image has {} bands, expected 1 or 3+",
            bands.num_bands()
        )));
    };

    Ok((rgb_data, width, height))
}

/// Load image with specific band selection for RGB channels.
///
/// This is used for SAM encoding with custom band selection on hyperspectral images.
async fn load_image_rgb_with_bands(
    state: &AppState,
    image_id: &str,
    red_band: u32,
    green_band: u32,
    blue_band: u32,
) -> Result<(Vec<u8>, u32, u32, SamBands), Error> {
    let image_path = find_image(state, image_id)?;

    // Load all bands
    let loader = state
        .loaders
        .find_loader(&image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_id.to_string()))?;

    let bands = loader.load_bands(&image_path).await?;

    let width = bands.width;
    let height = bands.height;
    let pixel_count = (width * height) as usize;
    let num_bands = bands.num_bands();

    // Clamp band indices to valid range
    let resolved = SamBands {
        red: red_band,
        green: green_band,
        blue: blue_band,
    }
    .clamp_to_num_bands(num_bands);
    let red_idx = resolved.red as usize;
    let green_idx = resolved.green as usize;
    let blue_idx = resolved.blue as usize;

    tracing::info!(
        "Extracting bands [{}, {}, {}] from {} total bands for SAM",
        red_idx,
        green_idx,
        blue_idx,
        num_bands
    );

    // Extract selected bands as RGB
    let r = &bands.bands[red_idx];
    let g = &bands.bands[green_idx];
    let b = &bands.bands[blue_idx];

    let mut rgb_data = Vec::with_capacity(pixel_count * 3);
    for i in 0..pixel_count {
        // Convert f32 [0.0, 1.0] to u8 [0, 255]
        rgb_data.push((r[i].clamp(0.0, 1.0) * 255.0) as u8);
        rgb_data.push((g[i].clamp(0.0, 1.0) * 255.0) as u8);
        rgb_data.push((b[i].clamp(0.0, 1.0) * 255.0) as u8);
    }

    Ok((rgb_data, width, height, resolved))
}

/// Spawn a task to stream an image.
fn spawn_stream_task(
    state: Arc<AppState>,
    streams: Arc<RwLock<ConnectionStreams>>,
    tx: mpsc::Sender<Message>,
    request_id: u32,
    image_id: String,
    level: u32,
    progressive: bool,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        let result = stream_image_mux(
            state.clone(),
            tx.clone(),
            request_id,
            &image_id,
            level,
            progressive,
        )
        .await;

        // Handle errors
        if let Err(e) = result {
            tracing::error!("Stream {} error: {}", request_id, e);
            let error = ProtocolError::error(ErrorCode::InternalError, e.to_string());
            tx.send(Message::Binary(
                encode_stream_error(request_id, &error).into(),
            ))
            .await
            .ok();
        }

        // Remove ourselves from active streams
        streams.write().await.remove(request_id);
        tracing::debug!("Stream {} completed and removed", request_id);
    })
}

/// Stream an image with multiplexed protocol.
async fn stream_image_mux(
    state: Arc<AppState>,
    tx: mpsc::Sender<Message>,
    request_id: u32,
    image_id: &str,
    level: u32,
    progressive: bool,
) -> Result<(), Error> {
    // Find the image file
    let image_path = find_image(&state, image_id)?;
    let image_hash = compute_image_hash(&image_path)?;

    // Check pyramid status
    let pyramid_status = state.pyramid_storage.get_status(&image_hash).await;

    // Get full dimensions and max level
    let (max_level, full_width, full_height) = if pyramid_status == PyramidStatus::Ready {
        let metadata = state.pyramid_storage.load_metadata(&image_hash).await?;
        (
            metadata.levels.len().saturating_sub(1) as u32,
            metadata.full_width,
            metadata.full_height,
        )
    } else {
        let loader = state
            .loaders
            .find_loader(&image_path)
            .ok_or_else(|| Error::UnsupportedFormat(image_id.to_string()))?;
        let metadata = loader.load_metadata(&image_path).await?;
        (
            crate::pyramid::calculate_num_levels(metadata.width, metadata.height).saturating_sub(1),
            metadata.width,
            metadata.height,
        )
    };

    let target_level = level.min(max_level);

    if progressive {
        tracing::info!(
            "Stream {}: progressive {} from level {} to {} (full: {}x{})",
            request_id,
            image_id,
            max_level,
            target_level,
            full_width,
            full_height
        );

        // Stream from highest (smallest) to target (largest)
        for (idx, current_level) in (target_level..=max_level).rev().enumerate() {
            let send_reset = idx == 0;
            stream_level_mux(
                state.clone(),
                &tx,
                request_id,
                StreamLevelRequest {
                    image_id,
                    image_path: &image_path,
                    image_hash: &image_hash,
                    level: current_level,
                    full_width,
                    full_height,
                    send_reset,
                },
            )
            .await?;
        }
    } else {
        // Single level
        stream_level_mux(
            state.clone(),
            &tx,
            request_id,
            StreamLevelRequest {
                image_id,
                image_path: &image_path,
                image_hash: &image_hash,
                level: target_level,
                full_width,
                full_height,
                send_reset: true,
            },
        )
        .await?;
    }

    // Send stream complete
    tx.send(Message::Binary(encode_stream_complete(request_id).into()))
        .await
        .ok();

    tracing::info!("Stream {} complete for '{}'", request_id, image_id);
    Ok(())
}

/// Stream a single pyramid level with multiplexed protocol.
async fn stream_level_mux(
    state: Arc<AppState>,
    tx: &mpsc::Sender<Message>,
    request_id: u32,
    request: StreamLevelRequest<'_>,
) -> Result<(), Error> {
    let StreamLevelRequest {
        image_id,
        image_path,
        image_hash,
        level,
        full_width,
        full_height,
        send_reset,
    } = request;

    let pyramid_status = state.pyramid_storage.get_status(image_hash).await;

    match pyramid_status {
        PyramidStatus::Ready => {
            stream_from_pyramid_mux(&state, tx, request_id, image_hash, level, send_reset).await
        }
        PyramidStatus::Building => {
            tracing::debug!(
                "Pyramid for '{}' is building; streaming level {} from source",
                image_id,
                level
            );
            stream_from_source_mux(
                &state,
                tx,
                request_id,
                StreamSourceRequest {
                    image_path,
                    image_id,
                    level,
                    full_width,
                    full_height,
                    send_reset,
                },
            )
            .await
        }
        _ => {
            // Stream from source and trigger pyramid build
            if matches!(
                pyramid_status,
                PyramidStatus::Pending | PyramidStatus::Failed
            ) && !state.pyramid_tasks.read().await.contains_key(image_hash)
            {
                spawn_pyramid_task(state.clone(), image_path, image_hash).await;
            }

            stream_from_source_mux(
                &state,
                tx,
                request_id,
                StreamSourceRequest {
                    image_path,
                    image_id,
                    level,
                    full_width,
                    full_height,
                    send_reset,
                },
            )
            .await
        }
    }
}

/// Stream from pyramid cache with multiplexed protocol.
async fn stream_from_pyramid_mux(
    state: &AppState,
    tx: &mpsc::Sender<Message>,
    request_id: u32,
    image_hash: &str,
    level: u32,
    send_reset: bool,
) -> Result<(), Error> {
    use hvat_common::PyramidLevel;

    let metadata = state.pyramid_storage.load_metadata(image_hash).await?;
    let max_level = metadata.levels.len().saturating_sub(1) as u32;
    let level = PyramidLevel::from_u32_clamped(level.min(max_level));

    let level_info = metadata
        .levels
        .get(level.as_usize())
        .ok_or_else(|| Error::Internal(format!("Pyramid level {} not found", level.as_u8())))?;

    // Send Reset if needed
    if send_reset {
        tx.send(Message::Binary(encode_reset_mux(request_id).into()))
            .await
            .map_err(|_| Error::Internal("Failed to send reset".to_string()))?;
    }

    // Send metadata
    let stream_meta = StreamMetadata {
        width: level_info.width,
        height: level_info.height,
        num_bands: metadata.num_bands as u32,
        num_layers: level_info.num_layers,
        full_width: metadata.full_width,
        full_height: metadata.full_height,
    };
    tx.send(Message::Binary(stream_meta.to_bytes_mux(request_id).into()))
        .await
        .map_err(|_| Error::Internal("Failed to send metadata".to_string()))?;

    // Load and stream layers
    let layers = state
        .pyramid_storage
        .load_level(image_hash, level.as_u32())
        .await?;

    let chunk_rows = state.config.stream_chunk_rows;

    for (layer_idx, rgba_data) in layers {
        let bytes_per_row = (level_info.width * 4) as usize;
        let total_rows = level_info.height;
        let mut row = 0u32;
        let mut chunks_since_yield = 0u32;

        while row < total_rows {
            let rows_to_send = chunk_rows.min(total_rows - row);
            let start_byte = (row as usize) * bytes_per_row;
            let end_byte = ((row + rows_to_send) as usize) * bytes_per_row;

            let chunk_data = &rgba_data[start_byte..end_byte];
            let msg = encode_layer_chunk_mux(
                request_id,
                layer_idx as u16,
                row,
                row + rows_to_send,
                chunk_data,
            );

            if tx.send(Message::Binary(msg.into())).await.is_err() {
                return Ok(());
            }

            row += rows_to_send;
            chunks_since_yield += 1;
            if chunks_since_yield >= YIELD_EVERY_N_CHUNKS {
                chunks_since_yield = 0;
                tokio::task::yield_now().await;
            }
        }

        let msg = encode_layer_complete_mux(request_id, layer_idx as u16);
        if tx.send(Message::Binary(msg.into())).await.is_err() {
            return Ok(());
        }
    }

    let msg = encode_level_complete_mux(request_id, level.as_u8());
    tx.send(Message::Binary(msg.into())).await.ok();

    Ok(())
}

/// Stream from source file with multiplexed protocol.
async fn stream_from_source_mux(
    state: &AppState,
    tx: &mpsc::Sender<Message>,
    request_id: u32,
    request: StreamSourceRequest<'_>,
) -> Result<(), Error> {
    let StreamSourceRequest {
        image_path,
        image_id,
        level,
        full_width,
        full_height,
        send_reset,
    } = request;

    let loader = state
        .loaders
        .find_loader(image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_id.to_string()))?;

    let bands = loader.load_bands(image_path).await?;

    // Calculate target size
    let (target_width, target_height) = calculate_level_size(bands.width, bands.height, level);

    // Downsample if needed
    let bands = if target_width != bands.width || target_height != bands.height {
        downsample_bands(&bands, target_width, target_height)
    } else {
        bands
    };

    // Send Reset if needed
    if send_reset {
        tx.send(Message::Binary(encode_reset_mux(request_id).into()))
            .await
            .map_err(|_| Error::Internal("Failed to send reset".to_string()))?;
    }

    // Send metadata
    let metadata = StreamMetadata {
        width: bands.width,
        height: bands.height,
        num_bands: bands.num_bands() as u32,
        num_layers: bands.num_layers(),
        full_width,
        full_height,
    };
    tx.send(Message::Binary(metadata.to_bytes_mux(request_id).into()))
        .await
        .map_err(|_| Error::Internal("Failed to send metadata".to_string()))?;

    // Pack and stream layers
    let layers = pack_bands_to_rgba_layers(&bands.bands, bands.width, bands.height);
    let chunk_rows = state.config.stream_chunk_rows;

    for (layer_idx, rgba_data) in layers {
        let bytes_per_row = (bands.width * 4) as usize;
        let total_rows = bands.height;
        let mut row = 0u32;
        let mut chunks_since_yield = 0u32;

        while row < total_rows {
            let rows_to_send = chunk_rows.min(total_rows - row);
            let start_byte = (row as usize) * bytes_per_row;
            let end_byte = ((row + rows_to_send) as usize) * bytes_per_row;

            let chunk_data = &rgba_data[start_byte..end_byte];
            let msg = encode_layer_chunk_mux(
                request_id,
                layer_idx as u16,
                row,
                row + rows_to_send,
                chunk_data,
            );

            if tx.send(Message::Binary(msg.into())).await.is_err() {
                return Ok(());
            }

            row += rows_to_send;
            chunks_since_yield += 1;
            if chunks_since_yield >= YIELD_EVERY_N_CHUNKS {
                chunks_since_yield = 0;
                tokio::task::yield_now().await;
            }
        }

        let msg = encode_layer_complete_mux(request_id, layer_idx as u16);
        if tx.send(Message::Binary(msg.into())).await.is_err() {
            return Ok(());
        }
    }

    let msg = encode_level_complete_mux(request_id, level as u8);
    tx.send(Message::Binary(msg.into())).await.ok();

    Ok(())
}

/// Calculate target size for a pyramid level.
fn calculate_level_size(full_width: u32, full_height: u32, level: u32) -> (u32, u32) {
    if level == 0 {
        return (full_width, full_height);
    }
    let scale = 1u32 << level;
    let width = (full_width / scale).max(1);
    let height = (full_height / scale).max(1);
    (width, height)
}

/// Build server capabilities.
fn build_server_capabilities(state: &AppState) -> ServerCapabilities {
    use hvat_common::{DownloadMode, ServerFeatures, ServerInfo, ServerLimits};

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
            inference: inference_enabled,
            sam: false,
        },
        download_mode: DownloadMode::Chunked,
        models,
    };
    capabilities.features.sam = capabilities.find_sam_prompt_model().is_some();
    capabilities
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ServerConfig;
    use crate::state::AppState;

    #[test]
    fn prepared_dimensions_eviction_is_lru_by_access() {
        let mut streams = ConnectionStreams::new(4, 2);

        for i in 0..MAX_PREPARED_DIMENSIONS {
            streams.cache_prepared_dimensions(format!("k{}", i), i as u32, i as u32);
        }

        // Touch k0 so it is no longer the least-recently-used entry.
        assert_eq!(streams.prepared_dimensions("k0"), Some((0, 0)));

        // Insert one more key and ensure k1 (the true LRU) is evicted.
        streams.cache_prepared_dimensions("k_new".to_string(), 999, 999);
        assert!(streams.prepared_dimensions("k1").is_none());
        assert_eq!(streams.prepared_dimensions("k0"), Some((0, 0)));
        assert_eq!(streams.prepared_dimensions("k_new"), Some((999, 999)));
        assert_eq!(streams.prepared_dimensions.len(), MAX_PREPARED_DIMENSIONS);
    }

    #[test]
    fn set_image_keeps_prepared_dimensions_for_other_images() {
        let mut streams = ConnectionStreams::new(4, 2);
        streams.cache_prepared_dimensions("img_a#0:1:2".to_string(), 100, 200);
        streams.cache_prepared_dimensions("img_b#0:1:2".to_string(), 300, 400);

        streams.set_image("img_a".to_string());
        streams.set_image("img_b".to_string());

        assert_eq!(streams.prepared_dimensions("img_a#0:1:2"), Some((100, 200)));
        assert_eq!(streams.prepared_dimensions("img_b#0:1:2"), Some((300, 400)));
    }

    #[test]
    fn capabilities_without_models_disable_inference_and_sam() {
        let state = AppState::new(ServerConfig::default());
        let caps = build_server_capabilities(&state);

        assert!(caps.features.streaming);
        assert!(caps.features.project_state);
        assert!(caps.features.downloads);
        assert!(caps.supports_chunked_downloads());
        assert!(!caps.features.inference);
        assert!(!caps.features.sam);
        assert!(caps.models.is_empty());
        assert!(!caps.supports_inference());
        assert!(!caps.supports_sam());
    }
}
