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
//! { "action": "start_stream_mux", "request_id": 1, "image_id": "img.png", "level": 0, "progressive": true }
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
use tokio::sync::{RwLock, mpsc};
use tokio::task::JoinHandle;

use crate::error::Error;
use crate::packer::{downsample_bands, pack_bands_to_rgba_layers};
use crate::protocol::{
    ClientMessage, ErrorCode, PROTOCOL_VERSION, ProtocolError, ServerCapabilities, StreamMetadata,
    encode_capabilities, encode_layer_chunk_mux, encode_layer_complete_mux,
    encode_level_complete_mux, encode_ping, encode_reset_mux, encode_stream_complete,
    encode_stream_error,
};
use crate::pyramid::{PyramidStatus, compute_image_hash};
use crate::state::AppState;
use crate::utils::find_image;

/// State for a single active stream within a multiplexed connection.
struct ActiveStream {
    /// Task handle for the streaming coroutine
    task: JoinHandle<()>,
    /// Image ID being streamed
    image_id: String,
}

/// Manages all active streams for a single WebSocket connection.
struct ConnectionStreams {
    /// Active streams indexed by request_id
    streams: HashMap<u32, ActiveStream>,
    /// Maximum concurrent streams allowed
    max_streams: u32,
}

impl ConnectionStreams {
    fn new(max_streams: u32) -> Self {
        Self {
            streams: HashMap::new(),
            max_streams,
        }
    }

    /// Check if we can start a new stream.
    fn can_start_stream(&self) -> bool {
        (self.streams.len() as u32) < self.max_streams
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

    /// Get the number of active streams.
    fn len(&self) -> usize {
        self.streams.len()
    }
}

/// Handle a multiplexed WebSocket connection.
///
/// This is the entry point for the `/api/ws` endpoint. It manages multiple
/// concurrent image streams over a single WebSocket connection.
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
            let _ = sender
                .send(Message::Binary(encode_stream_error(0, &error).into()))
                .await;
            let _ = sender.close().await;
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
    )));

    // Build and send server capabilities immediately
    let capabilities = build_server_capabilities(&state);
    let capabilities_bytes = encode_capabilities(&capabilities);

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
                            let _ = tx
                                .send(Message::Binary(encode_stream_error(0, &error).into()))
                                .await;
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

    // Cancel all active streams
    streams.write().await.cancel_all();

    // Clean up
    drop(tx);
    let _ = send_task.await;
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
    match msg {
        ClientMessage::StartStreamMux {
            request_id,
            image_id,
            level,
            progressive,
        } => {
            // Check if we can start a new stream
            {
                let streams_guard = streams.read().await;
                if !streams_guard.can_start_stream() {
                    let error = ProtocolError::retryable(
                        ErrorCode::RateLimited,
                        format!(
                            "Maximum concurrent streams ({}) reached",
                            streams_guard.len()
                        ),
                        1000,
                    );
                    let _ = tx
                        .send(Message::Binary(
                            encode_stream_error(request_id, &error).into(),
                        ))
                        .await;
                    return;
                }
            }

            tracing::info!(
                "Mux connection {}: starting stream {} for '{}' (level={}, progressive={})",
                connection_id,
                request_id,
                image_id,
                level,
                progressive
            );

            // Spawn the stream task
            let task = spawn_stream_task(
                state.clone(),
                streams.clone(),
                tx.clone(),
                request_id,
                image_id.clone(),
                level,
                progressive,
            );

            // Register the stream
            streams
                .write()
                .await
                .insert(request_id, ActiveStream { task, image_id });
        }

        ClientMessage::CancelStream { request_id } => {
            let cancelled = streams.write().await.cancel(request_id);
            if cancelled {
                tracing::info!(
                    "Mux connection {}: cancelled stream {}",
                    connection_id,
                    request_id
                );
            } else {
                tracing::debug!(
                    "Mux connection {}: stream {} not found (already complete?)",
                    connection_id,
                    request_id
                );
            }
        }

        // Legacy messages - not supported on multiplexed endpoint
        ClientMessage::StartStream { .. } | ClientMessage::Cancel => {
            let error = ProtocolError::error(
                ErrorCode::InvalidAction,
                "Legacy protocol not supported on this endpoint. Use start_stream_mux instead.",
            );
            let _ = tx
                .send(Message::Binary(encode_stream_error(0, &error).into()))
                .await;
        }

        // SAM messages - use request_id = 0 for now (TODO: add request_id to SAM)
        ClientMessage::SamEmbed | ClientMessage::SamSegment { .. } => {
            let error = ProtocolError::error(
                ErrorCode::InvalidAction,
                "SAM operations not yet supported on multiplexed endpoint",
            );
            let _ = tx
                .send(Message::Binary(encode_stream_error(0, &error).into()))
                .await;
        }

        ClientMessage::Pong { .. } => {
            // Handled in the main receive loop
        }
    }
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
            let _ = tx
                .send(Message::Binary(
                    encode_stream_error(request_id, &error).into(),
                ))
                .await;
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
                image_id,
                &image_path,
                &image_hash,
                current_level,
                full_width,
                full_height,
                send_reset,
            )
            .await?;
        }
    } else {
        // Single level
        stream_level_mux(
            state.clone(),
            &tx,
            request_id,
            image_id,
            &image_path,
            &image_hash,
            target_level,
            full_width,
            full_height,
            true,
        )
        .await?;
    }

    // Send stream complete
    let _ = tx
        .send(Message::Binary(encode_stream_complete(request_id).into()))
        .await;

    tracing::info!("Stream {} complete for '{}'", request_id, image_id);
    Ok(())
}

/// Stream a single pyramid level with multiplexed protocol.
async fn stream_level_mux(
    state: Arc<AppState>,
    tx: &mpsc::Sender<Message>,
    request_id: u32,
    image_id: &str,
    image_path: &Path,
    image_hash: &str,
    level: u32,
    full_width: u32,
    full_height: u32,
    send_reset: bool,
) -> Result<(), Error> {
    let pyramid_status = state.pyramid_storage.get_status(image_hash).await;

    match pyramid_status {
        PyramidStatus::Ready => {
            stream_from_pyramid_mux(&state, tx, request_id, image_hash, level, send_reset).await
        }
        PyramidStatus::Building => {
            // Send retryable error
            let (progress, eta_seconds) = state
                .pyramid_storage
                .get_progress(image_hash)
                .await
                .unwrap_or((0.0, None));

            let error = ProtocolError::retryable(
                ErrorCode::PyramidNotReady,
                format!("Pyramid building ({:.0}% complete)", progress * 100.0),
                5000,
            )
            .with_context(hvat_common::ErrorContext::Pyramid {
                image_id: image_id.to_string(),
                progress: Some(progress),
                eta_seconds,
            });

            tx.send(Message::Binary(
                encode_stream_error(request_id, &error).into(),
            ))
            .await
            .map_err(|_| Error::Internal("Failed to send error".to_string()))?;

            Ok(())
        }
        _ => {
            // Stream from source and trigger pyramid build
            if pyramid_status == PyramidStatus::Pending || pyramid_status == PyramidStatus::Failed {
                if !state.pyramid_tasks.read().await.contains_key(image_hash) {
                    super::websocket::spawn_pyramid_task(state.clone(), image_path, image_hash)
                        .await;
                }
            }

            stream_from_source_mux(
                &state,
                tx,
                request_id,
                image_path,
                image_id,
                level,
                full_width,
                full_height,
                send_reset,
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
            tokio::task::yield_now().await;
        }

        let msg = encode_layer_complete_mux(request_id, layer_idx as u16);
        if tx.send(Message::Binary(msg.into())).await.is_err() {
            return Ok(());
        }
    }

    let msg = encode_level_complete_mux(request_id, level.as_u8());
    let _ = tx.send(Message::Binary(msg.into())).await;

    Ok(())
}

/// Stream from source file with multiplexed protocol.
async fn stream_from_source_mux(
    state: &AppState,
    tx: &mpsc::Sender<Message>,
    request_id: u32,
    image_path: &Path,
    image_id: &str,
    level: u32,
    full_width: u32,
    full_height: u32,
    send_reset: bool,
) -> Result<(), Error> {
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
            tokio::task::yield_now().await;
        }

        let msg = encode_layer_complete_mux(request_id, layer_idx as u16);
        if tx.send(Message::Binary(msg.into())).await.is_err() {
            return Ok(());
        }
    }

    let msg = encode_level_complete_mux(request_id, level as u8);
    let _ = tx.send(Message::Binary(msg.into())).await;

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

/// Get current timestamp in milliseconds.
fn current_timestamp_ms() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Build server capabilities.
fn build_server_capabilities(state: &AppState) -> ServerCapabilities {
    ServerCapabilities {
        protocol_version: PROTOCOL_VERSION,
        max_image_size: state.config.max_cache_memory,
        max_pyramid_levels: hvat_common::MAX_PYRAMID_LEVEL,
        sam_enabled: state.sam_engine.is_some(),
        sam_model: if state.sam_engine.is_some() {
            state.config.sam_variant.name().to_string()
        } else {
            String::new()
        },
        max_concurrent_streams: state.config.max_user_streams as u32,
    }
}
