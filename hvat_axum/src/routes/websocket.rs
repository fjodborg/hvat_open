//! WebSocket handler for binary image streaming.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use hvat_common::pixel_count_u32;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::error::Error;
use crate::packer::{downsample_bands, pack_bands_to_rgba_layers};
use crate::protocol::{
    ClientMessage, ErrorCode, ProtocolError, SamPoint as ProtocolSamPoint, ServerCapabilities,
    ServerResponse, StreamMetadata, encode_capabilities, encode_error, encode_layer_chunk,
    encode_layer_complete, encode_level_complete, encode_ping, encode_reset,
};
use crate::pyramid::{PyramidStatus, compute_image_hash};
use crate::sam::{CachedEmbedding, EncoderOutput, SamPoint as BackendSamPoint};
use crate::state::AppState;
use crate::utils::find_image;

/// Handle a WebSocket connection for image streaming.
pub async fn handle_websocket(socket: WebSocket, state: Arc<AppState>, image_id: String) {
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
                .send(Message::Binary(encode_error(&error).into()))
                .await;
            let _ = sender.close().await;
            tracing::warn!(
                "Rejected connection: max connections ({}) reached",
                state.config.max_connections
            );
            return;
        }
    };

    tracing::info!(
        "WebSocket connection {} established for image '{}' (active: {})",
        connection_id,
        image_id,
        state.connection_count()
    );

    // Run the connection handler with cleanup on exit
    let result = handle_websocket_inner(socket, state.clone(), image_id, connection_id).await;

    // Always release the connection slot
    state.release_connection();
    tracing::info!(
        "WebSocket connection {} closed (active: {})",
        connection_id,
        state.connection_count()
    );

    if let Err(e) = result {
        tracing::error!("WebSocket connection {} error: {}", connection_id, e);
    }
}

/// Inner WebSocket handler with ping/pong keepalive.
async fn handle_websocket_inner(
    socket: WebSocket,
    state: Arc<AppState>,
    image_id: String,
    connection_id: u64,
) -> Result<(), Error> {
    let (mut sender, mut receiver) = socket.split();

    // Channel for sending messages to the client
    let (tx, mut rx) = mpsc::channel::<Message>(32);

    // Shared state for ping/pong tracking
    let last_pong = Arc::new(AtomicU64::new(current_timestamp_ms()));
    let connection_alive = Arc::new(AtomicBool::new(true));

    // Build and send server capabilities immediately on connect
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

        // Forward all other messages
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

                // Check if client has timed out (no pong received)
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

                // Send ping
                let ping_msg = encode_ping(now);
                if tx_ping
                    .send(Message::Binary(ping_msg.into()))
                    .await
                    .is_err()
                {
                    connection_alive_ping.store(false, Ordering::SeqCst);
                    break;
                }

                tracing::trace!("Sent ping to connection {}", connection_id);
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
                    tracing::debug!("Received text message: {}", text);
                    // Parse client message
                    match serde_json::from_str::<ClientMessage>(&text) {
                        Ok(client_msg) => {
                            // Handle Pong specially - update last_pong timestamp
                            if let ClientMessage::Pong { timestamp } = &client_msg {
                                last_pong.store(current_timestamp_ms(), Ordering::SeqCst);
                                tracing::trace!(
                                    "Received pong from connection {} (ts: {})",
                                    connection_id,
                                    timestamp
                                );
                                continue;
                            }

                            if let Err(e) = handle_client_message(
                                client_msg,
                                state.clone(),
                                &image_id,
                                tx.clone(),
                            )
                            .await
                            {
                                // Send error response
                                let error =
                                    ProtocolError::error(ErrorCode::InternalError, e.to_string());
                                let error_bytes = encode_error(&error);
                                let _ = tx.send(Message::Binary(error_bytes.into())).await;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Invalid client message: {}", e);
                            let response = ServerResponse::Error {
                                message: format!("Invalid message: {}", e),
                            };
                            if let Ok(json) = serde_json::to_string(&response) {
                                let _ = tx.send(Message::Text(json.into())).await;
                            }
                        }
                    }
                }
                Message::Close(_) => {
                    tracing::debug!("Received close frame from connection {}", connection_id);
                    break;
                }
                _ => {} // Ignore binary messages from client
            },
            Err(e) => {
                tracing::warn!(
                    "WebSocket receive error on connection {}: {}",
                    connection_id,
                    e
                );
                break;
            }
        }
    }

    // Signal shutdown to other tasks
    connection_alive.store(false, Ordering::SeqCst);

    // Clean up
    drop(tx);

    // Wait for tasks to finish
    let _ = send_task.await;
    if let Some(ping_task) = ping_task {
        ping_task.abort();
    }

    Ok(())
}

/// Get current timestamp in milliseconds (monotonic).
fn current_timestamp_ms() -> u64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Handle a parsed client message.
async fn handle_client_message(
    msg: ClientMessage,
    state: Arc<AppState>,
    image_id: &str,
    tx: mpsc::Sender<Message>,
) -> Result<(), Error> {
    match msg {
        ClientMessage::StartStream { level, progressive } => {
            // image_id comes from the WebSocket URL path
            if progressive {
                tracing::info!(
                    "Progressive loading requested for {}, target level {}",
                    image_id,
                    level
                );

                // Stream progressively from highest level (thumbnail) down to target level
                let image_path = find_image(&state, image_id)?;
                let image_hash = compute_image_hash(&image_path)?;

                // Check pyramid status to determine max level and full dimensions
                let pyramid_status = state.pyramid_storage.get_status(&image_hash).await;

                let (max_level, full_width, full_height) = if pyramid_status == PyramidStatus::Ready
                {
                    // Use cached pyramid metadata
                    let metadata = state.pyramid_storage.load_metadata(&image_hash).await?;
                    (
                        metadata.levels.len().saturating_sub(1) as u32,
                        metadata.full_width,
                        metadata.full_height,
                    )
                } else {
                    // Calculate max level from image dimensions
                    let loader = state
                        .loaders
                        .find_loader(&image_path)
                        .ok_or_else(|| Error::UnsupportedFormat(image_id.to_string()))?;
                    let metadata = loader.load_metadata(&image_path).await?;
                    (
                        crate::pyramid::calculate_num_levels(metadata.width, metadata.height)
                            .saturating_sub(1),
                        metadata.width,
                        metadata.height,
                    )
                };

                let target_level = level.min(max_level);

                // Stream from highest (smallest/thumbnail) to target (largest/full-res)
                // Higher level number = smaller image
                tracing::info!(
                    "Starting progressive stream for {}: levels {} down to {} (full: {}x{})",
                    image_id,
                    max_level,
                    target_level,
                    full_width,
                    full_height
                );
                for (idx, current_level) in (target_level..=max_level).rev().enumerate() {
                    tracing::info!(
                        "Streaming progressive level {}/{} for {} (idx={}, full: {}x{})",
                        current_level,
                        max_level,
                        image_id,
                        idx,
                        full_width,
                        full_height
                    );
                    // Only send Reset before first level
                    let send_reset = idx == 0;
                    stream_image(
                        state.clone(),
                        image_id,
                        current_level,
                        full_width,
                        full_height,
                        tx.clone(),
                        send_reset,
                    )
                    .await?;
                    tracing::info!(
                        "Completed streaming level {} for {}",
                        current_level,
                        image_id
                    );
                }
                tracing::info!("Progressive stream complete for {}", image_id);
            } else {
                // Non-progressive: just stream the requested level
                // Get full dimensions for metadata
                let image_path = find_image(&state, image_id)?;
                let loader = state
                    .loaders
                    .find_loader(&image_path)
                    .ok_or_else(|| Error::UnsupportedFormat(image_id.to_string()))?;
                let metadata = loader.load_metadata(&image_path).await?;
                stream_image(
                    state.clone(),
                    image_id,
                    level,
                    metadata.width,
                    metadata.height,
                    tx,
                    true,
                )
                .await?;
            }
        }

        ClientMessage::Cancel => {
            // Cancellation is handled by dropping the tx channel
            tracing::info!("Stream cancelled for {}", image_id);
        }

        ClientMessage::SamEmbed => {
            handle_sam_embed(&state, image_id, tx).await?;
        }

        ClientMessage::SamSegment { points, box_prompt } => {
            handle_sam_segment(&state, image_id, points, box_prompt, tx).await?;
        }

        ClientMessage::Pong { .. } => {
            // Pong is handled in the receive loop for accurate timing
        }
    }

    Ok(())
}

/// Stream an image at a specific pyramid level.
///
/// # Arguments
/// * `send_reset` - If true, sends Reset message before streaming (should be true for first level only)
/// * `full_width` - Full resolution width (for progressive loading metadata)
/// * `full_height` - Full resolution height (for progressive loading metadata)
async fn stream_image(
    state: Arc<AppState>,
    image_id: &str,
    level: u32,
    full_width: u32,
    full_height: u32,
    tx: mpsc::Sender<Message>,
    send_reset: bool,
) -> Result<(), Error> {
    // Find the image file
    let image_path = find_image(&state, image_id)?;

    // Compute hash for pyramid cache lookup
    let image_hash = compute_image_hash(&image_path)?;

    // Check if pyramid is cached
    let pyramid_status = state.pyramid_storage.get_status(&image_hash).await;

    match pyramid_status {
        PyramidStatus::Ready => {
            // Stream from cached pyramid
            stream_from_pyramid(&state, &image_hash, level, tx, send_reset).await
        }
        PyramidStatus::Building => {
            // Pyramid is being built - send error with progress so client can show progress bar
            let (progress, eta_seconds) = state
                .pyramid_storage
                .get_progress(&image_hash)
                .await
                .unwrap_or((0.0, None));

            let error = ProtocolError::retryable(
                ErrorCode::PyramidNotReady,
                format!(
                    "Image pyramid is being built ({:.0}% complete)",
                    progress * 100.0
                ),
                5000, // Retry after 5 seconds
            )
            .with_context(hvat_common::ErrorContext::Pyramid {
                image_id: image_id.to_string(),
                progress: Some(progress),
                eta_seconds,
            });

            let error_bytes = encode_error(&error);
            tx.send(Message::Binary(error_bytes.into()))
                .await
                .map_err(|_| Error::Internal("Failed to send error".to_string()))?;

            Ok(())
        }
        _ => {
            // Not cached (Pending or Failed) - stream from source and start building pyramid
            if pyramid_status == PyramidStatus::Pending || pyramid_status == PyramidStatus::Failed {
                // Start building pyramid in background (if not already)
                if !state.pyramid_tasks.read().await.contains_key(&image_hash) {
                    spawn_pyramid_task(state.clone(), &image_path, &image_hash).await;
                }
            }

            // Stream directly from source
            stream_from_source(
                &state,
                &image_path,
                image_id,
                level,
                full_width,
                full_height,
                tx,
                send_reset,
            )
            .await
        }
    }
}

/// Stream from a cached pyramid level.
async fn stream_from_pyramid(
    state: &AppState,
    image_hash: &str,
    level: u32,
    tx: mpsc::Sender<Message>,
    send_reset: bool,
) -> Result<(), Error> {
    use hvat_common::PyramidLevel;

    // Load pyramid metadata
    let metadata = state.pyramid_storage.load_metadata(image_hash).await?;

    // Find the requested level (clamp to available levels)
    let max_level = metadata.levels.len().saturating_sub(1) as u32;
    let level = PyramidLevel::from_u32_clamped(level.min(max_level));

    let level_info = metadata.levels.get(level.as_usize()).ok_or_else(|| {
        Error::Internal(format!(
            "Pyramid level {} not found (available: 0-{})",
            level.as_u8(),
            metadata.levels.len().saturating_sub(1)
        ))
    })?;

    // Send Reset message to clear display before new image (only for first level in progressive mode)
    if send_reset {
        tx.send(Message::Binary(encode_reset().into()))
            .await
            .map_err(|_| Error::Internal("Failed to send reset".to_string()))?;
    }

    // Send stream metadata with full-res dimensions for progressive loading
    let stream_meta = StreamMetadata {
        width: level_info.width,
        height: level_info.height,
        num_bands: metadata.num_bands as u32,
        num_layers: level_info.num_layers,
        full_width: metadata.full_width,
        full_height: metadata.full_height,
    };
    tx.send(Message::Binary(stream_meta.to_bytes().into()))
        .await
        .map_err(|_| Error::Internal("Failed to send metadata".to_string()))?;

    // Load and stream the level data
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
            let msg = encode_layer_chunk(layer_idx as u16, row, row + rows_to_send, chunk_data);

            if tx.send(Message::Binary(msg.into())).await.is_err() {
                return Ok(());
            }

            row += rows_to_send;
            tokio::task::yield_now().await;
        }

        let msg = encode_layer_complete(layer_idx as u16);
        if tx.send(Message::Binary(msg.into())).await.is_err() {
            return Ok(());
        }
    }

    let msg = encode_level_complete(level.as_u8());
    let _ = tx.send(Message::Binary(msg.into())).await;

    Ok(())
}

/// Stream directly from source file (no pyramid cache).
///
/// # Arguments
/// * `full_width` - Full resolution width (for progressive loading metadata)
/// * `full_height` - Full resolution height (for progressive loading metadata)
async fn stream_from_source(
    state: &AppState,
    image_path: &std::path::Path,
    image_id: &str,
    level: u32,
    full_width: u32,
    full_height: u32,
    tx: mpsc::Sender<Message>,
    send_reset: bool,
) -> Result<(), Error> {
    // Find a loader
    let loader = state
        .loaders
        .find_loader(image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_id.to_string()))?;

    // Load band data
    let bands = loader.load_bands(image_path).await?;

    // Calculate target size for this pyramid level
    let (target_width, target_height) = calculate_level_size(bands.width, bands.height, level);

    // Downsample if needed
    let bands = if target_width != bands.width || target_height != bands.height {
        downsample_bands(&bands, target_width, target_height)
    } else {
        bands
    };

    // Send Reset message to clear display before new image (only for first level in progressive mode)
    if send_reset {
        tx.send(Message::Binary(encode_reset().into()))
            .await
            .map_err(|_| Error::Internal("Failed to send reset".to_string()))?;
    }

    // Send metadata with full-res dimensions for progressive loading
    let metadata = StreamMetadata {
        width: bands.width,
        height: bands.height,
        num_bands: bands.num_bands() as u32,
        num_layers: bands.num_layers(),
        full_width,
        full_height,
    };
    tx.send(Message::Binary(metadata.to_bytes().into()))
        .await
        .map_err(|_| Error::Internal("Failed to send metadata".to_string()))?;

    // Pack bands to RGBA layers
    let layers = pack_bands_to_rgba_layers(&bands.bands, bands.width, bands.height);

    // Stream each layer in chunks
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
            let msg = encode_layer_chunk(layer_idx as u16, row, row + rows_to_send, chunk_data);

            if tx.send(Message::Binary(msg.into())).await.is_err() {
                // Client disconnected
                return Ok(());
            }

            row += rows_to_send;

            // Yield to allow other tasks to run
            tokio::task::yield_now().await;
        }

        // Send layer complete
        let msg = encode_layer_complete(layer_idx as u16);
        if tx.send(Message::Binary(msg.into())).await.is_err() {
            return Ok(());
        }
    }

    // Send level complete
    let msg = encode_level_complete(level as u8);
    let _ = tx.send(Message::Binary(msg.into())).await;

    Ok(())
}

/// Calculate the target size for a pyramid level.
///
/// Level 0 = full resolution
/// Level 1 = 1/2 resolution
/// Level 2 = 1/4 resolution
/// etc.
fn calculate_level_size(full_width: u32, full_height: u32, level: u32) -> (u32, u32) {
    if level == 0 {
        // Level 0 is always full resolution
        return (full_width, full_height);
    }

    // Each level halves the resolution
    let scale = 1u32 << level; // 2^level

    // Ensure minimum size of 1x1
    let width = (full_width / scale).max(1);
    let height = (full_height / scale).max(1);

    (width, height)
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
                        let _ = storage.mark_failed(&hash, &e.to_string()).await;
                    } else {
                        tracing::info!("Pyramid {} built successfully", hash);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to load bands for pyramid: {}", e);
                    let _ = storage.mark_failed(&hash, &e.to_string()).await;
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

// ============================================================================
// SAM Handlers
// ============================================================================

/// Handle SAM embedding request.
///
/// Computes the image embedding (encoder pass) and caches it for future
/// segmentation requests. The embedding is keyed by image hash.
async fn handle_sam_embed(
    state: &AppState,
    image_id: &str,
    tx: mpsc::Sender<Message>,
) -> Result<(), Error> {
    // Check if SAM is enabled
    let sam_engine = state
        .sam_engine
        .as_ref()
        .ok_or_else(|| Error::Internal("SAM is not enabled on this server".to_string()))?;

    // Find the image
    let image_path = find_image(state, image_id)?;
    let image_hash = compute_image_hash(&image_path)?;

    // Check if embedding is already cached
    if state.embedding_cache.contains(&image_hash).await {
        tracing::info!("SAM embedding cache hit for {}", image_id);
        let response = ServerResponse::SamEmbeddingReady {
            image_id: image_id.to_string(),
        };
        if let Ok(json) = serde_json::to_string(&response) {
            let _ = tx.send(Message::Text(json.into())).await;
        }
        return Ok(());
    }

    tracing::info!("Computing SAM embedding for {}", image_id);

    // Load the image as RGB
    let (rgb_data, width, height) = load_image_rgb(state, &image_path).await?;

    // Compute embedding (encoder pass)
    let encoder_output = sam_engine
        .encode_image(&rgb_data, width, height)
        .await
        .map_err(|e| Error::Internal(format!("SAM encoding failed: {}", e)))?;

    // Cache the embedding with all encoder outputs
    let cached = CachedEmbedding {
        image_embed: encoder_output.image_embed,
        high_res_feats_0: encoder_output.high_res_feats_0,
        high_res_feats_1: encoder_output.high_res_feats_1,
        width,
        height,
    };
    state.embedding_cache.insert(image_hash, cached).await;

    tracing::info!("SAM embedding computed and cached for {}", image_id);

    // Send success response
    let response = ServerResponse::SamEmbeddingReady {
        image_id: image_id.to_string(),
    };
    if let Ok(json) = serde_json::to_string(&response) {
        let _ = tx.send(Message::Text(json.into())).await;
    }

    Ok(())
}

/// Handle SAM segmentation request.
///
/// Uses a cached embedding (or computes one if missing) to decode masks
/// based on point and box prompts.
async fn handle_sam_segment(
    state: &AppState,
    image_id: &str,
    points: Vec<ProtocolSamPoint>,
    box_prompt: Option<[f32; 4]>,
    tx: mpsc::Sender<Message>,
) -> Result<(), Error> {
    // Check if SAM is enabled
    let sam_engine = state
        .sam_engine
        .as_ref()
        .ok_or_else(|| Error::Internal("SAM is not enabled on this server".to_string()))?;

    // Find the image
    let image_path = find_image(state, image_id)?;
    let image_hash = compute_image_hash(&image_path)?;

    // Get or compute embedding
    let cached = if let Some(cached) = state.embedding_cache.get_and_touch(&image_hash).await {
        tracing::debug!("SAM embedding cache hit for segment request");
        cached
    } else {
        tracing::info!("Computing SAM embedding on-demand for {}", image_id);

        // Load image and compute embedding
        let (rgb_data, width, height) = load_image_rgb(state, &image_path).await?;
        let encoder_output = sam_engine
            .encode_image(&rgb_data, width, height)
            .await
            .map_err(|e| Error::Internal(format!("SAM encoding failed: {}", e)))?;

        let cached = CachedEmbedding {
            image_embed: encoder_output.image_embed,
            high_res_feats_0: encoder_output.high_res_feats_0,
            high_res_feats_1: encoder_output.high_res_feats_1,
            width,
            height,
        };

        // Cache for future requests
        state
            .embedding_cache
            .insert(image_hash.clone(), cached.clone())
            .await;

        cached
    };

    // Convert protocol points to backend points
    let backend_points: Vec<BackendSamPoint> = points
        .into_iter()
        .map(|p| BackendSamPoint {
            x: p.x,
            y: p.y,
            label: p.label,
        })
        .collect();

    // Reconstruct EncoderOutput from cached embedding
    let encoder_output = EncoderOutput {
        image_embed: cached.image_embed,
        high_res_feats_0: cached.high_res_feats_0,
        high_res_feats_1: cached.high_res_feats_1,
    };

    // Run mask decoder
    let result = sam_engine
        .decode_mask(
            &encoder_output,
            cached.width,
            cached.height,
            &backend_points,
            box_prompt,
        )
        .await
        .map_err(|e| Error::Internal(format!("SAM decoding failed: {}", e)))?;

    // Convert polygons to flat arrays for the protocol
    // Backend gives us Vec<Vec<[f32; 2]>>, protocol expects Vec<Vec<f32>> (flattened)
    let flattened_polygons: Vec<Vec<f32>> = result
        .polygons
        .into_iter()
        .map(|polygon| polygon.into_iter().flat_map(|[x, y]| [x, y]).collect())
        .collect();

    let num_masks = result.iou_scores.len();

    // Send result
    let response = ServerResponse::SamMask {
        request_id: Uuid::new_v4().to_string(),
        polygons: flattened_polygons,
        iou_scores: result.iou_scores,
    };
    if let Ok(json) = serde_json::to_string(&response) {
        let _ = tx.send(Message::Text(json.into())).await;
    }

    tracing::info!(
        "SAM segmentation complete for {}: {} masks, {} points",
        image_id,
        num_masks,
        backend_points.len()
    );

    Ok(())
}

/// Load image as RGB data for SAM processing.
///
/// This loads the first 3 bands (or grayscale for single-band images)
/// and converts them to RGB format for SAM.
async fn load_image_rgb(
    state: &AppState,
    image_path: &std::path::Path,
) -> Result<(Vec<u8>, u32, u32), Error> {
    // Find a loader
    let loader = state
        .loaders
        .find_loader(image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_path.to_string_lossy().to_string()))?;

    // Load bands
    let bands = loader.load_bands(image_path).await?;

    let width = bands.width;
    let height = bands.height;
    let num_pixels = pixel_count_u32(width, height);

    // Convert to RGB
    let rgb_data = if bands.bands.len() >= 3 {
        // Use first 3 bands as RGB
        let mut rgb = Vec::with_capacity(num_pixels * 3);
        for i in 0..num_pixels {
            rgb.push(normalize_to_u8(bands.bands[0][i]));
            rgb.push(normalize_to_u8(bands.bands[1][i]));
            rgb.push(normalize_to_u8(bands.bands[2][i]));
        }
        rgb
    } else if !bands.bands.is_empty() {
        // Grayscale - replicate to RGB
        let mut rgb = Vec::with_capacity(num_pixels * 3);
        for i in 0..num_pixels {
            let val = normalize_to_u8(bands.bands[0][i]);
            rgb.push(val);
            rgb.push(val);
            rgb.push(val);
        }
        rgb
    } else {
        return Err(Error::Internal("Image has no bands".to_string()));
    };

    Ok((rgb_data, width, height))
}

/// Normalize a float value to u8 [0, 255].
///
/// Assumes the input is in [0, 1] range (common for hyperspectral data).
/// Clamps values outside this range.
fn normalize_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0) as u8
}

// ============================================================================
// Capabilities
// ============================================================================

/// Build server capabilities from current configuration.
///
/// This is sent to clients on WebSocket connect to inform them about
/// server features and protocol version.
fn build_server_capabilities(state: &AppState) -> ServerCapabilities {
    let mut caps = ServerCapabilities {
        protocol_version: hvat_common::PROTOCOL_VERSION,
        max_image_size: state.config.max_cache_memory,
        max_pyramid_levels: hvat_common::MAX_PYRAMID_LEVEL,
        sam_enabled: state.sam_engine.is_some(),
        sam_model: String::new(),
        max_concurrent_streams: state.config.max_user_streams as u32,
    };

    // Add SAM model info if enabled
    if state.sam_engine.is_some() {
        caps.sam_model = state.config.sam_variant.name().to_string();
    }

    caps
}
