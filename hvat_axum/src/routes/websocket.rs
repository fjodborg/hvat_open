//! WebSocket handler for binary image streaming.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;

use crate::error::Error;
use crate::packer::{downsample_bands, pack_bands_to_rgba_layers};
use crate::protocol::{
    ClientMessage, ServerResponse, StreamMetadata, encode_error, encode_layer_chunk,
    encode_layer_complete, encode_level_complete,
};
use crate::pyramid::{PyramidStatus, compute_image_hash};
use crate::state::AppState;

/// Handle a WebSocket connection for image streaming.
pub async fn handle_websocket(socket: WebSocket, state: Arc<AppState>, image_id: String) {
    let (mut sender, mut receiver) = socket.split();

    // Channel for sending messages to the client
    let (tx, mut rx) = mpsc::channel::<Message>(32);

    // Spawn task to forward messages to WebSocket
    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sender.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Process incoming messages
    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Text(text) => {
                // Parse client message
                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(client_msg) => {
                        if let Err(e) =
                            handle_client_message(client_msg, &state, &image_id, tx.clone()).await
                        {
                            // Send error response
                            let error_bytes = encode_error(&e.to_string());
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
            Message::Close(_) => break,
            _ => {} // Ignore binary messages from client
        }
    }

    // Clean up
    drop(tx);
    let _ = send_task.await;
}

/// Handle a parsed client message.
async fn handle_client_message(
    msg: ClientMessage,
    state: &AppState,
    image_id: &str,
    tx: mpsc::Sender<Message>,
) -> Result<(), Error> {
    match msg {
        ClientMessage::StartStream {
            image_id: req_id,
            level,
        } => {
            // Verify image_id matches (or use the one from the URL)
            let id = if req_id.is_empty() { image_id } else { &req_id };
            stream_image(state, id, level, tx).await?;
        }

        ClientMessage::CancelStream => {
            // Cancellation is handled by dropping the tx channel
            tracing::info!("Stream cancelled for {}", image_id);
        }

        ClientMessage::ChangeLevel { level } => {
            // For now, just start a new stream at the new level
            stream_image(state, image_id, level, tx).await?;
        }

        ClientMessage::SaveAnnotations {
            image_id: _,
            annotations: _,
            categories: _,
        } => {
            // TODO: Implement annotation storage
            let response = ServerResponse::Error {
                message: "Annotation storage not yet implemented".to_string(),
            };
            if let Ok(json) = serde_json::to_string(&response) {
                let _ = tx.send(Message::Text(json.into())).await;
            }
        }

        ClientMessage::LoadAnnotations { image_id: _ } => {
            // TODO: Implement annotation loading
            let response = ServerResponse::Error {
                message: "Annotation loading not yet implemented".to_string(),
            };
            if let Ok(json) = serde_json::to_string(&response) {
                let _ = tx.send(Message::Text(json.into())).await;
            }
        }

        ClientMessage::SamSegment { .. } | ClientMessage::SamEmbed { .. } => {
            // TODO: Implement SAM integration
            let response = ServerResponse::Error {
                message: "SAM integration not yet implemented".to_string(),
            };
            if let Ok(json) = serde_json::to_string(&response) {
                let _ = tx.send(Message::Text(json.into())).await;
            }
        }
    }

    Ok(())
}

/// Stream an image at a specific pyramid level.
async fn stream_image(
    state: &AppState,
    image_id: &str,
    level: u32,
    tx: mpsc::Sender<Message>,
) -> Result<(), Error> {
    // Find the image file
    let image_path = find_image(state, image_id)?;

    // Compute hash for pyramid cache lookup
    let image_hash = compute_image_hash(&image_path)?;

    // Check if pyramid is cached
    let pyramid_status = state.pyramid_storage.get_status(&image_hash).await;

    match pyramid_status {
        PyramidStatus::Ready => {
            // Stream from cached pyramid
            stream_from_pyramid(state, &image_hash, level, tx).await
        }
        PyramidStatus::Building => {
            // Pyramid is being built, stream directly from source for now
            tracing::info!("Pyramid {} is building, streaming from source", image_hash);
            stream_from_source(state, &image_path, image_id, level, tx).await
        }
        PyramidStatus::Pending | PyramidStatus::Failed => {
            // Start building pyramid in background
            let builder = state.pyramid_builder.clone();
            let storage = state.pyramid_storage.clone();
            let loaders = state.loaders.clone();
            let path_clone = image_path.clone();
            let hash_clone = image_hash.clone();

            tokio::spawn(async move {
                tracing::info!("Starting pyramid generation for {}", hash_clone);
                if let Some(loader) = loaders.find_loader(&path_clone) {
                    match loader.load_bands(&path_clone).await {
                        Ok(bands) => {
                            if let Err(e) = builder.build_and_save(&hash_clone, &bands).await {
                                tracing::error!("Failed to build pyramid: {}", e);
                                let _ = storage.mark_failed(&hash_clone, &e.to_string()).await;
                            } else {
                                tracing::info!("Pyramid {} built successfully", hash_clone);
                            }
                        }
                        Err(e) => {
                            tracing::error!("Failed to load bands for pyramid: {}", e);
                            let _ = storage.mark_failed(&hash_clone, &e.to_string()).await;
                        }
                    }
                }
            });

            // Stream directly from source while pyramid is being built
            stream_from_source(state, &image_path, image_id, level, tx).await
        }
    }
}

/// Stream from a cached pyramid level.
async fn stream_from_pyramid(
    state: &AppState,
    image_hash: &str,
    level: u32,
    tx: mpsc::Sender<Message>,
) -> Result<(), Error> {
    // Load pyramid metadata
    let metadata = state.pyramid_storage.load_metadata(image_hash).await?;

    // Find the requested level (clamp to available levels)
    let max_level = metadata.levels.len().saturating_sub(1) as u32;
    let level = level.min(max_level);

    let level_info = &metadata.levels[level as usize];

    // Send stream metadata
    let stream_meta = StreamMetadata {
        width: level_info.width,
        height: level_info.height,
        num_bands: metadata.num_bands as u32,
        num_layers: level_info.num_layers,
    };
    tx.send(Message::Binary(stream_meta.to_bytes().into()))
        .await
        .map_err(|_| Error::Internal("Failed to send metadata".to_string()))?;

    // Load and stream the level data
    let layers = state.pyramid_storage.load_level(image_hash, level).await?;

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

    let msg = encode_level_complete(level as u8);
    let _ = tx.send(Message::Binary(msg.into())).await;

    Ok(())
}

/// Stream directly from source file (no pyramid cache).
async fn stream_from_source(
    state: &AppState,
    image_path: &std::path::Path,
    image_id: &str,
    level: u32,
    tx: mpsc::Sender<Message>,
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

    // Send metadata
    let metadata = StreamMetadata {
        width: bands.width,
        height: bands.height,
        num_bands: bands.num_bands() as u32,
        num_layers: bands.num_layers(),
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

/// Find an image file by ID.
fn find_image(state: &AppState, image_id: &str) -> Result<std::path::PathBuf, Error> {
    let data_dir = &state.config.data_dir;

    if let Some(path) = search_for_image(data_dir, image_id, state) {
        return Ok(path);
    }

    Err(Error::ImageNotFound(image_id.to_string()))
}

/// Recursively search for an image matching the ID.
fn search_for_image(
    dir: &std::path::Path,
    image_id: &str,
    state: &AppState,
) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = search_for_image(&path, image_id, state) {
                return Some(found);
            }
        } else if state.loaders.supports(&path) {
            let relative_path = path
                .strip_prefix(&state.config.data_dir)
                .unwrap_or(&path)
                .to_string_lossy();

            let file_id = make_url_safe(&relative_path);
            if file_id == image_id {
                return Some(path);
            }
        }
    }

    None
}

/// Make a string URL-safe.
fn make_url_safe(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
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
