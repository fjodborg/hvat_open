//! WebSocket handler for binary image streaming.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::error::Error;
use crate::packer::{downsample_bands, pack_bands_to_rgba_layers};
use crate::protocol::{
    ClientMessage, SamPoint as ProtocolSamPoint, ServerResponse, StreamMetadata, encode_error,
    encode_layer_chunk, encode_layer_complete, encode_level_complete,
};
use crate::pyramid::{PyramidStatus, compute_image_hash};
use crate::sam::{CachedEmbedding, EncoderOutput, SamPoint as BackendSamPoint};
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
                tracing::debug!("Received text message: {}", text);
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

        ClientMessage::SamEmbed { image_id: req_id } => {
            let id = if req_id.is_empty() { image_id } else { &req_id };
            handle_sam_embed(state, id, tx).await?;
        }

        ClientMessage::SamSegment {
            image_id: req_id,
            points,
            box_prompt,
        } => {
            let id = if req_id.is_empty() { image_id } else { &req_id };
            handle_sam_segment(state, id, points, box_prompt, tx).await?;
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
    let num_pixels = (width * height) as usize;

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
