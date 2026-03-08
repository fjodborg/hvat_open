//! Image streaming logic — load from source, pack to RGBA, and stream chunks.
//!
//! This provides a ready-to-use streaming implementation that backends can use
//! directly or customize. Streams images from source files without pyramid caching.

use std::path::Path;

use axum::extract::ws::Message;
use tokio::sync::mpsc;

use crate::common::catalog::find_supported_image_by_id;
use crate::common::config::ServerConfig;
use crate::common::error::Error;
use crate::common::loaders::ImageLoaderRegistry;
use crate::common::packer::{calculate_level_size, downsample_bands, pack_bands_to_rgba_layers};
use crate::common::protocol::{
    PROTOCOL_VERSION, StreamMetadata, encode_layer_chunk_mux, encode_layer_complete_mux,
    encode_level_complete_mux, encode_reset_mux, encode_stream_complete,
};

const YIELD_EVERY_N_CHUNKS: u32 = 8;

/// Stream an image to the client, loading directly from source.
///
/// Handles progressive streaming (multiple levels from thumbnail to target)
/// and single-level streaming. Downsamples on-the-fly for non-zero levels.
pub async fn stream_image(
    data_dir: &Path,
    loaders: &ImageLoaderRegistry,
    chunk_rows: u32,
    tx: mpsc::Sender<Message>,
    request_id: u32,
    image_id: &str,
    level: u32,
    progressive: bool,
) -> Result<(), Error> {
    let image_path = find_supported_image_by_id(data_dir, image_id, |path| loaders.supports(path))
        .ok_or_else(|| Error::ImageNotFound(image_id.to_string()))?;

    let loader = loaders
        .find_loader(&image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_id.to_string()))?;

    let bands = loader.load_bands(&image_path).await?;
    let full_width = bands.width;
    let full_height = bands.height;

    // Calculate how many levels are possible
    let max_level = calculate_num_levels(full_width, full_height).saturating_sub(1);
    let target_level = level.min(max_level);

    if progressive && target_level > 0 {
        tracing::info!(
            "Stream {}: progressive {} from level {} to {} (full: {}x{})",
            request_id,
            image_id,
            max_level,
            target_level,
            full_width,
            full_height
        );

        for (idx, current_level) in (target_level..=max_level).rev().enumerate() {
            let send_reset = idx == 0;
            stream_single_level(
                &bands,
                full_width,
                full_height,
                current_level,
                send_reset,
                chunk_rows,
                &tx,
                request_id,
            )
            .await?;
        }
    } else {
        stream_single_level(
            &bands,
            full_width,
            full_height,
            target_level,
            true,
            chunk_rows,
            &tx,
            request_id,
        )
        .await?;
    }

    tx.send(Message::Binary(encode_stream_complete(request_id).into()))
        .await
        .ok();

    tracing::info!("Stream {} complete for '{}'", request_id, image_id);
    Ok(())
}

async fn stream_single_level(
    source_bands: &crate::common::loaders::BandData,
    full_width: u32,
    full_height: u32,
    level: u32,
    send_reset: bool,
    chunk_rows: u32,
    tx: &mpsc::Sender<Message>,
    request_id: u32,
) -> Result<(), Error> {
    let (target_width, target_height) = calculate_level_size(full_width, full_height, level);

    let bands = if target_width != source_bands.width || target_height != source_bands.height {
        downsample_bands(source_bands, target_width, target_height)
    } else {
        source_bands.clone()
    };

    if send_reset {
        tx.send(Message::Binary(encode_reset_mux(request_id).into()))
            .await
            .map_err(|_| Error::Internal("Failed to send reset".to_string()))?;
    }

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

    let layers = pack_bands_to_rgba_layers(&bands.bands, bands.width, bands.height);

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

/// Calculate the number of pyramid levels for an image.
const MIN_THUMBNAIL_SIZE: u32 = 256;

fn calculate_num_levels(width: u32, height: u32) -> u32 {
    let max_dim = width.max(height);
    if max_dim <= MIN_THUMBNAIL_SIZE {
        return 1;
    }
    let levels = (max_dim as f32 / MIN_THUMBNAIL_SIZE as f32).log2().ceil() as u32;
    levels + 1
}

/// Generate a thumbnail PNG for an image on-the-fly.
///
/// Loads the image, takes the first 3 bands (or fewer) as RGB, downscales to
/// fit within `max_dim`, and encodes as PNG.
pub async fn generate_thumbnail(
    data_dir: &Path,
    loaders: &ImageLoaderRegistry,
    image_id: &str,
    max_dim: u32,
) -> Result<Vec<u8>, Error> {
    let image_path = find_supported_image_by_id(data_dir, image_id, |path| loaders.supports(path))
        .ok_or_else(|| Error::ImageNotFound(image_id.to_string()))?;

    let loader = loaders
        .find_loader(&image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_id.to_string()))?;

    let bands = loader.load_bands(&image_path).await?;
    let (tw, th) = thumbnail_size(bands.width, bands.height, max_dim);
    let downsampled = if tw != bands.width || th != bands.height {
        downsample_bands(&bands, tw, th)
    } else {
        bands
    };

    // Take first 3 bands as RGB (or grayscale if only 1 band)
    let num_bands = downsampled.bands.len();
    let pixel_count = (tw * th) as usize;
    let mut rgba = vec![255u8; pixel_count * 4];
    for i in 0..pixel_count {
        let r = (downsampled.bands[0][i] * 255.0).clamp(0.0, 255.0) as u8;
        let g = if num_bands >= 3 {
            (downsampled.bands[1][i] * 255.0).clamp(0.0, 255.0) as u8
        } else {
            r
        };
        let b = if num_bands >= 3 {
            (downsampled.bands[2][i] * 255.0).clamp(0.0, 255.0) as u8
        } else {
            r
        };
        rgba[i * 4] = r;
        rgba[i * 4 + 1] = g;
        rgba[i * 4 + 2] = b;
    }

    let img = image::RgbaImage::from_raw(tw, th, rgba)
        .ok_or_else(|| Error::Internal("Failed to create image buffer".to_string()))?;
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .map_err(|e| Error::Internal(format!("PNG encode failed: {e}")))?;

    Ok(buf.into_inner())
}

fn thumbnail_size(width: u32, height: u32, max_dim: u32) -> (u32, u32) {
    let max = width.max(height);
    if max <= max_dim {
        return (width, height);
    }
    let scale = max_dim as f64 / max as f64;
    (
        (width as f64 * scale).round().max(1.0) as u32,
        (height as f64 * scale).round().max(1.0) as u32,
    )
}

/// Build default server capabilities for a streaming-only backend.
pub fn build_default_capabilities(config: &ServerConfig) -> hvat_common::ServerCapabilities {
    use hvat_common::{DownloadMode, ServerFeatures, ServerInfo, ServerLimits};

    hvat_common::ServerCapabilities {
        protocol_version: PROTOCOL_VERSION,
        server: ServerInfo {
            name: "hvat-backend-simple".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        },
        limits: ServerLimits {
            max_image_size: config.max_cache_memory,
            max_pyramid_levels: hvat_common::MAX_PYRAMID_LEVEL,
            max_concurrent_streams: config.max_user_streams as u32,
            max_concurrent_inferences: 0,
        },
        features: ServerFeatures {
            streaming: true,
            project_state: true,
            downloads: true,
            thumbnails: true,
            inference: false,
            sam: false,
            progressive_streaming: false,
        },
        download_mode: DownloadMode::SingleZip,
        models: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::config::ServerConfig;

    #[test]
    fn default_capabilities_use_single_zip_download_mode() {
        let caps = build_default_capabilities(&ServerConfig::default());
        assert!(caps.features.downloads);
        assert_eq!(caps.download_mode, hvat_common::DownloadMode::SingleZip);
        assert!(!caps.supports_chunked_downloads());
    }
}
