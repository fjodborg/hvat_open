//! Server startup tasks including thumbnail pre-generation.

use std::path::Path;
use std::sync::Arc;

use crate::pyramid::{PyramidStatus, compute_image_hash};
use crate::state::AppState;

/// Pre-generate pyramids/thumbnails for all images in the data directory.
///
/// This runs at startup to ensure thumbnails are ready for fast browsing.
/// Images that already have cached pyramids are skipped.
pub async fn pregenerate_pyramids(state: Arc<AppState>) {
    let data_dir = &state.config.data_dir;

    tracing::info!(
        "Starting pyramid pre-generation for: {}",
        data_dir.display()
    );

    // Collect all image paths
    let mut image_paths = Vec::new();
    collect_image_paths(data_dir, &state, &mut image_paths);

    let total = image_paths.len();
    tracing::info!("Found {} images to process", total);

    let mut generated = 0;
    let mut skipped = 0;
    let mut failed = 0;

    for (idx, path) in image_paths.iter().enumerate() {
        let relative = path
            .strip_prefix(data_dir)
            .unwrap_or(path)
            .display()
            .to_string();

        // Check if pyramid already exists
        match compute_image_hash(path) {
            Ok(hash) => {
                let status = state.pyramid_storage.get_status(&hash).await;

                match status {
                    PyramidStatus::Ready => {
                        tracing::debug!("[{}/{}] Skipping (cached): {}", idx + 1, total, relative);
                        skipped += 1;
                        continue;
                    }
                    PyramidStatus::Building => {
                        tracing::debug!(
                            "[{}/{}] Skipping (building): {}",
                            idx + 1,
                            total,
                            relative
                        );
                        skipped += 1;
                        continue;
                    }
                    _ => {}
                }

                // Need to generate pyramid
                tracing::info!("[{}/{}] Generating pyramid: {}", idx + 1, total, relative);

                // Find loader for this file
                if let Some(loader) = state.loaders.find_loader(path) {
                    match loader.load_bands(path).await {
                        Ok(bands) => {
                            match state.pyramid_builder.build_and_save(&hash, &bands).await {
                                Ok(meta) => {
                                    tracing::info!(
                                        "  -> Generated {} levels, thumbnail: {}x{}",
                                        meta.levels.len(),
                                        meta.levels.last().map(|l| l.width).unwrap_or(0),
                                        meta.levels.last().map(|l| l.height).unwrap_or(0)
                                    );
                                    generated += 1;
                                }
                                Err(e) => {
                                    tracing::warn!("  -> Failed to save pyramid: {}", e);
                                    failed += 1;
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("  -> Failed to load bands: {}", e);
                            failed += 1;
                        }
                    }
                } else {
                    tracing::warn!("  -> No loader found for: {}", relative);
                    failed += 1;
                }
            }
            Err(e) => {
                tracing::warn!("Failed to compute hash for {}: {}", relative, e);
                failed += 1;
            }
        }
    }

    tracing::info!(
        "Pyramid pre-generation complete: {} generated, {} skipped (cached), {} failed",
        generated,
        skipped,
        failed
    );
}

/// Recursively collect all image paths from a directory.
fn collect_image_paths(dir: &Path, state: &AppState, paths: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_image_paths(&path, state, paths);
            } else if state.loaders.supports(&path) {
                paths.push(path);
            }
        }
    }
}
