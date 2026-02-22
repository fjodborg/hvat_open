//! Server startup tasks including thumbnail pre-generation.

use std::path::PathBuf;
use std::sync::Arc;

use futures::stream::{self, StreamExt};
use hvat_backend_helper::catalog::collect_supported_image_paths;

use crate::pyramid::{PyramidStatus, compute_image_hash};
use crate::state::AppState;

enum PregenOutcome {
    Generated,
    Skipped,
    Failed,
}

/// Pre-generate pyramids/thumbnails for all images in the data directory.
///
/// This runs at startup to ensure thumbnails are ready for fast browsing.
/// Images that already have cached pyramids are skipped.
pub async fn pregenerate_pyramids(state: Arc<AppState>) {
    let data_dir = state.config.data_dir.clone();

    tracing::info!(
        "Starting pyramid pre-generation for: {}",
        data_dir.display()
    );

    // Collect all image paths
    let image_paths = collect_supported_image_paths(&data_dir, |path| state.loaders.supports(path));

    let total = image_paths.len();
    let concurrency_limit = state.config.pyramid_concurrency.max(1);
    tracing::info!(
        "Found {} images to process (concurrency={})",
        total,
        concurrency_limit
    );

    let mut generated = 0;
    let mut skipped = 0;
    let mut failed = 0;

    let mut tasks = stream::iter(image_paths.into_iter().enumerate().map(|(idx, path)| {
        let state = state.clone();
        let data_dir = data_dir.clone();
        async move { pregenerate_single_image(state, data_dir, path, idx + 1, total).await }
    }))
    .buffer_unordered(concurrency_limit);

    while let Some(outcome) = tasks.next().await {
        match outcome {
            PregenOutcome::Generated => generated += 1,
            PregenOutcome::Skipped => skipped += 1,
            PregenOutcome::Failed => failed += 1,
        }
    }

    tracing::info!(
        "Pyramid pre-generation complete: {} generated, {} skipped (cached), {} failed",
        generated,
        skipped,
        failed
    );
}

#[allow(clippy::cognitive_complexity)]
async fn pregenerate_single_image(
    state: Arc<AppState>,
    data_dir: PathBuf,
    path: PathBuf,
    index: usize,
    total: usize,
) -> PregenOutcome {
    let relative = path
        .strip_prefix(&data_dir)
        .unwrap_or(&path)
        .display()
        .to_string();

    // Check if pyramid already exists
    let hash = match compute_image_hash(&path) {
        Ok(hash) => hash,
        Err(e) => {
            tracing::warn!("Failed to compute hash for {}: {}", relative, e);
            return PregenOutcome::Failed;
        }
    };

    let status = state.pyramid_storage.get_status(&hash).await;
    match status {
        PyramidStatus::Ready => {
            tracing::debug!("[{}/{}] Skipping (cached): {}", index, total, relative);
            return PregenOutcome::Skipped;
        }
        PyramidStatus::Building => {
            tracing::debug!("[{}/{}] Skipping (building): {}", index, total, relative);
            return PregenOutcome::Skipped;
        }
        _ => {}
    }

    tracing::info!("[{}/{}] Generating pyramid: {}", index, total, relative);

    // Find loader for this file
    let Some(loader) = state.loaders.find_loader(&path) else {
        tracing::warn!("  -> No loader found for: {}", relative);
        return PregenOutcome::Failed;
    };

    let bands = match loader.load_bands(&path).await {
        Ok(bands) => bands,
        Err(e) => {
            tracing::warn!("  -> Failed to load bands: {}", e);
            return PregenOutcome::Failed;
        }
    };

    match state.pyramid_builder.build_and_save(&hash, &bands).await {
        Ok(meta) => {
            tracing::info!(
                "  -> Generated {} levels, thumbnail: {}x{}",
                meta.levels.len(),
                meta.levels.last().map(|l| l.width).unwrap_or(0),
                meta.levels.last().map(|l| l.height).unwrap_or(0)
            );
            PregenOutcome::Generated
        }
        Err(e) => {
            tracing::warn!("  -> Failed to save pyramid: {}", e);
            if let Err(mark_err) = state
                .pyramid_storage
                .mark_failed(&hash, &e.to_string())
                .await
            {
                tracing::warn!("  -> Failed to mark pyramid as failed: {}", mark_err);
            }
            PregenOutcome::Failed
        }
    }
}
