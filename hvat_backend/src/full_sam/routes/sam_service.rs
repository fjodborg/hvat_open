//! Shared SAM route helpers for REST handlers.

use crate::error::Error;
use crate::state::AppState;
use crate::utils::find_image;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SamBands {
    pub(crate) red: u32,
    pub(crate) green: u32,
    pub(crate) blue: u32,
}

impl SamBands {
    pub(crate) const fn default_rgb() -> Self {
        Self {
            red: 0,
            green: 1,
            blue: 2,
        }
    }

    pub(crate) fn embedding_key(self, image_id: &str) -> String {
        format!("{}#{}:{}:{}", image_id, self.red, self.green, self.blue)
    }

    pub(crate) fn clamp_to_num_bands(self, num_bands: usize) -> Self {
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

/// Resolve SAM band selection against image metadata so keys are canonical.
///
/// Returns `(resolved_bands, width, height)` from a single metadata read,
/// allowing callers to skip a separate `load_metadata` call for dimensions.
pub(crate) async fn resolve_sam_bands(
    state: &AppState,
    image_id: &str,
    requested: SamBands,
) -> Result<(SamBands, u32, u32), Error> {
    let image_path = find_image(state, image_id)?;
    let loader = state
        .loaders
        .find_loader(&image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_id.to_string()))?;
    let metadata = loader.load_metadata(&image_path).await?;
    let resolved = requested.clamp_to_num_bands(metadata.num_bands);
    Ok((resolved, metadata.width, metadata.height))
}

/// Load image with specific band selection for RGB channels.
pub(crate) async fn load_image_rgb_with_bands(
    state: &AppState,
    image_id: &str,
    red_band: u32,
    green_band: u32,
    blue_band: u32,
) -> Result<(Vec<u8>, u32, u32, SamBands), Error> {
    let image_path = find_image(state, image_id)?;

    let loader = state
        .loaders
        .find_loader(&image_path)
        .ok_or_else(|| Error::UnsupportedFormat(image_id.to_string()))?;

    let bands = loader.load_bands(&image_path).await?;

    let width = bands.width;
    let height = bands.height;
    let pixel_count = (width * height) as usize;
    let num_bands = bands.num_bands();

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

    let r = &bands.bands[red_idx];
    let g = &bands.bands[green_idx];
    let b = &bands.bands[blue_idx];

    let mut rgb_data = Vec::with_capacity(pixel_count * 3);
    for i in 0..pixel_count {
        rgb_data.push((r[i].clamp(0.0, 1.0) * 255.0) as u8);
        rgb_data.push((g[i].clamp(0.0, 1.0) * 255.0) as u8);
        rgb_data.push((b[i].clamp(0.0, 1.0) * 255.0) as u8);
    }

    Ok((rgb_data, width, height, resolved))
}
