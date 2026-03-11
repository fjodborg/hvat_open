//! Shared utility functions for hvat_backend.

use std::path::PathBuf;

use crate::common::catalog::find_supported_image_by_id;

use crate::error::{Error, Result};
use crate::state::AppState;

/// Make a string URL-safe by replacing non-alphanumeric characters with underscores.
///
/// Preserves `-` and `_` characters.
pub fn make_url_safe(s: &str) -> String {
    crate::common::catalog::make_url_safe(s)
}

/// Find an image file by its URL-safe ID.
///
/// Searches recursively in the data directory for a file whose URL-safe path matches.
pub fn find_image(state: &AppState, image_id: &str) -> Result<PathBuf> {
    find_supported_image_by_id(&state.config.data_dir, image_id, |path| {
        state.loaders.supports(path)
    })
    .ok_or_else(|| Error::ImageNotFound(image_id.to_string()))
}

#[cfg(test)]
#[path = "utils.test.rs"]
mod tests;
