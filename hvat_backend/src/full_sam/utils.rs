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
mod tests {
    use super::*;

    #[test]
    fn test_make_url_safe() {
        assert_eq!(make_url_safe("hello"), "hello");
        assert_eq!(make_url_safe("hello world"), "hello_world");
        assert_eq!(make_url_safe("path/to/file.png"), "path_to_file_png");
        assert_eq!(make_url_safe("file-name_123"), "file-name_123");
        assert_eq!(make_url_safe(""), "");
        assert_eq!(make_url_safe("a/b/c"), "a_b_c");
        // Unicode alphanumeric characters are preserved (is_alphanumeric is Unicode-aware)
        assert_eq!(make_url_safe("日本語"), "日本語");
    }

    #[test]
    fn test_make_url_safe_preserves_alphanumeric() {
        let input = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        assert_eq!(make_url_safe(input), input);
    }

    #[test]
    fn test_make_url_safe_special_chars() {
        assert_eq!(make_url_safe("file.png"), "file_png");
        assert_eq!(make_url_safe("path\\to\\file"), "path_to_file");
        assert_eq!(make_url_safe("file name (1).jpg"), "file_name__1__jpg");
    }
}
