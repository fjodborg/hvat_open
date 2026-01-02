//! Shared utility functions for hvat_axum.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};
use crate::state::AppState;

/// Make a string URL-safe by replacing non-alphanumeric characters with underscores.
///
/// Preserves `-` and `_` characters.
pub fn make_url_safe(s: &str) -> String {
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

/// Find an image file by its URL-safe ID.
///
/// Searches recursively in the data directory for a file whose URL-safe path matches.
pub fn find_image(state: &AppState, image_id: &str) -> Result<PathBuf> {
    let data_dir = &state.config.data_dir;

    search_for_image(data_dir, data_dir, image_id, state)
        .ok_or_else(|| Error::ImageNotFound(image_id.to_string()))
}

/// Recursively search for an image matching the ID.
fn search_for_image(
    dir: &Path,
    base_dir: &Path,
    image_id: &str,
    state: &AppState,
) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if let Some(found) = search_for_image(&path, base_dir, image_id, state) {
                return Some(found);
            }
        } else if state.loaders.supports(&path) {
            let relative_path = path
                .strip_prefix(base_dir)
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
