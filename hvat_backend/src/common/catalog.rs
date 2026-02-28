use std::path::{Path, PathBuf};

use serde::Serialize;

/// List entry for an image in a data directory.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ImageInfo {
    /// URL-safe identifier derived from relative path.
    pub id: String,
    /// Display name (filename).
    pub name: String,
    /// Relative path inside the data directory.
    pub path: String,
    /// Uppercase extension or "UNKNOWN".
    pub format: String,
}

/// Make a string URL-safe by replacing non-alphanumeric characters with `_`.
#[must_use]
pub fn make_url_safe(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

/// Count recursively all supported image files under `data_dir`.
#[must_use]
pub fn count_supported_images<F>(data_dir: &Path, is_supported: F) -> usize
where
    F: Fn(&Path) -> bool,
{
    if !data_dir.exists() {
        return 0;
    }

    let mut count = 0usize;
    visit_supported_files(data_dir, &is_supported, &mut |_, _| {
        count += 1;
    });
    count
}

/// List recursively all supported images under `data_dir`, sorted by relative path.
#[must_use]
pub fn list_supported_images<F>(data_dir: &Path, is_supported: F) -> Vec<ImageInfo>
where
    F: Fn(&Path) -> bool,
{
    if !data_dir.exists() {
        return Vec::new();
    }

    let mut images = Vec::new();
    visit_supported_files(data_dir, &is_supported, &mut |base_dir, path| {
        let relative_path = normalized_relative_path(base_dir, path);
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("unknown")
            .to_string();
        let format = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_uppercase())
            .unwrap_or_else(|| "UNKNOWN".to_string());
        let id = make_url_safe(&relative_path);

        images.push(ImageInfo {
            id,
            name,
            path: relative_path,
            format,
        });
    });

    images.sort_by(|a, b| a.path.cmp(&b.path));
    images
}

/// Collect recursively all supported image paths under `data_dir`.
#[must_use]
pub fn collect_supported_image_paths<F>(data_dir: &Path, is_supported: F) -> Vec<PathBuf>
where
    F: Fn(&Path) -> bool,
{
    if !data_dir.exists() {
        return Vec::new();
    }

    let mut paths = Vec::new();
    visit_supported_files(data_dir, &is_supported, &mut |_, path| {
        paths.push(path.to_path_buf());
    });
    paths
}

/// Find a supported file by URL-safe `image_id`.
#[must_use]
pub fn find_supported_image_by_id<F>(
    data_dir: &Path,
    image_id: &str,
    is_supported: F,
) -> Option<PathBuf>
where
    F: Fn(&Path) -> bool,
{
    if !data_dir.exists() {
        return None;
    }

    let mut found = None;
    visit_supported_files(data_dir, &is_supported, &mut |base_dir, path| {
        if found.is_some() {
            return;
        }
        let relative_path = normalized_relative_path(base_dir, path);
        if make_url_safe(&relative_path) == image_id {
            found = Some(path.to_path_buf());
        }
    });
    found
}

fn visit_supported_files<F, V>(base_dir: &Path, is_supported: &F, visitor: &mut V)
where
    F: Fn(&Path) -> bool,
    V: FnMut(&Path, &Path),
{
    visit_supported_files_recursive(base_dir, base_dir, is_supported, visitor);
}

fn visit_supported_files_recursive<F, V>(
    dir: &Path,
    base_dir: &Path,
    is_supported: &F,
    visitor: &mut V,
) where
    F: Fn(&Path) -> bool,
    V: FnMut(&Path, &Path),
{
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                visit_supported_files_recursive(&path, base_dir, is_supported, visitor);
            } else if is_supported(&path) {
                visitor(base_dir, &path);
            }
        }
    }
}

fn normalized_relative_path(base_dir: &Path, path: &Path) -> String {
    path.strip_prefix(base_dir)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::make_url_safe;

    #[test]
    fn make_url_safe_preserves_alphanumerics() {
        let input = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        assert_eq!(make_url_safe(input), input);
    }

    #[test]
    fn make_url_safe_replaces_special_chars() {
        assert_eq!(make_url_safe("file.png"), "file_png");
        assert_eq!(make_url_safe("path/to/file"), "path_to_file");
        assert_eq!(make_url_safe("path\\to\\file"), "path_to_file");
        assert_eq!(make_url_safe("file name (1).jpg"), "file_name__1__jpg");
    }

    #[test]
    fn make_url_safe_keeps_unicode_letters() {
        assert_eq!(make_url_safe("日本語"), "日本語");
    }
}
