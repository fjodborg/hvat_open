use std::path::{Component, Path, PathBuf};

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
///
/// NOTE: This is retained for legacy compatibility lookups.
/// New IDs should use `encode_image_id_from_relative`.
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

const IMAGE_ID_PREFIX: &str = "id:";

/// Encode a normalized relative path into a stable, collision-free image ID.
///
/// Format: `id:{hex-utf8-bytes}`
#[must_use]
pub fn encode_image_id_from_relative(relative_path: &str) -> String {
    let mut out = String::with_capacity(IMAGE_ID_PREFIX.len() + relative_path.len() * 2);
    out.push_str(IMAGE_ID_PREFIX);
    for byte in relative_path.as_bytes() {
        out.push(hex_nibble(byte >> 4));
        out.push(hex_nibble(byte & 0x0f));
    }
    out
}

/// Decode a stable image ID back into a normalized relative path.
#[must_use]
pub fn decode_relative_path_from_image_id(image_id: &str) -> Option<String> {
    let encoded = image_id.strip_prefix(IMAGE_ID_PREFIX)?;
    if encoded.len() % 2 != 0 {
        return None;
    }

    let mut bytes = Vec::with_capacity(encoded.len() / 2);
    for chunk in encoded.as_bytes().chunks_exact(2) {
        let hi = from_hex_digit(chunk[0])?;
        let lo = from_hex_digit(chunk[1])?;
        bytes.push((hi << 4) | lo);
    }

    String::from_utf8(bytes).ok()
}

fn hex_nibble(nibble: u8) -> char {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    HEX[nibble as usize] as char
}

fn from_hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(10 + (b - b'a')),
        b'A'..=b'F' => Some(10 + (b - b'A')),
        _ => None,
    }
}

fn is_safe_relative_path(relative_path: &str) -> bool {
    let path = Path::new(relative_path);
    if path.as_os_str().is_empty() || path.is_absolute() {
        return false;
    }
    !path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    })
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
        let id = encode_image_id_from_relative(&relative_path);

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

    // Fast path for new stable IDs.
    if let Some(relative_path) = decode_relative_path_from_image_id(image_id)
        && is_safe_relative_path(&relative_path)
    {
        let candidate = data_dir.join(&relative_path);
        if candidate.is_file() && is_supported(&candidate) {
            return Some(candidate);
        }
    }

    // Legacy fallback for old lossy IDs.
    let mut found = None;
    visit_supported_files(data_dir, &is_supported, &mut |base_dir, path| {
        if found.is_none() {
            let relative_path = normalized_relative_path(base_dir, path);
            if make_url_safe(&relative_path) == image_id {
                found = Some(path.to_path_buf());
            }
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
    use super::{
        decode_relative_path_from_image_id, encode_image_id_from_relative,
        find_supported_image_by_id, list_supported_images, make_url_safe,
    };
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn create_temp_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "hvat_backend_catalog_{}_{}",
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

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

    #[test]
    fn stable_image_id_roundtrip() {
        let relative = "nested/path/日本語 file.png";
        let encoded = encode_image_id_from_relative(relative);
        assert!(encoded.starts_with("id:"));
        assert_eq!(
            decode_relative_path_from_image_id(&encoded),
            Some(relative.to_string())
        );
    }

    #[test]
    fn stable_image_ids_avoid_legacy_collisions() {
        let a = "a/b.png";
        let b = "a_b.png";
        assert_eq!(make_url_safe(a), make_url_safe(b));
        assert_ne!(
            encode_image_id_from_relative(a),
            encode_image_id_from_relative(b)
        );
    }

    #[test]
    fn find_supported_image_by_id_accepts_stable_and_legacy_ids() {
        let dir = create_temp_dir();
        let nested = dir.join("sub");
        std::fs::create_dir_all(&nested).expect("create nested dir");
        let file_path = nested.join("file.png");
        std::fs::write(&file_path, b"png").expect("write file");

        let is_supported = |path: &std::path::Path| {
            path.extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("png"))
        };

        let listed = list_supported_images(&dir, is_supported);
        assert_eq!(listed.len(), 1);
        let stable_id = listed[0].id.clone();

        let stable_match = find_supported_image_by_id(&dir, &stable_id, is_supported)
            .expect("stable id should resolve");
        assert_eq!(stable_match, file_path);

        let legacy_id = make_url_safe("sub/file.png");
        let legacy_match = find_supported_image_by_id(&dir, &legacy_id, is_supported)
            .expect("legacy id should resolve");
        assert_eq!(legacy_match, file_path);

        let _ = std::fs::remove_dir_all(dir);
    }
}
