use super::{
    decode_relative_path_from_image_id, encode_image_id_from_relative, find_supported_image_by_id,
    list_supported_images, make_url_safe,
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
