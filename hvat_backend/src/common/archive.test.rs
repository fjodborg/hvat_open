use super::*;
use std::io::Read;
use std::time::{SystemTime, UNIX_EPOCH};

fn create_temp_dir() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock should be monotonic")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "hvat_backend_archive_{}_{}",
        std::process::id(),
        nanos
    ));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

#[test]
fn zip64_enabled_for_entries_over_4gib() {
    assert!(!requires_zip64(ZIP32_MAX_BYTES));
    assert!(requires_zip64(ZIP32_MAX_BYTES + 1));
}

#[test]
fn build_images_zip_writes_expected_entry_contents() {
    let dir = create_temp_dir();
    let source_file = dir.join("source.bin");
    std::fs::write(&source_file, b"abc123").expect("write source file");

    let entries = vec![ImageEntry {
        relative_path: "nested/source.bin".to_string(),
        file_path: source_file,
        size_bytes: 6,
    }];

    let zip_bytes = build_images_zip(&entries).expect("build zip");
    let mut archive =
        zip::ZipArchive::new(Cursor::new(zip_bytes)).expect("open generated archive");
    let mut zipped = archive
        .by_name("nested/source.bin")
        .expect("zip entry exists");
    let mut content = String::new();
    zipped
        .read_to_string(&mut content)
        .expect("read zipped content");
    assert_eq!(content, "abc123");

    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn build_images_zip_error_includes_relative_path_context() {
    let dir = create_temp_dir();
    let missing = dir.join("missing.bin");
    let entries = vec![ImageEntry {
        relative_path: "missing.bin".to_string(),
        file_path: missing,
        size_bytes: 16,
    }];

    let err = build_images_zip(&entries).expect_err("missing file should fail");
    assert!(err.contains("Failed to open"));
    assert!(err.contains("missing.bin"));

    let _ = std::fs::remove_dir_all(dir);
}
