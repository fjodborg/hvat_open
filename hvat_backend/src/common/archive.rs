use std::io::Cursor;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub enum ArchiveError {
    NoImages,
    BuildFailed(String),
}

#[derive(Debug)]
pub struct ProjectArchive {
    pub filename: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct ImageEntry {
    pub relative_path: String,
    pub file_path: PathBuf,
    pub size_bytes: u64,
}

// ============================================================================
// Single ZIP archive (download all)
// ============================================================================

const ZIP32_MAX_BYTES: u64 = u32::MAX as u64;

pub fn build_project_archive<F>(
    data_dir: &Path,
    project_name: &str,
    is_supported: F,
) -> Result<ProjectArchive, ArchiveError>
where
    F: Fn(&Path) -> bool,
{
    let mut entries = collect_image_entries(data_dir, &is_supported);
    if entries.is_empty() {
        return Err(ArchiveError::NoImages);
    }

    entries.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    let bytes = build_images_zip(&entries).map_err(ArchiveError::BuildFailed)?;
    let filename = format!("{}_images.zip", sanitize_filename_component(project_name));

    Ok(ProjectArchive { filename, bytes })
}

// ============================================================================
// Chunked download plan
// ============================================================================

pub const DEFAULT_PART_SIZE_MB: u64 = 256;
pub const MIN_PART_SIZE_MB: u64 = 16;
pub const MAX_PART_SIZE_MB: u64 = 1024;

#[derive(Debug, Serialize)]
pub struct DownloadPlanResponse {
    pub total_files: usize,
    pub total_bytes: u64,
    pub part_size_bytes: u64,
    pub part_count: usize,
    pub parts: Vec<DownloadPartInfo>,
}

#[derive(Debug, Serialize)]
pub struct DownloadPartInfo {
    pub index: usize,
    pub filename: String,
    pub file_count: usize,
    pub total_bytes: u64,
}

#[derive(Debug, Deserialize, Default)]
pub struct DownloadQuery {
    #[serde(default)]
    pub part_size_mb: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct PlannedDownloadPart {
    pub index: usize,
    pub filename: String,
    pub entries: Vec<ImageEntry>,
    pub total_bytes: u64,
}

#[derive(Debug)]
pub struct PlannedDownload {
    pub part_size_bytes: u64,
    pub total_files: usize,
    pub total_bytes: u64,
    pub parts: Vec<PlannedDownloadPart>,
}

#[must_use]
pub fn resolve_part_size_bytes(part_size_mb: Option<u64>) -> u64 {
    let mb = part_size_mb
        .unwrap_or(DEFAULT_PART_SIZE_MB)
        .clamp(MIN_PART_SIZE_MB, MAX_PART_SIZE_MB);
    mb.saturating_mul(1024 * 1024)
}

/// Build a chunked download plan from a data directory.
pub fn build_chunked_download_plan<F>(
    data_dir: &Path,
    project_name: &str,
    part_size_bytes: u64,
    is_supported: F,
) -> Result<PlannedDownload, ArchiveError>
where
    F: Fn(&Path) -> bool,
{
    let mut entries = collect_image_entries(data_dir, &is_supported);
    if entries.is_empty() {
        return Err(ArchiveError::NoImages);
    }
    entries.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));

    let total_files = entries.len();
    let total_bytes = entries.iter().map(|e| e.size_bytes).sum::<u64>();
    let partitions = partition_image_entries(entries, part_size_bytes);

    let base_name = sanitize_filename_component(project_name);
    let parts = partitions
        .into_iter()
        .enumerate()
        .map(|(index, part_entries)| {
            let total_bytes = part_entries.iter().map(|e| e.size_bytes).sum::<u64>();
            PlannedDownloadPart {
                index,
                filename: format!("{}_images_part{:03}.zip", base_name, index + 1),
                entries: part_entries,
                total_bytes,
            }
        })
        .collect();

    Ok(PlannedDownload {
        part_size_bytes,
        total_files,
        total_bytes,
        parts,
    })
}

/// Build a ZIP archive from a set of image entries.
pub fn build_images_zip(entries: &[ImageEntry]) -> Result<Vec<u8>, String> {
    let cursor = Cursor::new(Vec::<u8>::new());
    let mut zip = zip::ZipWriter::new(cursor);
    let base_options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Stored)
        .unix_permissions(0o644);

    for entry in entries {
        let options = base_options
            .clone()
            .large_file(requires_zip64(entry.size_bytes));
        zip.start_file(&entry.relative_path, options)
            .map_err(|e| format!("Failed to add '{}' to zip: {}", entry.relative_path, e))?;

        let mut input = std::fs::File::open(&entry.file_path).map_err(|e| {
            format!(
                "Failed to open '{}' from '{}': {}",
                entry.relative_path,
                entry.file_path.display(),
                e
            )
        })?;
        std::io::copy(&mut input, &mut zip)
            .map_err(|e| format!("Failed to write '{}' to zip: {}", entry.relative_path, e))?;
    }

    zip.finish()
        .map(|cursor| cursor.into_inner())
        .map_err(|e| format!("Failed to finalize zip: {}", e))
}

fn requires_zip64(size_bytes: u64) -> bool {
    size_bytes > ZIP32_MAX_BYTES
}

// ============================================================================
// Internal helpers
// ============================================================================

fn collect_image_entries<F>(base_dir: &Path, is_supported: &F) -> Vec<ImageEntry>
where
    F: Fn(&Path) -> bool,
{
    let mut entries = Vec::new();
    collect_image_entries_recursive(base_dir, base_dir, is_supported, &mut entries);
    entries
}

fn collect_image_entries_recursive<F>(
    dir: &Path,
    base_dir: &Path,
    is_supported: &F,
    out: &mut Vec<ImageEntry>,
) where
    F: Fn(&Path) -> bool,
{
    if let Ok(read_dir) = std::fs::read_dir(dir) {
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_image_entries_recursive(&path, base_dir, is_supported, out);
                continue;
            }
            if !(is_supported)(&path) {
                continue;
            }

            let relative = path
                .strip_prefix(base_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let size_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            out.push(ImageEntry {
                relative_path: relative,
                file_path: path,
                size_bytes,
            });
        }
    }
}

fn partition_image_entries(entries: Vec<ImageEntry>, part_size_bytes: u64) -> Vec<Vec<ImageEntry>> {
    let mut parts: Vec<Vec<ImageEntry>> = Vec::new();
    let mut current_part = Vec::new();
    let mut current_size = 0u64;

    for entry in entries {
        let entry_size = entry.size_bytes.max(1);
        let would_overflow =
            !current_part.is_empty() && current_size.saturating_add(entry_size) > part_size_bytes;
        if would_overflow {
            parts.push(current_part);
            current_part = Vec::new();
            current_size = 0;
        }

        current_size = current_size.saturating_add(entry_size);
        current_part.push(entry);
    }

    if !current_part.is_empty() {
        parts.push(current_part);
    }

    parts
}

fn sanitize_filename_component(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return "project".to_string();
    }

    let sanitized: String = trimmed
        .chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => ch,
        })
        .collect();

    if sanitized.trim().is_empty() {
        "project".to_string()
    } else {
        sanitized
    }
}

#[cfg(test)]
mod tests {
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
}
