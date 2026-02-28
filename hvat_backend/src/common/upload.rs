use std::path::Path;

use axum::{extract::Multipart, http::StatusCode};
use tokio::io::AsyncWriteExt;
use uuid::Uuid;

pub const UPLOAD_FIELD_TARGET_ROOT: &str = "target_root";
pub const UPLOAD_FIELD_FILES: &str = "files";
pub const MAX_UPLOAD_FILES: usize = 2_000;
pub const MAX_UPLOAD_FILE_BYTES: usize = 512 * 1024 * 1024;
pub const MAX_UPLOAD_TOTAL_BYTES: usize = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct UploadedFiles {
    pub files: Vec<String>,
}

impl UploadedFiles {
    #[must_use]
    pub fn uploaded_count(&self) -> usize {
        self.files.len()
    }
}

pub fn build_upload_relative_path(
    target_root: &str,
    client_file_name: &str,
) -> std::result::Result<String, (StatusCode, String)> {
    let mut segments = split_upload_segments(target_root, true)?;
    segments.extend(split_upload_segments(client_file_name, false)?);

    if segments.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Upload path resolved to empty".to_string(),
        ));
    }

    Ok(segments.join("/"))
}

pub fn normalize_upload_path(
    raw: &str,
    allow_empty: bool,
) -> std::result::Result<String, (StatusCode, String)> {
    let segments = split_upload_segments(raw, allow_empty)?;
    Ok(segments.join("/"))
}

pub fn split_upload_segments(
    raw: &str,
    allow_empty: bool,
) -> std::result::Result<Vec<String>, (StatusCode, String)> {
    let normalized = raw.trim().replace('\\', "/");

    if normalized.is_empty() {
        if allow_empty {
            return Ok(Vec::new());
        }
        return Err((StatusCode::BAD_REQUEST, "Upload path is empty".to_string()));
    }

    if normalized.starts_with('/') {
        return Err((
            StatusCode::BAD_REQUEST,
            "Absolute upload paths are not allowed".to_string(),
        ));
    }

    let mut segments = Vec::new();
    for segment in normalized.split('/') {
        let part = segment.trim();
        if part.is_empty() || part == "." {
            continue;
        }
        if part == ".." {
            return Err((
                StatusCode::BAD_REQUEST,
                "Upload path contains parent traversal ('..')".to_string(),
            ));
        }
        if part.contains('\0') {
            return Err((
                StatusCode::BAD_REQUEST,
                "Upload path contains NUL byte".to_string(),
            ));
        }
        if part.contains(':') {
            return Err((
                StatusCode::BAD_REQUEST,
                "Upload path contains invalid ':' character".to_string(),
            ));
        }
        segments.push(part.to_string());
    }

    if !allow_empty && segments.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Upload path is empty".to_string()));
    }

    Ok(segments)
}

/// Save uploaded image files from multipart payload into `data_dir`.
///
/// Multipart fields:
/// - `target_root` (optional text field, must appear before files)
/// - `files` (one or more file fields)
///
/// The `is_supported` callback is used to validate file extensions/types.
pub async fn save_uploaded_images<F>(
    data_dir: &Path,
    mut multipart: Multipart,
    is_supported: F,
) -> std::result::Result<UploadedFiles, (StatusCode, String)>
where
    F: Fn(&Path) -> bool,
{
    let mut target_root = String::new();
    let mut total_bytes = 0usize;
    let mut file_count = 0usize;
    let mut uploaded_files = Vec::new();
    let mut seen_file_fields = false;

    while let Some(mut field) = multipart.next_field().await.map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid multipart payload: {err}"),
        )
    })? {
        match field.name() {
            Some(UPLOAD_FIELD_TARGET_ROOT) => {
                if seen_file_fields {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        "'target_root' must appear before file fields".to_string(),
                    ));
                }

                let raw_target = field.text().await.map_err(|err| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Failed to read target_root: {err}"),
                    )
                })?;
                target_root = normalize_upload_path(&raw_target, true)?;
            }
            Some(UPLOAD_FIELD_FILES) => {
                seen_file_fields = true;
                file_count += 1;
                if file_count > MAX_UPLOAD_FILES {
                    return Err((
                        StatusCode::PAYLOAD_TOO_LARGE,
                        format!("Upload exceeds maximum of {} files", MAX_UPLOAD_FILES),
                    ));
                }

                let client_file_name = field.file_name().ok_or_else(|| {
                    (
                        StatusCode::BAD_REQUEST,
                        "Each uploaded file must include a filename".to_string(),
                    )
                })?;
                let relative_path = build_upload_relative_path(&target_root, client_file_name)?;
                let destination = data_dir.join(&relative_path);

                if !is_supported(&destination) {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!("Unsupported file type: {}", relative_path),
                    ));
                }

                let parent = destination.parent().ok_or_else(|| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Invalid destination path: {}", relative_path),
                    )
                })?;

                tokio::fs::create_dir_all(parent).await.map_err(|err| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to create upload directory: {err}"),
                    )
                })?;

                let temp_name = format!(
                    ".{}.uploading-{}",
                    destination
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("upload"),
                    Uuid::new_v4(),
                );
                let temp_path = parent.join(temp_name);

                let mut output = tokio::fs::File::create(&temp_path).await.map_err(|err| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to create temp file '{}': {err}", relative_path),
                    )
                })?;

                let mut file_bytes = 0usize;
                while let Some(chunk) = field.chunk().await.map_err(|err| {
                    (
                        StatusCode::BAD_REQUEST,
                        format!("Failed to read upload chunk for '{}': {err}", relative_path),
                    )
                })? {
                    file_bytes = file_bytes.saturating_add(chunk.len());
                    total_bytes = total_bytes.saturating_add(chunk.len());

                    if file_bytes > MAX_UPLOAD_FILE_BYTES {
                        let _ = tokio::fs::remove_file(&temp_path).await;
                        return Err((
                            StatusCode::PAYLOAD_TOO_LARGE,
                            format!(
                                "File '{}' exceeds per-file limit ({} bytes)",
                                relative_path, MAX_UPLOAD_FILE_BYTES
                            ),
                        ));
                    }
                    if total_bytes > MAX_UPLOAD_TOTAL_BYTES {
                        let _ = tokio::fs::remove_file(&temp_path).await;
                        return Err((
                            StatusCode::PAYLOAD_TOO_LARGE,
                            format!(
                                "Upload exceeds total limit ({} bytes)",
                                MAX_UPLOAD_TOTAL_BYTES
                            ),
                        ));
                    }

                    output.write_all(&chunk).await.map_err(|err| {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed writing upload '{}': {err}", relative_path),
                        )
                    })?;
                }

                if file_bytes == 0 {
                    let _ = tokio::fs::remove_file(&temp_path).await;
                    return Err((
                        StatusCode::BAD_REQUEST,
                        format!("Uploaded file '{}' is empty", relative_path),
                    ));
                }

                output.flush().await.map_err(|err| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("Failed to flush upload '{}': {err}", relative_path),
                    )
                })?;
                drop(output);

                tokio::fs::rename(&temp_path, &destination)
                    .await
                    .map_err(|err| {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("Failed to finalize upload '{}': {err}", relative_path),
                        )
                    })?;

                uploaded_files.push(relative_path);
            }
            _ => {}
        }
    }

    if uploaded_files.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "No files were uploaded".to_string(),
        ));
    }

    Ok(UploadedFiles {
        files: uploaded_files,
    })
}
