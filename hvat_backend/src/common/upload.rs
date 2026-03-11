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

#[derive(Default)]
struct UploadSession {
    target_root: String,
    total_bytes: usize,
    file_count: usize,
    uploaded_files: Vec<String>,
    seen_file_fields: bool,
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
    let mut session = UploadSession::default();

    while let Some(mut field) = multipart.next_field().await.map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            format!("Invalid multipart payload: {err}"),
        )
    })? {
        match field.name() {
            Some(UPLOAD_FIELD_TARGET_ROOT) => {
                session.target_root =
                    parse_target_root_field(field, session.seen_file_fields).await?;
            }
            Some(UPLOAD_FIELD_FILES) => {
                session.seen_file_fields = true;
                session.file_count = session.file_count.saturating_add(1);
                enforce_file_count_limit(session.file_count)?;
                let relative_path =
                    save_uploaded_file_field(data_dir, &mut field, &is_supported, &mut session)
                        .await?;
                session.uploaded_files.push(relative_path);
            }
            _ => {}
        }
    }

    if session.uploaded_files.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "No files were uploaded".to_string(),
        ));
    }

    Ok(UploadedFiles {
        files: session.uploaded_files,
    })
}

async fn parse_target_root_field(
    field: axum::extract::multipart::Field<'_>,
    seen_file_fields: bool,
) -> std::result::Result<String, (StatusCode, String)> {
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
    normalize_upload_path(&raw_target, true)
}

fn enforce_file_count_limit(file_count: usize) -> std::result::Result<(), (StatusCode, String)> {
    if file_count > MAX_UPLOAD_FILES {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("Upload exceeds maximum of {} files", MAX_UPLOAD_FILES),
        ));
    }
    Ok(())
}

async fn save_uploaded_file_field<F>(
    data_dir: &Path,
    field: &mut axum::extract::multipart::Field<'_>,
    is_supported: &F,
    session: &mut UploadSession,
) -> std::result::Result<String, (StatusCode, String)>
where
    F: Fn(&Path) -> bool,
{
    let client_file_name = field.file_name().ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            "Each uploaded file must include a filename".to_string(),
        )
    })?;
    let relative_path = build_upload_relative_path(&session.target_root, client_file_name)?;
    let destination = data_dir.join(&relative_path);

    if !is_supported(&destination) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("Unsupported file type: {}", relative_path),
        ));
    }

    let temp_path = prepare_upload_temp_path(&destination, &relative_path).await?;
    let mut output = create_upload_output(&temp_path, &relative_path).await?;

    let file_bytes = stream_upload_chunks(
        field,
        &relative_path,
        &temp_path,
        &mut output,
        &mut session.total_bytes,
    )
    .await?;
    finalize_upload_file(file_bytes, &relative_path, &temp_path, &destination, output).await?;

    Ok(relative_path)
}

async fn prepare_upload_temp_path(
    destination: &Path,
    relative_path: &str,
) -> std::result::Result<std::path::PathBuf, (StatusCode, String)> {
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
    Ok(parent.join(temp_name))
}

async fn create_upload_output(
    temp_path: &Path,
    relative_path: &str,
) -> std::result::Result<tokio::fs::File, (StatusCode, String)> {
    tokio::fs::File::create(temp_path).await.map_err(|err| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Failed to create temp file '{}': {err}", relative_path),
        )
    })
}

async fn stream_upload_chunks(
    field: &mut axum::extract::multipart::Field<'_>,
    relative_path: &str,
    temp_path: &Path,
    output: &mut tokio::fs::File,
    total_bytes: &mut usize,
) -> std::result::Result<usize, (StatusCode, String)> {
    let mut file_bytes = 0usize;
    while let Some(chunk) = field.chunk().await.map_err(|err| {
        (
            StatusCode::BAD_REQUEST,
            format!("Failed to read upload chunk for '{}': {err}", relative_path),
        )
    })? {
        file_bytes = file_bytes.saturating_add(chunk.len());
        *total_bytes = total_bytes.saturating_add(chunk.len());

        if file_bytes > MAX_UPLOAD_FILE_BYTES {
            let _ = tokio::fs::remove_file(temp_path).await;
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "File '{}' exceeds per-file limit ({} bytes)",
                    relative_path, MAX_UPLOAD_FILE_BYTES
                ),
            ));
        }
        if *total_bytes > MAX_UPLOAD_TOTAL_BYTES {
            let _ = tokio::fs::remove_file(temp_path).await;
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
    Ok(file_bytes)
}

async fn finalize_upload_file(
    file_bytes: usize,
    relative_path: &str,
    temp_path: &Path,
    destination: &Path,
    mut output: tokio::fs::File,
) -> std::result::Result<(), (StatusCode, String)> {
    if file_bytes == 0 {
        let _ = tokio::fs::remove_file(temp_path).await;
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

    tokio::fs::rename(temp_path, destination)
        .await
        .map_err(|err| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to finalize upload '{}': {err}", relative_path),
            )
        })?;

    Ok(())
}
