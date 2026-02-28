use std::path::{Path, PathBuf};

use hvat_common::annotation_io::{ExportBundle, ImportBundle};

pub const MAX_PROJECT_STATE_BYTES: usize = 64 * 1024 * 1024;

const PROJECT_STATE_DIR: &str = ".hvat";
const PROJECT_STATE_FILE: &str = "project_annotations.hvatbundle.json";

#[must_use]
pub fn project_state_path(data_dir: &Path) -> PathBuf {
    data_dir.join(PROJECT_STATE_DIR).join(PROJECT_STATE_FILE)
}

#[must_use]
pub fn validate_project_state_payload(payload: &[u8]) -> bool {
    let import_valid = serde_json::from_slice::<ImportBundle>(payload)
        .map(|bundle| !bundle.files.is_empty())
        .unwrap_or(false);
    if import_valid {
        return true;
    }

    serde_json::from_slice::<ExportBundle>(payload)
        .map(|bundle| !bundle.files.is_empty())
        .unwrap_or(false)
}

pub async fn read_project_state(data_dir: &Path) -> std::io::Result<Vec<u8>> {
    let path = project_state_path(data_dir);
    tokio::fs::read(path).await
}

pub async fn write_project_state(data_dir: &Path, payload: &[u8]) -> std::io::Result<()> {
    let path = project_state_path(data_dir);
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }

    let temp_path = path.with_extension("tmp");
    tokio::fs::write(&temp_path, payload).await?;
    tokio::fs::rename(temp_path, path).await?;
    Ok(())
}
