//! Canonical dataset model for annotation interchange.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Stable image ID in the canonical dataset.
pub type DatasetImageId = u64;
/// Stable annotation ID in the canonical dataset.
pub type DatasetAnnotationId = u64;
/// Stable category ID in the canonical dataset.
pub type DatasetCategoryId = u64;

/// Dataset-level metadata.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DatasetInfo {
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub year: Option<u32>,
    #[serde(default)]
    pub contributor: String,
    #[serde(default)]
    pub date_created: String,
}

/// Canonical annotation dataset independent of UI or backend transport.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AnnotationDataset {
    #[serde(default)]
    pub info: DatasetInfo,
    #[serde(default)]
    pub images: Vec<ImageRecord>,
    #[serde(default)]
    pub categories: Vec<CategoryRecord>,
    #[serde(default)]
    pub annotations: Vec<AnnotationRecord>,
}

/// Image metadata record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageRecord {
    pub id: DatasetImageId,
    pub file_name: String,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub source_index: Option<usize>,
}

/// Category metadata record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CategoryRecord {
    pub id: DatasetCategoryId,
    pub name: String,
    #[serde(default)]
    pub color: Option<[u8; 3]>,
}

/// Annotation record with format-agnostic geometry and attributes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnnotationRecord {
    pub id: DatasetAnnotationId,
    pub image_id: DatasetImageId,
    pub category_id: DatasetCategoryId,
    pub geometry: Geometry,
    #[serde(default)]
    pub score: Option<f32>,
    #[serde(default)]
    pub attributes: BTreeMap<String, Value>,
}

/// Supported canonical geometry types.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Geometry {
    BoundingBox { x: f32, y: f32, w: f32, h: f32 },
    Polygon { points: Vec<(f32, f32)> },
    Point { x: f32, y: f32 },
}

/// Supported exchange formats.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FormatId {
    Coco = 1,
    Datumaro = 2,
    YoloDetect = 3,
    YoloSeg = 4,
}

/// Import/export capability flags for a format adapter.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormatCapabilities {
    pub supports_multi_image: bool,
    pub supports_bbox: bool,
    pub supports_polygon: bool,
    pub supports_point: bool,
    pub supports_attributes: bool,
}

/// A file emitted by an exporter or consumed by an importer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleFile {
    pub path: String,
    pub bytes: Vec<u8>,
    #[serde(default)]
    pub mime_type: Option<String>,
}

/// Exported bundle (single-file or multi-file).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExportBundle {
    #[serde(default)]
    pub files: Vec<BundleFile>,
}

/// Import bundle (single-file or multi-file).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImportBundle {
    #[serde(default)]
    pub files: Vec<BundleFile>,
}

impl ImportBundle {
    /// Find a file by exact path.
    pub fn file_by_path(&self, path: &str) -> Option<&BundleFile> {
        self.files.iter().find(|file| file.path == path)
    }

    /// Find the first file matching a suffix, e.g. ".json".
    pub fn first_by_suffix(&self, suffix: &str) -> Option<&BundleFile> {
        self.files.iter().find(|file| file.path.ends_with(suffix))
    }
}

/// Warning codes surfaced in I/O reports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IoWarningCode {
    UnsupportedShape,
    MissingImageDimensions,
    CategoryRemapped,
    InvalidPolygon,
    OutOfBoundsCoordinates,
    InvalidRecord,
}

/// Machine-readable warning with optional structured context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IoWarning {
    pub code: IoWarningCode,
    pub message: String,
    #[serde(default)]
    pub context: BTreeMap<String, Value>,
}

impl IoWarning {
    /// Convenience constructor for warning without context.
    pub fn new(code: IoWarningCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            context: BTreeMap::new(),
        }
    }
}

/// Aggregate report describing non-fatal import behavior.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IoReport {
    #[serde(default)]
    pub imported_images: usize,
    #[serde(default)]
    pub imported_annotations: usize,
    #[serde(default)]
    pub imported_categories: usize,
    #[serde(default)]
    pub skipped_annotations: usize,
    #[serde(default)]
    pub warnings: Vec<IoWarning>,
}

/// Import payload containing canonical dataset + report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImportResult {
    pub dataset: AnnotationDataset,
    pub report: IoReport,
}
