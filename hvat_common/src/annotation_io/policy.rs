//! Import/export policy knobs.

use serde::{Deserialize, Serialize};

/// Which portion of app data should be exported.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum ExportScope {
    #[default]
    CurrentImage,
    WholeProject,
    SelectedImageIndices(Vec<usize>),
}

/// Import target behavior.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub enum ImportTargetMode {
    #[default]
    CurrentImage,
    WholeProject,
}

/// Category merge semantics when importing into existing state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CategoryMergePolicy {
    #[default]
    MatchById,
    MatchByName,
    ReplaceAll,
}

/// Annotation merge semantics when importing into existing state.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnnotationMergePolicy {
    #[default]
    ReplaceTarget,
    MergeAppend,
}

/// Policy for handling unsupported geometry during conversion.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeConversionPolicy {
    #[default]
    Strict,
    AllowLossyWarn,
    SkipUnsupported,
}

/// Export behavior controls.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExportOptions {
    #[serde(default)]
    pub scope: ExportScope,
    #[serde(default)]
    pub shape_policy: ShapeConversionPolicy,
    #[serde(default)]
    pub clamp_to_image_bounds: bool,
    #[serde(default)]
    pub include_empty_images: bool,
}

/// Import behavior controls.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImportOptions {
    #[serde(default)]
    pub target_mode: ImportTargetMode,
    #[serde(default)]
    pub category_policy: CategoryMergePolicy,
    #[serde(default)]
    pub annotation_policy: AnnotationMergePolicy,
    #[serde(default)]
    pub shape_policy: ShapeConversionPolicy,
}
