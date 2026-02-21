//! Traits for pluggable annotation format adapters.

use super::error::AnnotationIoError;
use super::policy::{ExportOptions, ImportOptions};
use super::types::{
    AnnotationDataset, ExportBundle, FormatCapabilities, FormatId, ImportBundle, ImportResult,
};

/// Export adapter for one format.
pub trait AnnotationExporter: Send + Sync {
    fn format_id(&self) -> FormatId;

    fn export(
        &self,
        dataset: &AnnotationDataset,
        options: &ExportOptions,
    ) -> Result<ExportBundle, AnnotationIoError>;

    fn capabilities(&self) -> FormatCapabilities {
        FormatCapabilities::default()
    }
}

/// Import adapter for one format.
pub trait AnnotationImporter: Send + Sync {
    fn format_id(&self) -> FormatId;

    fn import(
        &self,
        bundle: &ImportBundle,
        options: &ImportOptions,
    ) -> Result<ImportResult, AnnotationIoError>;

    fn capabilities(&self) -> FormatCapabilities {
        FormatCapabilities::default()
    }
}

/// Combined format adapter implementing both directions.
pub trait AnnotationFormat: AnnotationExporter + AnnotationImporter {}

impl<T> AnnotationFormat for T where T: AnnotationExporter + AnnotationImporter {}
