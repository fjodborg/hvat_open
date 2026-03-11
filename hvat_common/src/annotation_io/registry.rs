//! Runtime registry for annotation import/export adapters.

use std::collections::HashMap;
use std::sync::Arc;

use super::error::AnnotationIoError;
use super::policy::{ExportOptions, ImportOptions};
use super::traits::{AnnotationExporter, AnnotationFormat, AnnotationImporter};
use super::types::{
    AnnotationDataset, ExportBundle, FormatCapabilities, FormatId, ImportBundle, ImportResult,
};

/// Registry of importers and exporters keyed by format.
#[derive(Default)]
pub struct FormatRegistry {
    exporters: HashMap<FormatId, Arc<dyn AnnotationExporter>>,
    importers: HashMap<FormatId, Arc<dyn AnnotationImporter>>,
}

impl FormatRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an exporter for a format.
    pub fn register_exporter(&mut self, exporter: Arc<dyn AnnotationExporter>) {
        self.exporters.insert(exporter.format_id(), exporter);
    }

    /// Register an importer for a format.
    pub fn register_importer(&mut self, importer: Arc<dyn AnnotationImporter>) {
        self.importers.insert(importer.format_id(), importer);
    }

    /// Register one adapter that supports both import and export.
    pub fn register_format<T>(&mut self, format: T)
    where
        T: AnnotationFormat + 'static,
    {
        let adapter = Arc::new(format);
        self.register_exporter(adapter.clone());
        self.register_importer(adapter);
    }

    /// Dispatch export to the format adapter.
    pub fn export(
        &self,
        format_id: FormatId,
        dataset: &AnnotationDataset,
        options: &ExportOptions,
    ) -> Result<ExportBundle, AnnotationIoError> {
        let exporter = self
            .exporters
            .get(&format_id)
            .ok_or(AnnotationIoError::UnsupportedFormat(format_id))?;
        exporter.export(dataset, options)
    }

    /// Dispatch import to the format adapter.
    pub fn import(
        &self,
        format_id: FormatId,
        bundle: &ImportBundle,
        options: &ImportOptions,
    ) -> Result<ImportResult, AnnotationIoError> {
        let importer = self
            .importers
            .get(&format_id)
            .ok_or(AnnotationIoError::UnsupportedFormat(format_id))?;
        importer.import(bundle, options)
    }

    /// Return exportable formats.
    pub fn export_formats(&self) -> Vec<FormatId> {
        let mut ids = self.exporters.keys().copied().collect::<Vec<_>>();
        ids.sort_by_key(|id| *id as u8);
        ids
    }

    /// Return importable formats.
    pub fn import_formats(&self) -> Vec<FormatId> {
        let mut ids = self.importers.keys().copied().collect::<Vec<_>>();
        ids.sort_by_key(|id| *id as u8);
        ids
    }

    /// Return merged capabilities for a format if any adapter is registered.
    pub fn capabilities(&self, format_id: FormatId) -> Option<FormatCapabilities> {
        let exporter = self.exporters.get(&format_id).map(|e| e.capabilities());
        let importer = self.importers.get(&format_id).map(|i| i.capabilities());

        match (exporter, importer) {
            (None, None) => None,
            (Some(cap), None) | (None, Some(cap)) => Some(cap),
            (Some(a), Some(b)) => Some(FormatCapabilities {
                supports_multi_image: a.supports_multi_image || b.supports_multi_image,
                supports_bbox: a.supports_bbox || b.supports_bbox,
                supports_polygon: a.supports_polygon || b.supports_polygon,
                supports_point: a.supports_point || b.supports_point,
                supports_attributes: a.supports_attributes || b.supports_attributes,
            }),
        }
    }
}

#[cfg(test)]
#[path = "registry.test.rs"]
mod tests;
